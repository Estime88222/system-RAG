"""
evaluation/metrics.py

Métriques d'évaluation du retrieval pour un pipeline RAG :
- Precision@k
- Recall@k
- MRR (Mean Reciprocal Rank)
- NDCG@k (Normalized Discounted Cumulative Gain)

Ces fonctions sont volontairement pures (pas de dépendance à Chroma, FastAPI, etc.)
pour rester testables isolément et réutilisables ailleurs.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """
    Proportion de documents pertinents parmi les k premiers documents renvoyés.

    Precision@k = (nb de pertinents dans le top k) / k

    Args:
        retrieved: liste ordonnée des IDs de documents renvoyés par le retriever.
        relevant: ensemble des IDs de documents réellement pertinents (vérité terrain).
        k: nombre de documents considérés (top k).

    Returns:
        Un score entre 0.0 et 1.0. Retourne 0.0 si top_k est vide.
    """
    if k <= 0:
        return 0.0

    top_k = retrieved[:k]
    if not top_k:
        return 0.0

    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """
    Proportion de documents pertinents (sur le total existant) retrouvés
    dans le top k.

    Recall@k = (nb de pertinents dans le top k) / (nb total de pertinents)

    Args:
        retrieved: liste ordonnée des IDs de documents renvoyés par le retriever.
        relevant: ensemble des IDs de documents réellement pertinents (vérité terrain).
        k: nombre de documents considérés (top k).

    Returns:
        Un score entre 0.0 et 1.0. Retourne 0.0 si relevant est vide.
    """
    if not relevant or k <= 0:
        return 0.0

    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str]) -> float:
    """
    Inverse de la position du premier document pertinent trouvé.

    Reciprocal Rank = 1 / position_du_premier_pertinent

    Args:
        retrieved: liste ordonnée des IDs de documents renvoyés par le retriever.
        relevant: ensemble des IDs de documents réellement pertinents (vérité terrain).

    Returns:
        Un score entre 0.0 et 1.0. Retourne 0.0 si aucun document pertinent
        n'est trouvé dans retrieved.
    """
    for position, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / position
    return 0.0


def dcg_at_k(retrieved: Sequence[str], relevance_scores: Dict[str, float], k: int) -> float:
    """
    Discounted Cumulative Gain sur les k premiers résultats.

    DCG@k = somme( pertinence(doc_i) / log2(i + 1) ) pour i de 1 à k

    Args:
        retrieved: liste ordonnée des IDs de documents renvoyés par le retriever.
        relevance_scores: dict {doc_id: score de pertinence}. Un doc absent
            du dict est considéré comme non pertinent (score 0).
        k: nombre de documents considérés (top k).

    Returns:
        La valeur du DCG (non normalisée, peut dépasser 1.0).
    """
    if k <= 0:
        return 0.0

    top_k = retrieved[:k]
    dcg = 0.0
    for position, doc in enumerate(top_k, start=1):
        relevance = relevance_scores.get(doc, 0)
        dcg += relevance / math.log2(position + 1)
    return dcg


def ndcg_at_k(retrieved: Sequence[str], relevance_scores: Dict[str, float], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain sur les k premiers résultats.

    NDCG@k = DCG@k / IDCG@k
    où IDCG@k est le DCG obtenu avec un classement idéal (documents triés
    par pertinence décroissante).

    Args:
        retrieved: liste ordonnée des IDs de documents renvoyés par le retriever.
        relevance_scores: dict {doc_id: score de pertinence}. Les scores peuvent
            être binaires (0/1) ou gradués (ex: 0 à 3).
        k: nombre de documents considérés (top k).

    Returns:
        Un score entre 0.0 et 1.0. Retourne 0.0 si aucun document pertinent
        n'existe (IDCG = 0).
    """
    if k <= 0:
        return 0.0

    dcg = dcg_at_k(retrieved, relevance_scores, k)

    ideal_order = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(
        relevance / math.log2(position + 1)
        for position, relevance in enumerate(ideal_order, start=1)
    )

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(test_set: List[dict], k: int = 5) -> Dict[str, float]:
    """
    Calcule les 4 métriques moyennées sur un jeu de test complet.

    Chaque élément de test_set doit être un dict avec au minimum :
        - "retrieved_docs": List[str]  (IDs renvoyés par le retriever, ordonnés)
        - "relevant_docs": Set[str]    (IDs réellement pertinents)
    et optionnellement :
        - "relevance_scores": Dict[str, float]  (pour un NDCG gradué ;
          si absent, on considère pertinent = 1, non pertinent = 0)

    Args:
        test_set: liste de questions de test avec leurs résultats de retrieval.
        k: nombre de documents considérés pour Precision@k, Recall@k, NDCG@k.

    Returns:
        Un dict avec les 4 métriques moyennées :
        {"Precision@k": ..., "Recall@k": ..., "MRR": ..., "NDCG@k": ...}
    """
    if not test_set:
        return {
            f"Precision@{k}": 0.0,
            f"Recall@{k}": 0.0,
            "MRR": 0.0,
            f"NDCG@{k}": 0.0,
        }

    precisions: List[float] = []
    recalls: List[float] = []
    reciprocal_ranks: List[float] = []
    ndcgs: List[float] = []

    for item in test_set:
        retrieved = item["retrieved_docs"]
        relevant = item["relevant_docs"]
        relevance_scores = item.get(
            "relevance_scores", {doc: 1 for doc in relevant}
        )

        precisions.append(precision_at_k(retrieved, relevant, k))
        recalls.append(recall_at_k(retrieved, relevant, k))
        reciprocal_ranks.append(reciprocal_rank(retrieved, relevant))
        ndcgs.append(ndcg_at_k(retrieved, relevance_scores, k))

    n = len(test_set)
    return {
        f"Precision@{k}": sum(precisions) / n,
        f"Recall@{k}": sum(recalls) / n,
        "MRR": sum(reciprocal_ranks) / n,
        f"NDCG@{k}": sum(ndcgs) / n,
    }


if __name__ == "__main__":
    # Petit exemple d'utilisation / vérification manuelle
    example_test_set = [
        {
            "retrieved_docs": ["doc_12", "doc_3", "doc_45", "doc_9", "doc_100"],
            "relevant_docs": {"doc_12", "doc_45", "doc_78"},
            "relevance_scores": {"doc_12": 3, "doc_45": 2, "doc_78": 1},
        },
    ]

    results = evaluate_retrieval(example_test_set, k=5)
    for metric_name, score in results.items():
        print(f"{metric_name}: {score:.3f}")