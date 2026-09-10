"""
evaluation/retrieval_eval.py

Connecte le retriever réel du projet (search_similar_chunks, basé sur
LangChain + Chroma) aux métriques définies dans metrics.py.

Comme les chunks retournés par le retriever n'ont pas d'ID unique dans
leurs métadonnées, on génère un ID stable par hash du contenu du chunk
(page_content). Deux chunks avec exactement le même texte auront le même
ID — c'est voulu, ça reflète le fait qu'ils sont interchangeables pour
l'évaluation.

Ce module :
1. Charge un jeu de test (questions + documents pertinents attendus, identifiés
   par le même hash de contenu).
2. Interroge le retriever pour chaque question.
3. Calcule les métriques (Precision@k, Recall@k, MRR, NDCG@k).
4. Retourne / affiche un rapport.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

# Permet d'importer les modules à la racine du projet (search),
# comme le fait déjà search/retrieval.py dans ce projet.
sys.path.append(str(Path(__file__).parent.parent))

from langchain_core.documents import Document

from evaluation.metrics import evaluate_retrieval
from src.retrieval.search import search_similar_chunks


def chunk_id(chunk: Document) -> str:
    """
    Génère un ID stable pour un chunk à partir d'un hash de son contenu.

    Utiliser le contenu (plutôt que la position ou l'objet Python) garantit
    que le même chunk aura toujours le même ID d'une exécution à l'autre,
    ce qui est indispensable pour comparer avec le jeu de test annoté.

    Args:
        chunk: un objet Document (LangChain) retourné par le retriever.

    Returns:
        Un hash SHA-256 tronqué (16 caractères) du texte du chunk.
    """
    content = chunk.page_content.strip()
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def get_retrieved_ids(query: str, k: int = 5) -> List[str]:
    """
    Interroge le retriever réel et retourne la liste ordonnée des IDs
    (hash de contenu) des chunks renvoyés, du plus au moins pertinent.

    Args:
        query: la question posée.
        k: nombre de chunks à récupérer.

    Returns:
        Liste ordonnée d'IDs de chunks (hash de contenu).
    """
    chunks = search_similar_chunks(query, k=k)
    return [chunk_id(chunk) for chunk in chunks]


def load_test_set(path: str | Path) -> List[dict]:
    """
    Charge le jeu de test depuis un fichier JSON.

    Format attendu dans le fichier JSON (liste d'objets) :
    [
        {
            "question": "Comment configurer le MQTT broker ?",
            "relevant_docs": ["a1b2c3d4e5f6...", "9f8e7d6c5b4a..."],
            "relevance_scores": {"a1b2c3d4e5f6...": 3, "9f8e7d6c5b4a...": 2}
        },
        ...
    ]

    Les IDs dans "relevant_docs" doivent être des hash de contenu générés
    avec chunk_id() (voir generate_test_set_template() pour construire ce
    fichier à partir de chunks réels).

    Le champ "relevance_scores" est optionnel.

    Args:
        path: chemin vers le fichier JSON du jeu de test.

    Returns:
        Liste de dicts, avec "relevant_docs" converti en set (attendu par metrics.py).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw_test_set = json.load(f)

    test_set = []
    for item in raw_test_set:
        test_set.append(
            {
                "question": item["question"],
                "relevant_docs": set(item["relevant_docs"]),
                "relevance_scores": item.get("relevance_scores", {}),
            }
        )
    return test_set


def run_retrieval_on_test_set(test_set: List[dict], k: int = 5) -> List[dict]:
    """
    Fait tourner le retriever sur chaque question du jeu de test et complète
    chaque item avec les chunks effectivement retrouvés (sous forme d'IDs).

    Args:
        test_set: jeu de test chargé via load_test_set().
        k: nombre de chunks à récupérer par question.

    Returns:
        Le même test_set, avec un champ "retrieved_docs" ajouté à chaque item.
    """
    enriched_test_set = []
    for item in test_set:
        retrieved = get_retrieved_ids(item["question"], k=k)
        enriched_item = dict(item)
        enriched_item["retrieved_docs"] = retrieved
        enriched_test_set.append(enriched_item)
    return enriched_test_set


def run_evaluation(test_set_path: str | Path, k: int = 5) -> Dict[str, float]:
    """
    Pipeline complet : charge le jeu de test, interroge le retriever,
    calcule les métriques.

    Args:
        test_set_path: chemin vers le fichier JSON du jeu de test.
        k: nombre de chunks considérés pour les métriques.

    Returns:
        Dict des métriques moyennées (Precision@k, Recall@k, MRR, NDCG@k).
    """
    test_set = load_test_set(test_set_path)
    test_set = run_retrieval_on_test_set(test_set, k=k)
    return evaluate_retrieval(test_set, k=k)


def generate_test_set_template(
    questions: List[str],
    k: int = 10,
    output_path: str | Path = "evaluation/test_set_template.json",
) -> None:
    """
    Utilitaire d'aide à l'annotation : pour une liste de questions, interroge
    le retriever et génère un fichier JSON pré-rempli avec les chunks trouvés
    (texte + ID) afin que tu puisses ensuite cocher manuellement lesquels sont
    réellement pertinents, sans avoir à chercher les IDs toi-même.

    Le fichier généré n'est PAS directement utilisable par load_test_set() :
    c'est une base de travail à éditer (retirer les chunks non pertinents,
    ajouter des relevance_scores si besoin).

    Args:
        questions: liste des questions de test.
        k: nombre de chunks candidats à proposer par question.
        output_path: chemin où écrire le template JSON.
    """
    template = []
    for question in questions:
        chunks = search_similar_chunks(question, k=k)
        candidates = [
            {
                "id": chunk_id(chunk),
                "source_file": chunk.metadata.get("source_file", "inconnu"),
                "preview": chunk.page_content[:150],
            }
            for chunk in chunks
        ]
        template.append({"question": question, "candidates": candidates})

    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"Template généré : {output_path}")
    print("Édite ce fichier pour ne garder que les chunks pertinents par question,")
    print("puis convertis-le au format attendu par load_test_set().")


if __name__ == "__main__":
    TEST_SET_PATH = Path(__file__).parent / "test_set.json"
    K = 5

    results = run_evaluation(TEST_SET_PATH, k=K)

    print(f"\nRésultats de l'évaluation du retrieval (k={K}) :\n")
    for metric_name, score in results.items():
        print(f"  {metric_name}: {score:.3f}")