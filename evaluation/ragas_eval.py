"""
evaluation/ragas_eval.py

Évalue le pipeline RAG complet (retrieval + génération) avec RAGAS.

Contrairement à retrieval_eval.py (qui n'évalue que le retriever),
ce module évalue la chaîne complète : question -> contexte récupéré
-> réponse générée par le LLM.

Métriques RAGAS utilisées :
- faithfulness       : la réponse est-elle fidèle aux contextes récupérés
                        (pas d'hallucination) ?
- answer_relevancy   : la réponse répond-elle bien à la question posée ?
- context_precision  : les contextes récupérés sont-ils pertinents ?
- context_recall     : le retriever a-t-il récupéré tout ce qu'il fallait
                        (nécessite un ground_truth) ?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, cast

# Permet d'importer les modules à la racine du projet (search, generation),
# comme le fait déjà search/retrieval.py dans ce projet.
sys.path.append(str(Path(__file__).parent.parent))

from datasets import Dataset
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.retrieval.search import search_similar_chunks, format_context

# ---------------------------------------------------------------------------
# Point d'adaptation : à ajuster selon l'implémentation réelle du module
# de génération du projet (ex. generation/generator.py ou similaire).
# On suppose ici une fonction generate_answer(question, context) -> str.
# ---------------------------------------------------------------------------
try:
    from src.generation.llm_client import generate_answer
except ImportError:
    generate_answer = None


# ---------------------------------------------------------------------------
# Configuration du LLM juge (DeepSeek, via l'API compatible OpenAI)
# et des embeddings juge (Ollama, réutilisés depuis le pipeline existant)
# ---------------------------------------------------------------------------
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


def get_judge_llm(api_key: str) -> ChatOpenAI:
    """
    Construit le LLM juge utilisé par RAGAS pour noter les réponses.

    RAGAS utilise OpenAI par défaut ; on le remplace ici par DeepSeek,
    qui expose une API compatible.

    Args:
        api_key: clé API DeepSeek.

    Returns:
        Une instance ChatOpenAI configurée pour pointer vers DeepSeek.
    """
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=SecretStr(api_key),
        base_url=DEEPSEEK_API_BASE,
    )


def get_judge_embeddings() -> OllamaEmbeddings:
    """
    Construit le modèle d'embeddings utilisé par RAGAS pour certaines
    métriques (ex. answer_relevancy). On réutilise le même modèle Ollama
    que le pipeline de production, pour rester cohérent.

    Returns:
        Une instance OllamaEmbeddings.
    """
    return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)


def load_ragas_test_set(path: str | Path) -> List[dict]:
    """
    Charge le jeu de test pour l'évaluation RAGAS.

    Format attendu dans le fichier JSON (liste d'objets) :
    [
        {
            "question": "Comment configurer le MQTT broker ?",
            "ground_truth": "Réponse de référence rédigée à la main."
        },
        ...
    ]

    Args:
        path: chemin vers le fichier JSON du jeu de test.

    Returns:
        Liste de dicts {"question": ..., "ground_truth": ...}.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_on_test_set(test_set: List[dict], k: int = 5) -> Dict[str, list]:
    """
    Fait tourner le pipeline complet (retrieval + génération) sur chaque
    question du jeu de test, et construit les listes nécessaires au
    Dataset RAGAS.

    Args:
        test_set: jeu de test chargé via load_ragas_test_set().
        k: nombre de chunks à récupérer par question.

    Returns:
        Un dict avec les clés "question", "answer", "contexts", "ground_truth",
        chacune associée à une liste alignée par index.
    """
    if generate_answer is None:
        raise RuntimeError(
            "Impossible d'importer generate_answer depuis generation.generator. "
            "Adapte l'import en haut de ce fichier selon ton implémentation réelle."
        )

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for item in test_set:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        chunks = search_similar_chunks(question, k=k)
        context_text = format_context(chunks)
        answer = generate_answer(question, context_text)

        questions.append(question)
        answers.append(answer)
        contexts_list.append([chunk.page_content for chunk in chunks])
        ground_truths.append(ground_truth)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }


def run_ragas_evaluation(
    test_set_path: str | Path,
    deepseek_api_key: str,
    k: int = 5,
) -> EvaluationResult:
    """
    Pipeline complet : charge le jeu de test, fait tourner le pipeline RAG,
    construit le Dataset RAGAS, et calcule les 4 métriques.

    Args:
        test_set_path: chemin vers le fichier JSON du jeu de test RAGAS.
        deepseek_api_key: clé API DeepSeek utilisée comme LLM juge.
        k: nombre de chunks à récupérer par question.

    Returns:
        Un objet EvaluationResult (scores moyens accessibles via
        result.to_pandas(), résultats détaillés par question inclus).
    """
    test_set = load_ragas_test_set(test_set_path)
    pipeline_output = run_pipeline_on_test_set(test_set, k=k)
    dataset = Dataset.from_dict(pipeline_output)

    judge_llm = get_judge_llm(deepseek_api_key)
    judge_embeddings = get_judge_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    # evaluate() peut typer son retour comme EvaluationResult | Executor selon
    # les stubs de ragas ; en usage synchrone standard (notre cas), c'est
    # toujours un EvaluationResult.
    return cast(EvaluationResult, result)


if __name__ == "__main__":
    import os

    TEST_SET_PATH = Path(__file__).parent / "ragas_test_set.json"
    K = 5

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Variable d'environnement DEEPSEEK_API_KEY manquante. "
            "Définis-la avant de lancer ce script."
        )

    result = run_ragas_evaluation(TEST_SET_PATH, deepseek_api_key=api_key, k=K)

    print("\nRésultats de l'évaluation RAGAS :\n")
    print(result)

    df = result.to_pandas()
    output_csv = Path(__file__).parent / "ragas_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nRésultats détaillés exportés dans : {output_csv}")