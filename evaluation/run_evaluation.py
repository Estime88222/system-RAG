"""
evaluation/run_evaluation.py

Point d'entrée unique de l'évaluation du pipeline RAG.

Lance :
1. L'évaluation du retrieval seul (Precision@k, Recall@k, MRR, NDCG@k)
   via retrieval_eval.py, sur evaluation/test_set.json.
2. L'évaluation du pipeline complet (retrieval + génération) via RAGAS
   (faithfulness, answer_relevancy, context_precision, context_recall)
   via ragas_eval.py, sur evaluation/ragas_test_set.json.

Puis affiche un rapport consolidé dans la console et l'exporte en Markdown.

Usage :
    python -m evaluation.run_evaluation
    python -m evaluation.run_evaluation --k 10
    python -m evaluation.run_evaluation --skip-ragas
    python -m evaluation.run_evaluation --skip-retrieval
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ragas.dataset_schema import EvaluationResult

from evaluation.retrieval_eval import run_evaluation as run_retrieval_evaluation
from evaluation.ragas_eval import run_ragas_evaluation

EVALUATION_DIR = Path(__file__).parent
DEFAULT_RETRIEVAL_TEST_SET = EVALUATION_DIR / "test_set.json"
DEFAULT_RAGAS_TEST_SET = EVALUATION_DIR / "ragas_test_set.json"
REPORTS_DIR = EVALUATION_DIR / "reports"

# Colonnes non numériques présentes dans le DataFrame RAGAS (question,
# réponse, contextes, etc.) — à exclure du calcul des scores moyens.
RAGAS_NON_METRIC_COLUMNS = {"question", "answer", "contexts", "ground_truth"}


def parse_args() -> argparse.Namespace:
    """Lit les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Évaluation du pipeline RAG")
    parser.add_argument(
        "--k", type=int, default=5, help="Nombre de chunks considérés (défaut : 5)"
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Ne pas lancer l'évaluation du retrieval seul",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Ne pas lancer l'évaluation RAGAS (pipeline complet)",
    )
    parser.add_argument(
        "--retrieval-test-set",
        type=Path,
        default=DEFAULT_RETRIEVAL_TEST_SET,
        help="Chemin vers le jeu de test du retrieval",
    )
    parser.add_argument(
        "--ragas-test-set",
        type=Path,
        default=DEFAULT_RAGAS_TEST_SET,
        help="Chemin vers le jeu de test RAGAS",
    )
    return parser.parse_args()


def run_retrieval_step(test_set_path: Path, k: int) -> Optional[Dict[str, float]]:
    """
    Lance l'évaluation du retrieval seul, avec gestion d'erreur pour ne pas
    bloquer le reste du rapport si ce jeu de test n'existe pas encore.

    Args:
        test_set_path: chemin vers le jeu de test du retrieval.
        k: nombre de chunks considérés.

    Returns:
        Dict des métriques, ou None si l'étape a échoué.
    """
    if not test_set_path.exists():
        print(f"⚠️  Jeu de test retrieval introuvable : {test_set_path} — étape ignorée.")
        return None

    print(f"\n▶ Évaluation du retrieval (k={k})...")
    try:
        results = run_retrieval_evaluation(test_set_path, k=k)
    except Exception as exc:
        print(f"❌ Échec de l'évaluation du retrieval : {exc}")
        return None

    for metric_name, score in results.items():
        print(f"  {metric_name}: {score:.3f}")
    return results


def run_ragas_step(test_set_path: Path, k: int) -> Optional[EvaluationResult]:
    """
    Lance l'évaluation RAGAS du pipeline complet, avec gestion d'erreur
    pour ne pas bloquer le reste du rapport.

    Args:
        test_set_path: chemin vers le jeu de test RAGAS.
        k: nombre de chunks considérés.

    Returns:
        Le résultat RAGAS (EvaluationResult), ou None si l'étape a échoué
        ou a été ignorée.
    """
    if not test_set_path.exists():
        print(f"⚠️  Jeu de test RAGAS introuvable : {test_set_path} — étape ignorée.")
        return None

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  Variable DEEPSEEK_API_KEY manquante — étape RAGAS ignorée.")
        return None

    print("\n▶ Évaluation RAGAS (pipeline complet)...")
    try:
        result = run_ragas_evaluation(test_set_path, deepseek_api_key=api_key, k=k)
    except Exception as exc:
        print(f"❌ Échec de l'évaluation RAGAS : {exc}")
        return None

    print(result)
    return result


def extract_ragas_mean_scores(result: EvaluationResult) -> Dict[str, float]:
    """
    Extrait les scores moyens par métrique à partir d'un EvaluationResult,
    en passant par le DataFrame détaillé (une ligne par question).

    Args:
        result: résultat retourné par run_ragas_evaluation().

    Returns:
        Dict {nom_métrique: score_moyen}, limité aux colonnes numériques
        de métriques (question/answer/contexts/ground_truth exclues).
    """
    df = result.to_pandas()
    metric_columns = [col for col in df.columns if col not in RAGAS_NON_METRIC_COLUMNS]
    return {col: float(df[col].mean()) for col in metric_columns}


def build_markdown_report(
    retrieval_results: Optional[Dict[str, float]],
    ragas_result: Optional[EvaluationResult],
    k: int,
) -> str:
    """
    Construit un rapport Markdown consolidé à partir des deux évaluations.

    Args:
        retrieval_results: résultats de run_retrieval_step(), ou None.
        ragas_result: résultat de run_ragas_step(), ou None.
        k: valeur de k utilisée pour cette évaluation.

    Returns:
        Le contenu Markdown du rapport, sous forme de chaîne de caractères.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# Rapport d'évaluation RAG — {timestamp}", ""]

    lines.append(f"## Retrieval (k={k})")
    if retrieval_results:
        lines.append("")
        lines.append("| Métrique | Score |")
        lines.append("|---|---|")
        for metric_name, score in retrieval_results.items():
            lines.append(f"| {metric_name} | {score:.3f} |")
    else:
        lines.append("\n_Non évalué (jeu de test manquant ou erreur)._")

    lines.append("\n## RAGAS (pipeline complet)")
    if ragas_result:
        ragas_scores = extract_ragas_mean_scores(ragas_result)
        lines.append("")
        lines.append("| Métrique | Score |")
        lines.append("|---|---|")
        for metric_name, score in ragas_scores.items():
            lines.append(f"| {metric_name} | {score:.3f} |")
    else:
        lines.append("\n_Non évalué (jeu de test manquant, clé API absente, ou erreur)._")

    return "\n".join(lines)


def save_markdown_report(content: str) -> Path:
    """
    Sauvegarde le rapport Markdown dans evaluation/reports/, avec un nom
    de fichier horodaté pour garder un historique des évaluations.

    Args:
        content: contenu Markdown du rapport.

    Returns:
        Le chemin du fichier créé.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"eval_report_{timestamp}.md"
    report_path.write_text(content, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()

    retrieval_results = None
    ragas_result = None

    if not args.skip_retrieval:
        retrieval_results = run_retrieval_step(args.retrieval_test_set, k=args.k)

    if not args.skip_ragas:
        ragas_result = run_ragas_step(args.ragas_test_set, k=args.k)

    report = build_markdown_report(retrieval_results, ragas_result, k=args.k)
    report_path = save_markdown_report(report)

    print(f"\n📄 Rapport consolidé enregistré dans : {report_path}")


if __name__ == "__main__":
    main()