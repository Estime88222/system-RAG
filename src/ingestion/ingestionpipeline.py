"""
Orchestration complète de la phase d'ingestion :
chargement des documents -> découpage -> indexation vectorielle.
C'est le SEUL point d'entrée à utiliser pour ingérer des données dans le RAG,
que ce soit des fichiers locaux ou du contenu web.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.loader import load_document_from_directory
from ingestion.splitter import split_documents
from vectorstore.indexer import index_if_needed, index_chunks


def run_ingestion(
    raw_dir: str = "data/raw",
    force: bool = False,
) -> None:
    """
    Exécute le pipeline d'ingestion complet.

    - raw_dir : dossier contenant les documents à ingérer
    - force : si True, réindexe même si la base contient déjà des données
    """
    print("=== Démarrage de l'ingestion ===")

    # 1. Chargement
    docs = load_document_from_directory(raw_dir)
    if not docs:
        print("Aucun document trouvé, ingestion annulée.")
        return

    # 2. Découpage
    chunks = split_documents(docs)

    # 3. Indexation (uniquement si nécessaire, sauf si force=True)
    index_if_needed(chunks, force=force)

    print("=== Ingestion terminée ===")


def run_web_ingestion(url: str, max_depth: int = 2) -> None:
    """
    Pipeline d'ingestion dédié au contenu web (utilise web_loader.py).
    Toujours indexé directement (pas de vérification is_first_run,
    car on veut pouvoir ajouter du contenu web à volonté).
    """
    from ingestion.web_scraper import load_website

    print(f"=== Ingestion web depuis {url} ===")

    docs = load_website(url, max_depth=max_depth)
    if not docs:
        print("Aucune page trouvée, ingestion annulée.")
        return

    chunks = split_documents(docs)
    index_chunks(chunks)

    print("=== Ingestion web terminée ===")


if __name__ == "__main__":
    run_ingestion()