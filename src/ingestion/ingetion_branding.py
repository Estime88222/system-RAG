"""Script d'indexation dédié au  document de branding.
Ajoute un flag doc_type dans les métadonnées pour pouvoir le récupérer spécifiquement depuis retrieval/search.py.

A lancer une seule fois """

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.loader import load_single_document
from ingestion.splitter import split_documents
from vectorstore.indexer import index_chunks, get_vectorstore

BRANDING_DIR = "data/raw/branding/BRAND_BOOK_TARA_2026-08-17_v2.5.pdf"  # dossier contenant le document de branding

def ingest_branding_document(file_path: str= BRANDING_DIR) -> None:
    print(f"=== Indexation du document de branding : {file_path} ===")

    docs = load_single_document(file_path)

    # C'est ICI que le flag est ajouté — sur ce document précis, pas sur les autres
    for doc in docs:
        doc.metadata["doc_type"] = "BRAND_BOOK_TARA_2026-08-17_v2.5.pdf"

    chunks = split_documents(docs)

    # On force l'indexation (même si la base n'est pas vide,
    # car ce document doit être ajouté même après le premier lancement)
    index_chunks(chunks)

    print(f"✓ Document de branding indexé ({len(chunks)} chunk(s))")

if __name__ == "__main__":
    ingest_branding_document()