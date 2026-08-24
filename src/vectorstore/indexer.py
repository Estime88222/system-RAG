"""
Module d'indexation : insère les chunks (avec leurs embeddings) dans ChromaDB,
base vectorielle locale embarquée — aucun serveur externe requis.
"""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingestion.embedder import get_embedder

import sys
sys.path.append(str(Path(__file__).parent.parent))
from ingestion.embedder import get_embedder

PERSIST_DIRECTORY = "data/processed" 
COLLECTION_NAME = "documents_rag"


def get_vectorstore() -> Chroma:
    """
    Retourne l'instance Chroma connectée au dossier de persistance local.
    Crée le dossier automatiquement à la première insertion si besoin.
    """
    embedder = get_embedder()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=PERSIST_DIRECTORY,
    )


def index_chunks(chunks: list[Document]) -> Chroma:
    """
    Insère une liste de chunks dans la base vectorielle locale.
    """
    vectorstore = get_vectorstore()

    ids = [f"chunk_{chunk.metadata.get('chunk_id', i)}_{chunk.metadata.get('source_file', 'unknown')}"
           for i, chunk in enumerate(chunks)]

    vectorstore.add_documents(documents=chunks, ids=ids)

    print(f"✓ {len(chunks)} chunk(s) indexé(s) dans '{COLLECTION_NAME}' ({PERSIST_DIRECTORY})")

    return vectorstore


def get_collection_count() -> int:
    """Retourne le nombre de vecteurs actuellement stockés."""
    vectorstore = get_vectorstore()
    return vectorstore._collection.count()

def is_first_run() -> bool:
    """
    Vérifie si la base vectorielle est vide (aucun document indexé).
    Retourne True si c'est le premier lancement (base vide), False sinon.
    """
    try:
        count = get_collection_count()
        return count == 0
    except Exception:
        # Si la collection n'existe même pas encore, c'est bien un premier lancement
        return True


def index_if_needed(chunks: list[Document], force: bool= False) -> Chroma:
    """
    N'indexe les chunks QUE si la base est actuellement vide.
    Évite de ré-indexer à chaque lancement du programme.
    """
    if is_first_run():
        print("→ Base vectorielle vide, indexation en cours...")
        return index_chunks(chunks)
    else:
        print(f"→ Base déjà indexée ({get_collection_count()} vecteur(s)), indexation ignorée.")
        return get_vectorstore()