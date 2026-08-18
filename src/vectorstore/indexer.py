"""
Module d'indexation : insère les chunks (avec leurs embeddings) dans ChromaDB,
base vectorielle locale embarquée — aucun serveur externe requis.
"""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document

import sys
sys.path.append(str(Path(__file__).parent.parent))
from ingestion.embedder import get_embedder

PERSIST_DIRECTORY = "data/processed" #"database/chroma_db" 
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

