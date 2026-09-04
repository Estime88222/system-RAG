"""
Module d'indexation : insère les chunks (avec leurs embeddings) dans ChromaDB,
base vectorielle locale embarquée — aucun serveur externe requis.
"""

import json
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


def _serialize_metadata_value(value):
    """Convertit les structures complexes en types compatibles ChromaDB."""
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple)):
        normalized = [_serialize_metadata_value(item) for item in value]
        if not normalized:
            return None
        if all(isinstance(item, (str, int, float, bool, type(None))) for item in normalized):
            return normalized
        return json.dumps(normalized, ensure_ascii=False)

    if isinstance(value, dict):
        if not value:
            return None
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    return str(value)


def sanitize_metadata_for_chroma(metadata: dict) -> dict:
    """Nettoie les métadonnées pour qu'elles soient acceptées par ChromaDB."""
    if not isinstance(metadata, dict):
        return {}

    sanitized = {}
    for key, value in metadata.items():
        cleaned = _serialize_metadata_value(value)
        if cleaned is None:
            continue
        sanitized[key] = cleaned
    return sanitized


def index_chunks(chunks: list[Document]) -> Chroma:
    """
    Insère une liste de chunks dans la base vectorielle locale.
    """
    vectorstore = get_vectorstore()

    cleaned_chunks = []
    for i, chunk in enumerate(chunks):
        clean_metadata = sanitize_metadata_for_chroma(chunk.metadata)
        cleaned_chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata=clean_metadata,
            )
        )

    ids = [f"chunk_{chunk.metadata.get('chunk_id', i)}_{chunk.metadata.get('source_file', 'unknown')}"
           for i, chunk in enumerate(chunks)]

    vectorstore.add_documents(documents=cleaned_chunks, ids=ids)

    print(f"✓ {len(chunks)} chunk(s) indexé(s) dans '{COLLECTION_NAME}' ({PERSIST_DIRECTORY})")

    return vectorstore


def get_collection_count() -> int:
    """Retourne le nombre de vecteurs actuellement stockés."""
    vectorstore = get_vectorstore()
    return vectorstore._collection.count()


def get_indexed_chunk_keys() -> set[tuple[str, str]]:
    """Retourne les clés déjà présentes en base pour identifier les chunks déjà indexés."""
    vectorstore = get_vectorstore()
    try:
        result = vectorstore._collection.get(include=["metadatas"])
    except Exception:
        return set()

    metadatas = result.get("metadatas") or []
    indexed = set()
    for meta in metadatas:
        if not isinstance(meta, dict):
            continue
        source = str(meta.get("source_file") or meta.get("source") or "unknown")
        chunk_id = str(meta.get("chunk_id") or "")
        indexed.add((source, chunk_id))
    return indexed


def get_new_chunks_to_index(chunks: list[Document]) -> list[Document]:
    """Filtre uniquement les chunks qui ne sont pas déjà présents dans ChromaDB."""
    if not chunks:
        return []

    indexed = get_indexed_chunk_keys()
    new_chunks = []
    for chunk in chunks:
        source = str(chunk.metadata.get("source_file") or chunk.metadata.get("source") or "unknown")
        chunk_id = str(chunk.metadata.get("chunk_id") or "")
        if (source, chunk_id) not in indexed:
            new_chunks.append(chunk)
    return new_chunks


def is_first_run() -> bool:
    """
    Ancienne logique conservée pour compatibilité, mais ne doit plus être utilisée
    comme décision d'indexation. On préfèrera `get_new_chunks_to_index()`.
    """
    try:
        return get_collection_count() == 0
    except Exception:
        return True


def index_if_needed(chunks: list[Document], force: bool = False) -> Chroma:
    """
    N'indexe que les chunks nouveaux par rapport à la base existante.
    Si force=True, ré-indexe tout le lot.
    """
    if force:
        print("→ Force indexation activée, tous les chunks seront réindexés.")
        return index_chunks(chunks)

    new_chunks = get_new_chunks_to_index(chunks)
    if not new_chunks:
        print(f"→ Aucun nouveau chunk détecté ({len(chunks)} chunk(s) déjà présent(s) dans la base), indexation ignorée.")
        return get_vectorstore()

    print(f"→ {len(new_chunks)} nouveau(x) chunk(s) détecté(s), indexation en cours...")
    return index_chunks(new_chunks)