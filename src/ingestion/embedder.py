"""
Module de génération des embeddings à partir des chunks.
Utilise l'API OpenAI (text-embedding-3-small) via langchain-openai.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Modèle d'embedding OpenAI
# "text-embedding-3-small" = bon rapport coût/performance, 1536 dimensions
EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedder() -> OpenAIEmbeddings:
    """
    Retourne l'objet embedder configuré.
    Centralise la config ici pour ne la changer qu'à un seul endroit
    si tu veux changer de modèle plus tard.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY manquante dans le fichier .env")

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
    )


def embed_chunks(chunks: list[Document]) -> list[list[float]]:
    """
    Génère les vecteurs pour une liste de chunks.
    Retourne une liste de vecteurs (un par chunk), dans le même ordre.
    """
    embedder = get_embedder()
    texts = [chunk.page_content for chunk in chunks]

    vectors = embedder.embed_documents(texts)

    print(f"✓ {len(vectors)} embedding(s) généré(s), dimension = {len(vectors[0]) if vectors else 0}")

    return vectors


def embed_query(query: str) -> list[float]:
    """
    Génère l'embedding d'une question utilisateur (au moment de la recherche).
    """
    embedder = get_embedder()
    return embedder.embed_query(query)
