"""
Module de recherche : interroge la base vectorielle Chroma
pour trouver les chunks les plus pertinents par rapport à une question.
"""

from pathlib import Path
from langchain_core.documents import Document

import sys
sys.path.append(str(Path(__file__).parent.parent))
from vectorstore.indexer import get_vectorstore

# Nombre de chunks à récupérer par défaut
TOP_K = 5


def search_similar_chunks(query: str, k: int = TOP_K) -> list[Document]:
    """
    Recherche les k chunks les plus proches sémantiquement de la question.
    Retourne les Document complets (texte + métadonnées), sans les scores.
    """
    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search(query, k=k)

    print(f"✓ {len(results)} chunk(s) trouvé(s) pour la requête : \"{query}\"")

    return results


def search_with_scores(query: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """
    Même recherche, mais retourne aussi le score de similarité pour chaque chunk.
    Utile pour debug/évaluation : voir à quel point chaque résultat est pertinent.
    Score = distance (plus bas = plus proche/pertinent, avec Chroma en cosine).
    """
    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_score(query, k=k)

    for doc, score in results:
        source = doc.metadata.get("source_file", "inconnu")
        print(f"  [score={score:.4f}] {source} — {doc.page_content[:80]}...")

    return results


def format_context(chunks: list[Document]) -> str:
    """
    Assemble les chunks récupérés en un seul bloc de texte,
    prêt à être injecté dans le prompt du LLM (generation/prompts.py).
    """
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source_file", "source inconnue")
        context_parts.append(f"[Extrait {i} — source: {source}]\n{chunk.page_content}")

    return "\n\n".join(context_parts)    