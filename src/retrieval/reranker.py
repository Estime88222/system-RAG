"""
Module de reranking : affine les résultats de la recherche vectorielle
en utilisant un cross-encoder, plus précis mais plus lent qu'une similarité vectorielle simple.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# Modèle cross-encoder léger et gratuit, tourne en local (CPU ok)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None  # chargé une seule fois (coûteux à recharger à chaque appel)


def get_reranker() -> CrossEncoder:
    """Charge le modèle une seule fois et le réutilise (singleton simple)."""
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL)
    return _model


def rerank_chunks(query: str, chunks: list[Document], top_n: int = 5) -> list[Document]:
    """
    Reclasse une liste de chunks par pertinence réelle vis-à-vis de la question,
    et ne garde que les top_n meilleurs.
    """
    if not chunks:
        return []

    reranker = get_reranker()

    # Le cross-encoder évalue chaque paire (question, chunk) individuellement
    pairs = [[query, chunk.page_content] for chunk in chunks]
    scores = reranker.predict(pairs)

    # Trie les chunks par score décroissant
    scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    top_chunks = [chunk for chunk, score in scored_chunks[:top_n]]

    print(f"✓ Reranking : {len(chunks)} chunk(s) → {len(top_chunks)} conservé(s)")
    for chunk, score in scored_chunks[:top_n]:
        source = chunk.metadata.get("source_file", "inconnu")
        print(f"  [score={score:.4f}] {source}")

    return top_chunks
