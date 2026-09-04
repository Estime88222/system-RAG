"""
Module de recherche : interroge la base vectorielle Chroma
pour trouver les chunks les plus pertinents par rapport à une question.
"""

import json
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


def _parse_serialized_metadata(value):
    """Restaure les champs sérialisés JSON en structures Python utilisables."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def format_navigation_from_chunks(chunks: list[Document]) -> str:
    """
    Extrait et formate les informations de navigation à partir des métadonnées des chunks.
    Crée une section NAVIGATION_INFO que le LLM peut utiliser pour guider les utilisateurs.
    """
    nav_info = None
    page_title = None
    page_type = None

    for chunk in chunks:
        meta = chunk.metadata
        page_structure = _parse_serialized_metadata(meta.get("page_structure"))
        if isinstance(page_structure, dict) and not nav_info:
            page_title = str(page_structure.get("title", ""))
            page_type = str(page_structure.get("type", ""))

        if not nav_info:
            sections = _parse_serialized_metadata(meta.get("sections", []))
            nav_menu = _parse_serialized_metadata(meta.get("nav_menu", []))
            internal_links = _parse_serialized_metadata(meta.get("internal_links", []))
            cta_buttons = _parse_serialized_metadata(meta.get("cta_buttons", []))
            breadcrumbs = _parse_serialized_metadata(meta.get("breadcrumbs", []))

            nav_info = {
                "sections": sections if isinstance(sections, list) else [],
                "nav_menu": nav_menu if isinstance(nav_menu, list) else [],
                "internal_links": internal_links if isinstance(internal_links, list) else [],
                "cta_buttons": cta_buttons if isinstance(cta_buttons, list) else [],
                "breadcrumbs": breadcrumbs if isinstance(breadcrumbs, list) else [],
            }

            if nav_info.get("nav_menu"):
                break

    if not nav_info or not any(nav_info.values()):
        return ""

    nav_parts: list[str] = []

    if page_type:
        nav_parts.append(f"📄 TYPE DE PAGE: {page_type}")

    if page_title:
        nav_parts.append(f"📌 TITRE: {page_title}")

    sections = nav_info.get("sections", [])
    if isinstance(sections, list) and sections:
        nav_parts.append("\n📋 STRUCTURE:")
        for section in sections[:5]:
            if not isinstance(section, dict):
                continue
            level = int(section.get("level", 2) or 2)
            heading = str(section.get("heading", ""))
            indent = "  " * max(level - 2, 0)
            nav_parts.append(f"{indent}→ {heading}")

    nav_menu = nav_info.get("nav_menu", [])
    if isinstance(nav_menu, list) and nav_menu:
        nav_parts.append("\n🔗 MENU PRINCIPAL:")
        for item in nav_menu[:6]:
            if not isinstance(item, dict):
                continue
            nav_parts.append(f"  • {item.get('text', '')}")

    cta_buttons = nav_info.get("cta_buttons", [])
    if isinstance(cta_buttons, list) and cta_buttons:
        nav_parts.append("\n🎯 ACTIONS:")
        for btn in cta_buttons[:4]:
            if not isinstance(btn, dict):
                continue
            nav_parts.append(f"  • {btn.get('text', '')}")

    internal_links = nav_info.get("internal_links", [])
    if isinstance(internal_links, list) and internal_links:
        nav_parts.append("\n🔀 PAGES CONNEXES:")
        for link in internal_links[:5]:
            if not isinstance(link, dict):
                continue
            nav_parts.append(f"  • {link.get('text', '')}")

    return "\n".join(nav_parts)

def get_branding_context(k: int = 3) -> str:
    """" Récupère systématiquement quelques chunks du document de branding,
    indépendamment de la question posée — pour garantir que le ton
    de marque soit toujours présent dans le contexte du LLM.
    """
    vectorstore = get_vectorstore()
    # CORRECTION : utiliser doc_type (underscore) pour correspondre à la métadonnée définie dans ingetion_branding.py
    results = vectorstore.similarity_search(
        query="ton style identité marque guideline", 
        k=k, 
        filter={"doc_type": "BRAND_BOOK_TARA_2026-08-17_v2.5.pdf"}
    )
    if not results:
        print("⚠️ Aucun chunk du brand-book trouvé. Vérifiez que ingetion_branding.py a été exécuté.")
        return ""
    return format_context(results)