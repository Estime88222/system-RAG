"""
Point d'entrée principal : orchestre le pipeline RAG complet.
Question → Recherche → Contexte (+ Navigation) → Génération → Réponse
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from retrieval.search import (
    search_similar_chunks,
    format_context,
    get_branding_context,
    format_navigation_from_chunks,
)
from generation.llm_client import generate_answer
from ingestion.ingestionpipeline import run_ingestion


def ask(question: str, top_k: int = 5) -> str:
    """
    Fonction principale du RAG : prend une question, retourne une réponse
    ancrée dans les documents indexés.
    
    Pipeline:
    1. Recherche des chunks pertinents
    2. Extraction du contexte de contenu
    3. Extraction des infos de navigation pour guider l'utilisateur
    4. Récupération du brand-book pour le ton
    5. Construction du contexte enrichi
    6. Génération de la réponse
    """
    # 1. Récupérer un large groupe de candidats avant le reclassement.
    candidate_k = max(top_k * 3, 15)
    chunks = search_similar_chunks(question, k=candidate_k)

    try:
        from retrieval.reranker import rerank_chunks
        chunks = rerank_chunks(question, chunks, top_n=top_k)
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"⚠️ Reranking indisponible, recherche vectorielle utilisée : {exc}")
        chunks = chunks[:top_k]

    if not chunks:
        return "Je n'ai trouvé aucune information pertinente dans ma base de connaissances."

    # 2. Formatage du contenu
    content_context = format_context(chunks)

    # 3. Extraction des infos de navigation (pour guider l'utilisateur)
    navigation_context = format_navigation_from_chunks(chunks)
    
    # 4. Récupération du brand-book (ton de marque)
    branding_context = get_branding_context(k=3)

    # 5. Construction du contexte enrichi.
    # Le contenu métier est prioritaire; le brand-book ne sert qu'à régler le ton.
    full_context = "=== CONTENU PERTINENT ===\n"
    full_context += content_context

    if navigation_context:
        full_context += "\n\n=== INFORMATIONS DE NAVIGATION ===\n"
        full_context += navigation_context

    if branding_context:
        full_context += "\n\n=== RÈGLES DE TON DE MARQUE, À APPLIQUER EN DERNIER ===\n"
        full_context += branding_context

    # 6. Génération de la réponse via LLM
    answer = generate_answer(full_context, question)

    return answer


if __name__ == "__main__":

    run_ingestion()

    question = input("Pose ta question : ")
    reponse = ask(question)
    print(f"\n--- Réponse ---\n{reponse}")