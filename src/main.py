"""
Point d'entrée principal : orchestre le pipeline RAG complet.
Question → Recherche → Contexte → Génération DeepSeek → Réponse
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from retrieval.search import search_similar_chunks, format_context
from generation.llm_client import generate_answer


def ask(question: str, top_k: int = 5) -> str:
    """
    Fonction principale du RAG : prend une question, retourne une réponse
    ancrée dans les documents indexés.
    """
    # 1. Recherche des chunks pertinents
    chunks = search_similar_chunks(question, k=top_k)

    if not chunks:
        return "Je n'ai trouvé aucune information pertinente dans ma base de connaissances."

    # 2. Formatage du contexte
    context = format_context(chunks)

    # 3. Génération de la réponse via DeepSeek
    answer = generate_answer(context, question)

    return answer


if __name__ == "__main__":
    question = input("Pose ta question : ")
    reponse = ask(question)
    print(f"\n--- Réponse ---\n{reponse}")