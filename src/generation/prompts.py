"""
Templates de prompts utilisés pour la génération de réponses.
Centralise ici tout le texte envoyé au LLM, pour pouvoir l'ajuster
sans toucher à la logique de llm_client.py.
"""

from openai.types.chat import ChatCompletionMessageParam

# Prompt système : définit le comportement général de l'assistant
SYSTEM_PROMPT = """Tu es un assistant qui répond aux questions en te basant uniquement sur le contexte fourni.

Règles à respecter strictement :
- Réponds uniquement à partir des informations présentes dans le contexte ci-dessous.
- Si l'information n'est pas dans le contexte, dis clairement que tu ne sais pas plutôt que d'inventer une réponse.
- Ne fais aucune supposition au-delà de ce qui est écrit dans le contexte.
- Cite la source (nom du document) quand c'est pertinent.
- Réponds de façon claire et concise, en français."""


def build_user_prompt(context: str, question: str) -> str:
    """
    Construit le prompt utilisateur final, combinant le contexte récupéré
    (via retrieval/search.py) et la question posée.
    """
    return f"""Contexte :
{context}

Question : {question}"""


def build_messages(context: str, question: str) -> list[ChatCompletionMessageParam]:
    """
    Construit la liste de messages complète au format attendu par l'API
    (system + user), prête à être envoyée à llm_client.py.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(context, question)},
    ]

