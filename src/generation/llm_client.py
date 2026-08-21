"""
Client pour l'API Groq Cloud (Gratuit, compatible format OpenAI).
Gère l'appel au LLM pour la génération finale de réponse dans le pipeline RAG.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from prompts import build_messages

load_dotenv()

# Récupération de la clé Groq (gsk_...) dans votre fichier .env
GROK_API_KEY = os.getenv("GROK_API_KEY")

# CORRECTION : L'URL exacte
#  de l'API de Groq pour la compatibilité OpenAI
GROK_BASE_URL = "https://api.groq.com/openai/v1"

# Le modèle Meta performant et gratuit disponible sur Groq
MODEL = "openai/gpt-oss-120b"


def get_client() -> OpenAI:
    if not GROK_API_KEY:
        raise ValueError("Clé GROQ_API_KEY manquante dans le fichier .env")
    
    print(f"-> Clé API détectée par le script : {GROK_API_KEY[:8]} ...")
    
    return OpenAI(
        api_key=GROK_API_KEY,
        base_url=GROK_BASE_URL,
    )


def generate_answer(context: str, question: str) -> str:
    """
    Génère une réponse à partir du contexte récupéré (RAG) et de la question utilisateur.
    """
    client = get_client()
    messages = build_messages(context, question)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        stream=False,
    )
    
    # Syntaxe standard pour récupérer le contenu textuel de la réponse
    content = response.choices[0].message.content
    return content if content is not None else "aucune réponse générée"
