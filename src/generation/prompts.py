"""
Templates de prompts utilisés pour la génération de réponses.
Centralise ici tout le texte envoyé au LLM, pour pouvoir l'ajuster
sans toucher à la logique de llm_client.py.

Enrichi avec guidance de navigation : le LLM peut maintenant orienter les utilisateurs
vers les pages/sections pertinentes pour trouver les informations.
"""

from openai.types.chat import ChatCompletionMessageParam
import json

# Prompt système : définit le comportement général de l'assistant

SYSTEM_PROMPT = """Tu es Tara, une assistante qui répond uniquement à partir du contexte fourni.

SOURCES DU CONTEXTE :
- Le CONTENU MÉTIER fournit la réponse principale.
- Le BRAND-BOOK est une source de vérité pour l'identité, les valeurs, la mission,
  les engagements et le ton de Tara.
- Les INFORMATIONS DE NAVIGATION indiquent où trouver une information ou quelle
  action effectuer sur le site.

RÈGLES DE RÉPONSE :
- Réponds d'abord directement à la question.
- Utilise toutes les sources pertinentes du contexte.
- Le contenu métier et le brand-book peuvent constituer le contenu principal selon la question.
- La navigation doit être intégrée naturellement lorsqu'elle apporte une information utile.
- Ne laisse pas la navigation remplacer la réponse métier.
- Ne transforme pas systématiquement la réponse en guide du site.
- Si une page, un lien ou une action est pertinent, ajoute-le après l'explication principale.
- Si la question demande où trouver une information ou comment effectuer une action,
  la navigation peut devenir une partie importante de la réponse.
- Si la navigation n'est pas utile, ne la mentionne pas.
- Si l'information est absente, dis clairement que tu ne sais pas.
- N'invente aucune information.
- Réponds dans la langue de la question.
- En cas de contradiction entre les sources, signale-la clairement.

FORMAT RECOMMANDÉ :
1. Réponse directe et explication principale.
2. Informations de navigation uniquement si elles sont pertinentes.
3. Action ou lien utile, si disponible.

TOUJOURS : reste ancré dans le contexte fourni."""


def build_user_prompt(context: str, question: str) -> str:
    return f"""Voici les sources disponibles pour répondre.

Le contenu métier et le brand-book servent à construire la réponse.
La navigation sert à compléter la réponse lorsqu'elle est pertinente,
sans devenir automatiquement son format principal.

=== CONTEXTE ===
{context}

QUESTION: {question}

Réponds d'abord au fond de la question, puis ajoute une indication de navigation
seulement si elle apporte une réelle valeur."""


def format_navigation_info(navigation_data: dict) -> str:
    """
    Formate les données de navigation en texte lisible pour le LLM.
    
    Prend un dictionnaire de navigation et le transforme en description textuelle
    que le LLM peut utiliser pour guider les utilisateurs.
    
    Args:
        navigation_data: dict contenant sections, nav_menu, internal_links, cta_buttons, breadcrumbs
    
    Returns:
        str: Description formatée de la navigation
    """
    nav_text = []
    
    # 1. Structure de la page (sections)
    if navigation_data.get("sections"):
        nav_text.append("\n📋 STRUCTURE DE LA PAGE:")
        for section in navigation_data["sections"][:5]:  # Limiter à 5 sections
            indent = "  " * (section.get("level", 2) - 1)
            nav_text.append(f"{indent}→ {section.get('heading', 'Section')}")
    
    # 2. Menu de navigation principal
    if navigation_data.get("nav_menu"):
        nav_text.append("\n🔗 MENU PRINCIPAL:")
        for item in navigation_data["nav_menu"][:6]:  # Limiter à 6 items
            nav_text.append(f"  • {item.get('text', 'Lien')} — {item.get('url', '#')}")
    
    # 3. Fil d'Ariane
    if navigation_data.get("breadcrumbs"):
        breadcrumb_path = " > ".join([b.get('text', '') for b in navigation_data["breadcrumbs"]])
        nav_text.append(f"\n🗂️ CHEMIN ACTUEL: {breadcrumb_path}")
    
    # 4. Boutons d'appel à l'action
    if navigation_data.get("cta_buttons"):
        nav_text.append("\n🎯 ACTIONS DISPONIBLES:")
        for btn in navigation_data["cta_buttons"][:4]:  # Limiter à 4 boutons
            nav_text.append(f"  • [{btn.get('text', 'Action')}] → {btn.get('url', '#')}")
    
    # 5. Liens internes pertinents
    if navigation_data.get("internal_links"):
        nav_text.append("\n🔀 PAGES CONNEXES:")
        for link in navigation_data["internal_links"][:5]:  # Limiter à 5 liens
            nav_text.append(f"  • {link.get('text', 'Lien')} → {link.get('url', '#')}")
    
    return "\n".join(nav_text) if nav_text else ""


def extract_navigation_from_context(context: str) -> dict:
    """
    Extrait les informations de navigation du contexte si elles sont présentes
    (elles peuvent être formatées en tant que section spéciale dans le contexte).
    
    Args:
        context: Le contexte complet incluant potentiellement une section NAVIGATION_INFO
    
    Returns:
        dict: Informations de navigation ou dictionnaire vide si non trouvées
    """
    # Chercher une section marquée comme NAVIGATION_INFO
    if "NAVIGATION_INFO" in context:
        try:
            # Extraire la section entre les marqueurs
            start = context.find("[NAVIGATION_INFO]")
            end = context.find("[/NAVIGATION_INFO]")
            if start != -1 and end != -1:
                nav_json_str = context[start + len("[NAVIGATION_INFO]"):end].strip()
                return json.loads(nav_json_str)
        except (json.JSONDecodeError, ValueError):
            pass
    
    return {}


def build_messages(context: str, question: str) -> list[ChatCompletionMessageParam]:
    """
    Construit la liste de messages complète au format attendu par l'API
    (system + user), prête à être envoyée à llm_client.py.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(context, question)},
    ]

