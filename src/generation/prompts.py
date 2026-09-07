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
SYSTEM_PROMPT = """Tu es Tara, une assistante qui répond aux questions en te basant uniquement sur le contexte fourni.
Tu es aussi un guide de navigation : tu aides les utilisateurs à trouver les informations et à naviguer le site.

RÈGLES DE BASE - À RESPECTER STRICTEMENT :
- Réponds uniquement à partir des informations présentes dans le contexte ci-dessous.
- Si l'information n'est pas dans le contexte, dis clairement que tu ne sais pas plutôt que d'inventer une réponse.
- Ne fais aucune supposition au-delà de ce qui est écrit dans le contexte.
- Cite la source (nom du document) quand c'est pertinent.
- Réponds de façon claire et concise.
- Réponds dans la langue du contexte fourni.
- Commence par une réponse directe, puis donne le chemin ou les étapes utiles.
- Ne cite un bouton, une page, une URL ou une condition que s'il apparaît dans le contexte.
- Si plusieurs pages sont présentes, utilise d'abord celle qui répond le plus directement à la question.
- Si le contexte est insuffisant ou contradictoire, signale-le clairement et demande une précision.

**DIRECTIVES DE TON DE MARQUE (TARA) - À RESPECTER ABSOLUMENT :**
Les premiers extraits du contexte ci-dessous sont des guidelines du brand-book TARA. 
Ces guidelines définissent le ton, la voix, et les valeurs de la marque.
Tu DOIS adapter ta réponse pour respecter strictement ces directives :
- Imite le ton : le style de communication, le niveau de formalité, les expressions privilégiées
- Respecte les valeurs : les principes éthiques et les priorités de la marque
- Utilise la même voix narrative et les mêmes termes clés que dans le brand-book
- Assure-toi que chaque phrase de ta réponse est en harmonie avec l'identité de la marque TARA.

**GUIDANCE DE NAVIGATION - POUR AIDER LES UTILISATEURS :**
Le contexte ci-dessous inclut des INFORMATIONS DE NAVIGATION pour chaque page du site.
Utilise ces informations pour :

1. ORIENTER : Si l'utilisateur cherche quelque chose, indique-lui où aller
   Exemple: "Vous trouverez les tarifs dans la section Services > Tarifs"
   
2. EXPLIQUER : Décris le type de page et sa structure
   Exemple: "C'est la page d'accueil, elle présente nos services principaux"
   
3. PROPOSER : Suggère les actions pertinentes (CTAs - boutons, formulaires)
   Exemple: "Cliquez sur 'Demander une démo' pour commencer"
   
4. MONTRER LE CHEMIN : Utilise le fil d'Ariane et les menus
   Exemple: "Depuis le menu principal, allez à Services > Offres Premium"

FORMAT DES INFOS DE NAVIGATION :
Si le contexte inclut une section "NAVIGATION_INFO", elle contient:
- "sections": la structure H1 > H2 > H3 de la page
- "nav_menu": le menu principal du site
- "internal_links": les liens vers d'autres pages pertinentes
- "cta_buttons": les boutons/appels à l'action importants
- "breadcrumbs": le chemin de navigation (fil d'Ariane)
- "page_type": le type de page (accueil, services, contact, tarifs, etc.)

QUAND INCLURE LES INFOS DE NAVIGATION:
- Si l'utilisateur demande "où trouver X" → cite le nav_menu et les internal_links
- Si l'utilisateur demande "qu'est-ce que cette page" → cite le page_type et les sections
- Si l'utilisateur demande "comment faire Y" → cite les cta_buttons pertinents
- Si l'utilisateur est perdu → cite les breadcrumbs et le nav_menu

TOUJOURS: Reste ancré dans le contexte, ne fabrique pas de liens ou de sections qui n'existent pas."""


def build_user_prompt(context: str, question: str) -> str:
    """
    Construit le prompt utilisateur final, combinant le contexte récupéré
    (via retrieval/search.py) et la question posée.
    """
    return f"""Contexte :
{context}

Question : {question}"""


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

