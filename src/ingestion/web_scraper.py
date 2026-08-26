"""
Module de chargement de pages web via crawling récursif et sauvegarde dans un dossier local.
"""

import json
import os
import re
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_core.documents import Document


def clean_html(html: str) -> str:
    """
    Extrait le texte propre d'une page HTML, en retirant les scripts,
    les styles, et les balises de mise en forme qui polluent le contenu utile.
    """
    # Initialise BeautifulSoup pour analyser et manipuler l'arbre HTML de la page
    soup = BeautifulSoup(html, "html.parser")

    # Supprime définitivement les éléments HTML structurels, interactifs ou publicitaires
    # qui n'apportent aucune valeur sémantique au modèle d'IA (LLM)
    #for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
    for tag in soup(["style"]):
        tag.decompose()  # Découpe et détruit la balise ainsi que tout son contenu

    # Extrait tout le texte restant en forçant un saut de ligne entre chaque bloc textuel
    text = soup.get_text(separator="\n")

    # Nettoyage final : sépare le texte par ligne, retire les espaces superflus (trim),
    # et ignore complètement les lignes vides pour condenser l'information.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Rassemble les lignes nettoyées en un seul bloc textuel continu séparé par des sauts de ligne
    return "\n".join(lines)




def load_website(base_url: str,max_depth: int = 2,exclude_dirs: list[str] | None = None,) -> list[Document]:
    """
    Explore (crawl) un site web à partir d'une URL racine et retourne les pages
    découvertes sous forme d'une liste d'objets 'Document' LangChain.
    """
    # Configuration du chargeur récursif de LangChain
    loader = RecursiveUrlLoader(
        url=base_url,                  # Le point de départ de l'exploration du site
        max_depth=max_depth,          # Nombre maximum de clics successifs à suivre depuis l'accueil
        extractor=clean_html,         # La fonction de nettoyage (définie plus haut) appliquée à chaque page
        prevent_outside=True,         # Reste strictement sur le domaine initial (interdit de suivre des liens externes)
        exclude_dirs=exclude_dirs or [], # Chemins ou sous-dossiers spécifiques à ignorer (ex: ['/admin', '/private'])
        use_async=True,               # Active le téléchargement asynchrone (parallèle) pour accélérer le processus
        timeout=10,                   # Temps d'attente maximum en secondes par page avant d'abandonner
    )

    # Déclenche le crawling et télécharge toutes les pages correspondantes
    documents = loader.load()

    # Boucle sur chaque document récupéré pour enrichir sa structure de métadonnées
    for doc in documents:
        # Ajoute le type de source pour pouvoir filtrer plus tard par 'web', 'pdf', ou 'database' dans le RAG
        doc.metadata["source_type"] = "web"

        # Standardise la clé contenant l'URL d'origine de la page pour le référencement des sources
        doc.metadata["source_file"] = doc.metadata.get("source", base_url)

    print(f"✓ {len(documents)} page(s) web chargée(s) depuis {base_url}")

    # Retourne la liste finale des documents prêts à être découpés (chunking)
    return documents


def save_documents_to_folder(documents: list[Document], folder_path: str = "data/processed"):
    """
    Crée un dossier et sauvegarde chaque document dans un fichier JSON individuel.
    """
    # Crée le dossier s'il n'existe pas déjà
    
    os.makedirs(folder_path, exist_ok=True)

    for index, doc in enumerate(documents):
        # Récupère l'URL de la page
        url = doc.metadata.get("source", f"page_{index}")

        # Nettoie l'URL pour en faire un nom de fichier valide (remplace les caractères spéciaux par _)
        safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', url)

        # Limite la longueur du nom de fichier pour éviter les erreurs système
        safe_filename = safe_filename[-150:] + ".json"
        
        # Construit le chemin complet du fichier (ex: mon_dossier/https___site_com_page.json)
        file_path = os.path.join(folder_path, safe_filename)
        
        # Prépare les données à sauvegarder
        data_to_save = {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        
        # Écrit le fichier JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            
    print(f"📁 Tous les documents ({len(documents)}) ont été stockés dans le dossier : '{folder_path}'")
