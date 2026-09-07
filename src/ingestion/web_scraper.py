"""
Module de chargement de pages web via crawling récursif et sauvegarde dans un dossier local.
Enrichi pour capturer la navigation et la structure des pages (pour aide à la navigation).
"""

import json
import os
import re
import requests
from urllib.parse import urljoin, urlparse
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


NOM_HTML_EXTENSION =(".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp4", ".mp3", ".avi", ".mov", ".webm",
    ".csv", ".json", ".xml", ".txt",
    )


def is_scrapable_url(url: str, include_authenticated: bool = True) -> bool:
    """Vérifie si une URL est exploitable pour le scraping public ou authentifié."""
    if not url:
        return False

    parsed = urlparse(url)
    path = parsed.path.lower()

    if path.endswith(NOM_HTML_EXTENSION):
        return False

    blocked_segments = [
        "/api",
        "/_next",
        "/static",
        "/assets",
        "/media",
        "/favicon",
        "/manifest",
        "/robots.txt",
        "/sitemap",
    ]

    if any(path.startswith(block) for block in blocked_segments):
        return False

    if not include_authenticated:
        private_segments = [
            "/app",
            "/login",
            "/logout",
            "/auth",
            "/account",
            "/dashboard",
            "/collections",
        ]
        if any(path.startswith(block) for block in private_segments):
            return False

        if "login" in path or "auth" in path or "app" in path:
            return False

    return True


def is_public_content_url(url: str) -> bool:
    """Conserve uniquement le contenu public, pour compatibilité avec l'ancien appelant."""
    return is_scrapable_url(url, include_authenticated=False)


def save_authenticated_session(
    login_url: str,
    storage_state_path: str = "data/auth/storage_state.json",
) -> None:
    """Ouvre le login, puis sauvegarde la session après connexion manuelle."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright est requis: pip install playwright puis playwright install chromium"
        ) from exc

    os.makedirs(os.path.dirname(storage_state_path) or ".", exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(login_url, wait_until="domcontentloaded")
        input("Connectez-vous dans le navigateur, puis appuyez sur Entrée ici... ")
        page.context.storage_state(path=storage_state_path)
        browser.close()
    print(f"✓ Session authentifiée sauvegardée dans '{storage_state_path}'")


def load_authenticated_website(
    urls: list[str],
    storage_state_path: str = "data/auth/storage_state.json",
    max_depth: int = 2,
) -> list[Document]:
    """Crawl des pages publiques et privées avec une session navigateur persistée."""
    if not os.path.isfile(storage_state_path):
        raise FileNotFoundError(
            f"Session absente: {storage_state_path}. "
            "Exécutez save_authenticated_session() avant le crawl."
        )

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright est requis: pip install playwright puis playwright install chromium"
        ) from exc

    documents: list[Document] = []
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(url, 0) for url in urls]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(storage_state=storage_state_path)

        while queue:
            current_url, depth = queue.pop(0)
            normalized_url = current_url.split("#", 1)[0].rstrip("/") or current_url
            if normalized_url in visited or depth > max_depth:
                continue
            if not is_scrapable_url(normalized_url, include_authenticated=True):
                continue
            visited.add(normalized_url)

            page = context.new_page()
            try:
                try:
                    response = page.goto(
                        normalized_url,
                        wait_until="domcontentloaded",
                        timeout=30000,
                    )
                except PlaywrightTimeoutError:
                    # Certaines SPA ne terminent jamais complètement leur navigation.
                    response = None
                    if not page.url:
                        continue
                try:
                    page.get_by_text("Loading...", exact=True).wait_for(
                        state="hidden",
                        timeout=15000,
                    )
                except PlaywrightTimeoutError:
                    pass
                if response is None or response.status >= 400:
                    if not page.url:
                        continue
                final_url = page.url
                final_path = urlparse(final_url).path.lower()
                if "/login" in final_path or "/auth" in final_path:
                    print(f"⚠️ Session expirée ou refusée pour {normalized_url}")
                    continue

                html = page.content()
                text = clean_html(html)
                if not text:
                    continue

                structure = extract_page_structure(html, final_url, normalized_url)
                metadata = {
                    "source": final_url,
                    "source_type": "web_authenticated",
                    "source_file": final_url,
                    "page_structure": {
                        "title": structure["title"],
                        "type": structure["page_type"],
                    },
                    "sections": structure["sections"],
                    "internal_links": structure["internal_links"][:10],
                    "cta_buttons": structure["cta_buttons"],
                    "breadcrumbs": structure["breadcrumbs"],
                    "nav_menu": structure["nav_menu"],
                }
                documents.append(Document(page_content=text, metadata=metadata))

                if depth < max_depth:
                    for link in structure["internal_links"]:
                        link_url = link["url"]
                        if urlparse(link_url).netloc == urlparse(normalized_url).netloc:
                            queue.append((link_url, depth + 1))
            finally:
                page.close()

        context.close()
        browser.close()

    print(f"✓ {len(documents)} page(s) authentifiée(s) chargée(s)")
    return documents


def extract_page_structure(html: str, page_url: str, base_url: str) -> dict:
    """
    Extrait la structure, la navigation et le contexte d'une page HTML.
    Retourne un dictionnaire riche avec informations de navigation et d'aide.
    """
    soup = BeautifulSoup(html, "html.parser")
    
    structure = {
        "title": "",
        "page_type": "generic",
        "sections": [],
        "internal_links": [],
        "cta_buttons": [],
        "breadcrumbs": [],
        "nav_menu": [],
    }
    
    # 1. Extraire le titre principal (H1)
    h1 = soup.find("h1")
    if h1:
        structure["title"] = h1.get_text(strip=True)
    
    # 2. Détecter le type de page (basé sur l'URL et le contenu)
    path = urlparse(page_url).path.lower()
    if path in ['/', '']:
        structure["page_type"] = "accueil"
    elif 'contact' in path or 'nous-contacter' in path:
        structure["page_type"] = "contact"
    elif 'service' in path:
        structure["page_type"] = "services"
    elif 'produit' in path or 'produits' in path:
        structure["page_type"] = "produits"
    elif 'blog' in path or 'actualit' in path:
        structure["page_type"] = "blog"
    elif 'faq' in path or 'question' in path:
        structure["page_type"] = "faq"
    elif 'tarif' in path or 'prix' in path:
        structure["page_type"] = "tarifs"
    elif 'about' in path or 'qui-sommes' in path:
        structure["page_type"] = "about"
    
    # 3. Extraire la hiérarchie des headings (structure de la page)
    current_h1 = None
    current_h2 = None
    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
        text = heading.get_text(strip=True)
        level = int(heading.name[1])
        
        if level == 1:
            current_h1 = text
        elif level == 2:
            current_h2 = text
            if current_h1:
                structure["sections"].append({
                    "parent": current_h1,
                    "heading": current_h2,
                    "level": 2
                })
        elif level == 3:
            if current_h2:
                structure["sections"].append({
                    "parent": current_h2,
                    "heading": text,
                    "level": 3
                })
    
    # 4. Extraire les liens de navigation (nav)
    nav = soup.find("nav") or soup.find(class_=re.compile(r"navbar|menu|nav", re.I))
    if nav:
        for link in nav.find_all("a", href=True):
            href = str(link.get("href", "")).strip()
            text = link.get_text(strip=True)
            if href and text:
                structure["nav_menu"].append({
                    "text": text,
                    "url": urljoin(base_url, href)
                })
    
    # 5. Extraire les liens internes (contexte de navigation)
    for link in soup.find_all("a", href=True):
        href = str(link.get("href", "")).strip()
        text = link.get_text(strip=True)
        if href and text and not href.startswith("#"):
            full_url = urljoin(base_url, href)
            # Vérifier que c'est un lien interne
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                structure["internal_links"].append({
                    "text": text,
                    "url": full_url,
                    "context": "navigation"
                })
    
    # 6. Détecter les CTAs (boutons importants)
    for button_selector in ['button', '[role="button"]']:
        for button in soup.select(button_selector):
            text = button.get_text(strip=True)
            if text:
                # Chercher le lien associé au bouton
                link = button.find("a")
                if not link and button.parent:
                    link = button.parent.find("a")
                
                url = str(link.get("href", "#")).strip() if link else "#"
                if url != "#" and url:
                    url = urljoin(base_url, url)
                
                structure["cta_buttons"].append({
                    "text": text,
                    "url": url if url.startswith("http") else "#"
                })
    
    # 7. Extraire les breadcrumbs (fil d'Ariane)
    breadcrumb = soup.find(class_=re.compile(r"breadcrumb", re.I))
    if breadcrumb:
        for link in breadcrumb.find_all("a"):
            text = link.get_text(strip=True)
            href = str(link.get("href", "#")).strip()
            if text:
                structure["breadcrumbs"].append({
                    "text": text,
                    "url": urljoin(base_url, href)
                })
    
    return structure




def load_website(base_url: str, max_depth: int = 2, exclude_dirs: list[str] | None = None) -> list[Document]:
    """
    Explore un site web à partir d'une URL racine et retourne les pages
    découvertes sous forme d'une liste d'objets 'Document' LangChain.
    
    Enrichit chaque document avec :
    - Métadonnées de structure (titre, type de page)
    - Navigation (menus, liens internes)
    - Sections (hiérarchie H1>H2>H3)
    - CTAs (appels à l'action)
    - Breadcrumbs (fil d'Ariane)
    """
    # Configuration du chargeur récursif de LangChain
    loader = RecursiveUrlLoader(
        url=base_url,                  # Le point de départ de l'exploration du site
        max_depth=max_depth,          # Nombre maximum de clics successifs à suivre depuis l'accueil
        extractor=clean_html,         # La fonction de nettoyage appliquée au texte
        prevent_outside=True,         # Reste strictement sur le domaine initial
        exclude_dirs=exclude_dirs or [], # Chemins ou sous-dossiers spécifiques à ignorer
        use_async=True,               # Active le téléchargement asynchrone (parallèle)
        timeout=10,                   # Temps d'attente maximum en secondes par page
    )

    # Déclenche le crawling et télécharge toutes les pages correspondantes
    documents = loader.load()
    documents = [
        doc for doc in documents
        if is_public_content_url(str(doc.metadata.get("source", "")))
    ]

    # Enrichissement de chaque document avec infos de structure et navigation
    for doc in documents:
        # Métadonnées de base
        doc.metadata["source_type"] = "web"
        doc.metadata["source_file"] = doc.metadata.get("source", base_url)
        
        # === ENRICHISSEMENT: Extraire la structure pour la navigation ===
        # Note: Comme RecursiveUrlLoader nettoie déjà le HTML via clean_html(),
        # on doit ré-charger le HTML brut pour extraire la structure.
        # (Dans une version future, on pourrait modifier le loader pour conserver l'HTML brut)
        try:
            response = requests.get(doc.metadata.get("source", base_url), timeout=10)
            if response.status_code == 200:
                page_structure = extract_page_structure(response.text, doc.metadata.get("source", ""), base_url)
                
                # Ajouter la structure aux métadonnées
                doc.metadata["page_structure"] = {
                    "title": page_structure["title"],
                    "type": page_structure["page_type"],
                }
                doc.metadata["sections"] = page_structure["sections"]
                doc.metadata["internal_links"] = page_structure["internal_links"][:10]  # Limiter à 10 liens
                doc.metadata["cta_buttons"] = page_structure["cta_buttons"]
                doc.metadata["breadcrumbs"] = page_structure["breadcrumbs"]
                doc.metadata["nav_menu"] = page_structure["nav_menu"]
        except Exception as e:
            # Si l'extraction échoue, on continue sans ces métadonnées
            print(f"⚠️  Impossible d'extraire la structure pour {doc.metadata.get('source', 'URL inconnue')}: {e}")

    print(f"✓ {len(documents)} page(s) web chargée(s) depuis {base_url}")
    print(f"  Enrichies avec: structure de page, navigation, sections, CTAs")

    # Retourne la liste finale des documents prêts à être découpés (chunking)
    return documents


def save_documents_to_folder(documents: list[Document], folder_path: str = "data/raw/web"):
    """
    Crée un dossier et sauvegarde chaque document dans un fichier JSON individuel.
    Inclut toutes les métadonnées enrichies (structure, navigation, sections, etc.)
    """
    # Crée le dossier s'il n'existe pas déjà
    os.makedirs(folder_path, exist_ok=True)

    for index, doc in enumerate(documents):
        # Récupère l'URL de la page
        url = doc.metadata.get("source", f"page_{index}")

        # Nettoie l'URL pour en faire un nom de fichier valide
        safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', url)

        # Limite la longueur du nom de fichier pour éviter les erreurs système
        safe_filename = safe_filename[-150:] + ".json"
        
        # Construit le chemin complet du fichier
        file_path = os.path.join(folder_path, safe_filename)
        
        # Prépare les données à sauvegarder avec meilleure organisation
        data_to_save = {
            "page_content": doc.page_content,
            "page_info": {
                "url": doc.metadata.get("source", ""),
                "title": doc.metadata.get("page_structure", {}).get("title", ""),
                "type": doc.metadata.get("page_structure", {}).get("type", "generic"),
            },
            "navigation": {
                "sections": doc.metadata.get("sections", []),
                "breadcrumbs": doc.metadata.get("breadcrumbs", []),
                "nav_menu": doc.metadata.get("nav_menu", []),
                "internal_links": doc.metadata.get("internal_links", []),
                "cta_buttons": doc.metadata.get("cta_buttons", []),
            },
            "metadata": doc.metadata,  # Toutes les autres métadonnées
        }
        
        # Écrit le fichier JSON avec formatage lisible
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
    print(f"📁 {len(documents)} document(s) sauvegardé(s) dans '{folder_path}'")
    print(f"   ✓ Contenu de page")
    print(f"   ✓ Structure (sections, hiérarchie)")
    print(f"   ✓ Navigation (menus, liens internes, breadcrumbs)")
    print(f"   ✓ Appels à l'action (CTAs)")
    print(f"   ✓ Type de page (accueil, services, contact, etc.)")
