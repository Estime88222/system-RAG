"""
Orchestration complète de la phase d'ingestion :
chargement des documents -> découpage -> indexation vectorielle.
C'est le SEUL point d'entrée à utiliser pour ingérer des données dans le RAG,
que ce soit des fichiers locaux ou du contenu web.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.loader import load_document_from_directory
from ingestion.splitter import split_documents
from vectorstore.indexer import index_if_needed, index_chunks
from ingestion.web_scraper import load_website, save_documents_to_folder
#url à scanner pour recuperer les données du site
WEB_SOURCE = ["https://taramoney.com/app/"]


def run_ingestion(
    raw_dir: str = "data/raw",
    web_source: list[str]| None = None,
    force: bool = False,
) -> None:
    """
    Exécute le pipeline d'ingestion complet.

    - raw_dir : dossier contenant les documents à ingérer
    - force : si True, réindexe même si la base contient déjà des données
    """
    print("=== Démarrage de l'ingestion ===")

    if web_source is None:
        web_source = WEB_SOURCE

    all_docs=[]

    

    #2. chargement du contenu web 
    for url in web_source:
        docs_web = load_website(url, max_depth=3)
        if docs_web:
            save_documents_to_folder(docs_web,raw_dir)
    
    #Chargement des fichier 
    all_docs = load_document_from_directory(raw_dir)
    

    if not all_docs :
        print("aucun document trouvé ni en local ni sur le web, ingestion annulé ")
        return 

    # 2. Découpage
    chunks = split_documents(all_docs)

    # 3. Indexation (uniquement si nécessaire, sauf si force=True)
    index_if_needed(chunks, force=force)

    print("=== Ingestion terminée ===")

if __name__ == "__main__":
    run_ingestion()