"""
Module de chargement des documents bruts vers des objets Document LangChain
Support : PDF, TXT, DOCX, CSV
"""

import os 
import json
from pathlib import Path 
# importation des outils de spécification de Langchain pour lire chaque type de fichier
from langchain_community.document_loaders import(PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader) 
# importation de la structure de données standardisée de langchain 
from langchain_core.documents import Document

#dictionnnaire de correspondance (mapping), associe une extention à son type de lecture 
LOADER_MAPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
}

def load_single_document(file_path: str) -> list[Document]:
    """charge un seul fichier et retourne un listte de Document Langchain"""
    ext = Path(file_path).suffix.lower() # extraction l'extension du fichier convertis en miniscule

    if ext not in LOADER_MAPING:
        raise ValueError(f"Format non supporté {ext}(fichier : {file_path})")

    loader_class = LOADER_MAPING[ext] #stock le bon lecteur adapté à l'extension 

    if ext == ".txt":
        loader = loader_class(file_path, encoding="utf-8") # l'extension txt à besois qu'on lui spécifie l'encodage UTF-8 pour éviter les bugs d'accents
    else:
        loader = loader_class(file_path)

    documents = loader.load() # cree l'objet document 

    #Boucle sur les documents générés pour injecter le nom du fichier d'origine dans les métadonnées
    for doc in documents:
        doc.metadata["source_file"] = os.path.basename(file_path)

    return documents     # Renvoie la liste de documents (ex: un PDF de 5 pages renverra une liste de 5 objets Document)

def load_document_from_directory(raw_dir: str ="data/raw") -> list[Document]:
    """parcourt le dossier raw et chage tous les fichier supporté et
      retourne un liste unique de Document, pret pour le découpage"""

    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {raw_dir}")

    all_documents = []  # liste de tous les documents de tous les fichiers
    errors = []          # liste pour lister les fichiers qui ont planté lors de la lecture

    # .rglob("*") parcourt récursivement TOUS les fichiers et sous-dossiers du répertoire
    for file_path in raw_path.rglob("*"):
        if not file_path.is_file():
            continue

        # Le brand-book est indexé séparément avec son metadata `doc_type`.
        # L'inclure ici le rendrait récupérable comme un document métier ordinaire.
        if "branding" in {part.lower() for part in file_path.relative_to(raw_path).parts}:
            continue

        ext = file_path.suffix.lower()

        # On ignore silencieusement les extensions non gérées (ni erreur, ni chargement)
        if ext != ".json" and ext not in LOADER_MAPING:
            continue

        try:
            if ext == ".json":
                # Fichier issu du web scraping (save_documents_to_folder)
                docs = load_json_document(str(file_path))
            else:
                docs = load_single_document(str(file_path))

            all_documents.extend(docs)
            print(f"✓ Chargé : {file_path.name} ({len(docs)} page(s)/entrée(s))")

        except Exception as e:
            # Si le fichier est corrompu ou illisible, on capture l'erreur SANS bloquer le script
            errors.append((file_path.name, str(e)))
            print(f"✗ Erreur sur {file_path.name} : {e}")
            # pas de return ici — on continue avec les fichiers suivants

    # Résumé final affiché dans la console à la fin du scan
    if errors:
        print(f"\n⚠ {len(errors)} fichier(s) en erreur, {len(all_documents)} documents chargés au total")
    else:
        print(f"\n✓ Tous les fichiers chargés avec succès : {len(all_documents)} documents")

    return all_documents

def load_json_document(file_path: str) -> list[Document]:
    """
    Recharge un document précédemment sauvegardé au format JSON
    (via save_documents_to_folder), en reconstruisant l'objet Document original.
    """
    with open(file_path, "r", encoding="utf-8") as f :
        data = json.load(f)

    doc = Document(page_content=data["page_content"], metadata=data["metadata"])
    return [doc]