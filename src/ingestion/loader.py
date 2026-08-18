"""
Module de chargement des documents bruts vers des objets Document LangChain
Support : PDF, TXT, DOCX, CSV
"""

import os 
from pathlib import Path 
# importation des outils de spécification de Langchain pour lire chaque type de fichier
from langchain_community.document_loaders import(PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader) 
# importation de la structure de données standardisée de langchain 
from langchain_core.documents import Document

#dictionnnaire de correspondance (mapping), associe une extention à son type de lecture 
LOADER_MAPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
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

    all_documents = [] #liste de tous les document de tous les fichier 
    errors = [] # liste pour listé les fichier qui on planté lors de la lecture 

    # .rglob("*") parcourt récursivement TOUS les fichiers et sous-dossiers du répertoire
    for file_path in raw_path.rglob("*"):
        #on verifie que c'est bien un fichier et non un dossier et que son extension est supporté 

        if file_path.is_file() and file_path.suffix.lower() in LOADER_MAPING :
            try:
                #on va tenté de charger le fichier en appelant load_single_document()
                docs = load_single_document(str(file_path))
                #on ajoute le document touver à la liste des docments
                all_documents.extend(docs)
                print(f"✓ Chargé : {file_path.name} ({len(docs)} page(s)/entrée(s))")

            except Exception as e :
                # Si le fichier est corrompu ou illisible, on capture l'erreur sans bloquer le script
                errors.append((file_path.name, str(e)))
                print(f"✗ Erreur sur {file_path.name} : {e}")
                return errors
    # Résumé final affiché dans la console à la fin du scan
    if errors:
        print(f"\n⚠ {len(errors)} fichier(s) en erreur, {len(all_documents)} documents chargés au total")
    else:
        print(f"\n✓ Tous les fichiers chargés avec succès : {len(all_documents)} documents")

    # Renvoie la grande liste finale contenant l'intégralité des textes extraits
    return all_documents