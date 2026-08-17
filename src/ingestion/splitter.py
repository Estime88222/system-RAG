"""
Module de découpage (chunking) des documents chargés en morceaux
adaptés à l'embedding et à la recherche vectorielle.
"""

# Importation de l'outil de découpage de texte le plus performant et utilisé de LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Importation de la structure de données standardisée
from langchain_core.documents import Document

# Paramètres de configuration globaux (Modifiables selon vos besoins)
CHUNK_SIZE = 500       # Taille maximale de chaque morceau (mesurée ici en nombre de caractères)
CHUNK_OVERLAP = 50     # Chevauchement : nombre de caractères répétés entre la fin d'un morceau et le début du suivant


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Découpe une liste de Document en chunks plus petits.
    Conserve et enrichit les métadonnées d'origine sur chaque chunk.
    """
    
    # Configuration du découpeur "intelligent"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,          # Cible la taille max demandée
        chunk_overlap=chunk_overlap,    # Applique la répétition pour ne pas perdre le contexte
        length_function=len,            # Utilise la fonction standard Python len() pour compter les caractères
        
        # ORDRE DE PRIORITÉ DES SÉPARATEURS :
        # Le découpeur essaie d'abord de couper aux doubles sauts de ligne (paragraphes).
        # S'il n'y arrive pas sans dépasser 500 caractères, il tente aux simples sauts de ligne,
        # puis aux fins de phrases (points), puis aux espaces (mots), et enfin au caractère près.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Exécution du découpage automatique sur toute la liste de documents
    chunks = splitter.split_documents(documents)

    # Enrichissement des métadonnées pour chaque morceau généré
    # enumerate() permet d'obtenir à la fois l'indice (i) et le contenu du morceau (chunk)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i                         # Donne un numéro unique au morceau (ex: 0, 1, 2...)
        chunk.metadata["chunk_size"] = len(chunk.page_content) # Enregistre la taille exacte de ce morceau précis

    # Affichage des statistiques de découpage dans la console
    print(f"✓ {len(documents)} document(s) découpé(s) en {len(chunks)} chunk(s)")

    # Calcul et affichage de la taille moyenne réelle des morceaux (évite la division par zéro avec max(..., 1))
    print(f"  Taille moyenne : {sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)} caractères")

    # Renvoie la nouvelle liste contenant tous les petits morceaux de texte
    return chunks
