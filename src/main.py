from ingestion import load_document_from_directory, split_documents, embed_chunks

if __name__ == "__main__":

    docs = load_document_from_directory("data/raw")

    if docs:
        print("\n--- Aperçu du premier document ---")
        #print(f"Contenu (200 premiers caractères) : {docs[0].page_content[:200]}")
        total_chars = sum(len(doc.page_content) for doc in docs)
        print(f"Total characters: {total_chars}")
        print(f"Métadonnées : {docs[0].metadata}")
    
    if docs:
        chunks = split_documents(docs)

        print("\n--- Aperçu du premier chunk ---")
        print(f"Contenu : {chunks[0].page_content}")
        print(f"Métadonnées : {chunks[0].metadata}")   

    if docs:
        chunks = split_documents(docs)
        vectors = embed_chunks(chunks)

        print("\n--- Aperçu du premier vecteur ---")
        print(f"5 premières valeurs : {vectors[0][:5]}")
        print(f"Dimension totale : {len(vectors[0])}")

   