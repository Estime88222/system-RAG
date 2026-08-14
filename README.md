# Architecture

mon-projet-rag/
├── data/                       # Stockage des données (Base de connaissances)
│   ├── raw/                    # Documents bruts (PDF, TXT, DOCX, CSV)
│   └── processed/              # Textes nettoyés ou extraits (JSON, Markdown)
├── database/                   # Stockage de la base vectorielle locale
│   └── chroma_db/ or pinecone/ # Index et vecteurs persistés (ex: Chroma, FAISS)
├── src/                        # Code source principal de l'application
│   ├── __init__.py
│   ├── ingestion/              # Étape 1 : Pipeline de traitement des données
│   │   ├── loader.py           # Chargement des documents bruts
│   │   ├── splitter.py         # Découpage du texte en blocs (Chunking)
│   │   └── embedder.py         # Vectorisation via un modèle d'embedding
│   ├── retrieval/              # Étape 2 : Recherche d'informations
│   │   ├── search.py           # Requêtes dans la base de données vectorielle
│   │   └── reranker.py         # Optionnel : Tri et filtrage des résultats
│   ├── generation/             # Étape 3 : Synthèse avec le LLM
│   │   ├── llm_client.py       # Configuration du LLM (OpenAI, Ollama, Anthropic)
│   │   └── prompts.py          # Gestion des templates de prompts (System prompt)
│   └── main.py                 # Point d'entrée de la logique métier globale
├── api/                        # Optionnel : Exposition du RAG sous forme d'API
│   ├── app.py                  # Serveur FastAPI / Flask
│   └── routes.py               # Points d'accès (endpoints) pour l'utilisateur
├── tests/                      # Tests unitaires et d'intégration
│   ├── test_ingestion.py
│   └── test_retrieval.py
├── .env                        # Clés d'API (OPENAI_API_KEY) et variables secrètes
├── .gitignore                  # Pour ignorer /data, /database et le fichier .env
├── README.md                   # Documentation du projet
└── requirements.txt            # Dépendances (LangChain, LlamaIndex, Chromadb, etc.)

## Configuration de langchai

installation des dépendances de langchaine

'pip install deepagents "langchain[openai]" langchain-text-splitters requests numpy'
