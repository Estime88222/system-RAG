# Mon Projet RAG

Système de Retrieval-Augmented Generation (RAG) permettant d'interroger une base de connaissances personnalisée (PDF, TXT, DOCX, CSV) via un LLM, avec une base vectorielle locale.

## Objectif

Ce projet met en place un pipeline RAG complet :

1. Ingestion et découpage de documents
2. Vectorisation et stockage dans une base vectorielle locale
3. Recherche par similarité sémantique
4. Génération de réponses via un LLM, ancrées dans les documents fournis

À terme, ce RAG est destiné à alimenter un agent IA (branché sur un bot Telegram).

## Architecture

```Document brut → Chunking → Embedding → Base vectorielle
                                              ↓
Question utilisateur → Embedding → Recherche top-k → Contexte
                                                          ↓
                                    Prompt (contexte + question) → LLM → Réponse
```

## Stack technique

| Composant | Outil |
| --- | --- |
| Orchestration | LangChain |
| Chargement de documents | LangChain Document Loaders (PDF, TXT, DOCX, CSV) |
| Chunking | LangChain Text Splitters |
| Base vectorielle | ChromaDB (local) |
| Embeddings | Modèle configurable (local ou API) |
| LLM | Ollama (local) / OpenAI / Anthropic (configurable) |
| API | FastAPI |

## Prérequis

- Python 3.10+
- [Ollama](https://ollama.com) installé (si utilisation d'un LLM local)
- Un modèle Ollama compatible tool-calling si utilisé avec l'agent :

  ```bash
  ollama pull qwen2.5:7b
  ```

## Installation

```bash
# Cloner le projet
git clone <url-du-repo>
cd mon-projet-rag

# Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

Créer un fichier `.env` à la racine (voir `.env.example` si présent) :

```OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

> Le fichier `.env` ne doit jamais être commit (déjà exclu via `.gitignore`).

## Structure du projet

```mon-projet-rag/
├── data/
│   ├── raw/
│   └── processed/
├── database/
│   └── chroma_db/
├── src/
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── splitter.py
│   │   └── embedder.py
│   ├── vectorstore/
│   │   └── indexer.py
│   ├── retrieval/
│   │   ├── search.py
│   │   └── reranker.py
│   ├── generation/
│   │   ├── llm_client.py
│   │   └── prompts.py
│   └── main.py
├── api/
│   ├── app.py
│   └── routes.py
├── tests/
│   ├── test_ingestion.py
│   └── test_retrieval.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Utilisation

### 1. Ajouter des documents

Placer les fichiers à indexer dans `data/raw/` (formats supportés : `.pdf`, `.txt`, `.docx`, `.csv`).

### 2. Construire l'index vectoriel

```bash
python -m src.ingestion.loader
```

### 3. Lancer une requête de test

```bash
python -m src.main
```

### 4. Lancer l'API

```bash
uvicorn api.app:app --reload
```

L'API est alors disponible sur `http://localhost:8000`.

## Tests

```bash
pytest tests/
```

## Roadmap

- Chargement des documents (PDF, TXT, DOCX, CSV)
- Chunking configurable
- Embedding + indexation ChromaDB
- Pipeline de récupération (retrieval)
- Reranking des résultats
- Génération de réponses via LLM
- Évaluation du pipeline (Ragas)
- Exposition via API (FastAPI)
- Intégration à l'agent IA / bot Telegram

## Notes

- Les données (`data/`) et la base vectorielle (`database/`) sont exclues du versioning (voir `.gitignore`).
- Le fichier `main.py` est le point d'orchestration entre les étapes ingestion → retrieval → generation
