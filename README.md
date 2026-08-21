# Mon Projet RAG

Système de Retrieval-Augmented Generation (RAG) permettant d'interroger une base de connaissances personnalisée (PDF, TXT, DOCX, CSV) via un LLM, avec vectorisation locale et stockage vectoriel embarqué.

## Objectif

Ce projet met en place un pipeline RAG complet :
1. Ingestion et découpage de documents
2. Vectorisation et stockage local dans une base vectorielle
3. Recherche par similarité sémantique
4. Génération de réponses via un LLM, ancrées dans les documents fournis

À terme, ce RAG est destiné à alimenter un agent IA exposé en API, connectable à un bot Telegram ou tout autre client.

---

## Stack technique retenue (état final)

| Composant | Outil retenu | Alternatives testées / écartées |
|---|---|---|
| Orchestration | LangChain | — |
| Chargement de documents | LangChain Document Loaders (PDF, TXT, DOCX, CSV) | — |
| Chunking | `RecursiveCharacterTextSplitter` | — |
| Embeddings | **Ollama** (`nomic-embed-text`, 768 dim) | OpenAI (`text-embedding-3-small`) — écarté : quota payant, aucun crédit gratuit |
| Base vectorielle | **Chroma** (locale, embarquée) | PostgreSQL + pgvector — écarté après plusieurs blocages d'infrastructure (voir plus bas) |
| LLM de génération | **DeepSeek API** (`deepseek-chat`) / **Groq** (`openai/gpt-oss-120b`) en alternative gratuite | — |
| API d'exposition | FastAPI + Uvicorn | — |

---

## Installations réalisées

### Environnement Python

```bash
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
```

### Dépendances par module

```bash
# Ingestion (loaders + chunking)
pip install langchain langchain-community pypdf docx2txt unstructured langchain-text-splitters

# Embeddings — Ollama (solution retenue)
pip install langchain-ollama

# Embeddings — OpenAI (testé puis écarté)
pip install langchain-openai

# Vectorstore — Chroma (solution retenue)
pip install langchain-chroma chromadb

# Vectorstore — PostgreSQL/pgvector (testé puis écarté)
pip install langchain-postgres psycopg[binary] pgvector

# Génération — client compatible OpenAI (DeepSeek / Groq)
pip install openai python-dotenv

# Reranking (optionnel)
pip install sentence-transformers

# API
pip install fastapi uvicorn

# Web scraping (optionnel, pour alimenter le RAG depuis des sites web)
pip install beautifulsoup4
```

### Outils système

- **Ollama** — installé en local pour les embeddings (et testé pour la génération)
  ```bash
  ollama pull nomic-embed-text
  ```
- **Docker Desktop** — utilisé temporairement pour héberger PostgreSQL + pgvector (option finalement abandonnée au profit de Chroma)

---

## Choix techniques et justifications

### Embeddings : Ollama plutôt qu'OpenAI

- OpenAI (`text-embedding-3-small`, 1536 dimensions) a été la première solution testée, mais l'API a renvoyé une erreur `insufficient_quota` : compte sans moyen de paiement actif.
- **Ollama** (`nomic-embed-text`, 768 dimensions) a été retenu en remplacement : gratuit, illimité, local, aucune clé API, aucune donnée envoyée à l'extérieur.
- Compromis accepté : légèrement plus lent en CPU pur qu'une API cloud, mais sans risque de coût ou de quota.

### Base vectorielle : Chroma plutôt que PostgreSQL/pgvector

Une étude comparative (Chroma, pgvector, Qdrant, Weaviate, Milvus, Pinecone) a d'abord orienté le choix vers **pgvector**, pour sa cohérence avec une stack PostgreSQL déjà maîtrisée par ailleurs. Ce choix a cependant généré plusieurs blocages successifs (détaillés dans la section suivante), menant à un retour sur **Chroma** :

- Aucun serveur externe à gérer (base embarquée, stockée dans `database/chroma_db/`)
- Aucun port, aucune authentification, aucun risque de conflit réseau
- Suffisant pour le volume de documents actuel du projet
- pgvector reste une amélioration possible en V2 si le volume de données augmente significativement

### LLM de génération : DeepSeek (avec Groq en alternative gratuite)

- **DeepSeek API** (`deepseek-chat`) retenue par défaut, appelée via le SDK OpenAI redirigé vers `https://api.deepseek.com`.
- **Groq Cloud** testée comme alternative gratuite, via `https://api.groq.com/openai/v1`, avec le modèle `openai/gpt-oss-120b` (après dépréciation de `llama-3.3-70b-versatile` en cours de projet).
- Température de génération fixée à `0.2` dans les deux cas, pour privilégier la fidélité au contexte plutôt que la créativité (limiter le risque d'hallucination).

---

## Problèmes rencontrés, causes et solutions

| # | Problème | Cause | Solution appliquée |
|---|---|---|---|
| 1 | `ERREUR : l'extension « vector » n'est pas disponible` | pgvector n'est pas installé nativement avec l'installeur PostgreSQL officiel (EDB) sur Windows | Utilisation d'une image Docker officielle `pgvector/pgvector`, incluant l'extension préinstallée |
| 2 | `FATAL: authentification par mot de passe échouée` (PostgreSQL) | Conflit de port : le PostgreSQL natif (Windows, service `postgresql-x64-18`) et le conteneur Docker écoutaient tous deux sur le port `5432` — le service natif interceptait les connexions à la place du conteneur | Diagnostic via `Get-Service -Name postgresql*` ; changement du port du conteneur (`5433`) puis, finalement, abandon de PostgreSQL au profit de Chroma pour supprimer toute dépendance réseau |
| 3 | `openai.RateLimitError: insufficient_quota` | Compte OpenAI sans moyen de paiement actif (embeddings) | Migration vers Ollama (`nomic-embed-text`), solution locale et gratuite |
| 4 | `ollama._types.ResponseError: ... connectex: No connection could be made` | Sous-processus interne ("runner") d'Ollama qui n'a pas démarré correctement | Redémarrage complet du service Ollama (arrêt des process `ollama.exe`, relance) |
| 5 | Erreurs Pylance (`reportArgumentType`, `reportReturnType`, `reportTypedDictNotRequiredAccess`) | Typage strict des SDK (OpenAI, SQLAlchemy) : certains champs sont déclarés comme potentiellement `None` ou de type générique, même si dans notre usage précis ils sont toujours présents | Ajout de vérifications explicites (`if x is not None else ...`, `.get()` au lieu de l'accès direct par clé) |
| 6 | `openai.APIStatusError: 405 - Method not allowed` (Groq) | URL de base incorrecte (`https://groq.com` au lieu de l'endpoint API) | Correction de l'URL vers `https://api.groq.com/openai/v1` |
| 7 | `openai.NotFoundError: 404 - model_not_found` (Groq) | Modèle `llama-3.3-70b-versatile` déprécié par Groq en cours de projet | Migration vers le modèle recommandé en remplacement : `openai/gpt-oss-120b` |
| 8 | `ImportError: cannot import name 'rerank_chunks' from 'retrieval'` | Le fichier `retrieval/__init__.py` réexportait explicitement certaines fonctions, mais n'avait pas été mis à jour après l'ajout de `reranker.py` | Import direct depuis les modules (`from retrieval.search import ...`, `from retrieval.reranker import ...`) plutôt que via `__init__.py`, pour éviter la désynchronisation |
| 9 | `.env` mal interprété (mot de passe apparemment incorrect malgré une valeur juste) | Suspicion initiale d'un caractère invisible (retour chariot Windows `\r`) dans le fichier | Diagnostic par `repr()` de la variable chargée ; cause réelle finalement identifiée comme le conflit de port (problème #2), pas le fichier `.env` lui-même |

---

## Structure du projet

```mon-projet-rag/
├── data/
│   ├── raw/                     # Documents bruts (PDF, TXT, DOCX, CSV)
│   └── processed/               # Textes nettoyés/extraits
├── database/
│   └── chroma_db/               # Base vectorielle locale (Chroma)
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py             # Chargement des documents bruts
│   │   ├── splitter.py           # Découpage en chunks
│   │   └── embedder.py           # Génération des embeddings (Ollama)
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── indexer.py            # Indexation dans Chroma
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── search.py             # Recherche vectorielle + formatage du contexte
│   │   └── reranker.py           # Reranking par cross-encoder (optionnel)
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompts.py            # Templates de prompts (system + user)
│   │   └── llm_client.py         # Appel au LLM (DeepSeek / Groq)
│   └── main.py                    # Orchestration complète du pipeline (fonction ask())
├── api/
│   ├── app.py                     # Serveur FastAPI
│   └── routes.py                  # Endpoint POST /ask
├── tests/
│   ├── test_ingestion.py
│   └── test_retrieval.py
├── .env                            # Clés API et configuration (jamais commit)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Variables d'environnement (`.env`)

```# Embeddings (si retour vers OpenAI un jour)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# Génération — DeepSeek
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# Génération — Groq (alternative gratuite)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx

# Bot Telegram (intégration future)
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
```

> `POSTGRES_CONNECTION` n'est plus utilisée depuis le passage à Chroma, mais reste pertinente à documenter si le projet revient un jour vers pgvector.

---

## Utilisation

### 1. Ajouter des documents

Placer les fichiers à indexer dans `data/raw/` (formats supportés : `.pdf`, `.txt`, `.docx`, `.csv`).

### 2. Construire l'index vectoriel

```bash
python -m src.vectorstore.indexer
```

### 3. Interroger le RAG en ligne de commande

```bash
python -m src.main
```

### 4. Lancer l'API

```bash
uvicorn api.app:app --reload --port 8000
```

Documentation interactive disponible sur `http://localhost:8000/docs`.

---

## Roadmap

- [x] Chargement des documents (PDF, TXT, DOCX, CSV)
- [x] Chunking configurable
- [x] Embeddings locaux (Ollama)
- [x] Indexation vectorielle (Chroma)
- [x] Recherche par similarité
- [x] Génération de réponses (DeepSeek / Groq)
- [x] Pipeline complet assemblé (`main.py`)
- [x] Exposition via API (FastAPI)
- [ ] Reranking activé par défaut
- [ ] Évaluation automatisée (Ragas)
- [ ] Tests unitaires complets
- [ ] Ingestion de contenu web (`WebBaseLoader` / `RecursiveUrlLoader`)
- [ ] Intégration à la mini app Telegram
- [ ] Passage à l'échelle (pgvector/Qdrant si le volume de documents augmente)
