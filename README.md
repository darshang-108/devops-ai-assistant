# AI Engineering Knowledge Base

A **Retrieval-Augmented Generation (RAG)** system that lets engineering teams ask natural-language questions about internal documentation, codebases, and engineering knowledge.

---

## Architecture

```
User Question
     │
     ▼
  FastAPI  ──►  Embed query (SentenceTransformers)
     │                    │
     │                    ▼
     │          Vector DB search (Chroma / FAISS)
     │                    │
     │                    ▼
     │          Top-K relevant chunks
     │                    │
     ▼                    ▼
  LLM (OpenAI / Ollama)  ◄── context
     │
     ▼
  Answer + Sources
```

## Project Structure

```
ai_engineering_kb/
├── data/
│   ├── raw_docs/           ← place your .txt / .md / .pdf files here
│   └── processed_docs/
├── ingestion/
│   ├── load_documents.py   ← reads files from raw_docs/
│   └── chunk_documents.py  ← splits documents into chunks
├── embeddings/
│   └── embedder.py         ← SentenceTransformers wrapper
├── vector_store/
│   └── vector_db.py        ← Chroma & FAISS backends
├── retrieval/
│   └── retriever.py        ← embed query → search → return chunks
├── llm/
│   └── llm_client.py       ← send context + question to LLM
├── api/
│   └── main.py             ← FastAPI app (POST /query, POST /ingest, …)
├── utils/
│   └── config.py           ← central configuration
├── tests/
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add documents

Place `.txt`, `.md`, or `.pdf` files into `data/raw_docs/`.

### 3. Set environment variables

```bash
# For OpenAI
export OPENAI_API_KEY="sk-..."

# — OR — for a local Ollama instance
export LLM_PROVIDER="ollama"
export OLLAMA_MODEL="mistral"
```

### 4. Start the API server

```bash
cd ai_engineering_kb
uvicorn api.main:app --reload
```

The server starts at **http://localhost:8000**. Interactive docs at `/docs`.

### 5. Ingest documents

```bash
curl -X POST http://localhost:8000/ingest
```

### 6. Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do we deploy to production?"}'
```

## API Endpoints

| Method | Path      | Description                                |
|--------|-----------|--------------------------------------------|
| POST   | `/query`  | Ask a question, get an LLM-grounded answer |
| POST   | `/ingest` | (Re)ingest all documents from raw_docs/    |
| GET    | `/health` | Liveness check                             |
| GET    | `/stats`  | Vector store statistics                    |

## Configuration

All settings are centralised in `utils/config.py` and can be overridden via environment variables:

| Variable           | Default            | Description                      |
|--------------------|--------------------|----------------------------------|
| `LLM_PROVIDER`     | `openai`           | `openai` or `ollama`             |
| `OPENAI_API_KEY`   | —                  | Your OpenAI API key              |
| `OPENAI_MODEL`     | `gpt-3.5-turbo`    | OpenAI model name                |
| `OLLAMA_BASE_URL`  | `localhost:11434`  | Ollama server URL                |
| `OLLAMA_MODEL`     | `mistral`          | Ollama model name                |
| `API_HOST`         | `0.0.0.0`          | API bind address                 |
| `API_PORT`         | `8000`             | API port                         |

## Tech Stack

- **Python** – backend language
- **FastAPI** – HTTP API framework
- **LangChain** – text splitting utilities
- **SentenceTransformers** – embedding model (`all-MiniLM-L6-v2`)
- **ChromaDB** / **FAISS** – vector storage & similarity search
- **OpenAI** / **Ollama** – LLM answer generation
