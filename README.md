## Scribe RAG (Chainlit + FastAPI + Qdrant)

This project runs a Chainlit UI backed by a FastAPI RAG service. The RAG backend ingests documents, stores embeddings in Qdrant, and retrieves relevant chunks for each user query.

### Prerequisites
- Python 3.12+
- A running Postgres instance (local)
- A Qdrant Cloud cluster endpoint + API key
- Node.js (for Prisma CLI via `npx`)

### Install Dependencies
```
python -m pip install -r requirements.txt
```

Optional dev tools:
```
python -m pip install -r requirements-dev.txt
```

### Environment Variables
Create `.env` in the project root and set:
```
GEMINI_API_KEY=your_gemini_key

DATABASE_URL=postgresql://root:root@localhost:5432/postgres

QDRANT_URL=https://YOUR_QDRANT_CLUSTER_ENDPOINT
QDRANT_API_KEY=your_qdrant_api_key

RAG_BACKEND_URL=http://localhost:8001
RAG_TOP_K=5
RAG_DOCS_PATH=./docs
AI_LOG_FILE=logs/ai_frontend.logv
```

Notes:
- `RAG_DOCS_PATH` is optional. If set, the backend ingests that folder at startup.
- AI logs will be written to `logs/ai_frontend.log` and `logs/ai_backend.log` by default.
  - `AI_LOG_FILE` lets you override the log file path. Options:
  - Omit it to use defaults (`logs/ai_frontend.log` for UI and `logs/ai_backend.log` for backend).
  - Set it to a custom path to unify logs from both processes (e.g., `AI_LOG_FILE=logs/ai_all.log`).
  - Set different values per process if you want custom separation.

### Initialize Chainlit Database (Prisma)
Run once to create Chainlit tables in Postgres:
```
npx prisma@6.18.0 migrate deploy
```

### Run the Backend (FastAPI)
```
python -m uvicorn backend.main:app --reload --port 8001
```

### Run the UI (Chainlit)
```
python -m chainlit run frontend/main.py -w --port 8000
```

Open the UI at:
```
http://localhost:8000
```

### Folder Ingestion (Optional)
Put documents in `./docs` and set:
```
RAG_DOCS_PATH=./docs
```
Supported file types:
- `.txt`, `.md`, `.pdf`, `.docx`, `.doc`

### Troubleshooting
- If you see `relation "Thread" does not exist`, run the Prisma migration again.
- If RAG retrieval fails, verify `QDRANT_URL` uses `https://` and that `QDRANT_API_KEY` is set.
