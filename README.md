# PDF RAG System 📚

A complete Retrieval-Augmented Generation system for querying PDF documents using:
- **SentenceTransformer** (`all-MiniLM-L6-v2`) for local embeddings
- **ChromaDB** as the vector store
- **Mistral AI** as the LLM
- **FastAPI** as the backend
- **Single-file HTML frontend** (no build step)

---

## Project Structure

```
pdf-rag-system/
├── app/
│   ├── main.py                  # FastAPI app + lifespan + frontend serving
│   ├── config.py                # Pydantic settings (reads .env)
│   ├── api/routes/
│   │   ├── upload.py            # POST /api/upload, GET /api/documents, DELETE /api/documents/{name}
│   │   └── query.py             # POST /api/query, GET /api/stats
│   └── core/
│       ├── pdf_processor.py     # PyPDF extraction + LangChain chunking
│       ├── vector_store.py      # ChromaDB operations + SentenceTransformer embeddings
│       └── rag_pipeline.py      # RAG pipeline (retrieve → augment → generate)
├── frontend/
│   └── index.html               # Complete UI (no build needed)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Get a Mistral API Key
Go to https://console.mistral.ai and create an API key.

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and set your MISTRAL_API_KEY
```

### 3. Run with Docker (recommended)
```bash
cd docker
docker compose up --build
```

The app will be available at http://localhost:8000

---

## Run without Docker

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
uvicorn app.main:app --reload --port 8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload and index a PDF |
| GET | `/api/documents` | List all indexed documents |
| DELETE | `/api/documents/{filename}` | Remove a document |
| POST | `/api/query` | Ask a question |
| GET | `/api/stats` | Vector store statistics |
| GET | `/api/health` | Health check |
| GET | `/api/docs` | Swagger UI |

### Query Request Example
```json
{
  "question": "What are the main conclusions?",
  "source_filter": "report.pdf",   // optional: limit to one PDF
  "top_k": 4                        // optional: chunks to retrieve
}
```

---

## How RAG Works (step by step)

```
[PDF Upload]
     │
     ▼
PyPDF → Extract text from each page
     │
     ▼
RecursiveCharacterTextSplitter
→ Split into 500-char chunks with 50-char overlap
→ Overlap ensures context isn't lost at boundaries
     │
     ▼
SentenceTransformer (all-MiniLM-L6-v2)
→ Each chunk → 384-dimensional dense vector
→ Runs locally, no API cost
     │
     ▼
ChromaDB (HNSW cosine index)
→ Store (text, vector, metadata)

[Question]
     │
     ▼
SentenceTransformer → Question vector
     │
     ▼
ChromaDB cosine similarity search
→ Top-K most similar chunks
→ Filter chunks with similarity < 0.3
     │
     ▼
Build prompt: System + Context chunks + Question
     │
     ▼
Mistral AI (mistral-large-latest)
temperature=0.1 → factual, grounded answers
     │
     ▼
Return answer + sources + metadata
```

---

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | required | Your Mistral API key |
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `TOP_K_RESULTS` | 4 | Chunks to retrieve per query |
| `MISTRAL_MODEL` | `mistral-large-latest` | Mistral model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |

---

## Key Concepts Learned

| Concept | Where it's used |
|---------|----------------|
| **Chunking** | `pdf_processor.py` — RecursiveCharacterTextSplitter |
| **Dense Embeddings** | `vector_store.py` — SentenceTransformer |
| **Vector Similarity** | ChromaDB HNSW with cosine distance |
| **Context Grounding** | `rag_pipeline.py` — system prompt engineering |
| **Relevance Filtering** | Similarity score threshold (≥ 0.3) |
| **Persistent Storage** | ChromaDB PersistentClient on disk |
