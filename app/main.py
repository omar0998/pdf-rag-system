import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.routes import upload, query
from app.core.vector_store import get_embedding_model, get_chroma_client
import os

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy ChromaDB telemetry errors
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Lifespan — startup/shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load heavy models on startup so first request is fast."""
    logger.info("🚀 Starting PDF RAG System...")
    logger.info("Loading embedding model (this may take a moment on first run)...")
    get_embedding_model()   # pre-load SentenceTransformer
    get_chroma_client()     # connect to ChromaDB
    logger.info("✅ System ready.")
    yield
    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PDF RAG System",
    description=(
        "Upload PDFs and ask questions. "
        "Powered by SentenceTransformer embeddings + ChromaDB + Mistral AI."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS — allow frontend (served separately or on same origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
app.include_router(upload.router, prefix="/api", tags=["📄 Documents"])
app.include_router(query.router, prefix="/api", tags=["💬 Query"])


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "PDF RAG System"}


# ---------------------------------------------------------------------------
# Serve frontend (single HTML file)
# ---------------------------------------------------------------------------
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_PATH)
