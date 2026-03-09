from __future__ import annotations
import logging
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons — initialized once at module load
# ---------------------------------------------------------------------------

_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model (downloads on first call)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _embedding_model


def get_chroma_client() -> chromadb.PersistentClient:
    """Lazy-load the ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        logger.info(f"Connecting to ChromaDB at: {settings.CHROMA_DB_PATH}")
        _chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection():
    """Get or create the main ChromaDB collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity search
    )


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def add_chunks_to_store(chunks: list[str], pdf_name: str) -> int:
    """
    Embed chunks and store them in ChromaDB.

    How it works:
    - SentenceTransformer converts each text chunk into a dense vector
      (a list of floats) that captures its semantic meaning.
    - ChromaDB stores (text, vector, metadata) and indexes vectors using
      HNSW (Hierarchical Navigable Small World) for fast ANN search.
    """
    model = get_embedding_model()
    collection = get_collection()

    # Batch embed all chunks at once (much faster than one by one)
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    # Unique IDs — if same PDF is re-uploaded, old chunks get overwritten
    ids = [f"{pdf_name}::chunk::{i}" for i in range(len(chunks))]

    metadatas = [
        {"source": pdf_name, "chunk_index": i}
        for i in range(len(chunks))
    ]

    # ChromaDB upsert: inserts or updates if ID already exists
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    logger.info(f"Stored {len(chunks)} chunks for '{pdf_name}'")
    return len(chunks)


def search_similar_chunks(query: str, top_k: Optional[int] = None, source_filter: Optional[str] = None) -> list[dict]:
    """
    Semantic search: find chunks most relevant to the query.

    Process:
    1. Embed the query using the same model used for documents.
    2. ChromaDB computes cosine similarity between query vector and all stored vectors.
    3. Returns the top_k most similar chunks with their metadata.
    """
    k = top_k or settings.TOP_K_RESULTS
    model = get_embedding_model()
    collection = get_collection()

    # Check if collection has any documents
    if collection.count() == 0:
        return []

    query_embedding = model.encode([query]).tolist()

    where_filter = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
        where=where_filter,
    )

    # Package results with metadata
    chunks_with_meta = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks_with_meta.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "similarity_score": round(1 - dist, 4),  # cosine distance → similarity
        })

    return chunks_with_meta


def list_documents() -> list[dict]:
    """List all unique PDFs stored in ChromaDB."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    # Get all metadata to extract unique sources
    all_items = collection.get(include=["metadatas"])
    sources: dict[str, int] = {}

    for meta in all_items["metadatas"]:
        source = meta.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

    return [{"filename": name, "chunk_count": count} for name, count in sources.items()]


def delete_document(pdf_name: str) -> int:
    """Delete all chunks belonging to a specific PDF."""
    collection = get_collection()

    # Get IDs for this source
    results = collection.get(where={"source": pdf_name}, include=["metadatas"])

    if not results["ids"]:
        return 0

    collection.delete(ids=results["ids"])
    logger.info(f"Deleted {len(results['ids'])} chunks for '{pdf_name}'")
    return len(results["ids"])


def get_store_stats() -> dict:
    """Get overall statistics about the vector store."""
    collection = get_collection()
    docs = list_documents()
    return {
        "total_chunks": collection.count(),
        "total_documents": len(docs),
        "documents": docs,
    }
