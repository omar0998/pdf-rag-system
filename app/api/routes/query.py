from __future__ import annotations
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.core.rag_pipeline import answer_question
from app.core.vector_store import get_store_stats

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="Your question")
    source_filter: Optional[str] = Field(None, description="Limit search to a specific PDF filename")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the main topics covered in this document?",
                "source_filter": None,
                "top_k": 4,
            }
        }


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int
    model: str
    top_chunks: list[dict]


@router.post("/query", response_model=QueryResponse, summary="Ask a question about your PDFs")
async def query_documents(request: QueryRequest):
    """
    RAG Query Pipeline:
    1. Embed the question (SentenceTransformer)
    2. Find semantically similar chunks (ChromaDB cosine search)
    3. Build context-augmented prompt
    4. Generate answer (Mistral AI)
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(
            question=request.question,
            source_filter=request.source_filter,
            top_k=request.top_k,
        )
        return result

    except Exception as e:
        logger.error(f"Error in RAG query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/stats", summary="Get vector store statistics")
async def get_stats():
    """Returns info about stored documents and total chunk count."""
    try:
        return get_store_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
