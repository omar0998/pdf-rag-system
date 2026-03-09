import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.core.pdf_processor import process_pdf
from app.core.vector_store import add_chunks_to_store, delete_document, list_documents

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@router.post("/upload", summary="Upload and index a PDF file")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF → extract text → chunk → embed → store in ChromaDB.

    Pipeline stages:
    1. Validate file type and size
    2. Extract text with PyPDF
    3. Split into overlapping chunks (LangChain)
    4. Generate embeddings (SentenceTransformer)
    5. Upsert into ChromaDB
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file."
        )

    # Read file bytes
    file_bytes = await file.read()

    # Validate file size
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # Process PDF: extract text + chunk
        result = process_pdf(file_bytes, file.filename)

        # Store chunks in vector DB
        stored = add_chunks_to_store(result["chunks"], pdf_name=file.filename)

        return {
            "success": True,
            "message": f"'{file.filename}' processed and indexed successfully.",
            "filename": file.filename,
            "pages": result["page_count"],
            "chunks_created": stored,
            "total_characters": result["total_chars"],
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing PDF '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@router.get("/documents", summary="List all indexed documents")
async def list_documents_endpoint():
    """List all PDFs currently stored in the vector database."""
    try:
        docs = list_documents()
        return {
            "success": True,
            "count": len(docs),
            "documents": docs,
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{filename}", summary="Delete a document from the index")
async def delete_document_endpoint(filename: str):
    """Remove all chunks of a specific PDF from ChromaDB."""
    try:
        deleted = delete_document(filename)
        if deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{filename}' not found in the index."
            )
        return {
            "success": True,
            "message": f"Deleted '{filename}' ({deleted} chunks removed).",
            "chunks_deleted": deleted,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
