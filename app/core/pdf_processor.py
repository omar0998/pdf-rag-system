import io
import logging
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import settings

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, int]:
    """
    Extract text from PDF bytes.
    Returns (full_text, page_count)
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    page_count = len(reader.pages)
    full_text = ""

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n[Page {i + 1}]\n{text}\n"

    return full_text.strip(), page_count


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks using recursive character splitting.
    RecursiveCharacterTextSplitter tries to split on:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Periods (sentences)
    4. Spaces (words)
    This preserves semantic meaning better than fixed-size splitting.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_text(text)

    # Filter out very short chunks (less than 50 chars) — usually noise
    chunks = [c.strip() for c in chunks if len(c.strip()) >= 50]

    logger.info(f"Created {len(chunks)} chunks from text of {len(text)} characters")
    return chunks


def process_pdf(file_bytes: bytes, filename: str) -> dict:
    """
    Full PDF processing pipeline:
    PDF bytes → extracted text → chunks
    Returns processing metadata.
    """
    logger.info(f"Processing PDF: {filename}")

    # Step 1: Extract text
    text, page_count = extract_text_from_pdf(file_bytes)

    if not text:
        raise ValueError(f"Could not extract any text from '{filename}'. "
                         "The PDF might be scanned/image-based.")

    logger.info(f"Extracted {len(text)} characters from {page_count} pages")

    # Step 2: Chunk
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("Text extraction succeeded but chunking produced no results.")

    return {
        "filename": filename,
        "page_count": page_count,
        "total_chars": len(text),
        "chunks": chunks,
        "chunk_count": len(chunks),
    }
