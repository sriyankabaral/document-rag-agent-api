import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.services.chunker import chunk_text
from app.services.embedding_service import (
    generate_embeddings,
    get_embedding_dimension,
)
from app.services.text_extractor import extract_text_from_file


router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIRECTORY = Path("uploads")
ALLOWED_FILE_TYPES = {".pdf", ".txt"}


@router.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    chunking_method: str = Form("recursive"),
    embedding_model: str = Form(
        "sentence-transformers/all-MiniLM-L6-v2"
    ),
):
    try:
        original_filename = Path((file.filename or "").replace("\\", "/")).name
        file_type = Path(original_filename).suffix.lower()

        if file_type not in ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF and TXT files are allowed.",
            )

        UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

        saved_filename = f"{uuid4()}_{original_filename}"
        saved_path = UPLOAD_DIRECTORY / saved_filename

        with saved_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)
    finally:
        file.file.close()

    try:
        extracted_text = extract_text_from_file(str(saved_path), file_type)
    except ValueError as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if not extracted_text.strip():
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No readable text found in the uploaded file.",
        )

    text_preview = (
        extracted_text.replace("\r", " ").replace("\n", " ")[:300]
    )

    try:
        chunks = chunk_text(extracted_text, method=chunking_method)
    except ValueError as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if not chunks:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text was extracted, but no chunks could be created.",
        )

    sample_chunks = [
        chunk.replace("\r", " ").replace("\n", " ")[:300]
        for chunk in chunks[:3]
    ]

    try:
        embeddings = generate_embeddings(chunks, embedding_model)
        embedding_dimension = get_embedding_dimension(embedding_model)
    except ValueError as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate embeddings. Please try again.",
        ) from exc

    if len(embeddings) != len(chunks):
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding generation did not return one vector per chunk.",
        )

    sample_embedding = [
        round(float(value), 4) for value in embeddings[0][:5]
    ]

    return {
        "message": "File uploaded successfully",
        "original_filename": original_filename,
        "saved_filename": saved_filename,
        "saved_path": saved_path.as_posix(),
        "file_type": file_type,
        "text_length": len(extracted_text),
        "text_preview": text_preview,
        "chunking_method": chunking_method,
        "number_of_chunks": len(chunks),
        "sample_chunks": sample_chunks,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "number_of_embeddings": len(embeddings),
        "sample_embedding": sample_embedding,
    }
