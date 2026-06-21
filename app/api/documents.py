import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import DEFAULT_CHUNKING_METHOD, DEFAULT_EMBEDDING_MODEL
from app.db.database import get_db
from app.db.models import DocumentMetadata
from app.services.chunker import chunk_text
from app.services.embedding_service import (
    generate_embeddings,
    get_embedding_dimension,
)
from app.services.qdrant_service import (
    COLLECTION_NAME,
    search_similar_chunks,
    store_chunks_in_qdrant,
)
from app.services.text_extractor import extract_text_from_file


router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIRECTORY = Path("uploads")
ALLOWED_FILE_TYPES = {".pdf", ".txt"}


class DocumentSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    embedding_model: str = DEFAULT_EMBEDDING_MODEL


@router.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    chunking_method: str = Form(DEFAULT_CHUNKING_METHOD),
    embedding_model: str = Form(DEFAULT_EMBEDDING_MODEL),
    db: Session = Depends(get_db),
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

    document_metadata = DocumentMetadata(
        file_name=original_filename,
        file_type=file_type,
        chunking_method=chunking_method,
        embedding_model=embedding_model,
        number_of_chunks=len(chunks),
        qdrant_collection_name=COLLECTION_NAME,
    )

    try:
        db.add(document_metadata)
        db.flush()
    except Exception as exc:
        db.rollback()
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document metadata could not be prepared for saving.",
        ) from exc

    try:
        qdrant_point_ids = store_chunks_in_qdrant(
            chunks=chunks,
            embeddings=embeddings,
            original_filename=original_filename,
            saved_filename=saved_filename,
            file_type=file_type,
            chunking_method=chunking_method,
            embedding_model=embedding_model,
            document_id=document_metadata.id,
        )
    except Exception as exc:
        db.rollback()
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Failed to store document chunks in Qdrant. "
                "Make sure Qdrant is running."
            ),
        ) from exc

    try:
        db.commit()
        db.refresh(document_metadata)
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document was processed, but metadata could not be saved.",
        ) from exc

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
        "qdrant_collection_name": COLLECTION_NAME,
        "stored_vectors_count": len(qdrant_point_ids),
        "sample_qdrant_point_ids": qdrant_point_ids[:3],
        "document_id": document_metadata.id,
        "metadata_saved": True,
    }


@router.post("/search")
def search_documents(request: DocumentSearchRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    if not 1 <= request.top_k <= 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be between 1 and 10.",
        )

    try:
        query_embedding = generate_embeddings(
            [query],
            request.embedding_model,
        )[0]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate the query embedding.",
        ) from exc

    try:
        qdrant_results = search_similar_chunks(
            query_embedding=query_embedding,
            top_k=request.top_k,
            embedding_model=request.embedding_model,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    results = [
        {
            "score": result["score"],
            "original_filename": result["original_filename"],
            "chunk_index": result["chunk_index"],
            "chunk_text": result["chunk_text"],
            "chunking_method": result["chunking_method"],
            "embedding_model": result["embedding_model"],
        }
        for result in qdrant_results
    ]

    return {
        "query": query,
        "embedding_model": request.embedding_model,
        "top_k": request.top_k,
        "results_count": len(results),
        "results": results,
    }
