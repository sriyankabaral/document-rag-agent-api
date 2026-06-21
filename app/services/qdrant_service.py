import os
from uuid import uuid4

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_chunks")

qdrant_client = QdrantClient(url=QDRANT_URL)


def ensure_collection(collection_name: str, vector_size: int) -> None:
    if qdrant_client.collection_exists(collection_name):
        return

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def store_chunks_in_qdrant(
    chunks: list[str],
    embeddings: list[list[float]],
    original_filename: str,
    saved_filename: str,
    file_type: str,
    chunking_method: str,
    embedding_model: str,
    collection_name: str = COLLECTION_NAME,
    document_id: int | None = None,
) -> list[str]:
    if not chunks or not embeddings:
        raise ValueError("Chunks and embeddings cannot be empty.")

    if len(chunks) != len(embeddings):
        raise ValueError("The number of chunks and embeddings must match.")

    vector_size = len(embeddings[0])
    if vector_size == 0:
        raise ValueError("Embedding vectors cannot be empty.")

    ensure_collection(collection_name, vector_size)

    point_ids = [str(uuid4()) for _ in chunks]
    points = []
    for chunk_index, (point_id, chunk, embedding) in enumerate(
        zip(point_ids, chunks, embeddings)
    ):
        payload = {
            "file_name": original_filename,
            "original_filename": original_filename,
            "saved_filename": saved_filename,
            "file_type": file_type,
            "chunk_index": chunk_index,
            "chunk_text": chunk,
            "chunking_method": chunking_method,
            "embedding_model": embedding_model,
        }
        if document_id is not None:
            payload["document_id"] = document_id

        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        )

    qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
    )

    return point_ids


def search_similar_chunks(
    query_embedding: list[float],
    collection_name: str = COLLECTION_NAME,
    top_k: int = 5,
    exact: bool = False,
    embedding_model: str | None = None,
) -> list[dict]:
    if not query_embedding:
        raise ValueError("Query embedding cannot be empty.")

    try:
        if not qdrant_client.collection_exists(collection_name):
            raise ValueError(
                f"Qdrant collection '{collection_name}' does not exist."
            )

        response = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
            query_filter=(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="embedding_model",
                            match=models.MatchValue(value=embedding_model),
                        )
                    ]
                )
                if embedding_model
                else None
            ),
            search_params=(
                models.SearchParams(exact=True) if exact else None
            ),
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "Failed to search Qdrant. Make sure Qdrant is running."
        ) from exc

    results = []
    for point in response.points:
        payload = point.payload or {}
        results.append(
            {
                "score": float(point.score),
                "original_filename": payload.get("original_filename"),
                "saved_filename": payload.get("saved_filename"),
                "file_type": payload.get("file_type"),
                "chunk_index": payload.get("chunk_index"),
                "chunk_text": payload.get("chunk_text"),
                "chunking_method": payload.get("chunking_method"),
                "embedding_model": payload.get("embedding_model"),
                "document_id": payload.get("document_id"),
            }
        )

    return results
