import os
from uuid import uuid4

from qdrant_client import QdrantClient, models


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
    points = [
        models.PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "original_filename": original_filename,
                "saved_filename": saved_filename,
                "file_type": file_type,
                "chunk_index": chunk_index,
                "chunk_text": chunk,
                "chunking_method": chunking_method,
                "embedding_model": embedding_model,
            },
        )
        for chunk_index, (point_id, chunk, embedding) in enumerate(
            zip(point_ids, chunks, embeddings)
        )
    ]

    qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
    )

    return point_ids
