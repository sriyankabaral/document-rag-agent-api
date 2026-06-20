from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.embedding_service import generate_embeddings
from app.services.llm_service import generate_answer_with_ollama
from app.services.qdrant_service import search_similar_chunks


router = APIRouter(prefix="/agent", tags=["Agent"])


class AgentQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama3.2"


@router.post("/query")
def query_agent(request: AgentQueryRequest):
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
        search_results = search_similar_chunks(
            query_embedding=query_embedding,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    retrieved_results = [
        result for result in search_results if result.get("chunk_text")
    ]

    if not retrieved_results:
        return {
            "query": query,
            "answer": "No relevant context was found.",
            "llm_model": request.llm_model,
            "embedding_model": request.embedding_model,
            "retrieved_context_count": 0,
            "sources": [],
        }

    context_chunks = [
        result["chunk_text"] for result in retrieved_results
    ]

    try:
        answer = generate_answer_with_ollama(
            query=query,
            context_chunks=context_chunks,
            model_name=request.llm_model,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate an answer with Ollama.",
        ) from exc

    sources = [
        {
            "original_filename": result["original_filename"],
            "chunk_index": result["chunk_index"],
            "score": result["score"],
            "chunking_method": result["chunking_method"],
            "embedding_model": result["embedding_model"],
        }
        for result in retrieved_results
    ]

    return {
        "query": query,
        "answer": answer,
        "llm_model": request.llm_model,
        "embedding_model": request.embedding_model,
        "retrieved_context_count": len(retrieved_results),
        "sources": sources,
    }
