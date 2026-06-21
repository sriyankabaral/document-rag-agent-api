from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, model_validator

from app.services.agent_service import AgentExecutionError, run_agent
from app.services.redis_memory import ConversationMemoryError


router = APIRouter(prefix="/agent", tags=["Agent"])


class AgentQueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    conversation_id: str | None = None
    top_k: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama3.2"

    @model_validator(mode="after")
    def validate_conversation_identifier(self):
        session_id = (self.session_id or "").strip()
        conversation_id = (self.conversation_id or "").strip()

        if not session_id and not conversation_id:
            raise ValueError("session_id or conversation_id is required.")

        if session_id and conversation_id and session_id != conversation_id:
            raise ValueError(
                "session_id and conversation_id must match when both are set."
            )

        resolved_id = session_id or conversation_id
        if len(resolved_id) > 200:
            raise ValueError("The conversation identifier is too long.")

        self.session_id = resolved_id
        self.conversation_id = resolved_id
        return self


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
        result = run_agent(
            query=query,
            session_id=request.session_id,
            top_k=request.top_k,
            embedding_model=request.embedding_model,
            llm_model=request.llm_model,
        )
    except ConversationMemoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except AgentExecutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return {
        "query": query,
        "session_id": request.session_id,
        "answer": result.answer,
        "llm_model": request.llm_model,
        "embedding_model": request.embedding_model,
        "retrieved_context_count": result.retrieved_context_count,
        "sources": result.sources,
        "booking": result.booking,
    }
