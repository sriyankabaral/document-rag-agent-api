import json

from langchain_core.tools import tool
from pydantic import ValidationError

from app.config import (
    AGENT_NOTIFY_CANDIDATE_BY_DEFAULT,
    DEFAULT_EMBEDDING_MODEL,
)
from app.db.database import SessionLocal
from app.schemas.booking import (
    InterviewBookingRequest,
    InterviewBookingResponse,
)
from app.services.booking_service import (
    BookingPersistenceError,
    create_interview_booking,
)
from app.services.embedding_service import generate_embeddings
from app.services.qdrant_service import search_similar_chunks


@tool
def search_document_chunks(
    query: str,
    top_k: int = 5,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> str:
    """Search uploaded documents for chunks relevant to a user question."""
    try:
        query_embedding = generate_embeddings([query], embedding_model)[0]
        results = search_similar_chunks(
            query_embedding=query_embedding,
            top_k=top_k,
            embedding_model=embedding_model,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc), "results": []})

    return json.dumps({"results": results})


@tool
def book_interview_tool(
    full_name: str,
    email: str,
    interview_date: str,
    interview_time: str,
) -> str:
    """Book an interview after name, email, date, and time are confirmed."""
    try:
        booking_data = InterviewBookingRequest.model_validate(
            {
                "full_name": full_name,
                "email": email,
                "interview_date": interview_date,
                "interview_time": interview_time,
                "notify_candidate": AGENT_NOTIFY_CANDIDATE_BY_DEFAULT,
            }
        )
    except ValidationError as exc:
        return json.dumps(
            {"error": "Invalid booking details.", "details": str(exc)}
        )

    try:
        with SessionLocal() as db:
            booking = create_interview_booking(db, booking_data)
            booking_response = InterviewBookingResponse.model_validate(
                booking
            ).model_dump(mode="json")
    except BookingPersistenceError as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps({"booking": booking_response})
