from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.booking import (
    InterviewBookingRequest,
    InterviewBookingResponse,
)
from app.services.booking_service import (
    BookingPersistenceError,
    create_interview_booking,
)


router = APIRouter(tags=["Interview Booking"])


@router.post(
    "/book-interview",
    response_model=InterviewBookingResponse,
    status_code=status.HTTP_201_CREATED,
)
def book_interview(
    booking_data: InterviewBookingRequest,
    db: Session = Depends(get_db),
):
    try:
        return create_interview_booking(db, booking_data)
    except BookingPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
