from sqlalchemy.orm import Session

from app.db.models import InterviewBooking
from app.schemas.booking import InterviewBookingRequest
from app.services.email_service import EmailDeliveryResult, send_booking_emails


class BookingPersistenceError(Exception):
    pass


def create_interview_booking(
    db: Session,
    booking_data: InterviewBookingRequest,
) -> InterviewBooking:
    booking = InterviewBooking(
        full_name=booking_data.full_name,
        email=str(booking_data.email),
        interview_date=booking_data.interview_date,
        interview_time=booking_data.interview_time,
        notify_candidate=booking_data.notify_candidate,
        email_status="pending",
    )

    try:
        db.add(booking)
        db.commit()
        db.refresh(booking)
    except Exception as exc:
        db.rollback()
        raise BookingPersistenceError(
            "Interview booking could not be saved."
        ) from exc

    try:
        email_result = send_booking_emails(
            full_name=booking.full_name,
            candidate_email=booking.email,
            interview_date=booking.interview_date,
            interview_time=booking.interview_time,
            notify_candidate=booking.notify_candidate,
        )
    except Exception as exc:
        email_result = EmailDeliveryResult(
            admin_email=None,
            email_sent_admin=False,
            email_sent_candidate=False,
            email_status="failed",
            email_error=f"Unexpected email error: {exc}",
        )

    booking.admin_email = email_result.admin_email
    booking.email_sent_admin = email_result.email_sent_admin
    booking.email_sent_candidate = email_result.email_sent_candidate
    booking.email_status = email_result.email_status
    booking.email_error = email_result.email_error

    try:
        db.commit()
        db.refresh(booking)
    except Exception as exc:
        db.rollback()
        raise BookingPersistenceError(
            "Booking was saved, but email status could not be updated."
        ) from exc

    return booking
