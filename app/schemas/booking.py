from datetime import date, datetime, time

from pydantic import BaseModel, ConfigDict, EmailStr, field_validator


class InterviewBookingRequest(BaseModel):
    full_name: str
    email: EmailStr
    interview_date: date
    interview_time: time
    notify_candidate: bool = False

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: str) -> str:
        full_name = value.strip()
        if not full_name:
            raise ValueError("Full name cannot be empty.")
        return full_name


class InterviewBookingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    email: EmailStr
    interview_date: date
    interview_time: time
    notify_candidate: bool
    admin_email: str | None
    email_sent_admin: bool
    email_sent_candidate: bool
    email_status: str
    email_error: str | None
    created_at: datetime
