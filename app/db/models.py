from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, Date, DateTime, Integer, String, Text, Time

from app.db.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    id = Column(Integer, primary_key=True)
    file_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    upload_time = Column(DateTime(timezone=True), default=utc_now)
    chunking_method = Column(String, nullable=True)
    embedding_model = Column(String, nullable=True)
    number_of_chunks = Column(Integer, default=0)
    qdrant_collection_name = Column(String, nullable=True)


class InterviewBooking(Base):
    __tablename__ = "interview_bookings"

    id = Column(Integer, primary_key=True)
    full_name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    interview_date = Column(Date, nullable=False)
    interview_time = Column(Time, nullable=False)
    notify_candidate = Column(Boolean, default=False, nullable=False)
    admin_email = Column(String, nullable=True)
    email_sent_admin = Column(Boolean, default=False, nullable=False)
    email_sent_candidate = Column(Boolean, default=False, nullable=False)
    email_status = Column(String, default="pending", nullable=False)
    email_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
