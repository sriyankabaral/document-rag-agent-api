from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String

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
    preferred_date = Column(String, nullable=False)
    preferred_time = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)
