from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _migrate_interview_booking_table()


def _migrate_interview_booking_table():
    inspector = inspect(engine)
    if "interview_bookings" not in inspector.get_table_names():
        return

    columns = {
        column["name"]
        for column in inspector.get_columns("interview_bookings")
    }

    with engine.begin() as connection:
        if "preferred_date" in columns and "interview_date" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE interview_bookings "
                    "RENAME COLUMN preferred_date TO interview_date"
                )
            )
            columns.add("interview_date")

        if "preferred_time" in columns and "interview_time" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE interview_bookings "
                    "RENAME COLUMN preferred_time TO interview_time"
                )
            )
            columns.add("interview_time")

        new_columns = {
            "notify_candidate": "BOOLEAN NOT NULL DEFAULT 0",
            "admin_email": "VARCHAR",
            "email_sent_admin": "BOOLEAN NOT NULL DEFAULT 0",
            "email_sent_candidate": "BOOLEAN NOT NULL DEFAULT 0",
            "email_status": "VARCHAR NOT NULL DEFAULT 'pending'",
            "email_error": "TEXT",
        }

        for column_name, column_definition in new_columns.items():
            if column_name not in columns:
                connection.execute(
                    text(
                        f"ALTER TABLE interview_bookings ADD COLUMN "
                        f"{column_name} {column_definition}"
                    )
                )
