import os
import smtplib
from dataclasses import dataclass
from datetime import date, time
from email.message import EmailMessage

from dotenv import load_dotenv


load_dotenv()


@dataclass
class EmailDeliveryResult:
    admin_email: str | None
    email_sent_admin: bool
    email_sent_candidate: bool
    email_status: str
    email_error: str | None


@dataclass
class SMTPSettings:
    host: str
    port: int
    sender_email: str
    password: str
    admin_email: str
    use_tls: bool


def _parse_boolean(value: str) -> bool:
    normalized_value = value.strip().lower()
    if normalized_value in {"true", "1", "yes", "on"}:
        return True
    if normalized_value in {"false", "0", "no", "off"}:
        return False
    raise ValueError("SMTP_USE_TLS must be true or false.")


def _load_smtp_settings() -> SMTPSettings:
    values = {
        "SMTP_HOST": os.getenv("SMTP_HOST"),
        "SMTP_EMAIL": os.getenv("SMTP_EMAIL"),
        "SMTP_PASSWORD": os.getenv("SMTP_PASSWORD"),
        "SMTP_TO_EMAIL": os.getenv("SMTP_TO_EMAIL"),
    }
    missing_settings = [name for name, value in values.items() if not value]
    if missing_settings:
        raise ValueError(
            f"Missing SMTP settings: {', '.join(missing_settings)}."
        )

    try:
        port = int(os.getenv("SMTP_PORT", "587"))
    except ValueError as exc:
        raise ValueError("SMTP_PORT must be an integer.") from exc

    use_tls = _parse_boolean(os.getenv("SMTP_USE_TLS", "true"))

    return SMTPSettings(
        host=values["SMTP_HOST"],
        port=port,
        sender_email=values["SMTP_EMAIL"],
        password=values["SMTP_PASSWORD"],
        admin_email=values["SMTP_TO_EMAIL"],
        use_tls=use_tls,
    )


def _build_admin_email(
    settings: SMTPSettings,
    full_name: str,
    candidate_email: str,
    interview_date: date,
    interview_time: time,
    notify_candidate: bool,
) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = "New Interview Booking Request"
    message["From"] = settings.sender_email
    message["To"] = settings.admin_email
    message.set_content(
        "A new interview booking has been received.\n\n"
        f"Candidate: {full_name}\n"
        f"Candidate email: {candidate_email}\n"
        f"Interview date: {interview_date.isoformat()}\n"
        f"Interview time: {interview_time.strftime('%H:%M')}\n"
        f"Candidate notification enabled: {notify_candidate}\n"
    )
    return message


def _build_candidate_email(
    settings: SMTPSettings,
    full_name: str,
    candidate_email: str,
    interview_date: date,
    interview_time: time,
) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = "Interview Booking Confirmation"
    message["From"] = settings.sender_email
    message["To"] = candidate_email
    message.set_content(
        f"Hello {full_name},\n\n"
        "Your interview booking request has been received.\n\n"
        f"Interview date: {interview_date.isoformat()}\n"
        f"Interview time: {interview_time.strftime('%H:%M')}\n\n"
        "We will contact you if any additional information is needed.\n"
    )
    return message


def send_booking_emails(
    full_name: str,
    candidate_email: str,
    interview_date: date,
    interview_time: time,
    notify_candidate: bool,
) -> EmailDeliveryResult:
    admin_email = os.getenv("SMTP_TO_EMAIL") or None

    try:
        settings = _load_smtp_settings()
    except ValueError as exc:
        return EmailDeliveryResult(
            admin_email=admin_email,
            email_sent_admin=False,
            email_sent_candidate=False,
            email_status="failed",
            email_error=str(exc),
        )

    admin_sent = False
    candidate_sent = False
    errors = []

    try:
        with smtplib.SMTP(
            settings.host,
            settings.port,
            timeout=30,
        ) as smtp_server:
            if settings.use_tls:
                smtp_server.starttls()
            smtp_server.login(settings.sender_email, settings.password)

            try:
                smtp_server.send_message(
                    _build_admin_email(
                        settings=settings,
                        full_name=full_name,
                        candidate_email=candidate_email,
                        interview_date=interview_date,
                        interview_time=interview_time,
                        notify_candidate=notify_candidate,
                    )
                )
                admin_sent = True
            except Exception as exc:
                errors.append(f"Admin email failed: {exc}")

            if notify_candidate:
                try:
                    smtp_server.send_message(
                        _build_candidate_email(
                            settings=settings,
                            full_name=full_name,
                            candidate_email=candidate_email,
                            interview_date=interview_date,
                            interview_time=interview_time,
                        )
                    )
                    candidate_sent = True
                except Exception as exc:
                    errors.append(f"Candidate email failed: {exc}")
    except Exception as exc:
        errors.append(f"SMTP connection failed: {exc}")

    all_required_emails_sent = admin_sent and (
        candidate_sent if notify_candidate else True
    )
    if all_required_emails_sent:
        email_status = "sent"
    elif admin_sent or candidate_sent:
        email_status = "partial"
    else:
        email_status = "failed"

    return EmailDeliveryResult(
        admin_email=settings.admin_email,
        email_sent_admin=admin_sent,
        email_sent_candidate=candidate_sent,
        email_status=email_status,
        email_error="; ".join(errors) or None,
    )
