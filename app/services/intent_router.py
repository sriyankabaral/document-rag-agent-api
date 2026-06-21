import re
from enum import Enum

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class AgentIntent(str, Enum):
    BOOKING = "booking"
    DOCUMENT = "document"
    MEMORY = "memory"
    PERSONAL_INFO = "personal_info"
    GENERAL = "general"


BOOKING_PATTERNS = [
    r"\bbook\s+(?:an?\s+)?interview\b",
    r"\bschedule\s+(?:an?\s+)?interview\b",
    r"\barrange\s+(?:an?\s+)?interview\b",
    r"\bset\s+up\s+(?:an?\s+)?interview\b",
    r"\binterview\s+on\s+\d{4}-\d{2}-\d{2}\b",
]
DOCUMENT_PATTERNS = [
    r"\baccording\s+to\s+(?:the\s+)?document\b",
    r"\buploaded\s+(?:document|file)\b",
    r"\b(?:document|resume|policy|regulations?|knowledge base)\b",
    r"\bfile\s+content\b",
]
MEMORY_PATTERNS = [
    r"\bwhat\s+is\s+my\s+(?:name|email)\b",
    r"\bdo\s+you\s+remember\s+my\s+(?:name|email)\b",
    r"\bwhat\s+did\s+i\s+tell\s+you\b",
    r"\bwhat\s+information\s+did\s+i\s+provide\b",
    r"\bwhat\s+do\s+you\s+(?:know|remember)\s+about\s+me\b",
]
EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
NAME_PATTERN = re.compile(
    r"\bmy\s+name\s+is\s+(.+?)(?=\s+and\s+my\b|[,;.!?]|$)",
    re.IGNORECASE,
)


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _has_personal_information(text: str) -> bool:
    return bool(EMAIL_PATTERN.search(text) or NAME_PATTERN.search(text))


def _booking_conversation_is_active(history: list[BaseMessage]) -> bool:
    recent_assistant_messages = [
        str(message.content).lower()
        for message in history[-4:]
        if isinstance(message, AIMessage)
    ]
    booking_follow_up_phrases = (
        "complete the interview booking",
        "interview date",
        "interview time",
        "schedule the interview",
        "booking details",
    )
    return any(
        phrase in message
        for message in recent_assistant_messages
        for phrase in booking_follow_up_phrases
    )


def classify_agent_intent(
    query: str,
    history: list[BaseMessage],
) -> AgentIntent:
    if _matches_any(query, BOOKING_PATTERNS):
        return AgentIntent.BOOKING
    if _matches_any(query, DOCUMENT_PATTERNS):
        return AgentIntent.DOCUMENT
    if _matches_any(query, MEMORY_PATTERNS):
        return AgentIntent.MEMORY
    if _has_personal_information(query):
        return AgentIntent.PERSONAL_INFO
    if _booking_conversation_is_active(history):
        return AgentIntent.BOOKING
    return AgentIntent.GENERAL


def extract_personal_information(
    history: list[BaseMessage],
    current_query: str | None = None,
) -> dict[str, str]:
    details: dict[str, str] = {}
    texts = [
        str(message.content)
        for message in history
        if isinstance(message, HumanMessage)
    ]
    if current_query:
        texts.append(current_query)

    for text in texts:
        email_match = EMAIL_PATTERN.search(text)
        if email_match:
            details["email"] = email_match.group(0)

        name_match = NAME_PATTERN.search(text)
        if name_match:
            details["name"] = name_match.group(1).strip()

    return details


def answer_memory_question(
    query: str,
    history: list[BaseMessage],
) -> str:
    details = extract_personal_information(history)
    normalized_query = query.lower()

    if "email" in normalized_query:
        if details.get("email"):
            return f"Your email is {details['email']}."
        return "You have not provided an email in this conversation yet."

    if "name" in normalized_query:
        if details.get("name"):
            return f"Your name is {details['name']}."
        return "You have not provided your name in this conversation yet."

    known_details = []
    if details.get("name"):
        known_details.append(f"your name is {details['name']}")
    if details.get("email"):
        known_details.append(f"your email is {details['email']}")

    if known_details:
        return "You told me that " + " and ".join(known_details) + "."
    return "You have not provided personal information in this conversation yet."


def acknowledge_personal_information(
    query: str,
    history: list[BaseMessage],
) -> str:
    details = extract_personal_information(history, current_query=query)
    known_details = []
    if details.get("name"):
        known_details.append(f"your name is {details['name']}")
    if details.get("email"):
        known_details.append(f"your email is {details['email']}")

    if known_details:
        return "I will remember that " + " and ".join(known_details) + "."
    return "I will remember that information for this conversation."
