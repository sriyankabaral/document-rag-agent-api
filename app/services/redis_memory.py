import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from redis import Redis
from redis.exceptions import RedisError


load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_CONVERSATION_TTL_SECONDS = int(
    os.getenv("REDIS_CONVERSATION_TTL_SECONDS", "86400")
)
REDIS_HISTORY_LIMIT = int(os.getenv("REDIS_HISTORY_LIMIT", "20"))


class ConversationMemoryError(Exception):
    pass


@lru_cache(maxsize=1)
def get_redis_client() -> Redis:
    return Redis.from_url(REDIS_URL, decode_responses=True)


def _conversation_key(session_id: str) -> str:
    return f"agent:conversation:{session_id}"


def load_conversation(session_id: str) -> list[BaseMessage]:
    try:
        entries = get_redis_client().lrange(
            _conversation_key(session_id),
            -REDIS_HISTORY_LIMIT,
            -1,
        )
    except RedisError as exc:
        raise ConversationMemoryError(
            "Redis conversational memory is unavailable."
        ) from exc

    messages: list[BaseMessage] = []
    for entry in entries:
        try:
            item = json.loads(entry)
        except (TypeError, json.JSONDecodeError):
            continue

        if item.get("role") == "user":
            messages.append(HumanMessage(content=item.get("content", "")))
        elif item.get("role") == "assistant":
            messages.append(AIMessage(content=item.get("content", "")))

    return messages


def save_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    key = _conversation_key(session_id)
    entries = [
        json.dumps({"role": "user", "content": user_message}),
        json.dumps({"role": "assistant", "content": assistant_message}),
    ]

    try:
        pipeline = get_redis_client().pipeline()
        pipeline.rpush(key, *entries)
        pipeline.ltrim(key, -REDIS_HISTORY_LIMIT, -1)
        pipeline.expire(key, REDIS_CONVERSATION_TTL_SECONDS)
        pipeline.execute()
    except RedisError as exc:
        raise ConversationMemoryError(
            "Redis conversational memory could not be saved."
        ) from exc
