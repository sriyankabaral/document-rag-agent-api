import os

import requests
from dotenv import load_dotenv


load_dotenv()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
).rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def generate_answer_with_ollama(
    query: str,
    context_chunks: list[str],
    model_name: str | None = None,
) -> str:
    context = "\n\n".join(context_chunks)
    selected_model = model_name or OLLAMA_MODEL
    system_prompt = (
        "You are a helpful RAG assistant. Answer only using the provided "
        "document context. If the context does not contain the answer, say: "
        "The uploaded document does not contain enough information to answer "
        "this question. Keep the answer clear and concise."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{query}"

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(
            "Failed to connect to Ollama. Make sure Ollama is installed "
            "and running locally."
        ) from exc

    try:
        response_json = response.json()
        answer = response_json["message"]["content"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Ollama returned an unexpected response format."
        ) from exc

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Ollama returned an unexpected response format.")

    return answer.strip()
