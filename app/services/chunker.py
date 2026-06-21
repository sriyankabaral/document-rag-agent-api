import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import DEFAULT_CHUNKING_METHOD, SUPPORTED_CHUNKING_METHODS


def chunk_text(
    text: str,
    method: str = DEFAULT_CHUNKING_METHOD,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[str]:
    if method not in SUPPORTED_CHUNKING_METHODS:
        supported_methods = ", ".join(SUPPORTED_CHUNKING_METHODS)
        raise ValueError(
            f"Unsupported chunking method. Use one of: {supported_methods}."
        )

    if not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be at least zero and smaller than chunk_size."
        )

    if method == "recursive":
        return _chunk_text_recursively(text, chunk_size, chunk_overlap)

    if method == "custom":
        return _chunk_text_by_fixed_size(text, chunk_size, chunk_overlap)

    if method == "sentence":
        return _chunk_text_by_sentence(text, chunk_size)

    return []


def _chunk_text_recursively(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def _chunk_text_by_sentence(text: str, chunk_size: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        combined_text = f"{current_chunk} {sentence}".strip()

        if current_chunk and len(combined_text) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk = combined_text

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _chunk_text_by_fixed_size(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break
        start = end - chunk_overlap

    return chunks
