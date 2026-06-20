import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    method: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[str]:
    if not text.strip():
        return []

    if method == "recursive":
        return _chunk_text_recursively(text, chunk_size, chunk_overlap)

    if method == "sentence":
        return _chunk_text_by_sentence(text, chunk_size)

    raise ValueError(
        "Unsupported chunking method. Use 'recursive' or 'sentence'."
    )


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
