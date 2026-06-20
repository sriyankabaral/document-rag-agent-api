from functools import lru_cache

from sentence_transformers import SentenceTransformer


SUPPORTED_EMBEDDING_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
}


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str):
    if model_name not in SUPPORTED_EMBEDDING_MODELS:
        raise ValueError(
            "Unsupported embedding model. Use "
            "'sentence-transformers/all-MiniLM-L6-v2' or "
            "'BAAI/bge-small-en-v1.5'."
        )

    return SentenceTransformer(model_name)


def generate_embeddings(
    texts: list[str],
    model_name: str,
) -> list[list[float]]:
    if not texts:
        return []

    model = get_embedding_model(model_name)
    embeddings = model.encode(texts)
    return embeddings.tolist()


def get_embedding_dimension(model_name: str) -> int:
    model = get_embedding_model(model_name)
    dimension = model.get_sentence_embedding_dimension()

    if dimension is None:
        raise ValueError("Could not determine the embedding dimension.")

    return int(dimension)
