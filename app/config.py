AGENT_NOTIFY_CANDIDATE_BY_DEFAULT = True

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]

DEFAULT_CHUNKING_METHOD = "recursive"
SUPPORTED_CHUNKING_METHODS = ["recursive", "custom", "sentence"]
EVALUATION_CHUNKING_METHODS = ["recursive", "custom"]
