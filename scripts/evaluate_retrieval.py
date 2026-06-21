import argparse
import csv
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (  # noqa: E402
    EVALUATION_CHUNKING_METHODS,
    SUPPORTED_EMBEDDING_MODELS,
)
from app.services.chunker import chunk_text  # noqa: E402
from app.services.embedding_service import generate_embeddings  # noqa: E402
from app.services.llm_service import (  # noqa: E402
    OLLAMA_MODEL,
    generate_answer_with_ollama,
)
from app.services.qdrant_service import (  # noqa: E402
    qdrant_client,
    search_similar_chunks,
    store_chunks_in_qdrant,
)
from app.services.text_extractor import extract_text_from_file  # noqa: E402


CSV_COLUMNS = [
    "test_id",
    "question",
    "expected_document",
    "chunking_method",
    "embedding_model",
    "search_algorithm",
    "retrieved_sources",
    "correct_document_retrieved",
    "embedding_latency_ms",
    "search_latency_ms",
    "total_retrieval_latency_ms",
    "llm_generation_latency_ms",
    "end_to_end_latency_ms",
    "top_chunk_preview",
    "notes",
]


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    expected_document: str


@dataclass(frozen=True)
class EvaluationDocument:
    label: str
    path: Path
    text: str


QUESTIONS = [
    EvaluationQuestion(
        "What is Sriyanka's educational background?",
        "resume",
    ),
    EvaluationQuestion(
        "What AI or machine learning projects are mentioned in Sriyanka's resume?",
        "resume",
    ),
    EvaluationQuestion(
        "What are the main employee responsibilities in the IT company regulations?",
        "regulations",
    ),
    EvaluationQuestion(
        "What actions are prohibited according to the IT company regulations?",
        "regulations",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare chunking, embeddings, and Qdrant search modes."
    )
    parser.add_argument("--resume-path", type=Path, required=True)
    parser.add_argument("--regulations-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Measure Ollama answer-generation and end-to-end latency.",
    )
    parser.add_argument(
        "--llm-model",
        default=OLLAMA_MODEL,
        help=f"Ollama model used with --include-llm (default: {OLLAMA_MODEL}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "retrieval_evaluation_results.csv",
    )
    return parser.parse_args()


def load_document(label: str, path: Path) -> EvaluationDocument:
    resolved_path = path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Document not found: {resolved_path}")

    extension = resolved_path.suffix.lower()
    if extension not in {".pdf", ".txt"}:
        raise ValueError(f"Unsupported evaluation file: {resolved_path}")

    text = extract_text_from_file(str(resolved_path), extension)
    if not text.strip():
        raise ValueError(f"No readable text found in: {resolved_path}")

    return EvaluationDocument(label=label, path=resolved_path, text=text)


def evaluation_collection_name(
    chunking_method: str,
    embedding_model: str,
) -> str:
    model_label = "minilm" if "MiniLM" in embedding_model else "bge"
    return f"eval_{chunking_method}_{model_label}"


def prepare_collection(
    documents: list[EvaluationDocument],
    chunking_method: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    collection_name = evaluation_collection_name(
        chunking_method,
        embedding_model,
    )
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)

    for document in documents:
        chunks = chunk_text(
            document.text,
            method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        embeddings = generate_embeddings(chunks, embedding_model)
        store_chunks_in_qdrant(
            chunks=chunks,
            embeddings=embeddings,
            original_filename=document.path.name,
            saved_filename=document.path.name,
            file_type=document.path.suffix.lower(),
            chunking_method=chunking_method,
            embedding_model=embedding_model,
            collection_name=collection_name,
        )

    return collection_name


def unique_sources(results: list[dict]) -> list[str]:
    sources = []
    for result in results:
        source = result.get("original_filename")
        if source and source not in sources:
            sources.append(source)
    return sources


def run_evaluation(args: argparse.Namespace) -> list[dict]:
    documents = [
        load_document("resume", args.resume_path),
        load_document("regulations", args.regulations_path),
    ]
    expected_filenames = {
        document.label: document.path.name for document in documents
    }
    rows = []
    test_number = 1

    for chunking_method in EVALUATION_CHUNKING_METHODS:
        for embedding_model in SUPPORTED_EMBEDDING_MODELS:
            collection_name = prepare_collection(
                documents=documents,
                chunking_method=chunking_method,
                embedding_model=embedding_model,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            for question in QUESTIONS:
                for search_algorithm, exact in (
                    ("hnsw", False),
                    ("exact", True),
                ):
                    started_at = time.perf_counter()
                    query_embedding = generate_embeddings(
                        [question.question],
                        embedding_model,
                    )[0]
                    embedding_latency_ms = (
                        time.perf_counter() - started_at
                    ) * 1000

                    started_at = time.perf_counter()
                    results = search_similar_chunks(
                        query_embedding=query_embedding,
                        collection_name=collection_name,
                        top_k=args.top_k,
                        exact=exact,
                    )
                    search_latency_ms = (
                        time.perf_counter() - started_at
                    ) * 1000
                    total_retrieval_latency_ms = (
                        embedding_latency_ms + search_latency_ms
                    )
                    sources = unique_sources(results)
                    accuracy_sources = unique_sources(results[:3])
                    expected_filename = expected_filenames[
                        question.expected_document
                    ]
                    top_chunk = results[0].get("chunk_text", "") if results else ""
                    llm_latency: float | str = "not_measured"
                    end_to_end_latency: float | str = "not_measured"
                    notes = [
                        f"requested_top_k={args.top_k}",
                        "accuracy_cutoff=3",
                    ]

                    if args.include_llm:
                        notes.append(f"llm_model={args.llm_model}")
                        context_chunks = [
                            str(result.get("chunk_text", ""))
                            for result in results
                            if result.get("chunk_text")
                        ]
                        try:
                            started_at = time.perf_counter()
                            generate_answer_with_ollama(
                                query=question.question,
                                context_chunks=context_chunks,
                                model_name=args.llm_model,
                            )
                            measured_llm_latency = (
                                time.perf_counter() - started_at
                            ) * 1000
                            llm_latency = round(measured_llm_latency, 3)
                            end_to_end_latency = round(
                                total_retrieval_latency_ms
                                + measured_llm_latency,
                                3,
                            )
                        except Exception as exc:
                            notes.append(
                                f"llm_timing_failed={type(exc).__name__}"
                            )

                    rows.append(
                        {
                            "test_id": f"T{test_number:03d}",
                            "question": question.question,
                            "expected_document": question.expected_document,
                            "chunking_method": chunking_method,
                            "embedding_model": embedding_model,
                            "search_algorithm": search_algorithm,
                            "retrieved_sources": "; ".join(sources),
                            "correct_document_retrieved": (
                                expected_filename in accuracy_sources
                            ),
                            "embedding_latency_ms": round(
                                embedding_latency_ms,
                                3,
                            ),
                            "search_latency_ms": round(
                                search_latency_ms,
                                3,
                            ),
                            "total_retrieval_latency_ms": round(
                                total_retrieval_latency_ms,
                                3,
                            ),
                            "llm_generation_latency_ms": llm_latency,
                            "end_to_end_latency_ms": end_to_end_latency,
                            "top_chunk_preview": " ".join(
                                str(top_chunk).split()
                            )[:250],
                            "notes": "; ".join(notes),
                        }
                    )
                    test_number += 1

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_latency_groups(
    rows: list[dict],
    group_key: str,
    latency_key: str,
    heading: str,
) -> None:
    grouped_latencies: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        latency = row[latency_key]
        if isinstance(latency, (int, float)):
            grouped_latencies[str(row[group_key])].append(float(latency))

    print(f"\n{heading}")
    print("-" * 76)
    if not grouped_latencies:
        print("No successful measurements.")
        return
    for value, latencies in grouped_latencies.items():
        print(f"{value:<60} {mean(latencies):>10.3f} ms")


def print_average_latency(
    rows: list[dict],
    latency_key: str,
    heading: str,
) -> None:
    latencies = [
        float(row[latency_key])
        for row in rows
        if isinstance(row[latency_key], (int, float))
    ]
    print(f"\n{heading}")
    print("-" * 76)
    if latencies:
        print(f"{mean(latencies):.3f} ms")
    else:
        print("No successful measurements.")


def print_summary(
    rows: list[dict],
    output_path: Path,
    include_llm: bool,
) -> None:
    correct_tests = sum(
        1 for row in rows if row["correct_document_retrieved"]
    )
    accuracy = (correct_tests / len(rows) * 100) if rows else 0

    print("\nRetrieval Evaluation Summary")
    print("=" * 76)
    print(f"Total tests: {len(rows)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print_latency_groups(
        rows,
        "embedding_model",
        "embedding_latency_ms",
        "Average embedding latency by embedding model",
    )
    print_latency_groups(
        rows,
        "search_algorithm",
        "search_latency_ms",
        "Average Qdrant search latency by search algorithm",
    )
    print_latency_groups(
        rows,
        "embedding_model",
        "total_retrieval_latency_ms",
        "Average total retrieval latency by embedding model",
    )
    print_latency_groups(
        rows,
        "search_algorithm",
        "total_retrieval_latency_ms",
        "Average total retrieval latency by search algorithm",
    )
    if include_llm:
        print_average_latency(
            rows,
            "llm_generation_latency_ms",
            "Average LLM generation latency",
        )
        print_average_latency(
            rows,
            "end_to_end_latency_ms",
            "Average end-to-end latency",
        )
    print(f"\nCSV written to: {output_path.resolve()}")


def main() -> None:
    args = parse_args()
    if not 1 <= args.top_k <= 10:
        raise ValueError("--top-k must be between 1 and 10.")

    rows = run_evaluation(args)
    write_csv(rows, args.output)
    print_summary(rows, args.output, args.include_llm)


if __name__ == "__main__":
    main()
