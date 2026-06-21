# Retrieval Evaluation Findings

## Objective

Compare retrieval accuracy and latency across chunking strategies, embedding
models, and Qdrant search modes. Optionally measure Ollama generation and the
complete measured RAG response pipeline.

## Test Documents

- Sriyanka resume (`resume.pdf` or the path passed with `--resume-path`)
- IT company regulations (`it_company_regulations.txt` or the path passed
  with `--regulations-path`)

## Compared Chunking Strategies

- `recursive`: LangChain `RecursiveCharacterTextSplitter`, 500-character
  chunks with 100-character overlap by default.
- `custom`: fixed-size 500-character windows with 100-character overlap by
  default.

## Compared Embedding Models

- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`

## Compared Qdrant Search Algorithms

- `hnsw`: Qdrant's default approximate nearest-neighbor search.
- `exact`: Qdrant exact vector search via `SearchParams(exact=True)`.

## Evaluation Method

Each chunking/model combination is stored in a separate Qdrant collection.
Four fixed questions are searched with both Qdrant modes. A test is counted
as correct when the expected document appears among the top-k retrieved
sources. Accuracy always checks only the first three retrieved results, even
when a larger `--top-k` value is requested.

The latency fields have distinct meanings:

- `embedding_latency_ms`: query embedding generation time.
- `search_latency_ms`: Qdrant vector search time only.
- `total_retrieval_latency_ms`: embedding plus Qdrant search time.
- `llm_generation_latency_ms`: Ollama answer-generation time.
- `end_to_end_latency_ms`: embedding, search, and Ollama generation time.

LLM and end-to-end timings are recorded only when `--include-llm` is used.
They do not affect retrieval accuracy.

## Results

The evaluation ran 32 retrieval tests: four questions across four
chunking/model configurations and two Qdrant search modes. The expected
document appeared in the top three results for all 32 tests.

| Configuration | Accuracy | Embedding | Search | Total retrieval |
| --- | ---: | ---: | ---: | ---: |
| Recursive + MiniLM | 100% (8/8) | 69.090 ms | 133.179 ms | 202.270 ms |
| Recursive + BGE | 100% (8/8) | 55.919 ms | 76.622 ms | 132.541 ms |
| Custom + MiniLM | 100% (8/8) | 20.566 ms | 39.929 ms | 60.495 ms |
| Custom + BGE | 100% (8/8) | 40.654 ms | 45.435 ms | 86.090 ms |

| Overall comparison | Accuracy | Relevant average latency |
| --- | ---: | ---: |
| MiniLM embedding | 100% (16/16) | 44.828 ms embedding |
| BGE embedding | 100% (16/16) | 48.287 ms embedding |
| MiniLM total retrieval | 100% (16/16) | 131.382 ms retrieval |
| BGE total retrieval | 100% (16/16) | 109.315 ms retrieval |
| HNSW | 100% (16/16) | 63.443 ms search |
| Exact | 100% (16/16) | 84.139 ms search |

The run used `--include-llm`. Average Ollama generation latency was
12,183.539 ms, and average measured end-to-end latency was 12,303.888 ms.

## Observations

- Both chunking strategies retrieved the expected source for every question,
  so this test did not show an accuracy advantage for either strategy.
- Both embedding models achieved the same source-level accuracy. MiniLM
  generated query embeddings slightly faster, while BGE had lower total
  retrieval latency (109.315 ms versus 131.382 ms).
- Custom chunking had lower total retrieval latency than recursive chunking
  (73.292 ms versus 167.405 ms). Collection size and execution order can
  influence this result, so it should not be treated as a general claim that
  custom chunking is faster.
- HNSW search was 20.696 ms faster than exact search on average in this run.
  The dataset is too small to establish how either mode scales.
- Ollama generation dominated user-facing latency: its 12.184-second average
  was much larger than the 120.349 ms average retrieval time.
- No question missed its expected source within the top three results.

## Conclusion

All tested configurations were equally accurate at the document-source level.
Custom + MiniLM recorded the lowest average total retrieval latency in this
run at 60.495 ms, and HNSW recorded lower average search latency than exact
search. These results do not establish a universal winner: recursive chunking
preserves natural boundaries, and performance can change with collection
size and hardware. A larger labeled question set should be used before
changing the application's defaults.

## Limitations

- Only two documents and four questions were evaluated.
- Accuracy checks whether the expected document appears in the top three; it
  does not grade whether the exact best passage ranked first.
- End-to-end timing covers query embedding, Qdrant search, and Ollama answer
  generation. It does not include document ingestion or FastAPI request
  handling.
- Tests ran sequentially, so cache state and execution order may affect
  latency.
- Local CPU, memory pressure, Ollama model state, and background processes can
  cause latency to vary between runs.
- LLM timing measures duration, not the correctness of Ollama's answer.
