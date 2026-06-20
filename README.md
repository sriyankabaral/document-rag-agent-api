# document-rag-agent-api

Backend-only FastAPI project for a document RAG agent API.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the server:

```powershell
uvicorn app.main:app --reload
```

Test the health endpoint:

```powershell
curl http://127.0.0.1:8000/health
```

## Run Qdrant and Redis with Docker

Start Qdrant and Redis in the background:

```powershell
docker compose up -d
```

Check that the containers are running:

```powershell
docker compose ps
```

Stop and remove the containers:

```powershell
docker compose down
```

## SQLite Database

SQLite stores document-level metadata and interview bookings. The `app.db`
database file is created automatically when the FastAPI app starts and is
ignored by Git.

## Test Document Upload

Run the server and open http://127.0.0.1:8000/docs in your browser. In the
Swagger docs, open `POST /documents/upload`, select a `.pdf` or `.txt` file,
and execute the request. The API extracts text from PDF and TXT files after
upload. Scanned image-based PDFs may not produce text because OCR is not
included. The upload API also supports `recursive` and `sentence` text
chunking. Choose the `chunking_method` field in Swagger when testing
`POST /documents/upload`. The API then generates embeddings for every chunk
using `sentence-transformers/all-MiniLM-L6-v2` or
`BAAI/bge-small-en-v1.5`. The first request may take longer because the
selected embedding model must be downloaded. Uploaded chunks and embeddings
are stored in Qdrant. Qdrant must be running before testing the upload API:

```powershell
docker compose up -d
```

## Test Document Search

First upload a document, then call `POST /documents/search` from Swagger. For
example, search for `What projects are mentioned in this document?`. This
endpoint only retrieves relevant chunks; it does not generate a final LLM
answer yet.

## Test RAG Agent

`POST /agent/query` generates an answer using relevant Qdrant chunks and a
local Ollama model. Install Ollama, make sure it is running, and download the
model:

```powershell
ollama pull llama3.2
```

Use `ollama list` to check which models are available, and make sure
`llm_model` matches one of them. You can configure the local URL and default
model in a root `.env` file based on `.env.example`.

Example request body:

```json
{
  "query": "What AI projects are mentioned in the document?",
  "top_k": 5,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model": "llama3.2"
}
```
