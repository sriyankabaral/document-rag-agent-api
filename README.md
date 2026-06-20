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

SQLite stores document metadata and interview bookings. The `app.db` database
file is created automatically when the FastAPI app starts.

## Test Document Upload

Run the server and open http://127.0.0.1:8000/docs in your browser. In the
Swagger docs, open `POST /documents/upload`, select a `.pdf` or `.txt` file,
and execute the request. The API extracts text from PDF and TXT files after
upload. Scanned image-based PDFs may not produce text because OCR is not
included. The upload API also supports `recursive` and `sentence` text
chunking. Choose the `chunking_method` field in Swagger when testing
`POST /documents/upload`.
