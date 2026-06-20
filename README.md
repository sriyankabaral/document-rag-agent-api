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
