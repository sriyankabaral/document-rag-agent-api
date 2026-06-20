import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status


router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIRECTORY = Path("uploads")
ALLOWED_FILE_TYPES = {".pdf", ".txt"}


@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    try:
        original_filename = Path((file.filename or "").replace("\\", "/")).name
        file_type = Path(original_filename).suffix.lower()

        if file_type not in ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF and TXT files are allowed.",
            )

        UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

        saved_filename = f"{uuid4()}_{original_filename}"
        saved_path = UPLOAD_DIRECTORY / saved_filename

        with saved_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)
    finally:
        file.file.close()

    return {
        "message": "File uploaded successfully",
        "original_filename": original_filename,
        "saved_filename": saved_filename,
        "saved_path": saved_path.as_posix(),
        "file_type": file_type,
    }
