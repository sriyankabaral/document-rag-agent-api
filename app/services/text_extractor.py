from pathlib import Path

from pypdf import PdfReader


def extract_text_from_file(file_path: str, file_extension: str) -> str:
    file_extension = file_extension.lower()

    if file_extension not in {".pdf", ".txt"}:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    try:
        if file_extension == ".txt":
            return Path(file_path).read_text(encoding="utf-8")

        reader = PdfReader(file_path)
        page_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(page_text)
    except Exception as exc:
        raise ValueError(
            f"Failed to extract text from the {file_extension} file."
        ) from exc
