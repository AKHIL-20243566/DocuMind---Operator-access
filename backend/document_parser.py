"""Document parser — extracts text from PDF, DOCX, TXT, CSV, and Markdown files."""
import os
import csv
import io
import logging

logger = logging.getLogger(__name__)


def parse_file(file_path: str, filename: str = None) -> list[dict]:
    """Parse a file and return a list of document chunks with metadata."""
    if filename is None:
        filename = os.path.basename(file_path)

    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".pdf":
            return _parse_pdf(file_path, filename)
        elif ext == ".docx":
            return _parse_docx(file_path, filename)
        elif ext == ".txt":
            return _parse_txt(file_path, filename)
        elif ext == ".csv":
            return _parse_csv(file_path, filename)
        elif ext in (".md", ".markdown"):
            return _parse_markdown(file_path, filename)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return _parse_txt(file_path, filename)
    except Exception as e:
        logger.error(f"Error parsing {filename}: {e}")
        return []


def parse_bytes(content: bytes, filename: str) -> list[dict]:
    """Parse file content from bytes."""
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".pdf":
            return _parse_pdf_bytes(content, filename)
        elif ext == ".docx":
            return _parse_docx_bytes(content, filename)
        elif ext == ".txt":
            return _parse_txt_bytes(content, filename)
        elif ext == ".csv":
            return _parse_csv_bytes(content, filename)
        elif ext in (".md", ".markdown"):
            return _parse_txt_bytes(content, filename)
        else:
            return _parse_txt_bytes(content, filename)
    except Exception as e:
        logger.error(f"Error parsing {filename}: {e}")
        return []


# --- Chunking ---

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at sentence boundary
        if end < len(text):
            for sep in [". ", "\n\n", "\n", " "]:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size // 2:
                    end = start + last_sep + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


# --- PDF ---

def _parse_pdf(file_path: str, filename: str) -> list[dict]:
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed. Install with: pip install pypdf")
        return []

    reader = PdfReader(file_path)
    documents = []
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "source": filename,
                "page": page_num,
                "chunk_id": f"{filename}_p{page_num}_c{i}"
            })
    logger.info(f"Parsed PDF {filename}: {len(documents)} chunks from {len(reader.pages)} pages")
    return documents


def _parse_pdf_bytes(content: bytes, filename: str) -> list[dict]:
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed")
        return []

    reader = PdfReader(io.BytesIO(content))
    documents = []
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "source": filename,
                "page": page_num,
                "chunk_id": f"{filename}_p{page_num}_c{i}"
            })
    return documents


# --- DOCX ---

def _parse_docx(file_path: str, filename: str) -> list[dict]:
    try:
        from docx import Document
    except ImportError:
        logger.error("python-docx not installed. Install with: pip install python-docx")
        return []

    doc = Document(file_path)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    chunks = chunk_text(full_text)
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "text": chunk,
            "source": filename,
            "page": i + 1,
            "chunk_id": f"{filename}_c{i}"
        })
    logger.info(f"Parsed DOCX {filename}: {len(documents)} chunks")
    return documents


def _parse_docx_bytes(content: bytes, filename: str) -> list[dict]:
    try:
        from docx import Document
    except ImportError:
        return []

    doc = Document(io.BytesIO(content))
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    chunks = chunk_text(full_text)
    return [
        {"text": chunk, "source": filename, "page": i + 1, "chunk_id": f"{filename}_c{i}"}
        for i, chunk in enumerate(chunks)
    ]


# --- TXT / Markdown ---

def _parse_txt(file_path: str, filename: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return _text_to_chunks(text, filename)


def _parse_txt_bytes(content: bytes, filename: str) -> list[dict]:
    text = content.decode("utf-8", errors="ignore")
    return _text_to_chunks(text, filename)


def _parse_markdown(file_path: str, filename: str) -> list[dict]:
    return _parse_txt(file_path, filename)


def _text_to_chunks(text: str, filename: str) -> list[dict]:
    chunks = chunk_text(text)
    return [
        {"text": chunk, "source": filename, "page": i + 1, "chunk_id": f"{filename}_c{i}"}
        for i, chunk in enumerate(chunks)
    ]


# --- CSV ---

def _parse_csv(file_path: str, filename: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return _csv_to_chunks(f, filename)


def _parse_csv_bytes(content: bytes, filename: str) -> list[dict]:
    text = content.decode("utf-8", errors="ignore")
    return _csv_to_chunks(io.StringIO(text), filename)


def _csv_to_chunks(file_obj, filename: str) -> list[dict]:
    reader = csv.reader(file_obj)
    rows = list(reader)
    if not rows:
        return []

    headers = rows[0] if rows else []
    documents = []
    for row_num, row in enumerate(rows[1:], 2):
        if headers:
            text = " | ".join(f"{h}: {v}" for h, v in zip(headers, row) if v.strip())
        else:
            text = " | ".join(row)
        if text.strip():
            documents.append({
                "text": text,
                "source": filename,
                "page": row_num,
                "chunk_id": f"{filename}_r{row_num}"
            })
    logger.info(f"Parsed CSV {filename}: {len(documents)} rows")
    return documents


# --- Supported formats ---

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".md", ".markdown"}

def is_supported(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_EXTENSIONS
