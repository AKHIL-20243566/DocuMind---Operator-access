"""Document parser — extracts text from PDF, DOCX, TXT, CSV, and Markdown files."""
import os
import csv
import io
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _has_text(documents: list[dict]) -> bool:
    return any((d.get("text") or "").strip() for d in documents)


def _find_tesseract_cmd() -> str | None:
    """Find tesseract executable from PATH or common Windows install locations."""
    candidate_paths = [
        # Installed and exposed in PATH
        "tesseract",
        # Common machine-level path
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]

    for candidate in candidate_paths[1:]:
        if os.path.exists(candidate):
            return candidate

    # Keep PATH-based lookup last so explicit absolute paths win when available.
    return candidate_paths[0]


def _find_poppler_bin_dir() -> str | None:
    """Find Poppler bin dir from PATH aliases or common winget install layout."""
    # If aliases are already visible in PATH, pdf2image can resolve pdfinfo directly.
    if _command_exists("pdfinfo"):
        return None

    base = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if not base.exists():
        return None

    matches = sorted(base.glob("*oschwartz10612.Poppler*"), reverse=True)
    for match in matches:
        candidate = match / "poppler-25.07.0" / "Library" / "bin"
        if candidate.exists():
            return str(candidate)

        # Fallback for future versioned folders.
        for nested in match.glob("poppler-*/*/bin"):
            if nested.exists():
                return str(nested)
    return None


def _command_exists(command: str) -> bool:
    from shutil import which

    return which(command) is not None


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
    return parse_bytes_with_diagnostics(content, filename)["documents"]


def parse_bytes_with_diagnostics(content: bytes, filename: str) -> dict:
    """Parse file content from bytes and return extraction diagnostics."""
    ext = os.path.splitext(filename)[1].lower()

    diagnostics = {
        "success": False,
        "documents": [],
        "loader_used": None,
        "ocr_triggered": False,
        "error_code": None,
        "error_message": None,
        "status_messages": [],
    }

    try:
        if ext == ".pdf":
            return _parse_pdf_bytes_with_fallback(content, filename)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"):
            return _parse_image_bytes_with_diagnostics(content, filename)
        elif ext == ".docx":
            documents = _parse_docx_bytes(content, filename)
        elif ext == ".txt":
            documents = _parse_txt_bytes(content, filename)
        elif ext == ".csv":
            documents = _parse_csv_bytes(content, filename)
        elif ext in (".md", ".markdown"):
            documents = _parse_txt_bytes(content, filename)
        else:
            documents = _parse_txt_bytes(content, filename)

        diagnostics["documents"] = documents
        diagnostics["success"] = _has_text(documents)
        diagnostics["loader_used"] = ext.lstrip(".") or "txt"
        if diagnostics["success"]:
            diagnostics["status_messages"] = ["Extraction successful"]
        else:
            diagnostics["error_code"] = "NO_READABLE_TEXT"
            diagnostics["error_message"] = f"No readable text found in {filename}"
            diagnostics["status_messages"] = ["No readable text found"]
        return diagnostics
    except Exception as e:
        logger.error(f"Error parsing {filename}: {e}")
        diagnostics["error_code"] = "PARSER_FAILURE"
        diagnostics["error_message"] = f"Failed to parse {filename}: {e}"
        return diagnostics


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


# --- Image OCR (PNG / JPG / TIFF / BMP) ---

def _parse_image_bytes_with_diagnostics(content: bytes, filename: str) -> dict:
    """OCR a raw image file and return diagnostics.
    Owner: Anirudh (Data Engineer) — direct image ingestion pipeline.
    Supports PNG, JPG, JPEG, TIFF, BMP, WEBP via pytesseract + Pillow.
    """
    diagnostics = {
        "success": False,
        "documents": [],
        "loader_used": "ImageOCR",
        "ocr_triggered": True,
        "error_code": None,
        "error_message": None,
        "status_messages": ["Image detected", "Applying OCR..."],
    }

    try:
        import pytesseract
        from PIL import Image

        tesseract_cmd = _find_tesseract_cmd()
        if tesseract_cmd and tesseract_cmd != "tesseract":
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # Force a clear error early if Tesseract is missing
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:
            raise RuntimeError(
                "tesseract not found. Install Tesseract OCR and ensure tesseract.exe is in PATH."
            ) from exc

        image = Image.open(io.BytesIO(content))
        # Convert to RGB so tesseract handles all modes (RGBA, palette, etc.)
        image = image.convert("RGB")

        text = pytesseract.image_to_string(image) or ""
        chunks = chunk_text(text)

        if not chunks:
            diagnostics.update({
                "error_code": "NO_READABLE_TEXT",
                "error_message": f"No readable text found in image {filename}",
                "status_messages": diagnostics["status_messages"] + ["No readable text found"],
            })
            logger.error("Image OCR found no text | file=%s", filename)
            return diagnostics

        documents = [
            {
                "text": chunk,
                "source": filename,
                "page": 1,
                "chunk_id": f"{filename}_ocr_c{i}",
            }
            for i, chunk in enumerate(chunks)
        ]

        diagnostics.update({
            "success": True,
            "documents": documents,
            "status_messages": diagnostics["status_messages"] + ["Extraction successful"],
        })
        logger.info("Image OCR success | file=%s chunks=%d", filename, len(documents))
        return diagnostics

    except Exception as exc:
        error_text = str(exc).lower()
        is_dep_error = "tesseract" in error_text or "not found" in error_text or "pillow" in error_text

        if is_dep_error:
            diagnostics["error_code"] = "OCR_DEPENDENCY_MISSING"
            diagnostics["error_message"] = (
                "OCR engine unavailable. Install Tesseract OCR + Pillow and ensure tesseract.exe is in PATH."
            )
        else:
            diagnostics["error_code"] = "IMAGE_PARSE_FAILURE"
            diagnostics["error_message"] = f"Failed to process image {filename}: {exc}"

        diagnostics["status_messages"].append("OCR failed")
        logger.error("Image OCR failure | file=%s error=%s", filename, exc)
        return diagnostics


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


def _parse_pdf_bytes_with_fallback(content: bytes, filename: str) -> dict:
    """Parse PDF via layered fallback: PyPDFLoader -> UnstructuredPDFLoader -> OCR."""
    diagnostics = {
        "success": False,
        "documents": [],
        "loader_used": None,
        "ocr_triggered": False,
        "error_code": None,
        "error_message": None,
        "status_messages": [],
    }

    loader_errors = []

    # 1) pypdf byte parser (equivalent to PyPDFLoader text extraction stage)
    try:
        docs = _parse_pdf_bytes(content, filename)
        if _has_text(docs):
            diagnostics.update(
                {
                    "success": True,
                    "documents": docs,
                    "loader_used": "PyPDFLoader",
                    "status_messages": ["Extraction successful"],
                }
            )
            logger.info("PDF extraction success | file=%s loader=PyPDFLoader ocr_triggered=False chunks=%s", filename, len(docs))
            return diagnostics
        logger.warning("PDF extraction empty | file=%s loader=PyPDFLoader", filename)
        diagnostics["status_messages"].append("Scanned document detected")
    except Exception as exc:
        loader_errors.append(f"PyPDFLoader: {exc}")
        logger.warning("PDF extraction failure | file=%s loader=PyPDFLoader error=%s", filename, exc)

    # 2) Unstructured loader fallback (best-effort; optional dependency)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_pdf_path = tmp.name

        try:
            docs = _parse_pdf_with_unstructured_loader(tmp_pdf_path, filename)
            if _has_text(docs):
                diagnostics.update(
                    {
                        "success": True,
                        "documents": docs,
                        "loader_used": "UnstructuredPDFLoader",
                        "status_messages": ["Extraction successful"],
                    }
                )
                logger.info("PDF extraction success | file=%s loader=UnstructuredPDFLoader ocr_triggered=False chunks=%s", filename, len(docs))
                return diagnostics
            logger.warning("PDF extraction empty | file=%s loader=UnstructuredPDFLoader", filename)
        finally:
            try:
                os.unlink(tmp_pdf_path)
            except OSError:
                pass
    except Exception as exc:
        loader_errors.append(f"UnstructuredPDFLoader: {exc}")
        logger.warning("PDF extraction failure | file=%s loader=UnstructuredPDFLoader error=%s", filename, exc)

    # 3) OCR fallback
    diagnostics["ocr_triggered"] = True
    diagnostics["status_messages"] = ["Scanned document detected", "Applying OCR..."]
    try:
        docs = _parse_pdf_with_ocr(content, filename)
        if _has_text(docs):
            diagnostics.update(
                {
                    "success": True,
                    "documents": docs,
                    "loader_used": "OCR",
                    "status_messages": diagnostics["status_messages"] + ["Extraction successful"],
                }
            )
            logger.info("PDF extraction success | file=%s loader=OCR ocr_triggered=True chunks=%s", filename, len(docs))
            return diagnostics

        diagnostics.update(
            {
                "loader_used": "OCR",
                "error_code": "NO_READABLE_TEXT",
                "error_message": f"No readable text found in {filename}",
                "status_messages": diagnostics["status_messages"] + ["No readable text found"],
            }
        )
        logger.error("PDF extraction failure | file=%s loader=OCR ocr_triggered=True reason=no_readable_text", filename)
        return diagnostics
    except Exception as exc:
        loader_errors.append(f"OCR: {exc}")
        error_text = str(exc).lower()
        is_dependency_error = (
            "tesseract" in error_text
            or "poppler" in error_text
            or "pdfinfo" in error_text
            or "is not installed" in error_text
            or "not found" in error_text
        )

        if is_dependency_error:
            error_code = "OCR_DEPENDENCY_MISSING"
            error_message = (
                "OCR engine unavailable. Install Tesseract OCR and Poppler, then ensure both are in PATH."
            )
            status_messages = diagnostics["status_messages"] + ["OCR engine unavailable"]
        else:
            error_code = "CORRUPTED_OR_UNREADABLE_PDF"
            error_message = f"Failed to parse PDF {filename}. The file may be corrupted or unreadable."
            status_messages = diagnostics["status_messages"] + ["No readable text found"]

        diagnostics.update(
            {
                "loader_used": "OCR",
                "error_code": error_code,
                "error_message": error_message,
                "status_messages": status_messages,
            }
        )
        logger.error("PDF extraction failure | file=%s loader=OCR ocr_triggered=True error=%s", filename, exc)
        return diagnostics
    finally:
        if loader_errors:
            logger.info("PDF loader attempts | file=%s details=%s", filename, " | ".join(loader_errors))


def _parse_pdf_with_unstructured_loader(file_path: str, filename: str) -> list[dict]:
    from langchain_community.document_loaders import UnstructuredPDFLoader

    loader = UnstructuredPDFLoader(file_path)
    raw_docs = loader.load()

    documents = []
    for idx, doc in enumerate(raw_docs, 1):
        text = (doc.page_content or "").strip()
        if not text:
            continue
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(
                {
                    "text": chunk,
                    "source": filename,
                    "page": idx,
                    "chunk_id": f"{filename}_u{idx}_c{i}",
                }
            )
    return documents


def _parse_pdf_with_ocr(content: bytes, filename: str) -> list[dict]:
    import pytesseract
    from pdf2image import convert_from_bytes

    tesseract_cmd = _find_tesseract_cmd()
    if tesseract_cmd and tesseract_cmd != "tesseract":
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    poppler_path = _find_poppler_bin_dir()

    # Force a concrete check before processing pages to emit a clear error early.
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError(
            "tesseract not found. Install Tesseract OCR and ensure tesseract.exe is available in PATH."
        ) from exc

    if poppler_path:
        images = convert_from_bytes(content, poppler_path=poppler_path)
    else:
        images = convert_from_bytes(content)
    documents = []

    for page_num, image in enumerate(images, 1):
        text = pytesseract.image_to_string(image) or ""
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(
                {
                    "text": chunk,
                    "source": filename,
                    "page": page_num,
                    "chunk_id": f"{filename}_ocr_p{page_num}_c{i}",
                }
            )

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

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".csv", ".md", ".markdown",
    # Image formats — OCR via pytesseract + Pillow
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
}

def is_supported(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_EXTENSIONS
