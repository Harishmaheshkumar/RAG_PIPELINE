import json
from pathlib import Path
from io import BytesIO
import logging

# Logger setup
logger = logging.getLogger(__name__)

# Safe imports
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extracts text from PDF using pypdf."""
    if PdfReader is None:
        return "ERROR: pypdf library not installed."
    
    try:
        pdf = PdfReader(BytesIO(file_content))
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text if text.strip() else "ERROR: PDF is empty or scanned (no selectable text)."
    except Exception as e:
        logger.error(f"PDF Extraction Error: {e}")
        return f"ERROR: Failed to process PDF: {str(e)}"

def extract_text_from_docx(file_content: bytes) -> str:
    """Extracts text from DOCX using python-docx."""
    if Document is None:
        return "ERROR: python-docx library not installed."
    
    try:
        doc = Document(BytesIO(file_content))
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return text if text.strip() else "ERROR: DOCX file is empty."
    except Exception as e:
        logger.error(f"DOCX Extraction Error: {e}")
        return f"ERROR: Failed to process DOCX: {str(e)}"

def extract_text_from_file(filename: str, file_content: bytes) -> str:
    """
    Router to extract text based on file extension.
    Returns the extracted text or a string starting with 'ERROR:'.
    """
    if not file_content:
        return "ERROR: File is empty."

    ext = Path(filename).suffix.lower()
    
    try:
        if ext == ".pdf":
            return extract_text_from_pdf(file_content)
        elif ext in [".docx", ".doc"]:
            return extract_text_from_docx(file_content)
        elif ext in [".txt", ".md", ".py"]:
            return file_content.decode("utf-8", errors="ignore")
        elif ext == ".json":
            data = json.loads(file_content.decode("utf-8"))
            return json.dumps(data, indent=2)
        else:
            return f"ERROR: Unsupported file extension: {ext}"
    except Exception as e:
        logger.error(f"General Extraction Error for {filename}: {e}")
        return f"ERROR: {str(e)}"