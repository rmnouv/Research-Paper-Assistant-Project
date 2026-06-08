from pathlib import Path

from pypdf import PdfReader


def load_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Extract text from each page of a PDF.

    Returns 1-based page numbers so citations match the page numbers readers
    see in the PDF viewer.
    """

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((index, text))

    return pages
