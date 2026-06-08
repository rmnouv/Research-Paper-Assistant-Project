from pathlib import Path

from app.config import Settings
from app.schemas import AskResponse, Citation
from app.services.chunker import chunk_text
from app.services.document_loader import load_pdf_pages
from app.services.llm import LlmService
from app.services.reranker import Reranker
from app.services.vector_store import VectorStore


class RagService:
    """Coordinate ingestion, retrieval, reranking, and answer generation."""

    def __init__(self, settings: Settings) -> None:
        """Create the service dependencies from shared application settings."""

        self.settings = settings
        self.vector_store = VectorStore(settings)
        self.reranker = Reranker(settings)
        self.llm = LlmService(settings)

    def ingest_pdf(self, path: Path) -> int:
        """Load a PDF, chunk every page, and write the chunks to the vector store.

        Returns the number of chunks stored so API callers can report ingestion
        progress to the user.
        """

        all_chunks = []
        for page, text in load_pdf_pages(path):
            all_chunks.extend(
                chunk_text(
                    text,
                    page=page,
                    source=path.name,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
            )

        self.vector_store.add_chunks(all_chunks)
        return len(all_chunks)

    def answer(self, question: str) -> AskResponse:
        """Retrieve evidence for a question and generate a cited answer."""

        retrieved = self.vector_store.search(question, k=self.settings.retrieval_k)
        ranked = self.reranker.rank(question, retrieved)[: self.settings.rerank_k]
        citations = [
            Citation(
                source=item["source"],
                page=item["page"],
                chunk_id=item["id"],
                score=item["score"],
                text=item["text"],
            )
            for item in ranked
        ]
        return AskResponse(answer=self.llm.answer(question, citations), citations=citations)
