from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from app.config import Settings
from app.services.chunker import TextChunk


class VectorStore:
    """Store and search text chunks in a Chroma collection."""

    def __init__(self, settings: Settings) -> None:
        """Create the Chroma client and collection configured for this app."""

        self.settings = settings
        self.client = self._client()
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model
            ),
        )

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        """Insert or replace chunks in the collection.

        Chroma upsert uses each chunk id as the stable key, so re-ingesting the
        same PDF updates existing chunks instead of adding duplicates.
        """

        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {"source": chunk.source, "page": chunk.page}
                for chunk in chunks
            ],
        )

    def search(self, query: str, *, k: int) -> list[dict[str, Any]]:
        """Find the top matching chunks for a query.

        The returned dictionaries match the shape expected by the reranker and
        citation-building code in the RAG service.
        """

        results = self.collection.query(query_texts=[query], n_results=k)
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        items = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            items.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "source": metadata["source"],
                    "page": int(metadata["page"]),
                    # Chroma returns distance where lower is better; downstream
                    # code treats score as higher-is-better.
                    "score": 1.0 - float(distance),
                }
            )
        return items

    def _client(self):
        """Create either a remote Chroma HTTP client or a local persistent client."""

        if self.settings.chroma_host:
            return chromadb.HttpClient(
                host=self.settings.chroma_host,
                port=self.settings.chroma_port,
            )
        return chromadb.PersistentClient(path=self.settings.chroma_persist_dir)
