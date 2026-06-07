from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from app.config import Settings
from app.services.chunker import TextChunk


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = self._client()
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model
            ),
        )

    def add_chunks(self, chunks: list[TextChunk]) -> None:
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
                    "score": 1.0 - float(distance),
                }
            )
        return items

    def _client(self):
        if self.settings.chroma_host:
            return chromadb.HttpClient(
                host=self.settings.chroma_host,
                port=self.settings.chroma_port,
            )
        return chromadb.PersistentClient(path=self.settings.chroma_persist_dir)
