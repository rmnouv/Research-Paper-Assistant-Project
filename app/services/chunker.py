from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """A searchable slice of text with enough metadata to cite its source."""

    id: str
    text: str
    page: int
    source: str


def chunk_text(
    text: str,
    *,
    page: int,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Split page text into overlapping chunks suitable for vector search.

    The chunk ids are deterministic for a source/page/index combination, which
    lets the vector store upsert the same document without creating duplicates.
    """

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[TextChunk] = []
    start = 0
    index = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(
                TextChunk(
                    id=f"{source}:p{page}:c{index}",
                    text=chunk,
                    page=page,
                    source=source,
                )
            )
        if end == len(normalized):
            break

        # Keep neighboring chunks connected so answers can use context that
        # straddles the boundary between two chunks.
        start = max(end - chunk_overlap, start + 1)
        index += 1

    return chunks
