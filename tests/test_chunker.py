from app.services.chunker import chunk_text


def test_chunk_text_adds_source_page_and_overlap() -> None:
    text = " ".join(str(i) for i in range(80))

    chunks = chunk_text(
        text,
        page=3,
        source="paper.pdf",
        chunk_size=60,
        chunk_overlap=10,
    )

    assert len(chunks) > 1
    assert chunks[0].source == "paper.pdf"
    assert chunks[0].page == 3
    assert chunks[0].id == "paper.pdf:p3:c0"
