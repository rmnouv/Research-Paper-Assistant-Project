from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class Citation(BaseModel):
    source: str
    page: int
    chunk_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
