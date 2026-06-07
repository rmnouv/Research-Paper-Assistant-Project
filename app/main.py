from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.schemas import AskRequest, AskResponse, IngestResponse
from app.services.rag import RagService

app = FastAPI(title="Research Paper Assistant")


def get_rag(settings: Settings = Depends(get_settings)) -> RagService:
    return RagService(settings)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/papers", response_model=IngestResponse)
async def ingest_paper(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    rag: RagService = Depends(get_rag),
) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file.")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / Path(file.filename).name
    target.write_bytes(await file.read())

    chunks = rag.ingest_pdf(target)
    return IngestResponse(filename=target.name, chunks_indexed=chunks)


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest, rag: RagService = Depends(get_rag)) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    return rag.answer(request.question)
