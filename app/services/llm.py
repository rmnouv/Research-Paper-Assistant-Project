from app.config import Settings
from app.schemas import Citation


class LlmService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def answer(self, question: str, citations: list[Citation]) -> str:
        if not self.settings.openai_api_key:
            return (
                "OPENAI_API_KEY is not set, so I retrieved evidence but did not call an LLM. "
                "Set the key in .env to generate grounded answers."
            )

        from openai import OpenAI

        client = OpenAI(api_key=self.settings.openai_api_key)
        context = "\n\n".join(
            f"[{i}] {citation.source}, page {citation.page}: {citation.text}"
            for i, citation in enumerate(citations, start=1)
        )
        prompt = (
            "Answer the question using only the provided research paper excerpts. "
            "Cite sources inline with bracket numbers like [1]. If the excerpts do not contain "
            "the answer, say you do not know.\n\n"
            f"Question: {question}\n\n"
            f"Excerpts:\n{context}"
        )

        response = client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
