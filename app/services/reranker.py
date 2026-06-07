from typing import Any

from app.config import Settings


class Reranker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    def rank(self, question: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not items:
            return []

        model = self._get_model()
        pairs = [(question, item["text"]) for item in items]
        scores = model.predict(pairs)

        ranked = []
        for item, score in zip(items, scores, strict=True):
            ranked.append({**item, "score": float(score)})

        return sorted(ranked, key=lambda item: item["score"], reverse=True)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.settings.reranker_model)
        return self._model
