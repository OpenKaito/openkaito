from .abstract_model import AbstractRankingModel
from datetime import datetime, timezone
import math


class RecencyRankingModel(AbstractRankingModel):
    def __init__(self):
        super().__init__()

    def rank(self, query, documents):
        ranked_docs = sorted(
            documents,
            key=lambda doc: datetime.fromisoformat(doc["created_at"].rstrip("Z")),
            reverse=True,
        )
        return ranked_docs
