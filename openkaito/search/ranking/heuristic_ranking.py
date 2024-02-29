from .abstract_model import AbstractRankingModel
from datetime import datetime, timezone
import math


class HeuristicRankingModel(AbstractRankingModel):
    def __init__(self, length_weight=0.8, age_weight=0.2):
        super().__init__()
        self.length_weight = length_weight
        self.age_weight = age_weight

    def rank(self, query, documents):
        ranked_docs = sorted(
            documents,
            key=lambda doc: self.compute_score(query, doc),
            reverse=True,
        )
        return ranked_docs

    def compute_score(self, query, doc):
        now = datetime.now(timezone.utc)
        text_length = len(doc["text"])
        age = (
            now - datetime.fromisoformat(doc["created_at"].rstrip("Z"))
        ).total_seconds()
        return self.length_weight * self.text_length_score(
            text_length
        ) + self.age_weight * self.age_score(age)

    def text_length_score(self, text_length):
        return math.log(text_length + 1) / 10

    def age_score(self, age):
        return 60 * 60 / (age + 1)
