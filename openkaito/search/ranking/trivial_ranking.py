from .abstract_model import AbstractRankingModel


class TrivialRankingModel(AbstractRankingModel):
    def rank(self, query, documents):
        return documents
