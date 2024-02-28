
from abc import ABC, abstractmethod


class AbstractRankingModel(ABC):
    @abstractmethod
    def rank(self, query, documents):
        pass