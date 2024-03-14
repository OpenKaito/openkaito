import os

import bittensor as bt
from dotenv import load_dotenv


class SearchEngine:
    def __init__(self, search_client, ranking_model, twitter_crawler=None):
        load_dotenv()

        self.search_client = search_client
        self.init_indices()

        # for ranking recalled results
        self.ranking_model = ranking_model

        # optional, for crawling data
        self.twitter_crawler = twitter_crawler

    def init_indices(self):
        """
        Initializes the indices in the elasticsearch database.
        """
        index_name = "twitter"
        if not self.search_client.indices.exists(index=index_name):
            bt.logging.info("creating index...", index_name)
            self.search_client.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "id": {"type": "long"},
                            "text": {"type": "text"},
                            "created_at": {"type": "date"},
                            "username": {"type": "keyword"},
                            "url": {"type": "text"},
                            "quote_count": {"type": "long"},
                            "reply_count": {"type": "long"},
                            "retweet_count": {"type": "long"},
                            "favorite_count": {"type": "long"},
                        }
                    }
                },
            )

    def search(self, query_string, recall_size, result_size):
        """
        Search interface for this search engine
        """

        recalled_items = self.recall(query_string, recall_size)
        results = self.ranking_model.rank(query_string, recalled_items)
        return results[:result_size]

    def recall(self, query_string, recall_size):
        """
        Retrieve the results from the elasticsearch database.
        """

        try:
            response = self.search_client.search(
                index="twitter",
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "query_string": {
                                        "query": query_string,
                                        "default_field": "text",
                                        "default_operator": "AND",
                                    }
                                }
                            ],
                        }
                    },
                    "size": recall_size,
                },
            )
            documents = response["hits"]["hits"]
            results = []
            for document in documents if documents else []:
                doc = document["_source"]
                results.append(
                    {
                        "id": doc["id"],
                        "text": doc["text"],
                        "created_at": doc["created_at"],
                        "username": doc["username"],
                        "url": doc["url"],
                        "quote_count": doc["quote_count"],
                        "reply_count": doc["reply_count"],
                        "retweet_count": doc["retweet_count"],
                        "favorite_count": doc["favorite_count"],
                    }
                )
            bt.logging.info(f"retrieved {len(results)} results")
            bt.logging.trace(f"results: ")
            return results
        except Exception as e:
            bt.logging.error("recall error...", e)
            return []

    def crawl_and_index_data(self, query_string, max_size):
        """
        Crawls the data from the twitter crawler and indexes it in the elasticsearch database.
        """
        if self.twitter_crawler is None:
            bt.logging.warning(
                "Twitter crawler is not initialized. skipped crawling and indexing"
            )
        try:
            processed_docs = self.twitter_crawler.search(query_string, max_size)
            bt.logging.debug(f"crawled {len(processed_docs)} docs")
            bt.logging.trace(processed_docs)
        except Exception as e:
            bt.logging.error("crawling error...", e)
            processed_docs = []

        if len(processed_docs) > 0:
            try:
                bt.logging.info(f"bulk indexing {len(processed_docs)} docs")
                bulk_body = []
                for doc in processed_docs:
                    bulk_body.append(
                        {
                            "update": {
                                "_index": "twitter",
                                "_id": doc["id"],
                            }
                        }
                    )
                    bulk_body.append(
                        {
                            "doc": doc,
                            "doc_as_upsert": True,
                        }
                    )

                r = self.search_client.bulk(
                    body=bulk_body,
                    refresh=True,
                )
                bt.logging.trace("bulk update response...", r)
                if not r.get("errors"):
                    bt.logging.info("bulk update succeeded")
                else:
                    bt.logging.error("bulk update failed: ", r)
            except Exception as e:
                bt.logging.error("bulk update error...", e)
