import os
import traceback

import bittensor as bt
from dotenv import load_dotenv

from openkaito.protocol import SortType

from ..utils.embeddings import pad_tensor, text_embedding, MAX_EMBEDDING_DIM


class StructuredSearchEngine:
    def __init__(
        self,
        search_client,
        relevance_ranking_model,
        twitter_crawler=None,
        recall_size=50,
    ):
        load_dotenv()

        self.search_client = search_client
        self.init_indices()

        # for relevance ranking recalled results
        self.relevance_ranking_model = relevance_ranking_model

        self.recall_size = recall_size

        # optional, for crawling data
        self.twitter_crawler = twitter_crawler

    def twitter_doc_mapper(cls, doc):
        return {
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

        # recommended discord index schema mapping
        # (optional): miners may modify and improve it
        index_name = "discord"
        if not self.search_client.indices.exists(index=index_name):
            bt.logging.info("creating index...", index_name)
            self.search_client.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "server_id": {"type": "keyword"},
                            "server_name": {"type": "keyword"},
                            "channel_id": {"type": "keyword"},
                            "channel_name": {"type": "keyword"},
                            "channel_type": {"type": "keyword"},
                            "channel_category_id": {"type": "keyword"},
                            "channel_category": {"type": "keyword"},
                            "id": {"type": "long"},
                            "text": {"type": "text"},
                            "message_type": {"type": "keyword"},
                            "reference_message_id": {"type": "long"},
                            "is_pinned": {"type": "boolean"},
                            "created_at": {"type": "date"},
                            "modified_at": {"type": "date"},
                            "author_id": {"type": "keyword"},
                            "author_username": {"type": "keyword"},
                            "author_nickname": {"type": "keyword"},
                            "author_discriminator": {"type": "keyword"},
                            # add more fields for better indexing
                        }
                    }
                },
            )

    def search(self, search_query):
        """
        Structured search interface for this search engine

        Args:
        - search_query: A `StructuredSearchSynapse` or `SearchSynapse` object representing the search request sent by the validator.
        """

        result_size = search_query.size

        recalled_items = self.recall(
            search_query=search_query, recall_size=self.recall_size
        )

        ranking_model = self.relevance_ranking_model

        results = ranking_model.rank(search_query.query_string, recalled_items)

        return results[:result_size]

    def recall(self, search_query, recall_size):
        """
        Structured recall interface for this search engine
        """
        query_string = search_query.query_string

        es_query = {
            "query": {
                "bool": {
                    "must": [],
                }
            },
            "size": recall_size,
        }

        if search_query.query_string:
            es_query["query"]["bool"]["must"].append(
                {
                    "query_string": {
                        "query": query_string,
                        "default_field": "text",
                        "default_operator": "AND",
                    }
                }
            )

        if search_query.name == "StructuredSearchSynapse":
            if search_query.author_usernames:
                es_query["query"]["bool"]["must"].append(
                    {
                        "terms": {
                            "username": search_query.author_usernames,
                        }
                    }
                )

            time_filter = {}
            if search_query.earlier_than_timestamp:
                time_filter["lte"] = search_query.earlier_than_timestamp
            if search_query.later_than_timestamp:
                time_filter["gte"] = search_query.later_than_timestamp
            if time_filter:
                es_query["query"]["bool"]["must"].append(
                    {"range": {"created_at": time_filter}}
                )

        bt.logging.trace(f"es_query: {es_query}")

        try:
            response = self.search_client.search(
                index="twitter",
                body=es_query,
            )
            documents = response["hits"]["hits"]
            results = []
            for document in documents if documents else []:
                doc = document["_source"]
                results.append(self.twitter_doc_mapper(doc))
            bt.logging.info(f"retrieved {len(results)} results")
            bt.logging.trace(f"results: ")
            return results
        except Exception as e:
            bt.logging.error("recall error...", e)
            return []

    def discord_search(self, search_query):
        """
        Structured search interface for discord data
        """

        es_query = {
            "query": {
                "bool": {
                    "must": [],
                }
            },
            "size": search_query.size,
        }

        if search_query.query_string:
            es_query["query"]["bool"]["must"].append(
                {
                    "query_string": {
                        "query": search_query.query_string,
                        "default_field": "text",
                    }
                }
            )

        if search_query.author_usernames:
            es_query["query"]["bool"]["must"].append(
                {
                    "terms": {
                        "author_username": search_query.author_usernames,
                    }
                }
            )

        if search_query.server_name:
            es_query["query"]["bool"]["must"].append(
                {
                    "bool": {
                        "should": [
                            {"term": {"server_name": search_query.server_name}},
                            {"term": {"server_id": search_query.server_name}},
                        ]
                    }
                }
            )

        if search_query.channel_ids:
            es_query["query"]["bool"]["must"].append(
                {
                    "terms": {
                        "channel_id": search_query.channel_ids,
                    }
                }
            )

        if search_query.sort_by == SortType.RECENCY:
            es_query["sort"] = [{"created_at": {"order": "desc"}}]

        time_filter = {}
        if search_query.earlier_than_timestamp:
            time_filter["lte"] = search_query.earlier_than_timestamp
        if search_query.later_than_timestamp:
            time_filter["gte"] = search_query.later_than_timestamp
        if time_filter:
            es_query["query"]["bool"]["must"].append(
                {"range": {"created_at": time_filter}}
            )

        bt.logging.trace(f"es_query: {es_query}")

        search_client = self.search_client
        index_name = search_query.index_name if search_query.index_name else "discord"
        try:
            response = search_client.search(
                index=index_name,
                body=es_query,
            )
            documents = response["hits"]["hits"]
            results = []
            for document in documents if documents else []:
                doc = document["_source"]
                results.append(doc)
            bt.logging.info(f"retrieved {len(results)} results")
            bt.logging.trace(f"results: ")
            return results[: search_query.size]
        except Exception as e:
            bt.logging.error("retrieve error...", e)
            bt.logging.error(traceback.format_exc())

            return []

    def vector_search(self, query):
        topk = query.size
        query_string = query.query_string
        index_name = query.index_name if query.index_name else "eth_denver"

        embedding = text_embedding(query_string)[0]
        embedding = pad_tensor(embedding, max_len=MAX_EMBEDDING_DIM)
        body = {
            "knn": {
                "field": "embedding",
                "query_vector": embedding.tolist(),
                "k": topk,
                "num_candidates": 5 * topk,
            },
            "_source": {
                "excludes": ["embedding"],
            },
        }

        response = self.search_client.search(index=index_name, body=body)
        ranked_docs = [doc["_source"] for doc in response["hits"]["hits"]]
        # optional: you may implement yourselves additional post-processing filtering/ranking here

        return ranked_docs

    def crawl_and_index_data(self, query_string, author_usernames, max_size):
        """
        Crawls the data from the twitter crawler and indexes it in the elasticsearch database.
        """
        if self.twitter_crawler is None:
            bt.logging.warning(
                "Twitter crawler is not initialized. skipped crawling and indexing"
            )
        try:
            processed_docs = self.twitter_crawler.search(
                query_string, author_usernames, max_size
            )
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
