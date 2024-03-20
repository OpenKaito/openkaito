import argparse
import os
import random

import bittensor as bt
import openai
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from openkaito.crawlers.twitter.apidojo import ApiDojoTwitterCrawler
from openkaito.crawlers.twitter.microworlds import MicroworldsTwitterCrawler
from openkaito.evaluation.evaluator import Evaluator
from openkaito.protocol import SearchSynapse, SortType, StructuredSearchSynapse
from openkaito.search.engine import SearchEngine
from openkaito.search.ranking.heuristic_ranking import HeuristicRankingModel
from openkaito.search.structured_search_engine import StructuredSearchEngine
from openkaito.tasks import generate_author_index_task


def main():
    load_dotenv()
    bt.logging.set_debug(True)
    bt.logging.set_trace(True)

    # for ranking results evaluation
    llm_client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORGANIZATION"),
        max_retries=3,
    )

    twitter_crawler = None
    # # for integrity check
    # twitter_crawler = ApiDojoTwitterCrawler(os.environ["APIFY_API_KEY"])
    evaluator = Evaluator(llm_client, twitter_crawler)

    search_client = Elasticsearch(
        os.environ["ELASTICSEARCH_HOST"],
        basic_auth=(
            os.environ["ELASTICSEARCH_USERNAME"],
            os.environ["ELASTICSEARCH_PASSWORD"],
        ),
        verify_certs=False,
        ssl_show_warn=False,
    )

    search_engine = StructuredSearchEngine(
        search_client=search_client,
        relevance_ranking_model=HeuristicRankingModel(
            length_weight=0.8, age_weight=0.2
        ),
        twitter_crawler=None,
    )

    # search_query = StructuredSearchSynapse(
    #     size=10, author_usernames=["elonmusk", "nftbadger"]
    # )
    search_query = generate_author_index_task(size=10, num_authors=100)
    print(search_query)

    docs = search_engine.search(search_query=search_query)
    print("======documents======")
    print(docs)

    score = evaluator.llm_author_index_data_evaluation(docs)
    print("======LLM Score======")
    print(score)


if __name__ == "__main__":
    main()
