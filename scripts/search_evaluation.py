import argparse
import os

import bittensor as bt
import openai
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from openkaito.crawlers.twitter.apify import ApifyTwitterCrawler
from openkaito.evaluation.evaluator import Evaluator
from openkaito.search.engine import SearchEngine
from openkaito.search.ranking.heuristic_ranking import HeuristicRankingModel


def parse_args():
    parser = argparse.ArgumentParser(description="Miner Search Ranking Evaluation")
    parser.add_argument("--query", type=str, default="BTC", help="query string")
    parser.add_argument(
        "--search_recall_size", type=int, default=50, help="size of the recalled items"
    )
    parser.add_argument(
        "--size", type=int, default=5, help="size of the response items"
    )
    # parser.add_argument('--crawling', type=bool, default=False, action='store_true', help='crawling data before search')

    # --logging.debug, --logging.trace
    bt.logging.add_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    load_dotenv()

    # for ranking results evaluation
    llm_client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORGANIZATION"),
        max_retries=3,
    )

    # # for integrity check
    # twitter_crawler = ApifyTwitterCrawler(os.environ["APIFY_API_KEY"])
    evaluator = Evaluator(llm_client, None)

    search_client = Elasticsearch(
        os.environ["ELASTICSEARCH_HOST"],
        basic_auth=(
            os.environ["ELASTICSEARCH_USERNAME"],
            os.environ["ELASTICSEARCH_PASSWORD"],
        ),
        verify_certs=False,
        ssl_show_warn=False,
    )

    # for ranking recalled results
    ranking_model = HeuristicRankingModel(length_weight=0.8, age_weight=0.2)

    search_engine = SearchEngine(
        search_client=search_client, ranking_model=ranking_model, twitter_crawler=None
    )

    ranked_docs = search_engine.search(args.query, args.search_recall_size, args.size)
    print("======ranked documents======")
    print(ranked_docs)

    scores = evaluator.evaluate(args.query, args.size, [ranked_docs])
    print("======evaluation scores======")
    print(scores)


if __name__ == "__main__":
    main()
