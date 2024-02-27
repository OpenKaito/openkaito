from dotenv import load_dotenv
import os
import argparse
from elasticsearch import Elasticsearch
import openai

from otika.crawlers.twitter.apify import ApifyTwitterCrawler
from otika.evaluation.evaluator import Evaluator
from otika.search.engine import SearchEngine
from otika.search.ranking.heuristic_ranking import HeuristicRankingModel


def parse_args():
    parser = argparse.ArgumentParser(description="Miner Search Ranking Evaluation")
    parser.add_argument("--query", type=str, default="BTC", help="query string")
    parser.add_argument(
        "--size", type=int, default=5, help="size of the response items"
    )
    # parser.add_argument('--crawling', type=bool, default=False, action='store_true', help='crawling data before search')
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

    ranked_docs = search_engine.search(args.query, args.size)
    print(ranked_docs)

    scores = evaluator.evaluate(args.query, args.size, [ranked_docs])
    print(scores)


if __name__ == "__main__":
    main()
