from datetime import datetime
from apify_client import ApifyClient

import bittensor as bt

from otika.evaluation.utils import tweet_url_to_id


class ApifyTwitterCrawler:
    def __init__(self, api_key, timeout_secs=60):
        self.client = ApifyClient(api_key)

        self.timeout_secs = timeout_secs

        # microworlds actor id
        # users may use any other actor id that can crawl twitter data
        self.actor_id = "microworlds/twitter-scraper"

    def get_tweets_by_urls(self, urls: list):
        """
        Get tweets by urls.

        Args:
            urls (list): The list of urls to get tweets from.

        Returns:
            list: The list of tweets.
        """
        bt.logging.debug(f"Getting tweets by urls: {urls}")
        params = {
            "maxRequestRetries": 3,
            "searchMode": "live",
            "scrapeTweetReplies": True,
            "urls": urls,
            # because if url is a reply, the head tweet will also be included in the result
            "maxTweets": len(urls) * 2,
        }

        ids = [tweet_url_to_id(url) for url in urls]

        run = self.client.actor(self.actor_id).call(
            run_input=params, timeout_secs=self.timeout_secs
        )
        # filter out the original head tweet if requested url is reply
        results = [
            item
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items()
            if item["id_str"] in ids
        ]

        return results

    def search(self, query: str, max_size: int):
        """
        Searches for the given query on the crawled data.

        Args:
            query (str): The query to search for.
            max_size (int): The max number of results to return.

        Returns:
            list: The list of results.
        """
        bt.logging.debug(f"Crawling for query: '{query}' with size {max_size}")
        params = {
            "maxRequestRetries": 3,
            "searchMode": "live",
            "scrapeTweetReplies": True,
            "searchTerms": [query],
            "maxTweets": max_size,
        }

        run = self.client.actor(self.actor_id).call(
            run_input=params, timeout_secs=self.timeout_secs
        )
        bt.logging.debug(f"Apify Actor Run: {run}")

        results = [
            item
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items()
        ]
        bt.logging.trace(f"Apify Results: {results}")

        return results

    def process(self, results):
        """
        Process the results from the search.

        Args:
            results (list): The list of results to process.

        Returns:
            list: The list of processed results.
        """

        time_format = "%a %b %d %H:%M:%S %z %Y"
        results = [
            {
                "id": result["id_str"],
                "url": result["url"],
                "username": result["user"]["screen_name"],
                "text": result.get("full_text"),
                "created_at": datetime.strptime(
                    result.get("created_at"), time_format
                ).isoformat(),
                "quote_count": result.get("quote_count"),
                "reply_count": result.get("reply_count"),
                "retweet_count": result.get("retweet_count"),
                "favorite_count": result.get("favorite_count"),
            }
            for result in results
        ]
        bt.logging.debug(f"Processed results: {results}")
        return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    crawler = ApifyTwitterCrawler(os.environ["APIFY_API_KEY"])

    # r = crawler.search("BTC", 5)
    # print(crawler.process(r))

    r = crawler.get_tweets_by_urls(
        [
            "https://twitter.com/elonmusk/status/1762389336858022132"
            # "https://twitter.com/VitalikButerin/status/1759369749887332577",
            # "https://twitter.com/elonmusk/status/1760504129485705598",
        ]
    )
    print(crawler.process(r))
