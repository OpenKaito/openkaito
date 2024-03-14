from datetime import datetime
from apify_client import ApifyClient

import bittensor as bt

from openkaito.evaluation.utils import tweet_url_to_id


class ApiDojoTwitterCrawler:
    def __init__(self, api_key, timeout_secs=120):
        self.client = ApifyClient(api_key)

        self.timeout_secs = timeout_secs

        self.actor_id = "apidojo/tweet-scraper"

    def get_tweets_by_urls(self, urls: list):
        """
        Get tweets by urls.

        Args:
            urls (list): The urls to get tweets from.

        Returns:
            list: The list of tweet details.
        """

        params = {
            "startUrls": urls,
            "maxItems": len(urls),
            "maxTweetsPerQuery": 1,
            "onlyImage": False,
            "onlyQuote": False,
            "onlyTwitterBlue": False,
            "onlyVerifiedUsers": False,
            "onlyVideo": False,
        }

        run = self.client.actor(self.actor_id).call(
            run_input=params, timeout_secs=self.timeout_secs
        )
        return self.process_list(
            self.client.dataset(run["defaultDatasetId"]).iterate_items()
        )

    def get_tweets_by_ids_with_retries(self, ids: list, retries=2):
        """
        Get tweets by tweet ids with retries.

        Args:
            ids (list): The tweet ids to get tweets from.
            retries (int): The number of retries to make.

        Returns:
            dict: The dict of tweet id to tweet details.
        """
        result = {}
        remaining_ids = set(ids) - set(result.keys())
        while retries > 0 and len(remaining_ids) > 0:
            bt.logging.debug(f"Trying fetching ids: {remaining_ids}")
            urls = [f"https://x.com/x/status/{id}" for id in remaining_ids]
            tweets = self.get_tweets_by_urls(list(urls))
            for tweet in tweets:
                result[tweet["id"]] = tweet
            remaining_ids = set(remaining_ids) - set(result.keys())
            retries -= 1

        return result

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
        bt.logging.trace(f"Apify Actor Run: {run}")

        results = [
            item
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items()
        ]
        bt.logging.trace(f"Apify Results: {results}")

        return results

    def process_item(self, item):
        """
        Process the item.

        Args:
            item (dict): The item to process.

        Returns:
            dict: The processed item.
        """
        time_format = "%a %b %d %H:%M:%S %z %Y"
        return {
            "id": item["id"],
            "url": item["url"],
            "username": item["author"]["userName"],
            "text": item.get("text"),
            "created_at": datetime.strptime(
                item.get("createdAt"), time_format
            ).isoformat(),
            "quote_count": item.get("quoteCount"),
            "reply_count": item.get("replyCount"),
            "retweet_count": item.get("retweetCount"),
            "favorite_count": item.get("likeCount"),
        }

    def process_list(self, results):
        """
        Process the results from the search.

        Args:
            results (list): The list of results to process.

        Returns:
            list: The list of processed results.
        """
        return [self.process_item(result) for result in results]


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    crawler = ApiDojoTwitterCrawler(os.environ["APIFY_API_KEY"])

    # r = crawler.search("BTC", 5)
    # print(crawler.process_list(r))

    # r = crawler.get_tweets_by_urls_with_retry(
    #     [
    #         "https://twitter.com/pm_me_your_knee/status/1762448211875422690",
    #         "https://twitter.com/elonmusk/status/1762389336858022132",
    #         "https://twitter.com/VitalikButerin/status/1759369749887332577",
    #         "https://twitter.com/elonmusk/status/1760504129485705598",
    #     ],
    #     retries=1,
    # )
    r = crawler.get_tweets_by_ids_with_retries(
        [
            "1762448211875422690",
            "1762389336858022132",
            "1759369749887332577",
            "1760504129485705598",
        ],
        retries=2,
    )
    print(r)
