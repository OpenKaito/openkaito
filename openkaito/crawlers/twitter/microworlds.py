from datetime import datetime
from apify_client import ApifyClient

import bittensor as bt

from openkaito.evaluation.utils import tweet_url_to_id


class MicroworldsTwitterCrawler:
    def __init__(self, api_key, timeout_secs=60):
        self.client = ApifyClient(api_key)

        self.timeout_secs = timeout_secs

        # microworlds actor id
        # users may use any other actor id that can crawl twitter data
        self.actor_id = "microworlds/twitter-scraper"

    def get_tweet_by_url(self, url: str, max_size=20):
        """
        Get tweets by urls.

        Args:
            url (str): The url to get tweet from.

            # because if url is a reply, all tweets in the thread will also be included in the result
            max_size (int): The max number of tweets to return.

        Returns:
            dict: The tweet details.
        """
        bt.logging.debug(f"Getting tweet from url: {url}")
        params = {
            "maxRequestRetries": 3,
            "searchMode": "live",
            "scrapeTweetReplies": True,
            "urls": [url],
            # because if url is a reply, all tweets in the thread will also be included in the result
            "maxTweets": max_size,
        }
        tweet_id = tweet_url_to_id(url)

        run = self.client.actor(self.actor_id).call(
            run_input=params, timeout_secs=self.timeout_secs
        )
        # filter out the original head tweet if requested url is reply
        for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
            if item.get("id_str") == tweet_id:
                return item

        return None

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

        result = self.process_list(
            self.client.dataset(run["defaultDatasetId"]).iterate_items()
        )
        bt.logging.trace(f"Apify Actor Result: {result}")
        return result

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
            "id": item["id_str"],
            "url": item["url"],
            "username": item["user"]["screen_name"],
            "text": item.get("full_text"),
            "created_at": datetime.strptime(
                item.get("created_at"), time_format
            ).isoformat(),
            "quote_count": item.get("quote_count"),
            "reply_count": item.get("reply_count"),
            "retweet_count": item.get("retweet_count"),
            "favorite_count": item.get("favorite_count"),
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
    crawler = MicroworldsTwitterCrawler(os.environ["APIFY_API_KEY"])

    r = crawler.search("BTC", 5)
    print(r)

    # r = crawler.get_tweet_by_url(
    #     "https://twitter.com/pm_me_your_knee/status/1762448211875422690"
    #     # "https://twitter.com/elonmusk/status/1762389336858022132"
    #     # "https://twitter.com/VitalikButerin/status/1759369749887332577",
    #     # "https://twitter.com/elonmusk/status/1760504129485705598",
    # )
    # print(crawler.process_item(r))
