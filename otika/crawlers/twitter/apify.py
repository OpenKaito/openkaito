from loguru import logger
from datetime import datetime
from apify_client import ApifyClient


class ApifyTwitterCrawler:
    def __init__(self, api_key, timeout_secs=40):
        self.logger = logger
        self.client = ApifyClient(api_key)

        self.timeout_secs = timeout_secs

        # microworlds actor id
        # users may use any other actor id that can crawl twitter data
        self.actor_id = "microworlds/twitter-scraper"

    def search(self, query: str, max_length: int):
        """
        Searches for the given query on the crawled data.

        Args:
            query (str): The query to search for.
            length (int): The number of results to return.

        Returns:
            list: The list of results.
        """
        self.logger.info(f"Searching for query: '{query}' with length {max_length}")
        params = {
            "maxRequestRetries": 3,
            "searchMode": "live",
            "scrapeTweetReplies": True,
            "searchTerms": [query],
            "maxTweets": max_length,
        }

        run = self.client.actor(self.actor_id).call(
            run_input=params, timeout_secs=self.timeout_secs
        )
        self.logger.info(f"Apify Actor Run: {run}")

        results = [
            item
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items()
        ]
        self.logger.info(f"Apify Results: {results}")

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
                "created_at": datetime.strptime(result.get("created_at"), time_format),
                "quote_count": result.get("quote_count"),
                "reply_count": result.get("reply_count"),
                "retweet_count": result.get("retweet_count"),
                "favorite_count": result.get("favorite_count"),
            }
            for result in results
        ]
        self.logger.info(f"Processing results: {results}")
        return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    crawler = ApifyTwitterCrawler(os.environ["APIFY_API_KEY"])

    r = crawler.search("BTC", 5)

    # print(r)
    print(crawler.process(r))
