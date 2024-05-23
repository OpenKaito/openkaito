# The MIT License (MIT)
# Copyright © 2024 OpenKaito

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import time
import typing
from datetime import datetime

import bittensor as bt
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

import openkaito
from openkaito.base.miner import BaseMinerNeuron
from openkaito.crawlers.twitter.apidojo import ApiDojoTwitterCrawler
from openkaito.protocol import (
    DiscordSearchSynapse,
    SearchSynapse,
    StructuredSearchSynapse,
    SemanticSearchSynapse,
)
from openkaito.search.ranking import HeuristicRankingModel
from openkaito.search.structured_search_engine import StructuredSearchEngine
from openkaito.utils.version import compare_version, get_version


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self):
        super(Miner, self).__init__()

        load_dotenv()

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

        # optional, for crawling data
        twitter_crawler = (
            # MicroworldsTwitterCrawler(os.environ["APIFY_API_KEY"])
            ApiDojoTwitterCrawler(os.environ["APIFY_API_KEY"])
            if os.environ.get("APIFY_API_KEY")
            else None
        )

        self.structured_search_engine = StructuredSearchEngine(
            search_client=search_client,
            relevance_ranking_model=ranking_model,
            twitter_crawler=twitter_crawler,
            recall_size=self.config.neuron.search_recall_size,
        )

    async def forward_search(self, query: SearchSynapse) -> SearchSynapse:
        """
        Processes the incoming Search synapse by performing a search operation on the crawled data.

        Args:
            query (SearchSynapse): The synapse object containing the query information.

        Returns:
            SearchSynapse: The synapse object with the 'results' field set to list of the 'Document'.
        """
        start_time = datetime.now()
        bt.logging.info(f"received SearchSynapse: ", query)
        self.check_version(query)

        if not self.config.neuron.disable_crawling:
            crawl_size = max(self.config.neuron.crawl_size, query.size)
            self.structured_search_engine.crawl_and_index_data(
                query_string=query.query_string,
                author_usernames=None,
                # crawl and index more data than needed to ensure we have enough to rank
                max_size=crawl_size,
            )

        ranked_docs = self.structured_search_engine.search(query)

        bt.logging.debug(f"{len(ranked_docs)} ranked_docs", ranked_docs)
        query.results = ranked_docs
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        bt.logging.info(
            f"processed SearchSynapse in {elapsed_time} seconds",
        )
        return query

    async def forward_structured_search(
        self, query: StructuredSearchSynapse
    ) -> StructuredSearchSynapse:

        start_time = datetime.now()
        bt.logging.info(
            f"received StructuredSearchSynapse... timeout:{query.timeout}s ", query
        )
        self.check_version(query)

        # miners may adjust this timeout config by themselves according to their own crawler speed and latency
        if query.timeout > 12:
            # do crawling and indexing, otherwise search from the existing index directly
            crawl_size = max(self.config.neuron.crawl_size, query.size)
            self.structured_search_engine.crawl_and_index_data(
                query_string=query.query_string,
                author_usernames=query.author_usernames,
                # crawl and index more data than needed to ensure we have enough to rank
                max_size=crawl_size,
            )

        # disable crawling for structured search by default
        ranked_docs = self.structured_search_engine.search(query)
        bt.logging.debug(f"{len(ranked_docs)} ranked_docs", ranked_docs)
        query.results = ranked_docs
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        bt.logging.info(
            f"processed StructuredSearchSynapse in {elapsed_time} seconds",
        )
        return query

    async def forward_semantic_search(
        self, query: SemanticSearchSynapse
    ) -> SemanticSearchSynapse:

        start_time = datetime.now()
        bt.logging.info(
            f"received SemanticSearchSynapse... timeout:{query.timeout}s ", query
        )
        self.check_version(query)

        ranked_docs = self.structured_search_engine.vector_search(query)
        bt.logging.debug(f"{len(ranked_docs)} ranked_docs", ranked_docs)
        query.results = ranked_docs
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        bt.logging.info(
            f"processed SemanticSearchSynapse in {elapsed_time} seconds",
        )
        return query

    async def forward_discord_search(
        self, query: DiscordSearchSynapse
    ) -> DiscordSearchSynapse:

        start_time = datetime.now()
        bt.logging.info(
            f"received DiscordSearchSynapse... timeout:{query.timeout}s ", query
        )
        self.check_version(query)

        ranked_docs = self.structured_search_engine.discord_search(query)
        bt.logging.debug(f"{len(ranked_docs)} ranked_docs", ranked_docs)
        query.results = ranked_docs
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        bt.logging.info(
            f"processed DiscordSearchSynapse in {elapsed_time} seconds",
        )
        return query

    def print_info(self):
        metagraph = self.metagraph
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        log = (
            "Miner | "
            f"Epoch:{self.step} | "
            f"UID:{self.uid} | "
            f"Block:{self.block} | "
            f"Stake:{metagraph.S[self.uid]} | "
            f"Rank:{metagraph.R[self.uid]} | "
            f"Trust:{metagraph.T[self.uid]} | "
            f"Consensus:{metagraph.C[self.uid] } | "
            f"Incentive:{metagraph.I[self.uid]} | "
            f"Emission:{metagraph.E[self.uid]}"
        )
        bt.logging.info(log)

    def check_version(self, query):
        """
        Check the version of the incoming request and log a warning if it is newer than the miner's running version.
        """
        if (
            query.version is not None
            and compare_version(query.version, get_version()) > 0
        ):
            bt.logging.warning(
                f"Received request with version {query.version}, is newer than miner running version {get_version()}. You may updating the repo and restart the miner."
            )


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            miner.print_info()
            time.sleep(30)
