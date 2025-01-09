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
import openai
from datetime import datetime

import bittensor as bt
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import asyncio

import openkaito
from openkaito.base.miner import BaseMinerNeuron
from openkaito.crawlers.twitter.apidojo import ApiDojoTwitterCrawler
from openkaito.protocol import (
    DiscordSearchSynapse,
    SearchSynapse,
    SemanticSearchSynapse,
    StructuredSearchSynapse,
    TextEmbeddingSynapse,
    OfficialSynapse
)
from openkaito.search.ranking import HeuristicRankingModel
from openkaito.search.structured_search_engine import StructuredSearchEngine
from openkaito.utils.embeddings import openai_embeddings_tensor
from openkaito.utils.version import compare_version, get_version


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.
    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.
    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self):
        super(Miner, self).__init__()

        # TODO: remove this before merging
        self.dendrite = bt.dendrite(wallet=self.wallet)
        # skip intialization and wallet check when in debug mode, which only unit tests its forward methods

    # DEPRECATED: delete the function as no longer used
    async def forward_search(self, query: SearchSynapse) -> SearchSynapse:
        pass

    # DEPRECATED: delete the function as no longer used
    async def forward_structured_search(
        self, query: StructuredSearchSynapse
    ) -> StructuredSearchSynapse:
        pass

    # DEPRECATED: delete the function as no longer used
    async def forward_semantic_search(
        self, query: SemanticSearchSynapse
    ) -> SemanticSearchSynapse:
        pass

    # DEPRECATED: delete the function as no longer used
    async def forward_discord_search(
        self, query: DiscordSearchSynapse
    ) -> DiscordSearchSynapse:
        pass

    # Example of a text embedding function
    async def forward_text_embedding(
        self, query: TextEmbeddingSynapse
    ) -> TextEmbeddingSynapse:
        texts = query.texts
        dimensions = query.dimensions
        import openai

        client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT"),
            max_retries=3,
        )

        embeddings = openai_embeddings_tensor(
            client, texts, dimensions=dimensions, model="text-embedding-3-large"
        )
        query.results = embeddings.tolist()
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

    # TODO: Remove the function after testing
    async def test_send_official_synapse(self, validator_uid: int):
        query = OfficialSynapse(
            query_string=["Greetings from miner!"]
        )

        if validator_uid < 0 or validator_uid >= len(self.metagraph.axons):
            bt.logging.error(f"Invalid validator_uid: {validator_uid}")
            return
        validator_axon_endpoint = self.metagraph.axons[validator_uid]

        if not validator_axon_endpoint.is_serving:
            bt.logging.error(f"Validator at UID {validator_uid} is not serving.")
            return

        try:
            timeout_secs = 30
            responses = await self.dendrite(
                axons=[validator_axon_endpoint],
                synapse=query,
                deserialize=True,
                timeout=timeout_secs,
            )
            bt.logging.info(f"[Miner] OfficialSynapse responses: {responses}")

        except Exception as e:
            bt.logging.error(f"[Miner] Error sending OfficialSynapse: {e}")


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:

        # TODO: Remove the following block before merging
        miner_hotkey = miner.wallet.hotkey.ss58_address
        print(f"My Miner hotkey: {miner_hotkey}")
        # loop = asyncio.get_event_loop()
        # loop.run_until_complete(miner.test_send_official_synapse(144))

        time.sleep(120)

        while True:
            miner.print_info()
            time.sleep(30)
            # TODO: Remove the following block before merging
            # loop = asyncio.get_event_loop()
            # loop.run_until_complete(miner.test_send_official_synapse(144))
