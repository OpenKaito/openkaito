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


import json
import os
import random
import time
from datetime import datetime, timezone
from traceback import print_exception

# Bittensor
import bittensor as bt
import openai
import torch
from dotenv import load_dotenv

import openkaito
from openkaito.base.validator import BaseValidatorNeuron
from openkaito.crawlers.twitter.apify import ApifyTwitterCrawler
from openkaito.evaluation.evaluator import Evaluator
from openkaito.protocol import SearchSynapse
from openkaito.utils.uids import get_random_uids
from openkaito.utils.version import get_version


def random_query(input_file="queries.txt"):
    if not os.path.exists(input_file):
        bt.logging.error(f"Queries file not found at location: {input_file}")
        exit(1)
    lines = open(input_file).read().splitlines()
    return random.choice(lines)


class Validator(BaseValidatorNeuron):

    def __init__(self):
        super(Validator, self).__init__()
        load_dotenv()

        # for ranking results evaluation
        llm_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORGANIZATION"),
            max_retries=3,
        )

        # for integrity check
        twitter_crawler = ApifyTwitterCrawler(os.environ["APIFY_API_KEY"])

        self.evaluator = Evaluator(llm_client, twitter_crawler)

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        try:
            miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

            query_string = random_query()
            search_query = SearchSynapse(
                query_string=query_string,
                size=self.config.neuron.search_result_size,
                version=get_version(),
            )

            bt.logging.info(
                f"Sending search: {search_query} to miners: {[(uid, self.metagraph.axons[uid] )for uid in miner_uids]}"
            )

            # The dendrite client queries the network.
            responses = await self.dendrite(
                # Send the query to selected miner axons in the network.
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse=search_query,
                deserialize=True,
                # set the miner query timeout to be 120 seconds to allow more operations in miner
                timeout=120,
            )

            # Log the results for monitoring purposes.
            bt.logging.info(f"Received responses: {responses}")

            rewards = self.evaluator.evaluate(
                search_query.query_string, search_query.size, responses
            )

            bt.logging.info(f"Scored responses: {rewards} for {miner_uids}")
            # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
            self.update_scores(rewards, miner_uids)
        except Exception as e:
            bt.logging.error(f"Error during forward: {e}")

    def run(self):
        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

                # Sleep interval before the next iteration.
                time.sleep(self.config.neuron.search_request_interval)

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))
    

    def print_info(self):
        metagraph = self.metagraph
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        log = (
            "Validator | "
            f"Step:{self.step} | "
            f"UID:{self.uid} | "
            f"Block:{metagraph.block.item()} | "
            f"Stake:{metagraph.S[self.uid]} | "
            f"VTrust:{metagraph.Tv[self.uid]} | "
            f"Dividend:{metagraph.D[self.uid]} | "
            f"Emission:{metagraph.E[self.uid]}"
        )
        bt.logging.info(log)



# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            validator.print_info()
            time.sleep(30)
