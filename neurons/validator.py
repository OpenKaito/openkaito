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
import wandb
import tarfile
from datetime import datetime, timedelta, timezone
from traceback import print_exception
from pathlib import Path

# Bittensor
import bittensor as bt
import openai
import torch
from dotenv import load_dotenv

import openkaito
from openkaito import __version__
from openkaito.base.validator import BaseValidatorNeuron
from openkaito.crawlers.twitter.apidojo import ApiDojoTwitterCrawler
from openkaito.evaluation.evaluator import Evaluator
from openkaito.protocol import SearchSynapse, SemanticSearchSynapse
from openkaito.tasks import (
    generate_author_index_task,
    generate_discord_search_task,
    generate_question_from_eth_denver_segments,
    generate_structured_search_task,
    random_eth_denver_segments,
    random_query,
    generate_semantic_search_task,
)
from openkaito.utils.uids import get_random_uids
from openkaito.utils.version import get_version


class Validator(BaseValidatorNeuron):

    def __init__(self):
        super(Validator, self).__init__()
        load_dotenv()

        # for ranking results evaluation
        llm_client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            organization=os.getenv("OPENAI_ORGANIZATION"),
            max_retries=3,
        )
        self.llm_client = llm_client

        # for integrity check
        # twitter_crawler = MicroworldsTwitterCrawler(os.environ["APIFY_API_KEY"])
        twitter_crawler = ApiDojoTwitterCrawler(os.environ["APIFY_API_KEY"])

        self.evaluator = Evaluator(llm_client, twitter_crawler)

        with open("twitter_usernames.txt") as f:
            twitter_usernames = f.read().strip().splitlines()
        self.twitter_usernames = twitter_usernames

        self.init_eth_denver_dataset()

        netrc_path = Path.home() / ".netrc"
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is not None:
            bt.logging.info("WANDB_API_KEY is set")
        bt.logging.info("~/.netrc exists:", netrc_path.exists())

        if wandb_api_key is None and not netrc_path.exists():
            bt.logging.warning(
                "!!! WANDB_API_KEY not found in environment variables and `wandb login` didn't run. You are strongly recommended to setup wandb. We may enforce to make it required in the future."
            )
            self.config.neuron.wandb_off = True

        if not self.config.neuron.wandb_off:
            # wandb.login(key=os.environ["WANDB_API_KEY"], verify=True, relogin=True)
            wandb.init(
                project=f"sn{self.config.netuid}-validators",
                entity="openkaito",
                config={
                    "hotkey": self.wallet.hotkey.ss58_address,
                },
                name=f"validator-{self.uid}-{__version__}",
                resume="auto",
                dir=self.config.neuron.full_path,
                reinit=True,
            )

    def init_eth_denver_dataset(self):
        root_dir = __file__.split("neurons")[0]
        dataset_dir = root_dir + "datasets/eth_denver_dataset"
        self.eth_denver_dataset_dir = dataset_dir
        dataset_path = Path(dataset_dir)

        with tarfile.open(
            root_dir + "datasets/eth_denver_dataset.tar.gz", "r:gz"
        ) as tar:
            original_file_list = tar.getnames()
            original_file_list.remove("eth_denver_dataset")
            if len(list(dataset_path.glob("*.json"))) == len(original_file_list):
                bt.logging.info(
                    f"Eth Denver data already extracted to: {dataset_dir}, no need to re-extract"
                )
            else:
                tar.extractall(root_dir + "datasets")
                bt.logging.info(f"Eth Denver data extracted to: {dataset_dir}")

        bt.logging.info(
            f"{len(list(dataset_path.glob('*.json')))} files in {dataset_dir}"
        )

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

            random_number = random.random()

            # 10% discord task
            if random_number < 0.1:
                with open("bittensor_channels.json") as f:
                    channels = json.load(f)
                search_query = generate_discord_search_task(
                    query_string=None,
                    channel_ids=[random.choice(channels)["channel_id"]],
                    # earlier than 1 day messages to allow latency in validation groundtruth
                    earlier_than_timestamp=int(
                        (datetime.now() - timedelta(days=1)).timestamp() * 1000
                    ),
                    size=5,
                    version=get_version(),
                )
                search_query.timeout = 10
                bt.logging.info(
                    f"Sending {search_query.name}: {search_query.json()} to miner uids: {miner_uids}"
                )
            else:
                # 60% chance to send ETH Denver semantic search task
                if random_number < 0.7:
                    segments = random_eth_denver_segments(
                        self.eth_denver_dataset_dir, num_sources=3
                    )
                    bt.logging.debug(
                        f"{len(segments)} segments sampled from ETH Denver dataset."
                    )
                    bt.logging.trace(segments)
                    question = generate_question_from_eth_denver_segments(
                        self.llm_client, segments
                    )
                    search_query = generate_semantic_search_task(
                        query_string=question,
                        index_name="eth_denver",
                        version=get_version(),
                    )
                    # should be quick
                    search_query.timeout = 10
                    bt.logging.info(
                        f"Sending {search_query.name}: {search_query.query_string} to miner uids: {miner_uids}"
                    )

                # 20% chance to send index author data task with crawling and indexing
                elif random_number < 0.9:
                    search_query = generate_author_index_task(
                        size=10,  # author index data size
                        num_authors=2,
                    )
                    # this is a bootstrap task for users to crawl more data from the author list.
                    # miners may implement a more efficient way to crawl and index the author data in the background,
                    # instead of relying on the validator tasks
                    search_query.timeout = 90

                    bt.logging.info(
                        f"Sending {search_query.name}: author index data task, authors:{search_query.author_usernames} to miner uids: {miner_uids}"
                    )
                # 10% chance to send author search task without crawling
                elif random_number < 1:
                    search_query = generate_author_index_task(
                        size=10,  # author index data size
                        num_authors=2,
                    )
                    search_query.timeout = 10

                    bt.logging.info(
                        f"Sending {search_query.name}: author index data task, authors:{search_query.author_usernames} to miner uids: {miner_uids}"
                    )
                # 0% chance to send structured search task
                else:
                    search_query = generate_structured_search_task(
                        size=self.config.neuron.search_result_size,
                        author_usernames=random.sample(self.twitter_usernames, 100),
                    )
                    search_query.timeout = 90

                    bt.logging.info(
                        f"Sending {search_query.name}: {search_query.query_string} to miner uids: {miner_uids}"
                    )
            bt.logging.trace(
                f"miners: {[(uid, self.metagraph.axons[uid] )for uid in miner_uids]}"
            )

            # The dendrite client queries the network.
            responses = await self.dendrite(
                # Send the query to selected miner axons in the network.
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse=search_query,
                deserialize=True,
                timeout=search_query.timeout,
            )

            # Log the results for monitoring purposes.
            bt.logging.debug(f"Received responses: {responses}")

            if search_query.name == "SemanticSearchSynapse":
                rewards = self.evaluator.evaluate_semantic_search(
                    search_query, responses, self.eth_denver_dataset_dir
                )
            elif (
                search_query.name == "StructuredSearchSynapse"
                or search_query.name == "SearchSynapse"
            ):
                rewards = self.evaluator.evaluate(search_query, responses)
            elif search_query.name == "DiscordSearchSynapse":
                rewards = self.evaluator.evaluate_discord_search(
                    search_query, responses
                )
            else:
                bt.logging.error(f"Unknown search query name: {search_query.name}")
                rewards = torch.zeros(len(miner_uids))

            raw_scores = rewards.clone().detach()

            # relative scores in a batch
            rewards = rewards / (rewards.max() + 1e-5)

            bt.logging.info(f"Scored responses: {rewards} for {miner_uids}")

            self.update_scores(rewards, miner_uids)

            if not self.config.neuron.wandb_off:
                wandb_log = {
                    "synapse": search_query.json(),
                    "scores": {
                        uid.item(): reward.item()
                        for uid, reward in zip(miner_uids, rewards)
                    },
                    "raw_scores": {
                        uid.item(): raw_score.item()
                        for uid, raw_score in zip(miner_uids, raw_scores)
                    },
                    search_query.name
                    + "_scores": {
                        uid.item(): reward.item()
                        for uid, reward in zip(miner_uids, rewards)
                    },
                    search_query.name
                    + "_raw_scores": {
                        uid.item(): raw_score.item()
                        for uid, raw_score in zip(miner_uids, raw_scores)
                    },
                    # "responses": {
                    #     uid.item(): json.dumps(response)
                    #     for uid, response in zip(miner_uids, responses)
                    # },
                    search_query.name + "_avg_score": raw_scores.mean().item(),
                    "timestamp": int(datetime.now(timezone.utc).timestamp()),
                }
                wandb.log(wandb_log)
                bt.logging.debug("wandb_log", wandb_log)
            else:
                bt.logging.warning(
                    "!!! WANDB is not enabled. You are strongly recommended to obtain and set WANDB_API_KEY or run `wandb login`. We may enforce to make it required in the future."
                )

        except Exception as e:
            bt.logging.error(f"Error during forward: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))

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

        # In case of unforeseen errors, the validator will log the error and exit. (restart by pm2)
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))
            self.should_exit = True

    def print_info(self):
        metagraph = self.metagraph
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        log = (
            "Validator | "
            f"Step:{self.step} | "
            f"UID:{self.uid} | "
            f"Block:{self.block} | "
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
            if validator.should_exit:
                bt.logging.warning("Ending validator...")
                break

            time.sleep(30)
