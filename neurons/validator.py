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
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from traceback import print_exception

# Bittensor
import bittensor as bt
import openai
import torch
from dotenv import load_dotenv
import zlib

import openkaito
import wandb
from datasets import load_dataset
from openkaito import __version__
from openkaito.base.validator import BaseValidatorNeuron
from openkaito.crawlers.twitter.apidojo import ApiDojoTwitterCrawler
from openkaito.evaluation.evaluator import Evaluator
from openkaito.protocol import SearchSynapse, SemanticSearchSynapse
from openkaito.tasks import (
    generate_author_index_task,
    generate_discord_search_task,
    generate_discord_semantic_search_task_with_channel_id,
    generate_question_from_eth_conf_segments,
    generate_relevant_pairs,
    generate_semantic_search_task,
    generate_structured_search_task,
    generate_text_embedding_synapse,
    random_eth_conf_segments,
    random_query,
)
from openkaito.utils.uids import get_random_uids
from openkaito.utils.version import get_version
from openkaito.utils.embeddings import openai_embeddings_tensor


class Validator(BaseValidatorNeuron):
    def __init__(self):
        super(Validator, self).__init__()
        load_dotenv()

        # for ranking results evaluation
        llm_client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT"),
            max_retries=3,
        )
        self.llm_client = llm_client

        # for integrity check
        # twitter_crawler = MicroworldsTwitterCrawler(os.environ["APIFY_API_KEY"])
        # twitter_crawler = ApiDojoTwitterCrawler(os.environ["APIFY_API_KEY"])
        # deprecated since v0.7.0
        twitter_crawler = None

        self.evaluator = Evaluator(llm_client, twitter_crawler)

        with open("twitter_usernames.txt") as f:
            twitter_usernames = f.read().strip().splitlines()
        self.twitter_usernames = twitter_usernames

        self.init_eth_denver_dataset()
        self.init_eth_cc7_dataset()

        bt.logging.info("Loading FineWeb dataset")
        self.fineweb_dataset = load_dataset(
            "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
        )

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
                entity="sn-openkaito-openkaito",
                config={
                    "hotkey": self.wallet.hotkey.ss58_address,
                },
                name=f"validator-{self.uid}-{__version__}",
                # resume="auto",
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

    def init_eth_cc7_dataset(self):
        root_dir = __file__.split("neurons")[0]
        dataset_dir = root_dir + "datasets/eth_cc7_dataset"
        self.eth_cc7_dataset_dir = dataset_dir
        dataset_path = Path(dataset_dir)

        with tarfile.open(root_dir + "datasets/eth_cc7_dataset.tar.gz", "r:gz") as tar:
            original_file_list = tar.getnames()
            original_file_list.remove("eth_cc7_dataset")
            if len(list(dataset_path.glob("*.json"))) == len(original_file_list):
                bt.logging.info(
                    f"Eth CC[7] data already extracted to: {dataset_dir}, no need to re-extract"
                )
            else:
                tar.extractall(root_dir + "datasets")
                bt.logging.info(f"Eth CC[7] data extracted to: {dataset_dir}")

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

            query = None

            conf_dataset_dir = None
            discord_channel_id = None
            q_indices = None
            a_indices = None

            # Note: Currently, the active synapses are `SemanticSearchSynapse` and `TextEmbeddingSynapse`.

            # 80% chance to send text-embedding task
            if random_number < 0.8:
                bt.logging.info("Generating text-embedding relevant pairs...")
                pairs = generate_relevant_pairs(
                    self.fineweb_dataset,
                    num_articles=32,
                    num_pairs_per_article=2,
                    llm_client=self.llm_client,
                )
                bt.logging.info(f"Generated {len(pairs)} pairs")

                query, q_indices, a_indices = generate_text_embedding_synapse(pairs)
                # the payload might be large, need sometime for network transfer
                query.timeout = 60

                bt.logging.info(
                    f"Sending {query.name}: {query.texts} to miner uids: {miner_uids}"
                )
            # 10% chance to send ETH Denver semantic search task
            elif random_number < 0.9:
                conf_dataset_dir = self.eth_denver_dataset_dir
                segments = random_eth_conf_segments(conf_dataset_dir, num_sources=3)
                bt.logging.debug(
                    f"{len(segments)} segments sampled from ETH Denver dataset."
                )
                bt.logging.trace(segments)
                question = generate_question_from_eth_conf_segments(
                    self.llm_client, segments
                )
                query = generate_semantic_search_task(
                    query_string=question,
                    index_name="eth_denver",
                )
                # should be quick
                query.timeout = 15
                bt.logging.info(
                    f"Sending ETH Denver {query.name}: {query.query_string} to miner uids: {miner_uids}"
                )
            # 10% chance to send ETH CC[7] semantic search task
            else:
                conf_dataset_dir = self.eth_cc7_dataset_dir
                segments = random_eth_conf_segments(conf_dataset_dir, num_sources=3)
                bt.logging.debug(
                    f"{len(segments)} segments sampled from ETH CC[7] dataset."
                )
                bt.logging.trace(segments)
                question = generate_question_from_eth_conf_segments(
                    self.llm_client, segments
                )
                # must create a `eth_cc7` index in the miner
                query = generate_semantic_search_task(
                    query_string=question,
                    index_name="eth_cc7",
                )
                # should be quick
                query.timeout = 15
                bt.logging.info(
                    f"Sending ETH CC[7] {query.name}: {query.query_string} to miner uids: {miner_uids}"
                )

            # The dendrite client queries the network.
            responses = await self.dendrite(
                # Send the query to selected miner axons in the network.
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse=query,
                deserialize=True,
                timeout=query.timeout,
            )

            # Log the results for monitoring purposes.
            bt.logging.trace(f"Received responses: {responses}")

            if query.name == "SemanticSearchSynapse":
                rewards = self.evaluator.evaluate_semantic_search(
                    query, responses, conf_dataset_dir
                )
            elif (
                query.name == "StructuredSearchSynapse" or query.name == "SearchSynapse"
            ):
                rewards = self.evaluator.evaluate(query, responses)
            elif query.name == "DiscordSearchSynapse":
                rewards = self.evaluator.evaluate_discord_query_search(
                    query, responses, discord_channel_id
                )
            elif query.name == "TextEmbeddingSynapse":
                rewards, losses, top1_recalls, top3_recalls = (
                    self.evaluator.evaluate_text_embedding(
                        query, responses, q_indices, a_indices
                    )
                )
                openai_embeddings = openai_embeddings_tensor(
                    self.llm_client,
                    query.texts,
                    dimensions=query.dimensions,
                    model="text-embedding-3-large",
                )
                openai_reward, openai_loss, openai_top1_recall, openai_top3_recall = (
                    self.evaluator.evaluate_text_embedding(
                        query, [openai_embeddings.tolist()], q_indices, a_indices
                    )
                )

            else:
                bt.logging.error(f"Unknown search query name: {query.name}")
                rewards = torch.zeros(len(miner_uids))

            raw_scores = rewards.clone().detach()

            # relative scores in a batch
            rewards = rewards / (rewards.max() + 1e-5)

            bt.logging.info(f"Scored responses: {rewards} for {miner_uids}")

            self.update_scores(rewards, miner_uids)

            if not self.config.neuron.wandb_off:
                wandb_log = {
                    "synapse": zlib.compress(query.model_dump_json().encode()).hex(),
                    "scores": {
                        uid.item(): reward.item()
                        for uid, reward in zip(miner_uids, rewards)
                    },
                    "raw_scores": {
                        uid.item(): raw_score.item()
                        for uid, raw_score in zip(miner_uids, raw_scores)
                    },
                    query.name
                    + "_scores": {
                        uid.item(): reward.item()
                        for uid, reward in zip(miner_uids, rewards)
                    },
                    query.name
                    + "_raw_scores": {
                        uid.item(): raw_score.item()
                        for uid, raw_score in zip(miner_uids, raw_scores)
                    },
                    query.name
                    + "_responses": {
                        uid.item(): (
                            zlib.compress(json.dumps(response).encode()).hex()
                            if raw_score > 1e-5
                            else None
                        )
                        for uid, response, raw_score in zip(
                            miner_uids, responses, raw_scores
                        )
                    },
                    query.name + "_avg_score": raw_scores.mean().item(),
                    "timestamp": int(datetime.now(timezone.utc).timestamp()),
                }
                if query.name == "TextEmbeddingSynapse":
                    wandb_log.update(
                        {
                            "TextEmbeddingSynapse_losses": {
                                uid.item(): loss.item()
                                for uid, loss in zip(miner_uids, losses)
                            },
                            "TextEmbeddingSynapse_top1_recalls": {
                                uid.item(): top1_recall.item()
                                for uid, top1_recall in zip(miner_uids, top1_recalls)
                            },
                            "TextEmbeddingSynapse_top3_recalls": {
                                uid.item(): top3_recall.item()
                                for uid, top3_recall in zip(miner_uids, top3_recalls)
                            },
                            "TextEmbeddingSynapse_avg_loss": losses.nanmean().item(),
                            "TextEmbeddingSynapse_avg_top1_recall": top1_recalls.nanmean().item(),
                            "TextEmbeddingSynapse_avg_top3_recall": top3_recalls.nanmean().item(),
                            "TextEmbeddingSynapse_openai_raw_score": openai_reward.item(),
                            "TextEmbeddingSynapse_openai_loss": openai_loss.item(),
                            "TextEmbeddingSynapse_openai_top1_recall": openai_top1_recall.item(),
                            "TextEmbeddingSynapse_openai_top3_recall": openai_top3_recall.item(),
                        }
                    )

                log_size = len(json.dumps(wandb_log))
                bt.logging.debug(f"wandb_log original size: {log_size} bytes")

                # avoid exceeding wandb log size limit
                if log_size > 10_000_000:
                    wandb_log.pop("synapse")
                    log_size = len(json.dumps(wandb_log))
                    if log_size > 10_000_000:
                        wandb_log.pop(query.name + "_responses")
                        log_size = len(json.dumps(wandb_log))

                wandb.log(wandb_log)

                # clearer printing
                if query.name + "_responses" in wandb_log:
                    wandb_log.pop(query.name + "_responses")
                if "synapse" in wandb_log:
                    wandb_log.pop("synapse")

                log_size = len(json.dumps(wandb_log))
                bt.logging.debug("wandb_log", f"size: {log_size} bytes", wandb_log)
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
