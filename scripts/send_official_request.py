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
from datetime import datetime, timezone
from pathlib import Path
from traceback import print_exception
import asyncio

# Bittensor
import bittensor as bt
import openai
import torch
from dotenv import load_dotenv
import zlib
from typing import Tuple, List

import openkaito
import wandb
from openkaito import __version__
from openkaito.base.validator import BaseValidatorNeuron
from openkaito.evaluation.evaluator import Evaluator
from openkaito.protocol import SearchSynapse, SemanticSearchSynapse, TextEmbeddingSynapse, OfficialSynapse
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
from openkaito.utils.uids import get_random_uids, get_validator_uids
from openkaito.utils.version import get_version
from openkaito.utils.embeddings import openai_embeddings_tensor
from openkaito.utils.datasets_config import cached_datasets_from_config



class Validator(BaseValidatorNeuron):
    def __init__(self):
        super(Validator, self).__init__()
        load_dotenv()

        self.dendrite = bt.dendrite(wallet=self.wallet)

        # for ranking results evaluation
        llm_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT"),
            max_retries=3,
        )
        self.llm_client = llm_client
        twitter_crawler = None
        self.evaluator = Evaluator(llm_client, twitter_crawler)

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


        # TODO: remove this line before merge
        self.config.neuron.wandb_off = True
        if not self.config.neuron.wandb_off:
            # wandb.login(key=os.environ["WANDB_API_KEY"], verify=True, relogin=True)
            wandb.init(
                project=f"sn{self.config.netuid}-validators",
                entity="sn-openkaito-openkaito",
                config={
                    "hotkey": self.wallet.hotkey.ss58_address,
                    "spec_version": openkaito.__spec_version__,
                },
                name=f"validator-{self.uid}-{__version__}",
                # resume="auto",
                dir=self.config.neuron.full_path,
                reinit=True,
            )

    async def forward_official(self, synapse: OfficialSynapse) -> OfficialSynapse:
        bt.logging.info(f"[Message from Official] forward_official() got query: {synapse.query_string}")
        synapse.results = [
            f"Validator echo from forward_official: {synapse.query_string}"
        ]
        return synapse

    async def blacklist_official(self, synapse: OfficialSynapse) -> Tuple[bool, str]:
        if not synapse.dendrite.hotkey:
            return True, "Not a valid hotkey in `synapse.dendrite.hotkey`"
        if synapse.dendrite.hotkey in self.allowed_hotkeys:
            return False, f"Hotkey {synapse.dendrite.hotkey} in whitelist"
        else:
            return True, f"Hotkey {synapse.dendrite.hotkey} not in whitelist"

    async def priority_official(self, synapse: OfficialSynapse) -> float:
        return 1.0

    async def send_official_synapse(self, specified_validator_uids: List[int]):

        if specified_validator_uids:
            validator_uids = specified_validator_uids
        else:
            # NOTE: broadcast to all validators
            validator_uids = get_validator_uids(self, remove_self=True)
            bt.logging.info(f"Validator IDs: {validator_uids}")

        text_embedding_datasets = cached_datasets_from_config(branch="main")[
            "text_embedding_datasets"
        ]
        selected_dataset = random.choice(list(text_embedding_datasets.items()))

        pairs = generate_relevant_pairs(
            dataset=selected_dataset[1]["dataset"],
            num_articles=1, #TODO: change to random
            num_pairs_per_article=2, 
            llm_client=self.llm_client,
            text_field_name=selected_dataset[1]["text_field_name"],
            min_sentences=10,
        )

        embedding_synapse, _, _ = generate_text_embedding_synapse(
            pairs, dimensions=128
        )
        embedding_synapse.timeout = 30

        official_synapse = OfficialSynapse(
            texts=embedding_synapse.texts,
            dimensions=embedding_synapse.dimensions,
            normalized=embedding_synapse.normalized
        )
        official_synapse.timeout = 30

        for vuid in validator_uids:
  
            if vuid < 0 or vuid >= len(self.metagraph.axons):
                bt.logging.error(f"Invalid validator_uid: {vuid}")
                continue

            validator_axon = self.metagraph.axons[vuid]

            try:
                bt.logging.info(f"Sending OfficialSynapse to validator UID: {vuid}")
                responses = await self.dendrite(
                    axons=[validator_axon],
                    synapse=official_synapse,
                    deserialize=True,
                    timeout=official_synapse.timeout,
                )
                bt.logging.info(
                    f"Got {len(responses)} response(s) from validator {vuid}, result: {responses}"
                )
            except Exception as e:
                bt.logging.error(f"Error sending OfficialSynapse to validator {vuid}: {e}")


    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        # TODO: delete this when merge
        specified_validator_uids = [144]

        if specified_validator_uids:
            validator_uids = specified_validator_uids
        else:
            # NOTE: broadcast to all validators
            validator_uids = get_validator_uids(self, remove_self=True)
            bt.logging.info(f"Validator IDs: {validator_uids}")

        text_embedding_datasets = cached_datasets_from_config(branch="main")[
            "text_embedding_datasets"
        ]
        selected_dataset = random.choice(list(text_embedding_datasets.items()))

        pairs = generate_relevant_pairs(
            dataset=selected_dataset[1]["dataset"],
            num_articles=1, #TODO: change to random
            num_pairs_per_article=2, 
            llm_client=self.llm_client,
            text_field_name=selected_dataset[1]["text_field_name"],
            min_sentences=10,
        )

        embedding_synapse, _, _ = generate_text_embedding_synapse(
            pairs, dimensions=128
        )
        embedding_synapse.timeout = 30

        official_synapse = OfficialSynapse(
            texts=embedding_synapse.texts,
            dimensions=embedding_synapse.dimensions,
            normalized=embedding_synapse.normalized
        )
        official_synapse.timeout = 60

        for vuid in validator_uids:
  
            if vuid < 0 or vuid >= len(self.metagraph.axons):
                bt.logging.error(f"Invalid validator_uid: {vuid}")
                continue

            validator_axon = self.metagraph.axons[vuid]

            try:
                bt.logging.info(f"Sending OfficialSynapse to validator UID: {vuid}")
                responses = await self.dendrite(
                    axons=[validator_axon],
                    synapse=official_synapse,
                    deserialize=True,
                    timeout=official_synapse.timeout,
                )
                bt.logging.info(
                    f"Got {len(responses)} response(s) from validator {vuid}, result: {responses}"
                )
            except Exception as e:
                bt.logging.error(f"Error sending OfficialSynapse to validator {vuid}: {e}")
        # TODO: turn back this when merge
        """
        try:
            miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

            random_number = random.random()

            query = None

            conf_dataset_dir = None
            discord_channel_id = None
            q_indices = None
            a_indices = None
            selected_dataset = None

            # Note: Currently, the active synapses are `SemanticSearchSynapse` and `TextEmbeddingSynapse`.

            # 100% chance to send text-embedding task
            if random_number < 1 + 1e-5:
                bt.logging.info("Generating text-embedding relevant pairs...")

                text_embedding_datasets = cached_datasets_from_config(branch="main")[
                    "text_embedding_datasets"
                ]

                selected_dataset = random.choice(list(text_embedding_datasets.items()))
                bt.logging.info(f"Selected dataset: {selected_dataset[0]}")

                num_articles = random.randint(8, 32)
                bt.logging.info(f"Number of articles: {num_articles}")

                pairs = generate_relevant_pairs(
                    dataset=selected_dataset[1]["dataset"],
                    num_articles=num_articles,
                    num_pairs_per_article=4,
                    llm_client=self.llm_client,
                    text_field_name=selected_dataset[1]["text_field_name"],
                    min_sentences=10,
                )
                bt.logging.info(f"Generated {len(pairs)} pairs")

                query, q_indices, a_indices = generate_text_embedding_synapse(
                    pairs, dimensions=1024
                )
                # the payload might be large, need sometime for network transfer
                query.timeout = 60

                seperator = "\n---\n"
                bt.logging.info(
                    f"Sending {query.name}: {seperator.join(query.texts)} to miner uids: {miner_uids}"
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
                (
                    rewards,
                    losses,
                    top1_recalls,
                    top3_recalls,
                ) = self.evaluator.evaluate_text_embedding(
                    query, responses, q_indices, a_indices
                )
                openai_embeddings = openai_embeddings_tensor(
                    self.llm_client,
                    query.texts,
                    dimensions=query.dimensions,
                    model="text-embedding-3-large",
                )
                (
                    openai_reward,
                    openai_loss,
                    openai_top1_recall,
                    openai_top3_recall,
                ) = self.evaluator.evaluate_text_embedding(
                    query, [openai_embeddings.tolist()], q_indices, a_indices
                )

            else:
                bt.logging.error(f"Unknown search query name: {query.name}")
                rewards = torch.zeros(len(miner_uids))

            raw_scores = rewards.clone().detach()

            # relative scores in a batch
            rewards = rewards / (rewards.max() + 1e-5)

            bt.logging.info(f"Scored responses: {rewards} for {miner_uids}")

            self.update_scores(rewards, miner_uids)

            # TODO: remove this line before merge
            self.config.neuron.wandb_off = True
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
                            "TextEmbeddingSynapse_dataset": selected_dataset[0],
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
                            f"TextEmbeddingSynapse_{selected_dataset[0]}_losses": {
                                uid.item(): loss.item()
                                for uid, loss in zip(miner_uids, losses)
                            },
                            f"TextEmbeddingSynapse_{selected_dataset[0]}_top1_recalls": {
                                uid.item(): top1_recall.item()
                                for uid, top1_recall in zip(miner_uids, top1_recalls)
                            },
                            f"TextEmbeddingSynapse_{selected_dataset[0]}_top3_recalls": {
                                uid.item(): top3_recall.item()
                                for uid, top3_recall in zip(miner_uids, top3_recalls)
                            },
                            "TextEmbeddingSynapse_avg_loss": losses.nanmean().item(),
                            "TextEmbeddingSynapse_avg_top1_recall": top1_recalls.nanmean().item(),
                            "TextEmbeddingSynapse_avg_top3_recall": top3_recalls.nanmean().item(),
                            f"TextEmbeddingSynapse_{selected_dataset[0]}_avg_loss": losses.nanmean().item(),
                            f"TextEmbeddingSynapse_{selected_dataset[0]}_avg_top1_recall": top1_recalls.nanmean().item(),
                            f"TextEmbeddingSynapse_{selected_dataset[0]}_avg_top3_recall": top3_recalls.nanmean().item(),
                            "TextEmbeddingSynapse_openai_raw_score": openai_reward.item(),
                            "TextEmbeddingSynapse_openai_loss": openai_loss.item(),
                            "TextEmbeddingSynapse_openai_top1_recall": openai_top1_recall.item(),
                            "TextEmbeddingSynapse_openai_top3_recall": openai_top3_recall.item(),
                            f"TextEmbeddingSynapse_openai_{selected_dataset[0]}_raw_score": openai_reward.item(),
                            f"TextEmbeddingSynapse_openai_{selected_dataset[0]}_loss": openai_loss.item(),
                            f"TextEmbeddingSynapse_openai_{selected_dataset[0]}_top1_recall": openai_top1_recall.item(),
                            f"TextEmbeddingSynapse_openai_{selected_dataset[0]}_top3_recall": openai_top3_recall.item(),
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
        """
            

    def run(self):
        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")
        self.axon.start()
        bt.logging.info("Axon started and ready to handle OfficialSynapse requests.")

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
    intialization = True

    with Validator() as validator:

        miner_hotkey = validator.wallet.hotkey.ss58_address
        print(f"My Miner hotkey: {miner_hotkey}")

        # loop = asyncio.get_event_loop()
        # loop.run_until_complete(validator.send_official_synapse([144]))

        while True:
            if validator.should_exit:
                bt.logging.warning("Ending validator...")
                break
            # wait before the first print_info, to avoid websocket connection race condition
            if intialization:
                time.sleep(60 * 5)
                intialization = False

            # loop = asyncio.get_event_loop()
            # loop.run_until_complete(validator.send_official_synapse([144]))

            time.sleep(60)
            validator.print_info()
