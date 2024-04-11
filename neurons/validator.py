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
import pathlib
import random
import time
import wandb
import tarfile
from datetime import datetime, timezone
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
    generate_structured_search_task,
    random_query,
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
                entity="subnet-openkaito",
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

    def generate_question_from_eth_denver(self):
        dataset_dir = self.eth_denver_dataset_dir
        dataset_path = Path(dataset_dir)

        files = random.sample(list(dataset_path.glob("*.json")), 10)
        segments = []
        knowledge_text = ""
        for file in files:
            with open(file) as f:
                data = json.load(f)
                segments.append(data)
                knowledge_text += "Text: " + data["text"] + "\n\n"
        bt.logging.debug(f"{len(segments)} segments loaded")
        bt.logging.trace(segments)

        prompt = (
            "You are a crypto researcher, and you will be given a list of speaker transcript segments as your source of knowledge in ETH Denver 2024."
            "Analyze these speaker transcript segments from ETH Denver 2024, "
            "and generate several meaningful and profound questions that delve into the implications, strategies, "
            "and future directions discussed in the knowledge.\n\n"
            "Transcript segments:\n\n"
        )
        prompt += knowledge_text

        prompt += (
            "You need to generate one question with at most 15 words, based on the knowledge you have gained from the transcript segments."
            "Please answer with the question text only, without any additional context or explanation."
        )

        bt.logging.debug(f"Prompt: {prompt}")

        try:
            output = self.llm_client.chat.completions.create(
                model="gpt-4-turbo",
                # response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=2,
                timeout=60,
            )

            bt.logging.debug(
                f"generation questions LLM response: {output.choices[0].message.content}"
            )
            bt.logging.debug(
                f"LLM usage: {output.usage}, finish reason: {output.choices[0].finish_reason}"
            )
            return output.choices[0].message.content
        except Exception as e:
            bt.logging.error(f"Error during LLM completion: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))

        # try:
        #     questions = json.loads(output.choices[0].message.content)["questions"]
        #     assert len(questions) > 0
        #     for question in questions:
        #         assert isinstance(question, str)
        #     return questions
        # except Exception as e:
        #     if retries > 0:
        #         bt.logging.error(f"Error during questions parsing: {e}, retrying...")
        #         return self.generate_questions_from_eth_denver(retries=retries - 1)
        #     else:
        #         bt.logging.error(f"Error during questions parsing: {e}, giving up...")
        #         return ["What is the future of blockchain?"]

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
            # mixed tasks, deprecated SearchSynapse
            if random_number < 0:
                query_string = random_query(input_file="queries.txt")
                search_query = SearchSynapse(
                    query_string=query_string,
                    size=self.config.neuron.search_result_size,
                    version=get_version(),
                )
                search_query.timeout = 90
            else:
                # 20% chance to send senmantic seatch task
                if random_number < 0.2:
                    question = self.generate_question_from_eth_denver()
                    search_query = SemanticSearchSynapse(
                        query_string=question,
                        # top 10 results
                        size=10,
                        version=get_version(),
                    )
                    # should be quick
                    search_query.timeout = 10
                    bt.logging.info(
                        f"Sending {search_query.name}: {search_query.query_string} to miner uids: {miner_uids}"
                    )

                # 60% chance to send index author data task with crawling and indexing
                elif random_number < 0.8:
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
                elif random_number < 0.9:
                    search_query = generate_author_index_task(
                        size=10,  # author index data size
                        num_authors=2,
                    )
                    search_query.timeout = 10

                    bt.logging.info(
                        f"Sending {search_query.name}: author index data task, authors:{search_query.author_usernames} to miner uids: {miner_uids}"
                    )
                # 10% chance to send structured search task
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
            else:
                bt.logging.error(f"Unknown search query name: {search_query.name}")
                rewards = torch.zeros(len(miner_uids))

            bt.logging.info(f"Scored responses: {rewards} for {miner_uids}")

            self.update_scores(rewards, miner_uids)

            if not self.config.neuron.wandb_off:
                wandb_log = {
                    "synapse": search_query.json(),
                    "scores": {
                        uid.item(): reward.item()
                        for uid, reward in zip(miner_uids, rewards)
                    },
                    "responses": {
                        uid.item(): json.dumps(response)
                        for uid, response in zip(miner_uids, responses)
                    },
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
