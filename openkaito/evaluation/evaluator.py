import json
import os
from pathlib import Path
import random
from datetime import datetime, timezone
from dateutil import parser
from traceback import print_exception

import bittensor as bt
import dateutil.parser
import openai
import requests
import torch
import dateutil

from openkaito.protocol import SortType

from .utils import (
    ndcg_score,
    parse_llm_result,
    parse_llm_result_for_author_index,
    parse_llm_result_for_discord_msg,
    tweet_url_to_id,
)

DISCORD_MESSAGE_VALIDATE_API_URL = (
    "https://hx36hc3ne0.execute-api.us-west-2.amazonaws.com/dev/discord/{msg_id}"
)


class Evaluator:
    def __init__(self, llm_client, twitter_crawler=None) -> None:
        # for ranking results evaluation
        self.llm_client = llm_client

        # for integrity check
        self.twitter_crawler = twitter_crawler

        with open("twitter_usernames.txt", "r") as f:
            self.credit_twitter_author_usernames = set(f.read().strip().splitlines())
        bt.logging.info(
            f"loaded {len(self.credit_twitter_author_usernames)} credit_twitter_author_usernames"
        )

    def evaluate(self, query: bt.Synapse, responses: list):
        query_string = query.query_string
        size = query.size

        scores = torch.zeros(len(responses))

        zero_score_mask = torch.ones(len(responses))

        rank_scores = torch.zeros(len(responses))

        avg_ages = torch.zeros(len(responses))
        avg_age_scores = torch.zeros(len(responses))
        uniqueness_scores = torch.zeros(len(responses))
        credit_author_scores = torch.zeros(len(responses))

        now = datetime.now(timezone.utc)
        max_avg_age = 0

        spot_check_id_dict = dict()
        # quick integrity check and get spot_check_id_dict
        utcnow = datetime.now(timezone.utc)
        for i, response in enumerate(responses):
            try:
                if response is None or not response or len(response) > size:
                    zero_score_mask[i] = 0
                    continue
                for doc in response:
                    doc_id = doc["id"]
                    url_id = tweet_url_to_id(doc["url"])
                    if doc_id != url_id:
                        bt.logging.info(
                            f"Document id {doc_id} not match url id {url_id}"
                        )
                        zero_score_mask[i] = 0
                        break
                    if datetime.fromisoformat(doc["created_at"].rstrip("Z")) > utcnow:
                        bt.logging.info(
                            f"created_at {doc['created_at']} is in the future"
                        )
                        zero_score_mask[i] = 0
                        break
                spot_check_id_dict[i] = random.choice(response)["id"]
            except Exception as e:
                bt.logging.error(
                    f"Error while intitial checking {i}-th response: {e}, 0 score"
                )
                bt.logging.debug(print_exception(type(e), e, e.__traceback__))
                zero_score_mask[i] = 0

        if self.twitter_crawler is not None:
            bt.logging.debug(f"spot_check_id_dict: {spot_check_id_dict}")
            groundtruth_docs = self.twitter_crawler.get_tweets_by_ids_with_retries(
                list(set(spot_check_id_dict.values())), retries=2
            )
            bt.logging.debug(f"groundtruth_docs: {groundtruth_docs}")
            groundtruth_check = len(groundtruth_docs) > 0
            if not groundtruth_check:
                bt.logging.warning(
                    "groundtruth_docs is empty, apify scraper is likely to be down, skipping check"
                )
        else:
            groundtruth_check = False
            bt.logging.warning(
                "Twitter crawler is not initialized. spot content check is skipped."
            )

        for i, response in enumerate(responses):
            try:
                if zero_score_mask[i] == 0:
                    continue

                bt.logging.trace(f"Processing {i}-th response")
                if groundtruth_check:
                    # the spot check doc did not get fetched
                    if spot_check_id_dict[i] not in groundtruth_docs:
                        bt.logging.info(
                            f"spot check id {spot_check_id_dict[i]} can not be fetched in groundtruth_docs"
                        )
                        zero_score_mask[i] = 0
                        continue

                    # check all docs against groundtruth, if fetched
                    for doc in response:
                        if doc["id"] in groundtruth_docs:
                            bt.logging.trace(f"Checking doc {doc['id']}")
                            if not self.check_document(
                                doc, groundtruth_docs[doc["id"]]
                            ):
                                zero_score_mask[i] = 0
                                break

                if query.name == "StructuredSearchSynapse":
                    # for author index task
                    # check if the response is from the request author list
                    if query.author_usernames is not None:
                        if not all(
                            doc["username"] in query.author_usernames
                            for doc in response
                        ):
                            zero_score_mask[i] = 0
                            continue

                    # check if the response is within the time range filter
                    if query.earlier_than_timestamp is not None:
                        if not all(
                            get_datetime(doc["created_at"]).timestamp()
                            < query.earlier_than_timestamp
                            for doc in response
                        ):
                            zero_score_mask[i] = 0
                            continue
                    if query.later_than_timestamp is not None:
                        if not all(
                            get_datetime(doc["created_at"]).timestamp()
                            > query.later_than_timestamp
                            for doc in response
                        ):
                            zero_score_mask[i] = 0
                            continue

                    bt.logging.debug(
                        f"Integrity check passed for {i}-th response: ", response
                    )

                id_set = set()
                credit_username_count = 0
                for doc in response:
                    avg_ages[i] += (
                        now - datetime.fromisoformat(doc["created_at"].rstrip("Z"))
                    ).total_seconds()
                    id_set.add(doc["id"])
                    if doc["username"] in self.credit_twitter_author_usernames:
                        credit_username_count += 1
                avg_ages[i] /= len(response)
                max_avg_age = max(max_avg_age, avg_ages[i])

                uniqueness_scores[i] = len(id_set) / size
                credit_author_scores[i] = credit_username_count / size

                # index author data task
                if (
                    query.name == "StructuredSearchSynapse"
                    and query.author_usernames is not None
                ):
                    llm_ranking_scores = self.llm_author_index_data_evaluation(response)
                    # mean quality score
                    rank_scores[i] = sum(llm_ranking_scores) / len(llm_ranking_scores)
                else:
                    llm_ranking_scores = self.llm_keyword_ranking_evaluation(
                        query_string, response
                    )
                    rank_scores[i] = ndcg_score(llm_ranking_scores, size)

                bt.logging.info(f"Quality score: {rank_scores[i]}")
            except Exception as e:
                bt.logging.error(f"Error while processing {i}-th response: {e}")
                bt.logging.debug(print_exception(type(e), e, e.__traceback__))
                zero_score_mask[i] = 0

        # age contribution to encourage recency
        avg_age_scores = 1 - (avg_ages / (max_avg_age + 1))

        scores = avg_age_scores * 0.2 + rank_scores * 0.7 + credit_author_scores * 0.1
        scores = scores * uniqueness_scores

        # relative scores in a batch
        # scores = scores / (scores.max() + 1e-5)

        # return raw scores for tracking
        return scores * zero_score_mask

    def evaluate_semantic_search(
        self, query: bt.Synapse, responses: list, dataset_dir: str
    ):
        query_string = query.query_string
        size = query.size

        dataset_path = Path(dataset_dir)

        scores = torch.zeros(len(responses))

        zero_score_mask = torch.ones(len(responses))
        rank_scores = torch.zeros(len(responses))
        uniqueness_scores = torch.zeros(len(responses))

        for i, response in enumerate(responses):
            try:
                bt.logging.trace(f"Processing {i}-th response")
                if response is None or not response or len(response) > size:
                    zero_score_mask[i] = 0
                    continue

                id_set = set()
                groundtruth_docs = []
                for doc in response:
                    id_set.add(doc["doc_id"])
                    groundtruth_path = dataset_path / f"{doc['doc_id']}.json"
                    if groundtruth_path.exists():
                        with open(groundtruth_path, "r") as f:
                            groundtruth_docs.append(json.load(f))
                    else:
                        bt.logging.warning(
                            f"Groundtruth file {groundtruth_path} not found"
                        )
                        zero_score_mask[i] = 0
                        break

                if zero_score_mask[i] == 0:
                    continue

                uniqueness_scores[i] = len(id_set) / size

                llm_ranking_scores = self.llm_semantic_search_evaluation(
                    query_string, groundtruth_docs
                )
                rank_scores[i] = ndcg_score(llm_ranking_scores, size)

                bt.logging.info(f"Semantic search quality score: {rank_scores[i]}")
            except Exception as e:
                bt.logging.error(f"Error while processing {i}-th response: {e}")
                bt.logging.debug(print_exception(type(e), e, e.__traceback__))
                zero_score_mask[i] = 0

        scores = rank_scores * uniqueness_scores

        # relative scores in a batch
        # scores = scores / (scores.max() + 1e-5)

        # return raw scores for tracking
        return scores * zero_score_mask

    def evaluate_discord_search(self, query, responses):

        # query_string = query.query_string
        size = query.size

        scores = torch.zeros(len(responses))

        zero_score_mask = torch.ones(len(responses))

        rank_scores = torch.zeros(len(responses))

        avg_ages = torch.zeros(len(responses))
        avg_age_scores = torch.zeros(len(responses))
        uniqueness_scores = torch.zeros(len(responses))

        now = datetime.now(timezone.utc)
        max_avg_age = 0
        min_avg_age = float("inf")

        for i, response in enumerate(responses):
            try:
                if response is None or not response or len(response) > size:
                    zero_score_mask[i] = 0
                    continue

                # groundtruth integrity check
                for doc in response:
                    doc_id = doc["id"]
                    discord_msg_validate_url = DISCORD_MESSAGE_VALIDATE_API_URL.format(
                        msg_id=doc_id
                    )
                    try:
                        groundtruth = requests.get(discord_msg_validate_url).json()

                        if groundtruth["id"] != doc_id:
                            bt.logging.warning(
                                f"Discord message id {doc_id} not match url id {groundtruth['id']}"
                            )
                            zero_score_mask[i] = 0
                            break
                        if groundtruth["text"] != doc["text"]:
                            bt.logging.warning(
                                f"Document text {doc['text']} not match ground truth {groundtruth['text']}"
                            )
                            zero_score_mask[i] = 0
                            break
                        if groundtruth["author_username"] != doc["author_username"]:
                            bt.logging.warning(
                                f"Document author_username {doc['author_username']} not match ground truth {groundtruth['author_username']}"
                            )
                            zero_score_mask[i] = 0
                            break
                        if dateutil.parser.isoparse(
                            groundtruth["created_at"]
                        ) != dateutil.parser.isoparse(doc["created_at"]):
                            bt.logging.warning(
                                f"Document created_at {doc['created_at']} not match ground truth {groundtruth['created_at']}"
                            )
                            zero_score_mask[i] = 0
                            break

                    except Exception as e:
                        bt.logging.error(f"Error while validating discord message: {e}")
                        bt.logging.debug(print_exception(type(e), e, e.__traceback__))
                        zero_score_mask[i] = 0
                        break
            except Exception as e:
                bt.logging.error(
                    f"Error while intitial checking {i}-th response: {e}, 0 score"
                )
                bt.logging.debug(print_exception(type(e), e, e.__traceback__))
                zero_score_mask[i] = 0

        for i, response in enumerate(responses):
            try:
                if zero_score_mask[i] == 0:
                    continue

                bt.logging.debug(f"Processing {i}-th response")

                # for author index task
                # check if the response is from the request author list
                if query.author_usernames is not None:
                    if not all(
                        doc["author_username"] in query.author_usernames
                        for doc in response
                    ):
                        bt.logging.warning(
                            f"Author username not in query author usernames {query.author_usernames}"
                        )
                        zero_score_mask[i] = 0
                        continue

                if query.sort_by == SortType.RECENCY:
                    # check if the response is sorted by recency
                    if not all(
                        dateutil.parser.isoparse(a["created_at"])
                        > dateutil.parser.isoparse(b["created_at"])
                        for a, b in zip(response, response[1:])
                    ):
                        bt.logging.warning("Response not sorted by recency")
                        # not sorted by recency
                        zero_score_mask[i] = 0
                        continue

                if query.channel_ids is not None:
                    if not all(
                        doc["channel_id"] in query.channel_ids for doc in response
                    ):
                        bt.logging.warning(
                            f"Channel id not in query channel ids {query.channel_ids}"
                        )
                        zero_score_mask[i] = 0
                        continue

                # check if the response is within the time range filter
                if query.earlier_than_timestamp is not None:
                    if not all(
                        dateutil.parser.isoparse(doc["created_at"]).timestamp()
                        < query.earlier_than_timestamp
                        for doc in response
                    ):
                        bt.logging.warning(
                            f"created_at {doc['created_at']} is later than earlier_than_timestamp {query.earlier_than_timestamp}"
                        )
                        zero_score_mask[i] = 0
                        continue
                if query.later_than_timestamp is not None:
                    if not all(
                        dateutil.parser.isoparse(doc["created_at"]).timestamp()
                        > query.later_than_timestamp
                        for doc in response
                    ):
                        bt.logging.warning(
                            f"created_at {doc['created_at']} is earlier than later_than_timestamp {query.later_than_timestamp}"
                        )
                        zero_score_mask[i] = 0
                        continue

                bt.logging.debug(
                    f"Integrity check passed for {i}-th response: ", response
                )

                id_set = set()
                for doc in response:
                    avg_ages[i] += (
                        now - dateutil.parser.isoparse(doc["created_at"])
                    ).total_seconds()
                    id_set.add(doc["id"])
                avg_ages[i] /= len(response)
                max_avg_age = max(max_avg_age, avg_ages[i])
                min_avg_age = min(min_avg_age, avg_ages[i])

                uniqueness_scores[i] = len(id_set) / size
                if query.query_string is None:
                    llm_ranking_scores = self.llm_discord_message_evaluation(response)
                # currently the task does not contain query string, below is a placeholer/no-op
                else:
                    llm_ranking_scores = self.llm_keyword_ranking_evaluation(
                        query.query_string, response
                    )

                if query.sort_by == SortType.RECENCY:
                    # mean quality score
                    rank_scores[i] = sum(llm_ranking_scores) / size
                else:
                    # nDCG quality score
                    rank_scores[i] = ndcg_score(llm_ranking_scores, size)

                bt.logging.info(f"Quality score: {rank_scores[i]}")
            except Exception as e:
                bt.logging.error(f"Error while processing {i}-th response: {e}")
                bt.logging.debug(print_exception(type(e), e, e.__traceback__))
                zero_score_mask[i] = 0

        # age contribution to encourage recency
        # avg_age_scores = 1 - (avg_ages / (max_avg_age + 1))
        avg_age_scores = 1 - (avg_ages - min_avg_age) / (
            max_avg_age - min_avg_age + 1e-5
        )

        # recency counts up to 40%
        scores = avg_age_scores * 0.4 + rank_scores * 0.6
        scores = scores * uniqueness_scores

        # relative scores in a batch
        # scores = scores / (scores.max() + 1e-5)

        # return raw scores for tracking
        return scores * zero_score_mask

    def check_document(self, doc, groundtruth_doc):
        """
        This function checks the integrity of the document.
        """
        try:
            check_fields = ["text", "username"]
            for field in check_fields:
                if doc[field] != groundtruth_doc[field]:
                    bt.logging.info(
                        f"Document {field} {doc[field]} does not match ground truth {groundtruth_doc[field]}"
                    )
                    return False
            if datetime.fromisoformat(
                doc["created_at"].rstrip("Z")
            ) != datetime.fromisoformat(groundtruth_doc["created_at"].rstrip("Z")):
                bt.logging.info(
                    f"Document created_at {doc['created_at']} does not match ground truth {groundtruth_doc['created_at']}"
                )
                return False
            return True
        except Exception as e:
            bt.logging.error(f"Error while checking integrity of document: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))
            return False

    def llm_keyword_ranking_evaluation(self, query_string, docs, retries=3):
        """
        This function evaluates the ranking of the documents using the LLM.
        """
        try:
            newline = "\n"
            prompt_docs = "\n\n".join(
                [
                    f"ItemId: {i}\nTime: {doc['created_at'].split('T')[0]}\nText: {doc['text'][:1000].replace(newline, '  ')}"
                    for i, doc in enumerate(docs)
                ]
            )
            bt.logging.debug(
                f"Querying LLM of {query_string} with docs:\n" + prompt_docs
            )
            output = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": """Below are the metrics and definitions: 
outdated: Time-sensitive information that is no longer current or relevant.
off topic: Superficial content lacking depth and comprehensive insights.
somewhat relevant: Offers partial insight but lacks depth and comprehensive coverage.
relevant: Comprehensive, insightful content suitable for informed decision-making.""",
                    },
                    {
                        "role": "system",
                        "content": f"Current Time: {datetime.now().isoformat().split('T')[0]}",
                    },
                    {
                        "role": "system",
                        "content": """
Example 1:
ItemId: 0
Time: "2023-11-25" 
Text: Also driving the charm is Blast's unique design: Depositors start earning yields on the transferred ether alongside BLAST points. "Blast natively participates in ETH staking, and the staking yield is passed back to the L2's users and dapps," the team said in a post Tuesday. 'We've redesigned the L2 from the ground up so that if you have 1 ETH in your wallet on Blast, over time, it grows to 1.04, 1.08, 1.12 ETH automatically."
As such, Blast is invite-only as of Tuesday, requiring a code from invited users to gain access. Besides, the BLAST points can be redeemed starting in May.Blast raised over $20 million in a round led by Paradigm and Standard Crypto and is headed by pseudonymous figurehead @PacmanBlur, one of the co-founders of NFT marketplace Blur.
@PacmanBlur said in a separate post that Blast was an extension of the Blur ecosystem, letting Blur users earn yields on idle assets while improving the technical aspects required to offer sophisticated NFT products to users.
BLUR prices rose 12%% in the past 24 hours following the release of Blast

Query: Blast

Output:
item_id: 0
choice: relevant
reason: It is relevant as it deep dives into the Blast project.

Example 2:
ItemId: 1
Time: "2023-11-15"
Text: To celebrate, we've teamed up with artist @debbietea8 to release a commemorative piece of art on @arbitrum! 😍
Now available for free, exclusively in app! 🥳

Query: Arbitrum

Output:
item_id: 1
choice: off topic
reason: It is not directly related to Arbitrum as it just uses the arbitrum app.
""",
                    },
                    {
                        "role": "user",
                        "content": f"You will be given a list of documents with id and you have to rate them based on the relevance to the query. The documents are as follows:\n"
                        + prompt_docs,
                    },
                    {
                        "role": "user",
                        "content": f"Use the metric choices [outdated, off topic, somewhat relevant, relevant] to evaluate the text toward '{query_string}'?",
                    },
                    {
                        "role": "user",
                        "content": "Must answer in JSON format of a list of choices with item ids for all the given items: "
                        + "{'results': [{'item_id': the item id of choice, e.g. 0, 'reason': a very short explanation of your choice, 'choice':The choice of answer. }, {'item_id': 1, 'reason': explanation, 'choice': answer } , ... ] } ",
                    },
                ],
                temperature=0,
            )
            bt.logging.debug(f"LLM response: {output.choices[0].message.content}")
            bt.logging.debug(
                f"LLM usage: {output.usage}, finish reason: {output.choices[0].finish_reason}"
            )
        except Exception as e:
            bt.logging.error(f"Error while querying LLM: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))
            return 0

        try:
            result = json.loads(output.choices[0].message.content)
            bt.logging.debug(f"LLM result: {result}")
            ranking = parse_llm_result(result)
            bt.logging.info(f"LLM ranking: {ranking}")
            if len(ranking) != len(docs):
                raise ValueError(
                    f"Length of ranking {len(ranking)} does not match input docs length {len(docs)}"
                )
            # ranking_score = ndcg_score(ranking, size)
            # bt.logging.info(f"LLM Ranking score: {ranking_score}")
            # return ranking_score
            return ranking
        except Exception as e:
            bt.logging.error(f"Error while parsing LLM result: {e}, retrying...")
            if retries > 0:
                return self.llm_keyword_ranking_evaluation(
                    query_string, docs, retries - 1
                )
            else:
                bt.logging.error(
                    f"Failed to parse LLM result after retrying. Returning [0]."
                )
            return [0]

    def llm_author_index_data_evaluation(self, docs, retries=3):
        if docs is None or len(docs) == 0:
            return [0]
        try:
            newline = "\n"
            prompt_docs = "\n\n".join(
                [
                    f"ItemId: {i}\nTime: {doc['created_at'].split('T')[0]}\nText: {doc['text'][:1000].replace(newline, '  ')}"
                    for i, doc in enumerate(docs)
                ]
            )

            bt.logging.debug(
                f"Querying LLM of author index data with docs:\n" + prompt_docs
            )
            output = self.llm_client.chat.completions.create(
                model="gpt-4-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": """Below are the metrics and definitions: 
outdated: Time-sensitive information that is no longer current or relevant.
insightless: Superficial content lacking depth and comprehensive insights.
somewhat insightful: Offers partial insight but lacks depth and comprehensive coverage.
Insightful: Comprehensive, insightful content suitable for informed decision-making.""",
                    },
                    {
                        "role": "system",
                        "content": f"Current Time: {datetime.now().isoformat().split('T')[0]}",
                    },
                    {
                        "role": "system",
                        "content": """
Example 1:
ItemId: 0
Time: "2023-11-25" 
Text: Also driving the charm is Blast's unique design: Depositors start earning yields on the transferred ether alongside BLAST points. "Blast natively participates in ETH staking, and the staking yield is passed back to the L2's users and dapps," the team said in a post Tuesday. 'We've redesigned the L2 from the ground up so that if you have 1 ETH in your wallet on Blast, over time, it grows to 1.04, 1.08, 1.12 ETH automatically."
As such, Blast is invite-only as of Tuesday, requiring a code from invited users to gain access. Besides, the BLAST points can be redeemed starting in May.Blast raised over $20 million in a round led by Paradigm and Standard Crypto and is headed by pseudonymous figurehead @PacmanBlur, one of the co-founders of NFT marketplace Blur.
@PacmanBlur said in a separate post that Blast was an extension of the Blur ecosystem, letting Blur users earn yields on idle assets while improving the technical aspects required to offer sophisticated NFT products to users.
BLUR prices rose 12%% in the past 24 hours following the release of Blast


Output:
item_id: 0
choice: insightful
reason: It is contains insightful information about the Blast project.

Example 2:
ItemId: 1
Time: "2024-03-19"
Text: $SLERF to the moon!
$BOME $SOL $MUMU $BONK $BOPE $WIF $NAP 🥳

Output:
item_id: 1
choice: insightless
reason: It does not contain much meaningful information, just sentiment about some tickers.
""",
                    },
                    {
                        "role": "user",
                        "content": f"You will be given a list of documents with id and you have to rate them based on its information and insightfulness. The documents are as follows:\n"
                        + prompt_docs,
                    },
                    {
                        "role": "user",
                        "content": f"Use the metric choices [outdated, insightless, somewhat insightful, insightful] to evaluate the text.",
                    },
                    {
                        "role": "user",
                        "content": "Must answer in JSON format of a list of choices with item ids for all the given items: "
                        + "{'results': [{'item_id': the item id of choice, e.g. 0, 'reason': a very short explanation of your choice, 'choice':The choice of answer. }, {'item_id': 1, 'reason': explanation, 'choice': answer } , ... ] } ",
                    },
                ],
                temperature=0,
            )
            bt.logging.debug(f"LLM response: {output.choices[0].message.content}")
            bt.logging.debug(
                f"LLM usage: {output.usage}, finish reason: {output.choices[0].finish_reason}"
            )
        except Exception as e:
            bt.logging.error(f"Error while querying LLM: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))
            return 0

        try:
            result = json.loads(output.choices[0].message.content)
            # bt.logging.debug(f"LLM result: {result}")
            ranking = parse_llm_result_for_author_index(result)
            bt.logging.info(f"LLM ranking: {ranking}")
            if len(ranking) != len(docs):
                raise ValueError(
                    f"Length of ranking {len(ranking)} does not match input docs length {len(docs)}"
                )
            return ranking
        except Exception as e:
            bt.logging.error(f"Error while parsing LLM result: {e}, retrying...")
            if retries > 0:
                return self.llm_author_index_data_evaluation(docs, retries - 1)
            else:
                bt.logging.error(
                    f"Failed to parse LLM result after retrying. Returning [0]."
                )
            return [0]

    def llm_semantic_search_evaluation(self, query_string, docs, retries=3):
        if docs is None or len(docs) == 0:
            return [0]
        try:
            newline = "\n"
            prompt_docs = "\n\n".join(
                [
                    f"ItemId: {i}\nTalk Title: {doc['episode_title']}\nSpeaker: {doc['speaker']}\nText: {doc['text'][:2000].replace(newline, '  ')}"
                    for i, doc in enumerate(docs)
                ]
            )

            bt.logging.debug(
                f"Querying LLM of semantic search with docs:\n" + prompt_docs
            )
            output = self.llm_client.chat.completions.create(
                model="gpt-4-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": """Below are the metrics and definitions:
off topic: Superficial or unrelevant content that can not answer the given question.
somewhat relevant: Offers partial insight to partially answer the given question.
relevant: Comprehensive, insightful content suitable for answering the given question.""",
                    },
                    {
                        "role": "user",
                        "content": f"You will be given a list of documents with id and you have to rate them based on its information and relevance to the question. The documents are as follows:\n"
                        + prompt_docs,
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Use the metric choices [off topic, somewhat relevant, relevant] to evaluate whether the text can answer the given question:\n"
                            f"{query_string}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Must answer in JSON format of a list of choices with item ids for all the given items: "
                        + "{'results': [{'item_id': the item id of the text, e.g. 0, 'reason': a very short explanation of your choice, 'choice':The choice of answer. }, {'item_id': 1, 'reason': explanation, 'choice': answer } , ... ] } ",
                    },
                ],
                temperature=0,
            )
            bt.logging.debug(f"LLM response: {output.choices[0].message.content}")
            bt.logging.debug(
                f"LLM usage: {output.usage}, finish reason: {output.choices[0].finish_reason}"
            )
        except Exception as e:
            bt.logging.error(f"Error while querying LLM: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))
            return 0

        try:
            result = json.loads(output.choices[0].message.content)
            # bt.logging.debug(f"LLM result: {result}")
            ranking = parse_llm_result(result)
            bt.logging.info(f"LLM ranking: {ranking}")
            if len(ranking) != len(docs):
                raise ValueError(
                    f"Length of ranking {len(ranking)} does not match input docs length {len(docs)}"
                )
            return ranking
        except Exception as e:
            bt.logging.error(f"Error while parsing LLM result: {e}, retrying...")
            if retries > 0:
                return self.llm_semantic_search_evaluation(
                    query_string, docs, retries - 1
                )
            else:
                bt.logging.error(
                    f"Failed to parse LLM result after retrying. Returning [0]."
                )
            return [0]

    def llm_discord_message_evaluation(self, docs, retries=3):
        if docs is None or len(docs) == 0:
            return [0]
        try:
            newline = "\n"
            prompt_docs = "\n\n".join(
                [
                    f"ItemId: {i}\nTimestamp:{doc['created_at']}\nText: {doc['text'][:1000].replace(newline, '  ')}"
                    for i, doc in enumerate(docs)
                ]
            )

            bt.logging.debug(
                f"Querying LLM of discord message data with docs:\n" + prompt_docs
            )
            output = self.llm_client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": """You will be given a list of messages from a discord channel and you have to rate them based on its information and meaningfulness.
A message is meaningful if it is self-contained with valuable information, for example, it can be an announcement, a news, an instruction about code, or a opinion towards a subnet.
A message is meaningless if it contains no valuable information or is a piece of conversation taken out of contexts.
For example, meaningless messages can be a question without context, a response to an unknown question, log without explanation, code without context, very short messages, or an announcement from six months ago.
Note even if a message itself is informative, it should still be categorized into meaningless if it is part of a conversation or lacks context to understand the information based on the message itself.
""",
                    },
                    {
                        "role": "system",
                        "content": f"Current Time: {datetime.now().isoformat().split('T')[0]}",
                    },
                    {
                        "role": "user",
                        "content": f"You will be given a list of documents with id and timestamp, please rate them based on its information and meaningfulness. The documents are as follows:\n"
                        + prompt_docs,
                    },
                    {
                        "role": "user",
                        "content": f"Use the metric choices [meaningful, meaningless] to evaluate the text.",
                    },
                    {
                        "role": "user",
                        "content": "Must answer in JSON format of a list of choices with item ids for all the given items: "
                        + "{'results': [{'item_id': the item id of choice, e.g. 0, 'reason': a very short explanation of your choice, 'choice':The choice of answer. }, {'item_id': 1, 'reason': explanation, 'choice': answer } , ... ] } ",
                    },
                ],
                temperature=0,
            )
            bt.logging.debug(f"LLM response: {output.choices[0].message.content}")
            bt.logging.debug(
                f"LLM usage: {output.usage}, finish reason: {output.choices[0].finish_reason}"
            )
        except Exception as e:
            bt.logging.error(f"Error while querying LLM: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))
            return 0

        try:
            result = json.loads(output.choices[0].message.content)
            # bt.logging.debug(f"LLM result: {result}")
            ranking = parse_llm_result_for_discord_msg(result)
            bt.logging.info(f"LLM ranking: {ranking}")
            if len(ranking) != len(docs):
                raise ValueError(
                    f"Length of ranking {len(ranking)} does not match input docs length {len(docs)}"
                )
            return ranking
        except Exception as e:
            bt.logging.error(f"Error while parsing LLM result: {e}, retrying...")
            if retries > 0:
                return self.llm_discord_message_evaluation(docs, retries - 1)
            else:
                bt.logging.error(
                    f"Failed to parse LLM result after retrying. Returning [0]."
                )
            return [0]


def get_datetime(time_str: str):
    return datetime.fromisoformat(time_str.rstrip("Z"))
