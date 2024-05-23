import math


def ndcg_score(ranking, size):
    """
    This function calculates the NDCG score for the documents.
    """
    # ideal_ranking = sorted(ranking, reverse=True)

    # use all 1s as ideal ranking to take both RELEVANCE and RANKING into consideration

    ideal_ranking = [1] * size
    dcg = sum([r / math.log2(i + 1 + 1) for i, r in enumerate(ranking)])
    idcg = sum([r / math.log2(i + 1 + 1) for i, r in enumerate(ideal_ranking)])
    return dcg / idcg


def tweet_url_to_id(url):
    """
    This function converts a tweet URL to a tweet ID.
    """
    return url.split("?")[0].split("/")[-1]


def parse_llm_result(result):
    """
    This function parses the result from the LLM.
    """
    choice_mapping = {
        "outdated": 0,
        "off topic": 0,
        "somewhat relevant": 0.5,
        "relevant": 1,
    }
    return [choice_mapping[doc["choice"]] for doc in result["results"]]


def parse_llm_result_for_author_index(result):
    """
    This function parses the result from the LLM for author index.
    """
    choice_mapping = {
        "outdated": 0,
        "insightless": 0,
        "somewhat insightful": 0.5,
        "insightful": 1,
    }
    return [choice_mapping[doc["choice"]] for doc in result["results"]]


def parse_llm_result_for_discord_msg(result):
    choice_mapping = {
        "meaningless": 0,
        "meaningful": 1,
    }
    return [choice_mapping[doc["choice"]] for doc in result["results"]]
