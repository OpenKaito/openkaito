import os
import random
from datetime import datetime, timedelta

import bittensor as bt

from .protocol import SortType, StructuredSearchSynapse
from .utils.version import get_version


def random_query(input_file="queries.txt"):
    if not os.path.exists(input_file):
        bt.logging.error(f"Queries file not found at location: {input_file}")
        exit(1)
    lines = open(input_file).read().strip().splitlines()
    return random.choice(lines)

# The twitter usernames list is from a truncated snapshot of friendtech ( https://dune.com/cryptokoryo/friendtech )
def random_twitter_username(input_file="twitter_usernames.txt", num_authors: int = 2):
    if not os.path.exists(input_file):
        bt.logging.error(f"Twitter usernames file not found at location: {input_file}")
        exit(1)
    lines = open(input_file).read().strip().splitlines()
    return random.sample(lines, num_authors)


def random_datetime(start: datetime, end: datetime):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)


def random_past_datetime(start_days_ago: int = 365, end_days_ago: int = 10):

    return random_datetime(
        datetime.now() - timedelta(days=start_days_ago),
        datetime.now() - timedelta(days=end_days_ago),
    )


def generate_author_index_task(
    size: int = 5,
    num_authors: int = 2,
):
    author_usernames = random_twitter_username(num_authors=num_authors)
    return StructuredSearchSynapse(
        size=size,
        author_usernames=author_usernames,
        version=get_version(),
    )


def generate_structured_search_task(
    query_string: str = None,
    size: int = 5,
    sort_type: SortType = None,
    earlier_than: datetime = None,
    later_than: datetime = None,
    author_usernames: list = None,
) -> StructuredSearchSynapse:
    """
    Generates a structured search task for the validator to send to the miner.
    """
    random_number = random.random()

    # Randomly generate the query_string if not provided.
    if query_string is None:
        # 50% chance to generate a simple random query
        if random_number < 0.5:
            query_string = random_query()
        # 30% chance to generate a random query with OR
        elif random_number < 0.8:
            query_string = f"{random_query()} OR {random_query()}"
        # 20% chance to generate a random query with AND
        else:
            query_string = f"{random_query()} AND {random_query()}"

    # Randomly select the sort type if not provided.
    if sort_type is None:
        sort_type = SortType.RELEVANCE if random.random() < 0.5 else SortType.RECENCY

    # Randomly select the earlier_than and later_than if not provided.
    if later_than is None:
        # 0.5 ratio to set the later_than or not
        if random.random() < 0.5:
            later_than = random_past_datetime()
        else:
            later_than = None

    # Note: do not set the earlier_than by default if it is not provided.

    return StructuredSearchSynapse(
        query_string=query_string,
        size=size,
        sort_type=sort_type,
        earlier_than_timestamp=(earlier_than.timestamp() if earlier_than else None),
        later_than_timestamp=(later_than.timestamp() if later_than else None),
        author_usernames=author_usernames,
        version=get_version(),
    )


if __name__ == "__main__":
    task = generate_structured_search_task("BTC")
    print(task)
    print(task.name)
