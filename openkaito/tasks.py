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
    lines = open(input_file).read().splitlines()
    return random.choice(lines)


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


def generate_structured_search_task(
    query_string: str = None,
    size: int = 5,
    sort_type: SortType = None,
    created_earlier_than: datetime = None,
    created_later_than: datetime = None,
) -> StructuredSearchSynapse:
    """
    Generates a structured search task for the validator to send to the miner.
    """
    query_string = random_query() if query_string is None else query_string

    # Randomly select the sort type if not provided.
    if sort_type is None:
        sort_type = SortType.RELEVANCE if random.random() < 0.5 else SortType.RECENCY

    # Randomly select the created_earlier_than and created_later_than if not provided.
    if created_later_than is None:
        # 0.5 ratio to set the created_later_than or not
        if random.random() < 0.5:
            created_later_than = random_past_datetime()
        else:
            created_later_than = None

    # do not set the created_earlier_than by default if it is not provided.

    return StructuredSearchSynapse(
        query_string=query_string,
        size=size,
        sort_type=sort_type,
        created_earlier_than=created_earlier_than,
        created_later_than=created_later_than,
        version=get_version(),
    )


if __name__ == "__main__":
    task = generate_structured_search_task("BTC")
    print(task)
    print(task.name)
