import requests
from datasets import load_dataset
from cachetools import cached, LRUCache, TTLCache


def fetch_datasets_config(branch="main"):
    # dynamically fetch datasets config from github
    url = f"https://raw.githubusercontent.com/OpenKaito/openkaito/refs/heads/{branch}/datasets_config.json"
    response = requests.get(url)
    return response.json()


def load_datasets_from_config(configs):
    datasets = {}
    for config in configs:
        print("loading dataset: ", config)
        dataset = load_dataset(**config)
        datasets[f"{config.get('path')}|{config.get('name')}|{config.get('split')}"] = (
            dataset
        )
    return datasets


# Cache the datasets for 1 hour, then fetch config from github again
@cached(cache=TTLCache(maxsize=1, ttl=3600))
def cached_datasets_from_config(branch="main"):
    print("fetching datasets config from github")
    datasets_config = fetch_datasets_config(branch=branch)
    print("datasets config fetched", datasets_config)
    datasets = {}
    for key, value in datasets_config.items():
        datasets[key] = load_datasets_from_config(value)
    return datasets


if __name__ == "__main__":
    import random
    import time

    # datasets_config = fetch_datasets_config(branch="dataset_rotation")
    # print(datasets_config)

    # text_embedding_datasets = load_datasets_from_config(
    #     datasets_config["text_embedding_datasets"]
    # )
    # print(text_embedding_datasets)

    # chosen_dataset = random.choice(list(text_embedding_datasets.items()))
    # print(chosen_dataset[0])
    # print(chosen_dataset[1])

    cached_datasets_from_config(branch="dataset_rotation")
    print(cached_datasets_from_config(branch="dataset_rotation"))
    print("sleeping for 1 second")
    time.sleep(1)
    print(cached_datasets_from_config(branch="dataset_rotation"))

    print("sleeping for 3 seconds")
    time.sleep(3)
    print(cached_datasets_from_config(branch="dataset_rotation"))
