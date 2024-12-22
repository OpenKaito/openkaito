import requests
from cachetools import TTLCache, cached

from datasets import load_dataset


def fetch_datasets_config(branch="main"):
    # dynamically fetch datasets config from github
    url = f"https://raw.githubusercontent.com/OpenKaito/openkaito/refs/heads/{branch}/datasets_config.json"
    response = requests.get(url)
    return response.json()


def load_datasets_from_config(configs):
    datasets = {}
    for config in configs:
        print("loading dataset: ", config)
        text_field_name = config.pop("text_field_name")
        dataset = load_dataset(**config)
        datasets[f"{config.get('path')}|{config.get('name')}|{config.get('split')}"] = {
            "dataset": dataset,
            "text_field_name": text_field_name,
        }
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
    import json
    import os
    import random
    import time

    from dotenv import load_dotenv
    from openai import OpenAI

    from openkaito.tasks import generate_relevant_pairs

    load_dotenv()
    # datasets_config = fetch_datasets_config(branch="dataset_rotation")
    # datasets_config = json.load(open("datasets_config.json"))
    # print(datasets_config)

    # text_embedding_datasets = load_datasets_from_config(
    #     datasets_config["text_embedding_datasets"]
    # )
    # print(text_embedding_datasets)

    llm_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _test_text_embedding_dataset():
        datasets_config = json.load(open("datasets_config.json"))
        print(datasets_config)
        datasets = load_datasets_from_config(datasets_config["text_embedding_datasets"])
        print(datasets)
        # dataset_pairs = list(datasets.items())
        # selected_dataset = dataset_pairs[idx]
        # print(selected_dataset)
        # dataset = selected_dataset[1]["dataset"]
        # text_field_name = selected_dataset[1]["text_field_name"]
        # print(dataset.shuffle().take(1)[0])
        # for x in dataset.shuffle().take(1):
        #     print(x)
        for selected_dataset in datasets.items():
            print("using dataset", selected_dataset[0])
            pairs = generate_relevant_pairs(
                dataset=selected_dataset[1]["dataset"],
                num_articles=2,
                num_pairs_per_article=2,
                llm_client=llm_client,
                text_field_name=selected_dataset[1]["text_field_name"],
            )
            print(f"Generated {len(pairs)} pairs")
            print(pairs)

    def _test_ttl_datasets_config():
        cached_datasets_from_config(branch="dataset_rotation")
        print(cached_datasets_from_config(branch="dataset_rotation"))
        print("sleeping for 1 second")
        time.sleep(1)
        print(cached_datasets_from_config(branch="dataset_rotation"))

        print("sleeping for 3 seconds")
        time.sleep(3)
        print(cached_datasets_from_config(branch="dataset_rotation"))

    print("testing text embedding dataset")

    _test_text_embedding_dataset()
