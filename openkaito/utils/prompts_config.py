import requests
import random
from cachetools import TTLCache, cached


def fetch_prompts_config(branch="main"):
    # dynamically fetch prompts config from github
    url = f"https://raw.githubusercontent.com/OpenKaito/openkaito/refs/heads/{branch}/prompts_config.json"
    response = requests.get(url)
    return response.json()


# Cache the prompts for 1 hour, then fetch config from github again
@cached(cache=TTLCache(maxsize=1, ttl=3600))
def cached_prompts_from_config(branch="main"):
    print("fetching prompts config from github")
    prompts_config = fetch_prompts_config(branch=branch)
    print("prompts config fetched", prompts_config)
    return prompts_config


def random_dynamic_prompt(prompt_type):
    prompts_config = cached_prompts_from_config(branch="main")
    return random.choice(prompts_config[prompt_type])


if __name__ == "__main__":
    import random

    prompts_config = cached_prompts_from_config(branch="main")

    print(prompts_config)

    base_prompt = random.choice(prompts_config["text_embedding_prompts"])
    print(base_prompt)

    print("--------")
    for i in range(10):
        print(random_dynamic_prompt("text_embedding_prompts"))
