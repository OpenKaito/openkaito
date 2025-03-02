import requests
from cachetools import TTLCache, cached

@cached(cache=TTLCache(maxsize=1, ttl=3600))
def fetch_config(branch="main"):

    url = f"https://raw.githubusercontent.com/OpenKaito/openkaito/refs/heads/{branch}/burned_config.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # Return default values if config can't be fetched
        return {
            "burned_miner_address": None, # TODO: add the burned miner address here
            "burned_reward_percentage": 0.9
        }

if __name__ == "__main__":
    # Test the function
    config = fetch_config()
    print("Burned miner address:", config["burned_miner_address"])
    print("Burned reward percentage:", config["burned_reward_percentage"]) 