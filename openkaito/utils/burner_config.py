import requests
from cachetools import TTLCache, cached

@cached(cache=TTLCache(maxsize=1, ttl=3600))
def fetch_config(branch="main"):

    url = f"https://raw.githubusercontent.com/OpenKaito/openkaito/refs/heads/{branch}/burner_config.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # Return default values if config can't be fetched
        return {
            #"burner_miner_address": None, # TODO: add the burner miner address here
            "burner_miner_address": "5FHsM8ywifUgtSXbDQaSD9zzDYq76RqAvf9PnaSVg5q3Lc8G",
            "burner_reward_percentage": 0.95
        }

if __name__ == "__main__":
    # Test the function
    config = fetch_config()
    print("burner miner address:", config["burner_miner_address"])
    print("burner reward percentage:", config["burner_reward_percentage"]) 