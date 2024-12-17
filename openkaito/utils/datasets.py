import requests


def fetch_datasets_config(branch="main"):
    # dynamically fetch datasets config from github
    url = f"https://raw.githubusercontent.com/OpenKaito/openkaito/refs/heads/{branch}/datasets_config.json"
    response = requests.get(url)
    return response.json()


if __name__ == "__main__":
    fetch_datasets_config(branch="")
