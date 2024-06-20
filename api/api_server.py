import json
import os
import random
from typing import Annotated, List

import bittensor as bt
from dotenv import load_dotenv
from fastapi import FastAPI, Header
from loguru import logger

from openkaito.protocol import (
    DiscordSearchSynapse,
    SemanticSearchSynapse,
    StructuredSearchSynapse,
)

load_dotenv()


app = FastAPI()

api_keys = set(os.environ.get("OPENKAITO_VALIDATOR_API_KEYS", "").split(","))
logger.info(f"API keys: {api_keys}")

# FILL IN THE FOLLOWING
validator_wallet_name = (
    "validator"  # The wallet name of the validator, or os.environ.get("WALLET_NAME")
)
validator_hotkey_name = (
    "default"  # The hotkey name of the validator, or os.environ.get("HOTKEY_NAME")
)
subtensor_network = "main"  # The subtensor network, "finney", "test", "local", or os.environ.get("SUBTENSOR_NETWORK"),
netuid = 88  # 88 for testnet and 5 for mainnet


subtensor = bt.subtensor(network=subtensor_network)
wallet = bt.wallet(name=validator_wallet_name, hotkey=validator_hotkey_name)
metagraph = subtensor.metagraph(netuid=netuid)
metagraph.sync(subtensor=subtensor)
dendrite = bt.dendrite(wallet=wallet)


available_synapses = {
    "StructuredSearchSynapse": StructuredSearchSynapse,
    "SemanticSearchSynapse": SemanticSearchSynapse,
    "DiscordSearchSynapse": DiscordSearchSynapse,
}


## Authentication
## Request Header: x-api-key: <api_key>
## validator needs to specify a list of valid api keys in env vars, e.g., in `.env` file.
def validate_api_key(api_key: str):
    return api_key in api_keys


@app.get("/")
async def read_root(x_api_key: Annotated[str | None, Header()] = None):
    logger.info(f"API key: {x_api_key}")
    if not validate_api_key(x_api_key):
        return {"message": "Invalid API Key in header x-api-key."}
    return {"message": "This is bittensor OpenKaito API server."}


# accept a json string of a synapse and send it to the network
@app.get("/send_synapse")
async def send_synapse(
    synapse_json: str,  # can be one of StructuredSearchSynapse, SemanticSearchSynapse, DiscordSearchSynapse, using `model_dump_json()` method
    miner_uids: str = "random",  # comma separated list of uids, e.g, "0,1,2", or "random", or "top", default is "random"
    sampling_number: int = 3,  # used only when `miner_uids` is "top" or "random", number of miners to sample from the list of uids, default is 3. If `miner_uids` is a list of uids, this parameter is ignored.
    x_api_key: Annotated[str | None, Header()] = None,
):
    if not validate_api_key(x_api_key):
        return {"message": "Invalid API Key in header x-api-key."}

    try:
        synapse_type = json.loads(synapse_json).get("name")

        if synapse_type not in available_synapses:
            return {"message": f"Synapse type {synapse_type} is not available."}

        synapse = available_synapses[synapse_type].model_validate_json(synapse_json)
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"message": f"Failed to validate synapse_json: {e}"}

    logger.info(f"received synapse: {synapse.model_dump_json()}")

    if miner_uids == "random":
        miner_uids = random.sample(metagraph.uids.tolist(), sampling_number)
    elif miner_uids == "top":
        miner_uids = topk_incentive_uids(metagraph, sampling_number)
    else:
        try:
            miner_uids = list(map(int, miner_uids.split(",")))
        except Exception as e:
            logger.error(f"Error: {e}")
            return {
                "message": f"Failed to parse miner_uids: {e}, must be comma separated list of uids, e.g, `0,1,2`, or `random`, or `top`"
            }

    responses = await dendrite(
        axons=[metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        deserialize=True,
        timeout=synapse.timeout,
    )

    return {
        "responses": {uid: response for uid, response in zip(miner_uids, responses)},
        "synapse": synapse.model_dump_json(),
    }


# the metagraph can be requested to sync with the network via a POST request to `/sync_metagraph` endpoint.
@app.post("/sync_metagraph")
async def sync_metagraph(x_api_key: Annotated[str | None, Header()] = None):
    if not validate_api_key(x_api_key):
        return {"message": "Invalid API Key in header x-api-key."}
    try:
        metagraph.sync(subtensor=subtensor)
        return {"message": "Metagraph sync completed."}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"message": f"Failed to sync metagraph: {e}"}


@app.get("/topk_incentive_uids")
async def get_topk_incentive_uids(
    x_api_key: Annotated[str | None, Header()] = None,
    k: int = 10,
):
    if not validate_api_key(x_api_key):
        return {"message": "Invalid API Key in header x-api-key."}
    return topk_incentive_uids(metagraph, k)


def topk_incentive_uids(metagraph, k: int) -> List[int]:
    miners_uids = metagraph.uids.tolist()

    # Builds a dictionary of uids and their corresponding incentives
    all_miners_incentives = {
        "miners_uids": miners_uids,
        "incentives": list(map(lambda uid: metagraph.I[uid], miners_uids)),
    }

    # Zip the uids and their corresponding incentives into a list of tuples
    uid_incentive_pairs = list(
        zip(all_miners_incentives["miners_uids"], all_miners_incentives["incentives"])
    )

    # Sort the list of tuples by the incentive value in descending order
    uid_incentive_pairs_sorted = sorted(
        uid_incentive_pairs, key=lambda x: x[1], reverse=True
    )

    logger.info(f"Top {k} uids with highest incentives: {uid_incentive_pairs_sorted}")
    top_k_uids = [uid for uid, incentive in uid_incentive_pairs_sorted[:k]]

    return top_k_uids


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, loop="asyncio", host="0.0.0.0", port=8900)
