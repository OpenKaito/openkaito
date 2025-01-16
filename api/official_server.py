import json
import os
import random
import requests
from typing import Annotated, List, Optional

import torch
import bittensor as bt
from dotenv import load_dotenv
from fastapi import FastAPI, Header
from loguru import logger
from pydantic import BaseModel

from openkaito.protocol import (
    OfficialSynapse,
)
from openkaito.utils.version import get_version

# from openkaito.utils.uids import get_validator_uids, get_miners_uids

load_dotenv()

try:
    logger.info("Loading environment variables")
    logger.info("-----------------------------")
    validator_wallet_name = os.environ["VALIDATOR_API_WALLET_NAME"]
    validator_hotkey_name = os.environ["VALIDATOR_API_HOTKEY_NAME"]
    subtensor_network = os.environ["SUBTENSOR_NETWORK"]
    netuid = int(os.environ["NETUID"])
    api_keys = set(
        [key.strip() for key in os.environ["OPENKAITO_VALIDATOR_API_KEYS"].split(",")]
    )
    logger.info(f"Authorized API keys: {api_keys}")

    logger.info("Loading bittensor components")
    logger.info("-----------------------------")
    subtensor = bt.subtensor(network=subtensor_network)
    wallet = bt.wallet(name=validator_wallet_name, hotkey=validator_hotkey_name)
    metagraph = subtensor.metagraph(netuid=netuid)
    metagraph.sync(subtensor=subtensor)
    dendrite = bt.dendrite(wallet=wallet)


except KeyError as e:
    logger.error(f"Error: {e}")
    raise Exception("Please set up `.env`")

available_synapses = {
    "OfficialSynapse": OfficialSynapse,
}

app = FastAPI()


def validate_api_key(api_key: str):
    return api_key in api_keys


@app.get("/")
async def read_root(x_api_key: Annotated[str | None, Header()] = None):
    logger.info(f"API key: {x_api_key}")
    if not validate_api_key(x_api_key):
        return {"message": "Invalid API Key in header x-api-key."}
    return {"message": "This is bittensor OpenKaito API server."}


def get_topk_uids(uids: List[int], k: int) -> List[int]:
    uid_incentives = [(uid, metagraph.I[uid]) for uid in uids]
    uid_incentives_sorted = sorted(uid_incentives, key=lambda x: x[1], reverse=True)
    return [uid for uid, _ in uid_incentives_sorted[:k]]


def get_random_uids(uids: List[int], k: int) -> List[int]:
    k = min(k, len(uids))
    return random.sample(uids, k)


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    if not metagraph.axons[uid].is_serving:
        return False
    if metagraph.validator_permit[uid] and metagraph.S[uid] > vpermit_tao_limit:
        return False
    return True


@app.get("/get_miners_uids")
async def api_get_miners_uids(
    k: int,
    exclude: Annotated[str | None, None] = None,
    specified_miners: Annotated[str | None, None] = None,
):
    exclude_list = list(map(int, exclude.split(","))) if exclude else []
    specified_miners_list = (
        list(map(int, specified_miners.split(","))) if specified_miners else None
    )

    candidate_uids = []
    avail_uids = []

    specified_miners_set = set(specified_miners_list) if specified_miners_list else None

    for uid in range(metagraph.n.item()):
        uid_is_available = check_uid_availability(metagraph, uid, 4096)
        uid_is_not_excluded = uid not in exclude_list
        uid_is_in_specified = (
            specified_miners_set is None or uid in specified_miners_set
        )

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded and uid_is_in_specified:
                candidate_uids.append(uid)

    k = min(k, len(candidate_uids))
    selected_uids = random.sample(candidate_uids, k) if candidate_uids else []

    return {"available_uids": avail_uids, "selected_uids": selected_uids}


@app.get("/get_validator_uids")
async def api_get_validator_uids(remove_self: bool = True):
    logger.info("Getting validator UIDs from metagraph.")
    logger.info(f"Validator permit: {metagraph.validator_permit}")

    validator_permit = torch.tensor(metagraph.validator_permit, dtype=torch.bool)
    validator_uids = torch.where(validator_permit)[0].long()

    if remove_self:
        self_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        logger.info(f"Self UID: {self_uid}")
        validator_uids = validator_uids[validator_uids != self_uid]

    logger.info(f"Validator UIDs (after removal): {validator_uids.tolist()}")

    return {"validator_uids": validator_uids.tolist()}


def get_miners_uids(
    metagraph, k: int, exclude: List[int] = None, specified_miners: List[int] = None
) -> torch.LongTensor:
    candidate_uids = []
    avail_uids = []

    if specified_miners is not None:
        specified_miners_set = set(specified_miners)  # Use set for faster lookup
    else:
        specified_miners_set = None

    for uid in range(metagraph.n.item()):
        uid_is_available = check_uid_availability(metagraph, uid, 4096)
        uid_is_not_excluded = exclude is None or uid not in exclude
        uid_is_in_specified = (
            specified_miners_set is None or uid in specified_miners_set
        )

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded and uid_is_in_specified:
                candidate_uids.append(uid)

    k = min(k, len(candidate_uids))
    uids = torch.tensor(random.sample(candidate_uids, k))
    return uids


def get_validator_uids(metagraph, remove_self: bool = True) -> torch.LongTensor:
    bt.logging.info(f"Getting validator uids from metagraph.")
    bt.logging.info(f"Validator permit: {metagraph.validator_permit}")
    validator_permit = torch.tensor(metagraph.validator_permit, dtype=torch.bool)
    validator_uids = torch.where(validator_permit)[0].long()
    if remove_self:
        self_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Self UID: {self_uid}")
        validator_uids = validator_uids[validator_uids != self_uid]

    bt.logging.info(f"Validator UIDs (after removal): {validator_uids}")
    return validator_uids


class OfficialSynapseRequest(BaseModel):
    texts: List[str]
    dimensions: int = 512
    normalized: bool = True

    use_specified_validators: Optional[List[int]] = None
    validators_selection: str = "top"
    validators_number: int = 1

    use_specified_miners: List[int] = [142]  # None
    miners_selection: str = "top"  # "random", "top"
    miners_number: int = 1

    timeout: int = 30


@app.post("/official_synapse")
async def official_synapse(
    request: OfficialSynapseRequest,
    x_api_key: Annotated[str | None, Header()] = None,
):
    if not validate_api_key(x_api_key):
        return {"message": "Invalid API Key in header x-api-key."}

    if not request.use_specified_validators:
        if request.validators_selection == "top":
            chosen_validator_uids = get_validator_uids(metagraph, remove_self=True)[
                : request.validators_number
            ]
        elif request.validators_selection == "random":
            all_validator_uids = get_validator_uids(metagraph, remove_self=True)
            chosen_validator_uids = random.sample(
                all_validator_uids.tolist(),
                min(request.validators_number, len(all_validator_uids)),
            )
        else:
            return {"message": "Invalid validators_selection parameter"}
    else:
        chosen_validator_uids = request.use_specified_validators

    if not request.use_specified_miners:
        if request.miners_selection == "top":
            chosen_miner_uids = get_miners_uids(
                metagraph, k=request.miners_number
            ).tolist()
        elif request.miners_selection == "random":
            chosen_miner_uids = get_miners_uids(
                metagraph, k=request.miners_number
            ).tolist()
        else:
            return {"message": "Invalid miners_selection parameter"}
    else:
        chosen_miner_uids = request.use_specified_miners

    synapse = OfficialSynapse(
        texts=request.texts,
        dimensions=request.dimensions,
        normalized=request.normalized,
        miner_uids=chosen_miner_uids,
        version=get_version(),
    )
    synapse.timeout = request.timeout

    axons = []
    for uid in chosen_validator_uids:
        if uid < 0 or uid >= len(metagraph.axons):
            logger.error(f"Invalid uid: {uid}, skipping.")
            continue
        axons.append(metagraph.axons[uid])

    if not axons:
        return {"message": f"No valid axon found for UIDs: {chosen_validator_uids}"}

    logger.info(
        f"[OfficialSynapse Route] Forwarding OfficialSynapse to {len(axons)} axons => UIDs: {chosen_validator_uids}"
    )

    try:
        responses = await dendrite(
            axons=axons,
            synapse=synapse,
            deserialize=True,
            timeout=synapse.timeout,
        )
        logger.info(f"[OfficialSynapse Route] Received {len(responses)} response(s).")
        return {"chosen_uids": chosen_validator_uids, "responses": responses}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"message": f"Failed to send OfficialSynapse: {e}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, loop="asyncio", host="0.0.0.0", port=8988)
