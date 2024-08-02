import json

import requests

from openkaito.protocol import (
    DiscordSearchSynapse,
    SemanticSearchSynapse,
    SortType,
    StructuredSearchSynapse,
)
from openkaito.utils.version import get_version

# change to your api endpoint and x-api-key
OPENKAITO_API_ENDPOINT = "http://localhost:8900"
X_API_KEY = "key1"


# r = requests.post(
#     f"{OPENKAITO_API_ENDPOINT}/sync_metagraph",
#     headers={"x-api-key": X_API_KEY},
# )
# print(r.json())


# search_query = SemanticSearchSynapse(
#     query_string="what is the future of Ethereum?",
#     size=5,
#     version=get_version(),
# )

# search_query = StructuredSearchSynapse(
#     query_string="TAO",
#     size=5,
#     sort_by=SortType.RELEVANCE,
#     version=get_version(),
# )

search_query = DiscordSearchSynapse(
    query_string="What is recent anouncement in project Open Kaito?",
    size=5,
    version=get_version(),
)

req = {
    "synapse_json": search_query.model_dump_json(),
    "miner_uids": "0,1",
    # "miner_uids": "top",
    # "miner_uids": "random",
}

r = requests.get(
    f"{OPENKAITO_API_ENDPOINT}/send_synapse",
    params=req,
    headers={"x-api-key": X_API_KEY},
)
print(r)
print(json.dumps(r.json(), indent=2))
