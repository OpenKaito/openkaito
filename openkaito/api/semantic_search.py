import bittensor as bt
from typing import List, Optional, Union, Any, Dict
from openkaito.protocol import SemanticSearchSynapse
from bittensor.subnets import SubnetsAPI
from openkaito.utils.version import get_version


class SemanticSearchAPI(SubnetsAPI):
    def __init__(self, wallet: "bt.wallet"):
        super().__init__(wallet)
        self.netuid = 5
        self.name = "SemanticSearchAPI"

    def prepare_synapse(
        self,
        query_string: str,
        size: int = 5,
    ) -> SemanticSearchSynapse:
        synapse = SemanticSearchSynapse(
            query_string=query_string,
            size=size,
            index_name="eth_denver",
            version=get_version(),
        )
        return synapse

    def process_responses(self, responses: List[Union["bt.Synapse", Any]]) -> List[int]:
        print("Processing responses...", responses)
        outputs = []
        for response in responses:
            if response.dendrite.status_code != 200:
                continue
            outputs.extend(response.results)
        return outputs
