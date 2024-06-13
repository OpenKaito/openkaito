import os
import json
import time
import torch
import base64
import typing
import asyncio
import traceback
import bittensor as bt
import threading

from openkaito.utils.config import check_config, add_args, config

from openkaito.base.neuron import BaseNeuron

from openkaito.protocol import (
    DiscordSearchSynapse,
    StructuredSearchSynapse,
    SemanticSearchSynapse,
)

from openkaito.utils.uids import get_random_uids
from openkaito.utils.version import get_version
from openkaito.utils.misc import ttl_get_block


class ApiNeuron:
    """
    API node for storage network

    Attributes:
        subtensor (bt.subtensor): The interface to the Bittensor network's blockchain.
        wallet (bt.wallet): Cryptographic wallet containing keys for transactions and encryption.
        metagraph (bt.metagraph): Graph structure storing the state of the network.
        database (redis.StrictRedis): Database instance for storing metadata and proofs.
    """

    node_type = "validator_api"

    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    @property
    def block(self):
        return ttl_get_block(self)

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"

    def __init__(self):
        self.config = ApiNeuron.config()
        self.check_config(self.config)
        bt.logging(config=self.config.logging)
        bt.logging.debug(self.config)

        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.debug(str(self.subtensor))

        self.wallet = bt.wallet(config=self.config)
        bt.logging.debug(f"wallet: {str(self.wallet)}")

        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running neuron on subnet: {self.config.netuid} with uid {self.uid} using network: {self.subtensor.chain_endpoint}"
        )

        self.axon = bt.axon(wallet=self.wallet, config=self.config)

        self.axon.attach(
            forward_fn=self.forward_semantic_search,
            blacklist_fn=self.blacklist_semantic_search,
            priority_fn=self.priority_semantic_search,
        )
        bt.logging.info(f"Axon created: {self.axon}")


        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.debug(str(self.dendrite))

        self.last_sync_block = self.block - 10

        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

        self.step = 0

    def run(self):
        self.resync_metagraph()

        bt.logging.info(
            f"Serving API axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        self.axon.start()

        bt.logging.info(f"API neuron starting at block: {self.block}")
    

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.should_exit:
                while (
                    self.block - self.last_sync_block < self.config.neuron.epoch_length
                ):
                    # Wait before checking again.
                    time.sleep(1)

                    # Check if we should exit.
                    if self.should_exit:
                        break

                # Sync metagraph and potentially set weights.
                self.resync_metagraph()
                self.step += 1

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("API Neuron killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        if not self.is_running:
            bt.logging.debug("Starting API Neuron in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            # asyncio.new_event_loop().run_until_complete(self.run())
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        if self.is_running:
            bt.logging.debug("Stopping API Neuron in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    async def forward_semantic_search(
        self, query: SemanticSearchSynapse
    ) -> SemanticSearchSynapse:

        bt.logging.debug(f"Forwarding semantic search query: {query.json()}")

        # TODO: random select miners or top miners? (if top miners, will they be queried too much?)
        miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

        responses = await self.dendrite(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=query,
            deserialize=True,
            timeout=query.timeout if query.timeout else 10,
        )
        # do we run evaluation for api queries?

        # TODO: how to structure the results? (e.g. top 5 results(then we will need to do evaluatoin), or all results?)
        # How to respect the `size` parameter in synapse?
        query.results = dict(zip(miner_uids, responses))

        return query

    async def blacklist(self, synapse: bt.Synapse) -> typing.Tuple[bool, str]:

        return False, "Debug mode"

        # Default to deny all hotkeys that are not in the whitelist.
        return True, "Only whitelisted hotkeys are allowed"

    async def priority(self, synapse: bt.Synapse) -> float:
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority

    async def blacklist_structured_search(
        self, synapse: StructuredSearchSynapse
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_structured_search(
        self, synapse: StructuredSearchSynapse
    ) -> float:
        return await self.priority(synapse)

    async def blacklist_semantic_search(
        self, synapse: SemanticSearchSynapse
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_semantic_search(self, synapse: SemanticSearchSynapse) -> float:
        return await self.priority(synapse)

    async def blacklist_discord_search(
        self, synapse: DiscordSearchSynapse
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_discord_search(self, synapse: DiscordSearchSynapse) -> float:
        return await self.priority(synapse)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)
        self.last_sync_block = self.block
        bt.logging.info("resync_metagraph() done")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_run_thread()


if __name__ == "__main__":
    with ApiNeuron() as api_neuron:
        while True:
            time.sleep(30)
            bt.logging.debug(f"API neuron running at block {api_neuron.block}")
