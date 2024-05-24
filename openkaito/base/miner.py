# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import threading
import time
import traceback
import typing

import bittensor as bt
import torch

from openkaito.base.neuron import BaseNeuron
from openkaito.protocol import (
    DiscordSearchSynapse,
    SearchSynapse,
    SemanticSearchSynapse,
    StructuredSearchSynapse,
)
from openkaito.utils.config import config


class BaseMinerNeuron(BaseNeuron):
    """
    Base class for Bittensor miners.
    """

    neuron_type: str = "MinerNeuron"

    def __init__(self):
        super().__init__(config=self.config())

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            bt.logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        # The axon handles request processing, allowing validators to send this miner requests.
        self.axon = bt.axon(wallet=self.wallet, config=self.config)

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info(f"Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward_search,
            blacklist_fn=self.blacklist_search,
            priority_fn=self.priority_search,
        ).attach(
            forward_fn=self.forward_structured_search,
            blacklist_fn=self.blacklist_structured_search,
            priority_fn=self.priority_structured_search,
        ).attach(
            forward_fn=self.forward_semantic_search,
            blacklist_fn=self.blacklist_semantic_search,
            priority_fn=self.priority_semantic_search,
        ).attach(
            forward_fn=self.forward_discord_search,
            blacklist_fn=self.blacklist_discord_search,
            priority_fn=self.priority_discord_search,
        )
        bt.logging.info(f"Axon created: {self.axon}")

        self.last_sync_block = self.block - 1000

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        bt.logging.warning("wrong execution path: forward()")
        return synapse

    async def forward_search(self, query: SearchSynapse) -> SearchSynapse:
        bt.logging.warning("unimplemented: forward_search()")

    async def forward_structured_search(
        self, query: StructuredSearchSynapse
    ) -> StructuredSearchSynapse:
        bt.logging.warning("unimplemented: forward_structured_search()")

    async def forward_semantic_search(
        self, query: SemanticSearchSynapse
    ) -> SemanticSearchSynapse:
        bt.logging.warning("unimplemented: forward_semantic_search()")

    async def forward_discord_search(
        self, query: DiscordSearchSynapse
    ) -> DiscordSearchSynapse:
        bt.logging.warning("unimplemented: forward_discord_search()")

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the miner's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
        During each epoch of its operation, the miner waits for new blocks on the Bittensor network, updates its
        knowledge of the network (metagraph), and sets its weights. This process ensures the miner remains active
        and up-to-date with the network's latest state.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that miner is registered on the network.
        self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Start  starts the miner's axon, making it active on the network.
        self.axon.start()

        bt.logging.info(f"Miner starting at block: {self.block}")

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
                self.sync()
                self.step += 1

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the miner's background operations upon exiting the context.
        This method facilitates the use of the miner in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)
        self.last_sync_block = self.block
        bt.logging.info("resync_metagraph() done")

    def should_set_weights(self) -> bool:
        return False

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return self.block - self.last_sync_block > self.config.neuron.epoch_length

    async def blacklist(self, synapse: bt.Synapse) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (bt.Synapse): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        if not synapse.dendrite.hotkey:
            return True, "Hotkey not provided"
        registered = synapse.dendrite.hotkey in self.metagraph.hotkeys
        if self.config.blacklist.allow_non_registered and not registered:
            return False, "Allowing un-registered hotkey"
        elif not registered:
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, f"Unrecognized hotkey {synapse.dendrite.hotkey}"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        stake = self.metagraph.S[uid].item()
        if self.config.blacklist.validator_min_stake and stake < self.config.blacklist.validator_min_stake:
            bt.logging.warning(f"Blacklisting request from {synapse.dendrite.hotkey} [uid={uid}], not enough stake -- {stake}")
            return True, "Stake below minimum"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: bt.Synapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (bt.Synapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
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

    async def blacklist_search(self, synapse: SearchSynapse) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_search(self, synapse: SearchSynapse) -> float:
        return await self.priority(synapse)

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

    def save_state(self):
        pass

    def load_state(self):
        pass

    def save_state(self):
        pass

    def load_state(self):
        pass
