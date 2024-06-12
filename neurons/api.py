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


class ApiNeuron(BaseNeuron):
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

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"

    def __init__(self):
        self.config = ApiNeuron.config()
        self.check_config(self.config)
        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        bt.logging.debug(self.config)

        bt.logging.debug("loading subtensor")
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.debug(str(self.subtensor))

        bt.logging.debug("loading wallet")
        self.wallet = bt.wallet(config=self.config)

        if not self.subtensor.is_hotkey_registered_on_subnet(
            hotkey_ss58=self.wallet.hotkey.ss58_address, netuid=self.config.netuid
        ):
            raise Exception(
                f"Wallet not currently registered on netuid {self.config.netuid}, please first register wallet before running"
            )

        bt.logging.debug(f"wallet: {str(self.wallet)}")

        bt.logging.debug("init metagraph")
        self.metagraph = bt.metagraph(
            netuid=self.config.netuid, network=self.subtensor.network, sync=False
        )
        self.metagraph.sync(subtensor=self.subtensor)  # Sync metagraph with subtensor.
        bt.logging.debug(str(self.metagraph))

        self.my_subnet_uid = self.metagraph.hotkeys.index(
            self.wallet.hotkey.ss58_address
        )
        bt.logging.info(f"Running validator on uid: {self.my_subnet_uid}")

        bt.logging.debug("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            self.axon.attach(
                forward_fn=self.forward_semantic_search,
                blacklist_fn=self.blacklist_semantic_search,
                priority_fn=self.priority_semantic_search,
            )

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                self.axon.start()

            except Exception as e:
                bt.logging.error(f"Failed to serve Axon: {e}")
                pass

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize: {e}")
            pass

        bt.logging.debug("loading dendrite")
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.debug(str(self.dendrite))

        # Init the event loop.
        self.loop = asyncio.get_event_loop()

        self.last_sync_block = self.block - 10

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

        self.step = 0

    def run(self):
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

        # After all we have to ensure subtensor connection is closed properly
        finally:
            if hasattr(self, "subtensor"):
                bt.logging.debug("Closing subtensor connection")
                self.subtensor.close()

    def run_in_background_thread(self):
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        bt.logging.warning("forward()")
        return synapse
    
    def __enter__(self):
        self.run_in_background_thread()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_run_thread()


if __name__ == "__main__":
    ApiNeuron().run()
