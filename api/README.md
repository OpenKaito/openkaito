# OpenKaito Validator API

Validator can setup an API server to issue search queries to the network, for building Apps on OpenKaito subnet, interacting with other subnets, etc. The API server will be responsible for issuing search queries to the network and receiving ranked results from miners.

To setup the API server, you need to config via the `.env` file,

```bash
cn ${PROJECT_ROOT}/api
mv .env.example .env
vim .env
```

Then you can set some api keys in `.env` file separated by comma, as well other wallet and subtensor info.

```bash
OPENKAITO_VALIDATOR_API_KEYS="sn5_key1,sn5_key2,sn5_key3"

# Wallet and subtensor info
VALIDATOR_API_WALLET_NAME="validator"
VALIDATOR_API_HOTKEY_NAME="default"
SUBTENSOR_NETWORK="test"  # or "finney" for mainnet
NETUID="88"  # 88 for testnet and 5 for mainnet
```

modify your wallet and subtensor info in `api/api_server.py`, then run the following command:

```bash
fastapi run api_server.py --port 8900 # or any other port
```

Then you can refer to `api/text_embedding_query.sh` for sending text-embedding queries to the network.
