# Quickstart

## General

The `netuid` for `openkaito` is `5` on mainnet, and `88` on testnet.

### Prepare Wallet

Generally, for both validator and miner, you need to prepare your wallet and make your key registered in the subnet. You can find related guidance in the [running on mainnet](./docs/running_on_mainnet.md) section.

If you are new to bittensor subnet, you are recommended to start by following

- [running on staging](./docs/running_on_staging.md)
- [running on testnet](./docs/running_on_testnet.md).

### Install openkaito

In the root folder of this repository, run the following command to install openkaito:

```bash
pip install -e .
```

### Obtain Apify API Key

To use the provided crawler, you need to obtain an API key from [Apify](https://console.apify.com/). After obtaining the API key, you can write it down in the `.env` file.

```
APIFY_API_KEY='apify_api_xxxxxx'
```

The key is for both the miner and the validator. For miner, it is used to search and crawl the data needed. For validator, it is used to get the ground truth data for evaluation.

> **_NOTE:_** You can also create your own crawler to crawl the data you need. But you need to be aware that the data you crawl should be able to pass the integrity check of the validator.

## Miner Setup

Then you can prepare your dotenv file by:

```bash
cp .env.example .env
```

### Setup Elasticsearch

To be a miner of openkaito, you need to have an Elasticsearch instance running. You can install Elasticsearch by following the instructions [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html). That includes intalling [docker](https://docs.docker.com/engine/install/) and run the Elasticsearch docker image.

#### Install Docker

You can refer to the [docker official guide](https://docs.docker.com/engine/install/) to install docker.

In short, you can run the following command to install docker:

**Ubuntu**

```bash
# Uninstall old versions
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install Docker
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Verify that the Docker Engine installation is successful by running the hello-world image
sudo docker run hello-world
```

Then you have successfully installed and started Docker Engine.

**CentOS**

```bash
# Uninstall old versions
sudo yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine

# Install the yum-utils package (which provides the yum-config-manager utility) and set up the repository
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# Install Docker
sudo yum install docker

# Start Docker
sudo systemctl start docker

# Verify that the Docker Engine installation is successful by running the hello-world image
sudo docker run hello-world
```

Then you have successfully installed and started Docker Engine.

#### Run Elasticsearch with Docker

If you have [docker](https://docs.docker.com/engine/install/) installed, you can run the following command to start an Elasticsearch instance:

```bash
sudo docker network create elastic
sudo docker pull docker.elastic.co/elasticsearch/elasticsearch:8.12.1
sudo docker run --name elasticsearch --net elastic -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -t docker.elastic.co/elasticsearch/elasticsearch:8.12.1
```

After running the above command, you will be prompted the password for the built-in user `elastic`. You can write it down in the `.env` file.

```
ELASTICSEARCH_HOST="https://localhost:9200"
ELASTICSEARCH_USERNAME="elastic"
ELASTICSEARCH_PASSWORD="your_password"
```

#### Useful Commands

If you forget the password, you can reset it by running the following command:

```bash
sudo docker exec -it elasticsearch /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic
```

If the Elasticsearch instance exited unexpectedly, you can start it by running the following command:

```bash
sudo docker start elasticsearch
```

### Setup Semantic Search Dataset & Indexing

You may refer to and run `scripts/vector_index_eth_denver_dataset.py` to index the ETH Denver 2024 dataset for semantic search.

This script extracts the Eth Denver dataset (open-sourced by [Kaito AI](https://portal.kaito.ai/events/ETHDenver2024)), indexes the documents in Elasticsearch, and indexes the embeddings of the documents in Elasticsearch.
It also provides a test query to retrieve the top-k similar documents to the query.

This script is intentionally kept transparent and hackable, and miners may do their own customizations.

```bash
python scripts/vector_index_eth_denver_dataset.py
```

### Start the Miner

After setting up the Elasticsearch and obtaining the API key, you can start the miner by running the following command:

```bash
python neurons/miner.py --netuid 5 --subtensor.network finney --wallet.name miner --wallet.hotkey default --logging.debug --blacklist.force_validator_permit --axon.port 8091
```

The detailed commandline arguments for the `neurons/miner.py` can be obtained by `python neurons/miner.py --help`, and are as follows:

```bash
usage: miner.py [-h] [--no_prompt] [--wallet.name WALLET.NAME] [--wallet.hotkey WALLET.HOTKEY] [--wallet.path WALLET.PATH]
                [--subtensor.network SUBTENSOR.NETWORK] [--subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT] [--subtensor._mock SUBTENSOR._MOCK]
                [--logging.debug] [--logging.trace] [--logging.record_log] [--logging.logging_dir LOGGING.LOGGING_DIR] [--axon.port AXON.PORT]
                [--axon.ip AXON.IP] [--axon.external_port AXON.EXTERNAL_PORT] [--axon.external_ip AXON.EXTERNAL_IP]
                [--axon.max_workers AXON.MAX_WORKERS] [--netuid NETUID] [--neuron.name NEURON.NAME] [--neuron.device NEURON.DEVICE]
                [--neuron.epoch_length NEURON.EPOCH_LENGTH] [--neuron.events_retention_size NEURON.EVENTS_RETENTION_SIZE] [--neuron.dont_save_events]
                [--neuron.disable_crawling] [--neuron.crawl_size NEURON.CRAWL_SIZE] [--neuron.search_recall_size NEURON.SEARCH_RECALL_SIZE]
                [--blacklist.force_validator_permit] [--blacklist.allow_non_registered] [--config CONFIG] [--strict] [--no_version_checking]

options:
  -h, --help            show this help message and exit
  --no_prompt           Set true to avoid prompting the user.
  --wallet.name WALLET.NAME
                        The name of the wallet to unlock for running bittensor (name mock is reserved for mocking this wallet)
  --wallet.hotkey WALLET.HOTKEY
                        The name of the wallet's hotkey.
  --wallet.path WALLET.PATH
                        The path to your bittensor wallets
  --subtensor.network SUBTENSOR.NETWORK
                        The subtensor network flag. The likely choices are: -- finney (main network) -- test (test network) -- archive (archive
                        network +300 blocks) -- local (local running network) If this option is set it overloads subtensor.chain_endpoint with an
                        entry point node from that network.
  --subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT
                        The subtensor endpoint flag. If set, overrides the --network flag.
  --subtensor._mock SUBTENSOR._MOCK
                        If true, uses a mocked connection to the chain.
  --logging.debug       Turn on bittensor debugging information
  --logging.trace       Turn on bittensor trace level information
  --logging.record_log  Turns on logging to file.
  --logging.logging_dir LOGGING.LOGGING_DIR
                        Logging default root directory.
  --axon.port AXON.PORT
                        The local port this axon endpoint is bound to. i.e. 8091
  --axon.ip AXON.IP     The local ip this axon binds to. ie. [::]
  --axon.external_port AXON.EXTERNAL_PORT
                        The public port this axon broadcasts to the network. i.e. 8091
  --axon.external_ip AXON.EXTERNAL_IP
                        The external ip this axon broadcasts to the network to. ie. [::]
  --axon.max_workers AXON.MAX_WORKERS
                        The maximum number connection handler threads working simultaneously on this endpoint. The grpc server distributes new worker
                        threads to service requests up to this number.
  --netuid NETUID       Subnet netuid
  --neuron.name NEURON.NAME
                        Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name.
  --neuron.device NEURON.DEVICE
                        Device to run on.
  --neuron.epoch_length NEURON.EPOCH_LENGTH
                        The default epoch length (how often we set weights, measured in 12 second blocks).
  --neuron.events_retention_size NEURON.EVENTS_RETENTION_SIZE
                        Events retention size.
  --neuron.dont_save_events
                        If set, we dont save events to a log file.
  --neuron.disable_crawling
                        If set, we disable crawling when receiving a search request.
  --neuron.crawl_size NEURON.CRAWL_SIZE
                        The number of documents to crawl when receiving each query.
  --neuron.search_recall_size NEURON.SEARCH_RECALL_SIZE
                        The number of search results to retrieve for ranking.
  --blacklist.force_validator_permit
                        If set, we will force incoming requests to have a permit.
  --blacklist.allow_non_registered
                        If set, miners will accept queries from non registered entities. (Dangerous!)
  --config CONFIG       If set, defaults are overridden by passed file.
  --strict              If flagged, config will check that only exact arguments have been set.
  --no_version_checking
                        Set ``true`` to stop cli version checking.
```

### Notes

We provide `scripts/search_evaluation.py` to quick preview your search engine performance. You can run the following command to evaluate the search engine's performance:

```bash
python scripts/search_evaluation.py --query 'BTC' --size 5
```

Note the score may not always be the save even if everything is kept unchanged. But you can always improve your search engine performance systematically to gain a statistically significant improvement.

To obtain better miner performance, you can consider the following options:

- crawl and index more data, the more data you have, the better the search engine performance you can achieve
  - adjust the apify crawler running parameters, e.g., increase the size of on search crawling
  - implement a continuous crawler, instead of or in addition to the on search crawling, to crawl data and ingest into the Elasticsearch instance
- build better ranking model
  - customize the provied ranking model, e.g., tune the parameters for `length_weight` and `age_weight`
  - implement a better ranking model, e.g., integrating LLM
- improve the recall stage
  - tune the parameters for the search query, e.g., adjust the `size` parameter via `MINER_SEARCH_RECALL_SIZE` in `.env`
  - build better index for the data you crawled and ingested into Elasticsearch, e.g., Knowledge Graph
- any other advanced improvements you can think of

## Validator Setup

You may need to setup your wallet and hotkey according to [running on mainnet](./docs/running_on_mainnet.md).

### Obtain OpenAI API Key

To use the LLM ranking result evaluation, you need to obtain an API key from [OpenAI](https://platform.openai.com/). After obtaining the API key, you can write it down in the `.env` file.

```
OPENAI_API_KEY="sk-xxxxxx"
```

> Don't forget to also obtain and set the `APIFY_API_KEY` as mentioned in the above **General** section.

### Obtain WandB API Key

Log in to [Weights & Biases](https://wandb.ai) and generate a key in your account settings.

Set the key `WANDB_API_KEY` in the `.env` file.

```
WANDB_API_KEY="your_wandb_api_key"
```

### Install Dependencies

To enable validator auto update with github repo, you can install `pm2` and `jq`.

**Ubuntu**

```bash
# Install pm2 and jq
sudo apt-get install jq npm
sudo npm install -g pm2
```

**CentOS**

```bash
# Install jq
sudo yum install jq

# Install npm and pm2
sudo yum install nodejs20
sudo npm install -g pm2
```

### Start the Validator

You can start the validator by running the following command, enabling validator auto update with github repo:

```bash
# Run the run.sh to enable auto update
pm2 start run.sh --name openkaito_validator_autoupdate -- --netuid 5 --subtensor.network finney --wallet.name <your_wallet_name> --wallet.hotkey <your_hotkey> --logging.debug
```

You may pass in more command line arguments to the `validator` command, e.g., `--axon.port <your_axon_port>` etc., if needed.

This will run two PM2 process: one for the `neurons/validator.py` which is called `openkaito_validator_main_process`, and one for the `run.sh` script which is called `openkaito_validator_autoupdate`. The script will check for updates every 30 minutes, if there is an update then it will pull it, install it, restart `openkaito_validator_main_process` and then restart itself.

The detailed commandline arguments for the `validator` can be obtained by `python neurons/validator.py --help`, and are as follows:

```bash
usage: validator.py [-h] [--no_prompt] [--wallet.name WALLET.NAME] [--wallet.hotkey WALLET.HOTKEY] [--wallet.path WALLET.PATH]
                    [--subtensor.network SUBTENSOR.NETWORK] [--subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT] [--subtensor._mock SUBTENSOR._MOCK]
                    [--logging.debug] [--logging.trace] [--logging.record_log] [--logging.logging_dir LOGGING.LOGGING_DIR] [--axon.port AXON.PORT]
                    [--axon.ip AXON.IP] [--axon.external_port AXON.EXTERNAL_PORT] [--axon.external_ip AXON.EXTERNAL_IP]
                    [--axon.max_workers AXON.MAX_WORKERS] [--netuid NETUID] [--neuron.name NEURON.NAME] [--neuron.device NEURON.DEVICE]
                    [--neuron.epoch_length NEURON.EPOCH_LENGTH] [--neuron.events_retention_size NEURON.EVENTS_RETENTION_SIZE]
                    [--neuron.dont_save_events] [--neuron.num_concurrent_forwards NEURON.NUM_CONCURRENT_FORWARDS]
                    [--neuron.sample_size NEURON.SAMPLE_SIZE] [--neuron.search_request_interval NEURON.SEARCH_REQUEST_INTERVAL]
                    [--neuron.search_result_size NEURON.SEARCH_RESULT_SIZE] [--neuron.disable_set_weights]
                    [--neuron.moving_average_alpha NEURON.MOVING_AVERAGE_ALPHA] [--neuron.axon_off]
                    [--neuron.vpermit_tao_limit NEURON.VPERMIT_TAO_LIMIT] [--config CONFIG] [--strict] [--no_version_checking]

options:
  -h, --help            show this help message and exit
  --no_prompt           Set true to avoid prompting the user.
  --wallet.name WALLET.NAME
                        The name of the wallet to unlock for running bittensor (name mock is reserved for mocking this wallet)
  --wallet.hotkey WALLET.HOTKEY
                        The name of the wallet's hotkey.
  --wallet.path WALLET.PATH
                        The path to your bittensor wallets
  --subtensor.network SUBTENSOR.NETWORK
                        The subtensor network flag. The likely choices are: -- finney (main network) -- test (test network) -- archive (archive
                        network +300 blocks) -- local (local running network) If this option is set it overloads subtensor.chain_endpoint with an
                        entry point node from that network.
  --subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT
                        The subtensor endpoint flag. If set, overrides the --network flag.
  --subtensor._mock SUBTENSOR._MOCK
                        If true, uses a mocked connection to the chain.
  --logging.debug       Turn on bittensor debugging information
  --logging.trace       Turn on bittensor trace level information
  --logging.record_log  Turns on logging to file.
  --logging.logging_dir LOGGING.LOGGING_DIR
                        Logging default root directory.
  --axon.port AXON.PORT
                        The local port this axon endpoint is bound to. i.e. 8091
  --axon.ip AXON.IP     The local ip this axon binds to. ie. [::]
  --axon.external_port AXON.EXTERNAL_PORT
                        The public port this axon broadcasts to the network. i.e. 8091
  --axon.external_ip AXON.EXTERNAL_IP
                        The external ip this axon broadcasts to the network to. ie. [::]
  --axon.max_workers AXON.MAX_WORKERS
                        The maximum number connection handler threads working simultaneously on this endpoint. The grpc server distributes new worker
                        threads to service requests up to this number.
  --netuid NETUID       Subnet netuid
  --neuron.name NEURON.NAME
                        Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name.
  --neuron.device NEURON.DEVICE
                        Device to run on.
  --neuron.epoch_length NEURON.EPOCH_LENGTH
                        The default epoch length (how often we set weights, measured in 12 second blocks).
  --neuron.events_retention_size NEURON.EVENTS_RETENTION_SIZE
                        Events retention size.
  --neuron.dont_save_events
                        If set, we dont save events to a log file.
  --neuron.num_concurrent_forwards NEURON.NUM_CONCURRENT_FORWARDS
                        The number of concurrent forwards running at any time.
  --neuron.sample_size NEURON.SAMPLE_SIZE
                        The number of miners to query in a single step.
  --neuron.search_request_interval NEURON.SEARCH_REQUEST_INTERVAL
                        The interval seconds between search requests.
  --neuron.search_result_size NEURON.SEARCH_RESULT_SIZE
                        The number of search results required for each miner to return.
  --neuron.disable_set_weights
                        Disables setting weights.
  --neuron.moving_average_alpha NEURON.MOVING_AVERAGE_ALPHA
                        Moving average alpha parameter, how much to add of the new observation.
  --neuron.axon_off, --axon_off
                        Set this flag to not attempt to serve an Axon.
  --neuron.vpermit_tao_limit NEURON.VPERMIT_TAO_LIMIT
                        The maximum number of TAO allowed to query a validator with a vpermit.
  --config CONFIG       If set, defaults are overridden by passed file.
  --strict              If flagged, config will check that only exact arguments have been set.
  --no_version_checking
                        Set ``true`` to stop cli version checking.
```

### Monitor the Validator

To monitor your validator process, use the following pm2 commands to monitor the status and logs of your process:

```bash
pm2 status
pm2 logs <id>
```
