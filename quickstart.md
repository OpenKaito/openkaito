# Project Otika Quickstart

## General

On bittensor testnet, the netuid of subnet Project Otika is `88`.

### Prepare Wallet

Generally, for both validator and miner, you need to prepare your wallet and make your key registered in the subnet. You can find related guidance in the [running on mainnet](./docs/running_on_mainnet.md) section.

If you are new to bittensor subnet, you are recommended to start by following
    - [running on staging](./docs/running_on_staging.md)
    - [running on testnet](./docs/running_on_testnet.md).


## Miner Setup


### Install Otika
In the root folder of this repository, run the following command to install Otika:
```bash
pip install -e .
```

Then you can prepare your dotenv file by:
```bash
cp .env.example .env
```


### Setup Elasticsearch


To use Otika, you need to have an Elasticsearch instance running. You can install Elasticsearch by following the instructions [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html).

Basicall, if you have [docker](https://docs.docker.com/engine/install/) installed, you can run the following command to start an Elasticsearch instance:
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

If you forget the password, you can reset it by running the following command:
```bash
sudo docker exec -it elasticsearch /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic
```

### Obtain Crawler API Key

To use the provided crawler, you need to obtain an API key from [Apify](https://console.apify.com/). After obtaining the API key, you can write it down in the `.env` file.

```
APIFY_API_KEY='apify_api_xxxxxx'
```


### Start the Miner

After setting up the Elasticsearch and obtaining the API key, you can start the miner by running the following command:
```bash
python neurons/miner.py --netuid <netuid> --subtensor.network finney --wallet.name miner --wallet.hotkey default --logging.debug --blacklist.force_validator_permit
```


## Validator Setup

### Obtain OpenAi API Key

To use the LLM ranking result evaluation, you need to obtain an API key from [OpenAI](https://platform.openai.com/). After obtaining the API key, you can write it down in the `.env` file.

```
OPENAI_API_KEY="sk-xxxxxx"
```

### Configuration via .env

You can configure the validator by setting the following environment variables in the `.env` file:

```
VALIDATOR_LOOP_SLEEP=30    # The sleep interval between sending requests to the miner
VALIDATOR_SEARCH_QUERY_LENGTH=5  # The length of the search results required by the validator
```

### Start the Validator

You can start the validator by running the following command:

```bash
python neurons/validator.py --netuid <netuid> --subtensor.network finney --wallet.name validator --wallet.hotkey default --logging.debug --neuron.sample_size 10
```