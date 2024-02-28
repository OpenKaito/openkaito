# Project Otika Quickstart

## General

On bittensor testnet, the netuid of subnet Project Otika is `88`.

### Prepare Wallet

Generally, for both validator and miner, you need to prepare your wallet and make your key registered in the subnet. You can find related guidance in the [running on mainnet](./docs/running_on_mainnet.md) section.

If you are new to bittensor subnet, you are recommended to start by following
- [running on staging](./docs/running_on_staging.md)
- [running on testnet](./docs/running_on_testnet.md).



### Install Otika

In the root folder of this repository, run the following command to install Otika:
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


To use Otika, you need to have an Elasticsearch instance running. You can install Elasticsearch by following the instructions [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html). That includes intalling [docker](https://docs.docker.com/engine/install/) and run the Elasticsearch docker image.

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
sudo yum install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

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


### Start the Miner

After setting up the Elasticsearch and obtaining the API key, you can start the miner by running the following command:
```bash
python neurons/miner.py --netuid <netuid> --subtensor.network finney --wallet.name miner --wallet.hotkey default --logging.debug --blacklist.force_validator_permit
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

### Obtain OpenAi API Key

To use the LLM ranking result evaluation, you need to obtain an API key from [OpenAI](https://platform.openai.com/). After obtaining the API key, you can write it down in the `.env` file.

```
OPENAI_API_KEY="sk-xxxxxx"
```

### Configuration via .env

You can configure the validator by setting the following environment variables in the `.env` file:

```
VALIDATOR_LOOP_SLEEP=30    # The sleep interval between sending requests to the miner
VALIDATOR_SEARCH_QUERY_SIZE=5  # The size of the search results required by the validator
```

### Start the Validator

You can start the validator by running the following command:

```bash
python neurons/validator.py --netuid <netuid> --subtensor.network finney --wallet.name validator --wallet.hotkey default --logging.debug --neuron.sample_size 10 --neuron.axon_off
```

### Notes about validator auto-update

To enable validator auto update with github repo, you can install `pm2` and `jq`, then execute the `run.sh`.

```bash
# Install pm2 and jq
sudo apt-get install jq npm
sudo npm install -g pm2

# Run the run.sh to enable auto update
# You may need to modify the first few lines of `run.sh` to set the variables properly
./run.sh
```

To monitor your validator process, use the following pm2 commands to monitor the status and logs of your process:

```bash
pm2 status
pm2 logs <id>
```
