<div align="center">

# **OpenKaito - Decentralized Kaito AI** <!-- omit in toc -->

[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
---

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

## Installation

### Validator Installation

Please see [Validator Setup](https://github.com/MetaSearch-IO/decentralized-search/blob/main/quickstart.md#validator-setup) in the [quick start guide](https://github.com/MetaSearch-IO/decentralized-search/blob/main/quickstart.md).

### Miner Installation

Please see [Miner Setup](https://github.com/MetaSearch-IO/decentralized-search/blob/main/quickstart.md#miner-setup) in the [quick start guide](https://github.com/MetaSearch-IO/decentralized-search/blob/main/quickstart.md).

---

## Introduction

### The Problem
The internet is becoming more closed than open -
- **Paradigm Shift**: Data is increasingly generated and hosted on closed platforms rathen than the open web.
- **Data Restrictions**: More platforms are imposing greater restrictions on public user data (eg X, Reddit).
- **The Consequence**: A less open internet and greater centralization of power to platforms.

### OpenKaito's Mission
- Kaito AI is committed to democratizing access to information through its established platform. However, the in-house approach to data collection, indexing, AI training, and ranking imposes operational burdens and stifles broader public innovation.
- OpenKaito aims to build a decentralized indexing layer that powers smart search and feeds across both the open and closed web, curated and indexed by the community.

### An Analology to Google
- Google runs large-scale data acquisition on the open web, performs content understanding and indexing, and serves search queries through proprietary ranking algorithms.
- For OpenKaito, the community decides the valuable knowledge to acquire, provides their proprietary resources for data acquisition (e.g., IP addresses, access to closed platforms), and competes on the best indexing and ranking algorithm around the knowledge.

## App Ecosystem
OpenKaito operates as an application-agnostic infrastructure layer to provide indexed and ranked data, powering a series of applications. Here is a non-exhaustive list of projects based on OpenKaito:

[Ethereum Conference Knowledge Base (sponsored by Ethereum Foundation)](https://portal.kaito.ai/events/ETHDenver2024)
<p align="left">
  <img src="https://github.com/OpenKaito/openkaito/assets/106579566/d2e1feda-60b0-49b8-b16b-123227cf89a9" width="40%" height="auto">
</p>

[Bittensor Discord Search by Corcel)](https://playground.corcel.io/open-kaito/discord-search)
<p align="left">
  <img src="https://github.com/OpenKaito/openkaito/assets/106579566/761eef6b-63bb-469e-9a48-4d634a88c228" width="40%" height="auto">
</p>

[TaoBot Integration (under development)](https://x.com/taodotbot/status/1796174248395813336)

## Token Economy
OpenKaito aims to have a self-sustainable economy 
- Sponsors/doners will pay the network for maintaining the vertical search engine 
- Users will pay the network to access premium features (premium content, analytics, alerts, etc) 
- Miners and validators will be rewarded by doing the work that actually generates economic value in the eyes of sponsors and users 
- Economic surplus (fees from sponsors and users) will be shared among the participants in the network (miners, validators, subnet owner) 

As proof of concept we've already secured grants from the Ethereum Foundation and Starknet Foundation, and are in talks with other sponsors (including non-crypto entities and communities) who recognise the economic values of such vertical search engine for their communities.

## Background Knowledge

<p align="center">
  <img src="https://github.com/MetaSearch-IO/decentralized-search/assets/106579566/68a4c45d-72bc-4444-a5f6-2cc4d917871b" width="60%" height="auto">
</p>

### Inverted Index

An inverted index serves as the foundation of a search engine. The high level idea is to construct a reverse lookup table from a keyword to documents containing the keyword. A sophisticated search engine usually leverages NLP techniques (e.g. tokenization, stemming) and content understanding models (e.g. classification, tagging, categorization) to optimize keyword extractions.

A search query typically expresses logical constraint on keywords and can be fulfilled by operations on the inverted index. An inverted index is distributed by nature - keyword partition and document partition are the two common partition schemes.

### Search Ranking

**Retrieval ranking** ranks documents satisfying the retrieval condition based on a ranking criteria. It focuses on simple, indexable signals such as term frequency (TF) and inverse document frequency (IDF), along with a linear combination of more static signals that enhance the speed and relevance of search results.

**Re-ranking** ranks a smaller set of candidates selected by retrieval ranking with more expensive techniques. Modern re-ranking employs deep learning algorithms to analyze complex signals like user interaction data. In a decentralized environment, this presents opportunities for optimization through collective intelligence, where network participants contribute to the validation and improvement of the re-ranking process.

### Knowledge Graph

A Knowledge Graph in the context of search engines is a structured representation of real-world entities and their interrelations. It serves as a foundation for enhancing search queries and document understanding by understanding the context and the relationships between different pieces of information. In web3, contexts like the relationship between projects, influencers, etc. are critical to effective search & analytics, and its evolving nature makes it a great fit to be solved with collective intelligence.

## Towards a Decentralized Search Engine

Instead of building a decentralized version of every component of a search engine, we focus on posing search relevance as a validation-miner problem to encourage miners to come up with innovative solutions to data acquisition, indexing, ranking, and knowledge graph. To goal is that based on a fair and effective criterion, miners are incentivized to optimize components with the highest ROI, similar to how a search engineering team runs on A/B testing and failure analysis.

<p align="center">
  <img src="https://github.com/MetaSearch-IO/decentralized-search/assets/106579566/7fa302f8-585b-47db-8881-cf1a2133a814" width="60%" height="auto">
</p>

### Validator

**Search Queries:** Validators are responsible for issuing search queries to the network and expect a list of ranked results from miners. Search queries will follow a simple format that supports basic functionalities, including keywords, AND/OR semantics, sorting by date, sorting by relevance, date filtering, etc., as outlined in Appendix A.

**AI-based nDCG:** nDCG is a standard metric for evaluating search engines, taking into account both result relevance and their relative positions. The downside of nDCG is that it requires a human-annotated ideal result set, which can be expensive and slow. However, model-based nDCG has gained traction recently, thanks to advancements in large language models (LLMs). In our validator-miner scheme, we implement a cost-effective ML-based nDCG rater that leverages a distilled LLM-based nDCG evaluator.

**Evolution of nDCG Evaluator:** To ensure that our evaluation scheme accurately reflects true result relevance, the evaluator model will be continuously fine-tuned with real user engagement data from The Search App (outlined in the subsequent section) and regularly updated on HuggingFace according to a set schedule. Both the model parameters and the training mechanism will be open-sourced, with the potential for full decentralization using BitTensor's model training capabilities.

**Result Correctness:** To prevent fabricated results, validators will selectively verify the URLs of search results to ensure their consistency with the original sources.

### Miner

Miners fulfill search requests issued by validators by providing a ranked list of results and are encouraged to find innovative ways to enhance the quality of their results. While there is no prescribed method for implementing the search, we suggest the following basic framework as a starting point.

**Search Index:** We provide local ElasticSearch instances with a basic schema for a set of supported sources, such as Twitter and governance forums. By default, a search request is translated into an ElasticSearch query.

**Crawler:** A basic crawler, which periodically updates the search index, is provided based on Apify. However, for more cost-effective crawling, node owners are encouraged to develop their own crawler stack.

**Ranking Algorithm:** The default ranking algorithm is BM25, natively supported by ElasticSearch. It's important to note that BM25 relies on the term frequency (TF) and inverse document frequency (IDF) within the search index, so rankings may vary based on the content in a node’s search index.

### Reward Model

Rewards are based on the following criteria:

**Truthfulness:** Miners receive rewards only for providing authentic results from a specified set of sources and will incur penalties for serving fabricated data.

**Relevance:** Miners are rewarded for the content and contextual relevance of the results, as reflected by nDCG, where the ordering of results contributes to relevance.

**Recency:** Miners are rewarded for the timeliness of results—the more recent the content, the higher the reward.

**Diversity:** Rewards consider diversity at both the source level (e.g., one source versus multiple sources) and the content level (e.g., various opinions, different authors), which can be assessed using content clustering methods.

### Validator API Server

Validtor can setup an API server to issue search queries to the network, for building Apps on OpenKaito subnet, interacting with other subnets, etc. The API server will be responsible for issuing search queries to the network and receiving ranked results from miners.

To setup the API server, you can set some api keys in `.env` file separated by comma,

```bash
OPENKAITO_VALIDATOR_API_KEYS="key1,key2,key3"
```

modify your wallet and subtensor info in `api/api_server.py`, then run the following command:

```bash
fastapi run api/api_server.py --port 8900
```

Then you can refer to `api/sample_request.py` for sending search queries to the network.

### Indexing Data from Other Subnets

**SN13 dataverse** is a decentralized data scraping subnet. If you are running a miner for openkaito and happen to be a miner for SN13, you can use SN13 as an extra source of raw data to be indexed in OpenKaito. Please refer to `scripts/import_sn13_data.py` for more details.

```bash
$ python scripts/import_sn13_data.py -h
usage: import_sn13_data.py [-h] --db DB [--batch_size BATCH_SIZE] [--time_bucket_ids [TIME_BUCKET_IDS ...]]

Import SN13 Data

options:
  -h, --help            show this help message and exit
  --db DB               SN13 sqlite3 database file, e.g., ../data-universe/SqliteMinerStorage.sqlite
  --batch_size BATCH_SIZE
                        optional, batch size for importing data, default is 100
  --time_bucket_ids [TIME_BUCKET_IDS ...]
                        optional, a list of SN13 timeBucketId to be imported, seperate by space, e.g., 474957 474958 474959
```

## Engineering Roadmap

#### ~~[Done] Reward Model Adjustment and More Sources~~
~~Currently Twitter is the only source that we onboarded and the evaluation is based on relevance and recency so that we can better calibrate the reward model. The team is actively working on onboarding more sources (e.g. News, Governance, Audio) and more diversed ranking & evaluation signals. Stay tuned!~~

#### ~~[Done] Rich Semantics~~
~~We are working on supporting a varity of search semantics including AND/OR, filter, sort by, etc.~~

#### ~~[Done] Vector Retrieval and Embedding Model~~
~~We will be supporting vector retrieval for RAG use cases and add competition for embedding models~~

#### ~~[Done] Validator Tools~~
~~Validator API and various tooling for better access and diagoniss of OpenKaito data~~

#### QA Engine on Social Network Data
Extend the RAG capability beyond conference data to create a real-time QA engine for social network data.

#### Expansion Beyond Web3
Plan to expand into more verticals beyond Web3.

#### Revenue Distribution Mechanism
We plan to implement a system that redistributes economic surplus back to the community.


