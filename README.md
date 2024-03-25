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

- Kaito AI is committed to democratizing access to Web3 information through its established platform. However, the in-house approach to data collection, indexing, AI training, and ranking imposes operational burdens and stifles broader public innovation.
- Search engines are complex systems beyond a mere database or a ranking algorithm. A useful search engine must also possess low latency, presenting additional challenges to its decentralization. Subnet OpenKaito serves as Kaito AI’s foray into technical innovations to address these challenges. By leveraging BitTensor’s built-in Yuma consensus, we define search indexing as a miner-validator problem, where index relevance is evaluated by an AI-based nDCG evaluator that learns from real user engagement feedback. We also plan to introduce a seamless search and analytics product based on this decentralized search layer, featuring intelligent coordination and caching mechanisms on validator nodes.
- Our goal is to build a decentralize indexing layer powering smart search and analytics.

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

## Towards a Decentralized Web3 Search

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


## Coming Soon

### Reward Model Adjustment and More Sources
Currently Twitter is the only source that we onboarded and the evaluation is based on relevance and recency so that we can better calibrate the reward model. The team is actively working on onboarding more sources (e.g. News, Governance, Audio) and more diversed ranking & evaluation signals. Stay tuned!

### Rich Semantics
We are working on supporting a varity of search semantics including AND/OR, filter, sort by, etc.

### Vector Retrieval
We will be supporting vector retrieval for RAG use cases

### The OpenKaito App

To provide a fast and seamless user experience, we have designed a centralized client layer on top of validator nodes, leading to an end-to-end search and analytics product for Web3. This design stems from several key insights from the Kaito AI team's product research:

1. **Head-heavy Query Distribution:** Search queries to Kaito AI’s institutional product are predominantly head-heavy—the top queries, typically project names and tickers, account for the majority of traffic. This indicates that proactive fetching and aggregation of search results into local storage can enable a low-latency experience for most queries.
2. **Ticker-centered Feeds & Analytics:** A central aspect of Kaito AI's product offering is its feeds and analytics focused on tickers, which are powered by its search stack and curated by AI. This focus on tickers is well-suited for pre-fetching, aggregation, and caching.
3. **Tail Queries:** To ensure a smooth experience for less common queries, whose results cannot be pre-fetched, the search backend will store and index any aggregated results locally. The search client can build up this content index by indexing results from frequent queries, as well as by continuously issuing crypto-related queries to the Subnet (e.g., Crypto topics and narratives as currently indexed by Kaito AI).
