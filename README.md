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

> There is a legacy version of the project focusing on decentralized indexing of various data sources, see [here](./docs/openkaito_v0_legacy.md) for more details.

## Abstract

Bittensor Subnet 5's primary focus is the development of the world’s best performing and most generalizable text embedding model.

Leveraging an extensive Large Language Model (LLM)-augmented corpus for evaluation, miners are empowered to develop and deploy text-embedding models that surpass current state-of-the-art (SOTA) performance.

## Objectives & Contributions

The primary objective of Subnet 5 is to train and serve the best and most generalizable text-embedding models. Such text-embedding models can empower plenty of downstream applications such as semantic search, natural language understanding, and so on.

Miners will be responsible for training models using an extensive corpus of textual data and serving the model in a low-latency and high-throughput way. These models will be utilized to generate high-quality embeddings for diverse text inputs.

Validators will conduct rigorous evaluations of the models using multiple benchmarks. Performance comparisons will be made against existing SOTA text embedding models to ensure continuous improvement and competitiveness.

Subnet users will gain access to cutting-edge text embedding models that are most generic and exceed SOTA performance. These models will be made publicly available through the validator API of Bittensor Subnet 5, facilitating widespread adoption and integration into various applications.

## Incentive Mechanism

Miners will receive a batch of texts and embed them.

For the text embeddings, validators have the pairwise relevance information to evaluate them via the contrastive learning loss:

```math
\mathcal{L}_\text{InfoNCE} = - \mathbb{E} \left[\log \frac{f(\mathbf{x}, \mathbf{c})}{\sum_{\mathbf{x}' \in X} f(\mathbf{x}', \mathbf{c})} \right]
```

where $f(x,c) = \exp{(x \cdot c)}$ is an estimate of $\frac{p(x | c)}{p(x)}$, and $c$ is the target embedding, and $x$ is the positive sample, and $x'$ are negative samples.

This is to maximize the mutual information between positive pairs $x$ and $c$:

$I(\mathbf{x}; \mathbf{c}) = \sum_{\mathbf{x}, \mathbf{c}} p(\mathbf{x}, \mathbf{c}) \log\frac{p(\mathbf{x}, \mathbf{c})}{p(\mathbf{x})p(\mathbf{c})} = \sum_{\mathbf{x}, \mathbf{c}} p(\mathbf{x}, \mathbf{c})\log\frac{p(\mathbf{x}|\mathbf{c})}{p(\mathbf{x})}$

and minimize the mutual information between negative pairs $x'$ and $c$:  $I(\mathbf{x'}; \mathbf{c})$.

Gradually we can potentially add processing time into consideration to encourage faster embedding and lower latency.

## Computing Requirements

There are no hard requirements for miners’ equipment, as long as they can serve their text-embedding model in a low-latency and high-throughput manner.

To achieve this, miners typically need the following infrastructures:

Model Training:

- Machines with GPUs for fast training models on large datasets

Model Serving:

- Dedicated model inference server

## Subnet User Interface

Eventually, Subnet 5 will serve the text-embedding model via the subnet validator API.

The dev experience of using Subnet 5 Embedding API will be similar to the OpenAI text-embedding API [https://platform.openai.com/docs/guides/embeddings/embedding-models](https://platform.openai.com/docs/guides/embeddings/embedding-models).

## Development Roadmap

V1:

- The text-embedding model evaluation and incentive mechanism
- Subnet dashboard with model performance growing curve, and comparison to OpenAI text-embedding-3-small and text-embedding-3-large models as baselines
- Subnet API for serving the miners trained model to the subnet users.

V2 and further:

- Extending the dataset
- Extending the evaluation incentive model to tasks like document re-ranking
- Incorporating the documents’ pairwise distance in the evaluation
- …

## Appendix - Backgrounds

### Text Embedding Model

Text embedding models are fundamental to modern Natural Language Processing (NLP), representing words, phrases, or documents as dense vectors in a continuous space. These models have evolved significantly over time:

Classic Approaches:

- One-hot encoding and count-based methods (e.g., TF-IDF)
- Limited in capturing semantic relationships

Word Embeddings:

- Based on distributional semantics
- Key models: Word2Vec, GloVe, FastText
- Capture word similarities and relationships

Sentence and Document Embeddings:

- Extend word-level techniques to larger text units, dynamic representations based on context
- Examples: ELMo, BERT, GPT
- Better at handling polysemy and context-dependent meanings

Applications span various NLP tasks, including semantic similarity, machine translation, and sentiment analysis. Ongoing challenges include addressing bias and improving efficiency.

This evolution from simple representations to sophisticated contextual models has dramatically enhanced NLP capabilities, enabling a more nuanced understanding of language by machines.

### Vector-based Semantic Search

Vector-based semantic search evolved from traditional keyword-based methods to address limitations in understanding context and meaning. It leverages advances in natural language processing and machine learning to represent text as dense vectors in a high-dimensional space.

Key components of vector-based semantic search include:

- Text embedding (e.g., Word2Vec, GloVe, BERT, GPT)
- Efficient nearest-neighbor search algorithms (e.g., indexing vectors using HNSW)

By indexing documents with their embeddings, it is possible to:

- Capture semantic relationships between words and concepts
- Improve handling of synonyms and related terms
- More intuitive and context-aware search experiences

Vector-based semantic search has significantly enhanced information retrieval across various applications, offering more relevant results by understanding the intent behind queries rather than relying solely on exact keyword matches.
