import torch
import numpy as np

from transformers import BertTokenizer, BertModel

MAX_EMBEDDING_DIM = 1024


# padding tensor to MAX_EMBEDDING_DIM with zeros
# applicable to embeddings with shape (d) or (n, d) where d < MAX_EMBEDDING_DIM
def pad_tensor(tensor, max_len=MAX_EMBEDDING_DIM):
    """Pad tensor with zeros to max_len dimensions"""

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if tensor.ndim == 1:
        if tensor.shape[0] < max_len:
            pad = torch.zeros(max_len - tensor.shape[0], device=tensor.device)
            tensor = torch.cat((tensor, pad), dim=0)
    elif tensor.ndim == 2:
        if tensor.shape[1] < max_len:
            pad = torch.zeros(
                tensor.shape[0], max_len - tensor.shape[1], device=tensor.device
            )
            tensor = torch.cat((tensor, pad), dim=1)
    else:
        raise ValueError("Invalid tensor shape")
    return tensor


# for semantic search
# Note: you may consider more powerful embedding models here, or even finetune your own embedding model
# but make sure the query embedding is compatible with the indexed document embeddings
def text_embedding(text, model_name="bert-base-uncased"):
    """Get text embedding using Bert model"""

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    encoded_input = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        model_output = model(**encoded_input)

    # cls pooling
    # embedding = model_output[0][:, 0]

    # mean pooling
    embedding = torch.mean(model_output[0], dim=1)

    # Note: miners are encouraged to explore more pooling strategies, finetune the model, etc.

    return embedding


def openai_embeddings_tensor(
    client, texts, dimensions=64, model="text-embedding-3-large"
):

    # replace newlines, which can negatively affect performance.
    texts = [text.replace("\n", " ") for text in texts]

    embeddings = client.embeddings.create(
        input=texts, model=model, dimensions=dimensions
    ).data

    return torch.as_tensor([emb.embedding for emb in embeddings])
