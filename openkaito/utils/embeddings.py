import torch
import numpy as np

MAX_EMBEDDING_DIM = 2048


# padding tensor to MAX_EMBEDDING_DIM with zeros
# applicable to embeddings with shape (d) or (n, d) where d < MAX_EMBEDDING_DIM
def pad_tensor(tensor, max_len=MAX_EMBEDDING_DIM):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if tensor.ndim == 1:
        if tensor.shape[0] < max_len:
            pad = torch.zeros(max_len - tensor.shape[0])
            tensor = torch.cat((tensor, pad), dim=0)
    elif tensor.ndim == 2:
        if tensor.shape[1] < max_len:
            pad = torch.zeros(tensor.shape[0], max_len - tensor.shape[1])
            tensor = torch.cat((tensor, pad), dim=1)
    else:
        raise ValueError("Invalid tensor shape")
    return tensor
