from . import config
from . import misc
from . import uids
from . import embeddings


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
