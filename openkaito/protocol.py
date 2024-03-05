# The MIT License (MIT)
# Copyright © 2024 OpenKaito

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import bittensor as bt


import pydantic


class Version(pydantic.BaseModel):
    major: int
    minor: int
    patch: int

class SearchSynapse(bt.Synapse):
    """
    Search protocol representation for handling request and response communication between
    the miner and the validator.

    Attributes:
    - query_string: A string value representing the search request sent by the validator.
    - size: the maximal count size of results to return.
    - version: A `Version` object representing the version of the protocol.
    - results: A list of `Document` dict which, when filled, represents the response from the miner.
    """

    query_string: str
    size: int = pydantic.Field(5, ge=1, le=50)
    version: Optional[Version] = None

    results: Optional[List[Dict]] = None

    def deserialize(self) -> List[Dict]:
        return self.results
