# Query Language & Response API

A search query is identified with a query string and a sort parameter. Query string may contain search **keywords, logical operators, and time filter**. Sort parameter can be one of **relevancy**(default) and **recency**.

## Keywords

Keywords are text strings that you can use to search across crawled content, e.g., `ETH`, `bittensor`, etc.

## Logical Operators

Logical operators allow you to specify more than one criteria and to indicate the relationship between them. Whitespace characters outside of quotation marks act as implicit `OR` operators. Kaito decentralized search supports the following additional explicit logical operators:

| Operator | Description | Example |
| --- | --- | --- |
| AND | Matches if the content contains both keywords | `bittensor AND subnet` |
| OR | Matches if the content contains either keyword | `ETH \| BTC` |
| - | Matches if the content does not contain the keyword | `-BTC` |
| "..." | Matches if the content contains a quoted phrase (words in the quote appear in the same order) | `"decentralized exchanges"` |
| ( ) | Groups values or search keywords together | `(decentralized OR web3) AND (search engine)` |

## Time Filter

### with protocol fields

You can specify the time range of the content you want to search for using the `earlier_than_timestamp` and `later_than_timestamp` fields.

```python
    earlier_than_timestamp: int = pydantic.Field(None, ge=0)
    later_than_timestamp: int = pydantic.Field(None, ge=0)
```

### with query_string

You have the flexibility to tailor your search based on their creation time with varying degrees of precision using absolute time format in the `query_string`.

The format for specifying a time is:

`[yyyy]-[MM]-[dd]T[HH]:[mm]:[ss]`.

In this format, `[yyyy]` is the 4-digit year, `[MM]` is the 2-digit month, `[dd]` is the 2-digit day, `[HH]` is the 2-digit hour of a 24-hour clock, `[mm]` is the minute, and `[ss]` is the second. All times are in UTC.

Examples:

- All content in 2023:

```text
created_at:(>=2023-01-01 AND <=2023-12-31)
```

- Created before 2024 Jan 1st 8AM UTC

```text
created_at:<2024-24-01T08
```

- Created after 2024 Feb 1st 10AM UTC

```text
created_at:>2012-01-01
```

## Sort

The sort parameter can be one of **relevancy**(default) and **recency**.

```python
class SortType(str, Enum):
    RELEVANCE = "relevance"
    RECENCY = "recency"
```

### Response

The miner’s response is a list of ranked documents.

```json
[
  {
    "id": "1769458523031892221",
    "text": "You're witnessing the single largest transfer of wealth in human history\n\nThis will be written about in history books\n\nYour grandchildren will never question the validity of crypto\n\nBut you psych yourself out and sell to early every single time",
    "created_at": "2024-03-17T20:19:21+00:00",
    "username": "ozarknft",
    "url": "https://x.com/ozarknft/status/1769458523031892221",
    "quote_count": 0,
    "reply_count": 19,
    "retweet_count": 1,
    "favorite_count": 76
  },
  {
    "id": "1774941290477842843",
    "text": "RT @tradewithPhoton: Solana network congestion is very high at the moment, causing delayed and failed transactions across the chain.\n\nIf yo…",
    "created_at": "2024-04-01T23:25:55+00:00",
    "username": "ozarknft",
    "url": "https://x.com/ozarknft/status/1774941290477842843",
    "quote_count": 4,
    "reply_count": 15,
    "retweet_count": 17,
    "favorite_count": 142
  }
]
```

### Structured Search Synapse Protocol

The structured search protocol between miners and validators can follow this:

```python

class StructuredSearchSynapse(bt.Synapse):
    """
    Structured search protocol representation for handling request and response communication between
    the miner and the validator.

    Attributes:
    - query_string: A string value representing the search request sent by the validator.
    - size: the maximal count size of results to return.
    - sort_type: the type of sorting to use for the search results.
    - earlier_than_timestamp: A timestamp value representing the earliest time to search for.
    - later_than_timestamp: A timestamp value representing the latest time to search for.
    - version: A `Version` object representing the version of the protocol.
    """

    query_string: Optional[str] = None
    size: int = pydantic.Field(5, ge=1, le=50)

    earlier_than_timestamp: int = pydantic.Field(None, ge=0)
    later_than_timestamp: int = pydantic.Field(None, ge=0)

    author_usernames: Optional[List[str]] = None

    sort_by: Optional[SortType] = None

    custom_fields: Optional[Dict] = None

    version: Optional[Version] = None

    results: Optional[List[Dict]] = None

    def deserialize(self) -> List[Dict]:
        return self.results
```
