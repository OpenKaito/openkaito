import argparse
import json
import os
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from openkaito.evaluation.utils import tweet_url_to_id


def parse_args():
    parser = argparse.ArgumentParser(description="Import SN13 Data")
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="SN13 sqlite3 database file, e.g., ../data-universe/SqliteMinerStorage.sqlite",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="optional, batch size for importing data, default is 100",
    )
    parser.add_argument(
        "--time_bucket_ids",
        nargs="*",
        type=str,
        help="optional, a list of SN13 timeBucketId to be imported, seperated by space, e.g., 474957 474958 474959",
    )
    return parser.parse_args()


def format_time_bucket_ids(time_bucket_ids):
    return f"({', '.join(time_bucket_ids)})"


def init_twitter_index(es_client):
    index_name = "twitter"
    if not es_client.indices.exists(index=index_name):
        print("creating index...", index_name)
        es_client.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "id": {"type": "long"},
                        "text": {"type": "text"},
                        "created_at": {"type": "date"},
                        "username": {"type": "keyword"},
                        "url": {"type": "text"},
                        "quote_count": {"type": "long"},
                        "reply_count": {"type": "long"},
                        "retweet_count": {"type": "long"},
                        "favorite_count": {"type": "long"},
                    }
                }
            },
        )
        print(f"index {index_name} created")


def data_entity_to_twitter_doc(data_entity):
    if isinstance(data_entity, sqlite3.Row):
        data_entity = dict(data_entity)
    content = json.loads(data_entity["content"])
    return {
        "id": tweet_url_to_id(data_entity["uri"]),
        "text": content["text"],
        "created_at": datetime.fromisoformat(data_entity["datetime"]).isoformat(),
        "username": content["username"].lstrip("@"),  # remove leading @
        "url": data_entity["uri"],
        "quote_count": None,
        "reply_count": None,
        "retweet_count": None,
        "favorite_count": None,
    }


def main():
    args = parse_args()
    print(vars(args))
    load_dotenv()

    batch_size = args.batch_size

    es_client = Elasticsearch(
        os.environ["ELASTICSEARCH_HOST"],
        basic_auth=(
            os.environ["ELASTICSEARCH_USERNAME"],
            os.environ["ELASTICSEARCH_PASSWORD"],
        ),
        verify_certs=False,
        ssl_show_warn=False,
    )

    init_twitter_index(es_client)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # source=2 means Twitter(X) data
    sql_stmt = "SELECT * FROM DataEntity WHERE source=2"
    if args.time_bucket_ids and len(args.time_bucket_ids) > 0:
        sql_stmt += (
            f" AND timeBucketId IN {format_time_bucket_ids(args.time_bucket_ids)}"
        )
    print(sql_stmt)
    c.execute(sql_stmt)

    import_count = 0
    batch_rows = c.fetchmany(batch_size)
    while batch_rows:
        print(f"importing {len(batch_rows)} rows...")
        import_count += len(batch_rows)
        processed_docs = [data_entity_to_twitter_doc(row) for row in batch_rows]
        # print(processed_docs)
        bulk_body = []
        for doc in processed_docs:
            bulk_body.append(
                {
                    "update": {
                        "_index": "twitter",
                        "_id": doc["id"],
                    }
                }
            )
            bulk_body.append(
                {
                    "doc": doc,
                    "doc_as_upsert": True,
                }
            )

        r = es_client.bulk(
            body=bulk_body,
            refresh=True,
        )
        if not r.get("errors"):
            print(f"bulk update {len(processed_docs)} succeeded")
        else:
            print("bulk update failed: ", r)

        batch_rows = c.fetchmany(batch_size)

    print(f"imported {import_count} rows")


if __name__ == "__main__":
    main()
