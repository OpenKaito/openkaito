import argparse
import sqlite3


def parse_args():
    parser = argparse.ArgumentParser(description="Import SN13 Data")
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="SN13 sqlite3 database file, e.g., .../data-universe/SqliteMinerStorage.sqlite",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(vars(args))

    conn = sqlite3.connect(args.db)
    c = conn.cursor()
    # source=2 means Twitter(X) data
    c.execute("SELECT * FROM DataEntity WHERE source=2;")
    print(c.fetchall())


if __name__ == "__main__":
    main()
