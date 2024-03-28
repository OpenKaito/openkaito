import wandb
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Export wandb run metrics")
    parser.add_argument("--run_id", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    api = wandb.Api()

    # run is specified by <entity>/<project>/<run_id>
    run = api.run(f"subnet-openkaito/sn5-validators/{args.run_id}")

    metrics_dataframe = run.history()
    # metrics_dataframe.to_csv("metrics.csv")
    print(metrics_dataframe)
