import os, json, argparse
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config",
    )
    return parser.parse_args()


# datasets = ["IBD", "DR", "BC"]
def main():
    args = parse_args()

    with open(args.config, "r") as f:
        path_config = json.load(f)

    PATH = path_config["PATH"]

    path_save = os.path.join(PATH, "data")

    df = pd.read_csv(os.path.join(path_save, "data_original.csv"))

    df["age"] = df["age"] // 10

    df_label = df["DEATH_EVENT"]
    df_features = df.drop(columns=["DEATH_EVENT"])

    df_features.to_csv(os.path.join(path_save, "features_learning.csv"), index=False)
    df_label.to_csv(os.path.join(path_save, "label_learning.csv"), index=False)


if __name__ == "__main__":
    main()
