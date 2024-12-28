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


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        path_config = json.load(f)

    PATH = path_config["PATH"]

    path_save = os.path.join(PATH, "data")

    features = np.load(os.path.join(path_save, "features_original.npy"))
    label = np.load(os.path.join(path_save, "label_original.npy"))

    df_features = pd.DataFrame(
        features,
        columns=[f"feature_{i}" for i in np.arange(1, np.shape(features)[1] + 1)],
    )
    df_label = pd.DataFrame(label, columns=["label"])["label"]

    df_features.to_csv(os.path.join(path_save, "features.csv"), index=False)
    df_label.to_csv(os.path.join(path_save, "label.csv"), index=False)


if __name__ == "__main__":
    main()
