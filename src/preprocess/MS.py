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

    df = pd.read_csv(os.path.join(path_save, "data_original.csv"))
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["Sex"] = np.where(df["Sex"] == "Female", 1, 0)
    df["Marital"] = np.where(df["Marital"] == "Married", 1, 0)
    races = df["Race"].unique()
    for race in races[:-1]:
        df[race] = np.where(df["Race"] == race, 1, 0)

    df_features = df.drop(columns=["seqn", "Race", "MetabolicSyndrome"])
    df_label = df["MetabolicSyndrome"]

    df_features.to_csv(os.path.join(path_save, "features.csv"), index=False)
    df_label.to_csv(os.path.join(path_save, "label.csv"), index=False)


if __name__ == "__main__":
    main()
