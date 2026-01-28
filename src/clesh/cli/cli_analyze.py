"""
CLI entry point for CLE-SH analysis (modern mode).

Given features.csv and shap.npy, runs the analysis and prints a short summary.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from clesh.config import CLEConfig
from clesh.explainer import Explainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLE-SH: run analysis on SHAP values")
    parser.add_argument("--features", type=str, required=True, help="Path to features.csv")
    parser.add_argument("--shap", type=str, required=True, help="Path to shap.npy")

    parser.add_argument("--p-feature-selection", type=float, default=0.05)
    parser.add_argument("--p-univariate", type=float, default=0.05)
    parser.add_argument("--p-interaction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = CLEConfig(
        p_feature_selection=args.p_feature_selection,
        p_univariate=args.p_univariate,
        p_interaction=args.p_interaction,
    )

    X = pd.read_csv(Path(args.features))
    shap_values = np.load(Path(args.shap))

    explainer = Explainer(X=X, shap_values=shap_values, config=config)
    results = explainer.analyze()

    print(f"Selected features ({len(results.selected_features)}):")
    for i, name in enumerate(results.selected_features, start=1):
        ftype = results.feature_types.get(name, "unknown")
        print(f"{i}. {name} ({ftype})")

