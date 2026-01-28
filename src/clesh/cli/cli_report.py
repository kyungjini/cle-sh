"""
CLI entry point for CLE-SH PDF report generation (modern mode).

This CLI is intentionally modern-only to avoid path/import ambiguity.
It expects explicit inputs (features.csv + shap.npy) and writes plots + PDF.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from clesh.config import CLEConfig
from clesh.explainer import Explainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLE-SH: generate plots and a PDF report")
    parser.add_argument("--features", type=str, required=True, help="Path to features.csv")
    parser.add_argument("--shap", type=str, required=True, help="Path to shap.npy")
    parser.add_argument("--output", type=str, required=True, help="Output directory for plots/report")
    parser.add_argument("--label", type=str, default="Dataset", help="Label shown in the report")

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

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    explainer = Explainer(X=X, shap_values=shap_values, config=config)
    results = explainer.analyze()
    explainer.save_plots(out_dir, results=results, dpi=300)
    explainer.generate_report(out_dir / "clesh_report.pdf", label=args.label, results=results, plot_dir=out_dir)

