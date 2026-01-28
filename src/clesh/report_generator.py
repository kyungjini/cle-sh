"""
PDF report generation for CLE-SH analysis.

This module generates professional PDF reports from analysis results.
"""

import logging
from pathlib import Path
from typing import Union, Optional, Dict
from datetime import datetime

from fpdf import FPDF

from clesh.analyzer import AnalysisResults
from clesh.config import CLEConfig

logger = logging.getLogger(__name__)


def _format_p_value(p_val: float, threshold: float = 0.05) -> str:
    """
    Format p-value according to the rule:
    - If p < threshold: display as "p < {threshold}"
    - If p >= threshold: display as "p = {value}" (rounded to 5 decimal places)
    """
    if p_val is None or (isinstance(p_val, float) and (p_val != p_val)):  # NaN
        return "p = 1.00000"
    if p_val < threshold:
        return f"p < {threshold}"
    return f"p = {p_val:.5f}"


def _generate_univariate_description(feat_name: str, result: Dict, config: CLEConfig) -> str:
    """Generate human-readable description for univariate analysis."""
    feat_type = result.get("feature_type", "unknown")

    if feat_type == "continuous":
        best_func = result.get("best_function", "None")
        rmse = result.get("rmse", float("inf"))

        if best_func and best_func != "None" and rmse != float("inf"):
            func_descriptions = {
                "linear": "linear",
                "quadratic": "quadratic",
                "sigmoid": "sigmoid (threshold effect)",
            }
            func_desc = func_descriptions.get(best_func, best_func)

            if best_func == "sigmoid":
                return (
                    f"The impact of '{feat_name}' follows a non-linear {func_desc} pattern, "
                    f"indicating a threshold effect where the SHAP value shifts significantly "
                    f"at certain feature values (RMSE: {rmse:.4f})."
                )
            if best_func == "quadratic":
                return (
                    f"The impact of '{feat_name}' exhibits a {func_desc} relationship, "
                    f"suggesting a non-linear response with acceleration or deceleration "
                    f"effects (RMSE: {rmse:.4f})."
                )
            return (
                f"The impact of '{feat_name}' demonstrates a {func_desc} relationship, "
                f"indicating a consistent directional effect across feature values "
                f"(RMSE: {rmse:.4f})."
            )

        return (
            f"The impact of '{feat_name}' does not exhibit a statistically significant "
            f"functional relationship, suggesting minimal systematic variation in SHAP values."
        )

    # Binary or discrete
    stats_result = result.get("statistics", {})
    p_val = stats_result.get("p_value", 1.0)
    if p_val is None or (isinstance(p_val, float) and (p_val != p_val)):  # NaN
        p_val = 1.0
    p_val_str = _format_p_value(p_val, config.p_univariate)

    if stats_result.get("significant", False):
        return (
            f"The feature '{feat_name}' shows a statistically significant variation "
            f"in its impact across different categories ({p_val_str}), "
            f"indicating that the feature's contribution to model predictions differs "
            f"substantially between groups."
        )
    return (
        f"The feature '{feat_name}' does not show statistically significant variation "
        f"in its impact across categories ({p_val_str}), suggesting relatively "
        f"uniform contribution to model predictions regardless of category."
    )


def _generate_interaction_description(
    target_feat: str, inter_feat: str, result: Dict, config: CLEConfig
) -> str:
    """Generate human-readable description for interaction analysis."""
    stats_result = result.get("statistics", {})
    target_type = result.get("target_type", "unknown")
    p_val = stats_result.get("p_value", 1.0)
    p_val_str = _format_p_value(p_val, config.p_interaction)

    if target_type == "continuous":
        trend_change = stats_result.get("trend_change", False)
        if trend_change:
            return (
                f"The interaction between '{target_feat}' and '{inter_feat}' demonstrates a "
                f"significant trend change ({p_val_str}), revealing that '{inter_feat}' "
                f"fundamentally alters the reaction pattern of '{target_feat}'."
            )
        return (
            f"The interaction between '{target_feat}' and '{inter_feat}' does not show "
            f"statistically significant trend change ({p_val_str}), indicating that "
            f"'{inter_feat}' does not substantially modify the reaction pattern of "
            f"'{target_feat}'."
        )

    distribution_shift = stats_result.get("distribution_shift", False)
    if distribution_shift:
        return (
            f"The interaction between '{target_feat}' and '{inter_feat}' shows a significant "
            f"distribution shift ({p_val_str}), indicating that '{inter_feat}' modifies "
            f"the relative impact pattern of '{target_feat}' across different categories."
        )
    return (
        f"The interaction between '{target_feat}' and '{inter_feat}' does not show "
        f"statistically significant distribution shift ({p_val_str}), indicating that "
        f"'{inter_feat}' does not substantially modify the impact pattern of "
        f"'{target_feat}' across categories."
    )


class CLEPDF(FPDF):
    """Custom PDF class for CLE-SH reports."""

    def header(self):
        self.set_font("Arial", "B", 15)
        self.cell(0, 10, "CLE-SH Analysis Report", 0, 1, "C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")


def generate_pdf_report(
    results: AnalysisResults,
    output_path: Union[str, Path],
    label: str,
    config: CLEConfig,
    plot_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Generate PDF report from analysis results."""
    pdf = CLEPDF()
    pdf.add_page()

    plot_dir_path = Path(plot_dir) if plot_dir else None

    # PAGE 1
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"CLE-SH Results for {label}", 0, 1, "C")
    pdf.ln(2)

    pdf.set_font("Arial", "", 9)
    pdf.cell(0, 4, f"Date: {datetime.today().strftime('%Y-%m-%d')}", 0, 1)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 5, "Dataset Summary", 0, 1)
    pdf.set_font("Arial", "", 8)

    # Sample/feature count
    n_samples = 0
    if results.univariate_results:
        first_result = next(iter(results.univariate_results.values()))
        if "shap_values" in first_result:
            n_samples = len(first_result["shap_values"])
        elif "groups" in first_result:
            groups = first_result["groups"]
            n_samples = sum(len(g) for g in groups) if groups else 0
    n_features = len(results.feature_types) if results.feature_types else 0

    pdf.cell(0, 3, f"Samples: {n_samples} | Features: {n_features}", 0, 1)

    n_selected = len(results.selected_features)
    pdf.cell(0, 3, f"Selected: {n_selected} features (p < {config.p_feature_selection})", 0, 1)

    type_counts = {"discrete": 0, "continuous": 0, "binary": 0}
    for feat in results.selected_features:
        feat_type = results.feature_types.get(feat, "discrete")
        type_counts[feat_type] = type_counts.get(feat_type, 0) + 1
    pdf.cell(
        0,
        3,
        f"Types: Binary={type_counts['binary']}, Discrete={type_counts['discrete']}, "
        f"Continuous={type_counts['continuous']}",
        0,
        1,
    )

    current_y = pdf.get_y()

    # SHAP summary plot
    shap_plot_added = False
    if plot_dir_path:
        possible_shap_plots = [
            plot_dir_path / "shap_summary_plot.jpg",
            plot_dir_path / "shap_summary_plot.png",
            plot_dir_path.parent / "shap_summary_plot.jpg",
        ]
        for shap_plot in possible_shap_plots:
            if shap_plot.exists():
                shap_y = current_y + 5
                if shap_y < 150:
                    pdf.image(str(shap_plot), x=10, y=shap_y, w=190, h=0)
                    shap_plot_added = True
                    break

    if not shap_plot_added:
        pdf.set_y(current_y + 5)
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 10, "[SHAP Summary Plot not available]", 0, 1, "C")

    # Univariate section
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Univariate Analysis", 0, 1)
    pdf.ln(5)

    univariate_plot_dir = plot_dir_path / "univariate_analysis" if plot_dir_path else None
    item_count = 0

    for feat_name, result in results.univariate_results.items():
        if item_count > 0 and item_count % 2 == 0:
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Univariate Analysis", 0, 1)
            pdf.ln(5)

        item_y_start = pdf.get_y() if item_count % 2 == 0 else 140
        pdf.set_y(item_y_start)
        pdf.set_font("Arial", "B", 10)
        feat_type = result.get("feature_type", "unknown")
        pdf.cell(0, 5, f"{feat_name} ({feat_type})", 0, 1)
        pdf.set_font("Arial", "", 8)

        description = _generate_univariate_description(feat_name, result, config)
        pdf.multi_cell(pdf.epw, 3, description, 0, "L")

        current_y_after_text = pdf.get_y()
        max_y_for_item = 140 if item_count % 2 == 0 else 280

        plot_added = False
        if univariate_plot_dir:
            for ext in [".jpg", ".png", ".jpeg"]:
                plot_path = univariate_plot_dir / f"{feat_name}{ext}"
                if plot_path.exists():
                    img_y = current_y_after_text + 3
                    if img_y + 60 > max_y_for_item:
                        img_y = 140 if item_count % 2 == 0 else 25
                        if item_count % 2 != 0:
                            pdf.add_page()
                    img_width = 120
                    img_x = (210 - img_width) / 2
                    pdf.image(str(plot_path), x=img_x, y=img_y, w=img_width, h=0)
                    plot_added = True
                    pdf.set_y(img_y + 60)
                    break

        if not plot_added:
            pdf.set_y(current_y_after_text + 3)
            pdf.set_font("Arial", "I", 8)
            pdf.cell(0, 3, "[Plot not available]", 0, 1, "C")

        item_count += 1

    # Interaction section
    if results.interaction_results:
        interaction_item_count = 0
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Interaction Analysis", 0, 1)
        pdf.ln(8)

        interaction_plot_dir = plot_dir_path / "interaction_analysis" if plot_dir_path else None

        for _, result in results.interaction_results.items():
            if interaction_item_count > 0 and interaction_item_count % 2 == 0:
                pdf.add_page()

            interaction_y_start = 25 if interaction_item_count % 2 == 0 else 140
            target_feat = result.get("target_feature", "unknown")
            inter_feat = result.get("interaction_feature", "unknown")

            pdf.set_y(interaction_y_start)
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 5, f"{target_feat} x {inter_feat}", 0, 1)
            pdf.set_font("Arial", "", 8)

            description = _generate_interaction_description(target_feat, inter_feat, result, config)
            pdf.multi_cell(pdf.epw, 3, description, 0, "L")

            current_y_after_text = pdf.get_y()
            max_y_for_item = 140 if interaction_item_count % 2 == 0 else 280

            plot_added = False
            if interaction_plot_dir:
                for ext in [".jpg", ".png", ".jpeg"]:
                    plot_path = interaction_plot_dir / f"{target_feat}_{inter_feat}{ext}"
                    if plot_path.exists():
                        img_y = current_y_after_text + 3
                        if img_y + 60 > max_y_for_item:
                            img_y = 140 if interaction_item_count % 2 == 0 else 25
                            if interaction_item_count % 2 != 0:
                                pdf.add_page()
                        img_width = 120
                        img_x = (210 - img_width) / 2
                        pdf.image(str(plot_path), x=img_x, y=img_y, w=img_width, h=0)
                        plot_added = True
                        pdf.set_y(img_y + 60)
                        break

            if not plot_added:
                pdf.set_y(current_y_after_text + 3)
                pdf.set_font("Arial", "I", 8)
                pdf.cell(0, 3, "[Plot not available]", 0, 1, "C")

            interaction_item_count += 1

    output_path = Path(output_path)
    pdf.output(str(output_path))
    print(f"PDF report saved to {output_path}")

