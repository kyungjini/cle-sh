"""
Visualization module for CLE-SH analysis.

This module contains all matplotlib and seaborn plotting functions.
All functions return figure and axis objects without saving files.
"""

from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import shap


def plot_feature_selection(
    feat_counts: List[int],
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot feature selection curve.
    
    Args:
        feat_counts: List of feature counts at each selection point.
        ax: Optional matplotlib axis. If None, creates new figure.
    
    Returns:
        Tuple of (figure, axis).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    
    ax.plot(np.arange(1, len(feat_counts) + 1), feat_counts, color="black", marker="o")
    ax.set_title("Feature Selection")
    ax.set_xticks(np.arange(1, len(feat_counts) + 1))
    y_feat_cnt = [
        i if i in feat_counts else " " for i in range(1, max(feat_counts) + 1, 1)
    ]
    ax.set_yticks(np.arange(1, max(feat_counts) + 1), y_feat_cnt)
    ax.grid()
    plt.tight_layout()
    
    return fig, ax


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: Optional[int] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot SHAP summary plot.
    
    Args:
        shap_values: Array of SHAP values.
        X: Feature DataFrame.
        max_display: Maximum number of features to display.
        ax: Optional matplotlib axis. If None, creates new figure.
    
    Returns:
        Tuple of (figure, axis).
    """
    if ax is None:
        shap.summary_plot(shap_values, X, show=False, max_display=max_display)
        ax = plt.gca()
        fig = plt.gcf()  # Get current figure since shap.summary_plot returns None
    else:
        shap.summary_plot(shap_values, X, show=False, max_display=max_display, ax=ax)
        fig = ax.figure
    
    return fig, ax


def plot_discrete_univariate(
    groups: List[List[float]],
    group_labels: List[str],
    feature_name: str,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot boxplot for discrete/binary univariate analysis.
    
    Args:
        groups: List of SHAP value groups.
        group_labels: Labels for each group.
        feature_name: Name of the feature.
        ax: Optional matplotlib axis. If None, creates new figure.
    
    Returns:
        Tuple of (figure, axis).
    """
    if ax is None:
        # Fixed size for PDF report (fits 2 per page)
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    
    ax.boxplot(
        groups,
        flierprops={"marker": "o", "markersize": 2},
        medianprops=dict(color="black"),
    )
    ax.set_xticklabels(group_labels)
    ax.set_xlabel(feature_name, fontsize=9)
    ax.set_ylabel("SHAP value", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    
    return fig, ax


def plot_continuous_univariate(
    x: np.ndarray,
    y: np.ndarray,
    feature_name: str,
    best_function: Optional[str] = None,
    function_params: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot scatter plot with fitted function for continuous univariate analysis.
    
    Args:
        x: Feature values.
        y: SHAP values.
        feature_name: Name of the feature.
        best_function: Name of best fitting function ("linear", "quadratic", "sigmoid").
        function_params: Parameters for the function.
        ax: Optional matplotlib axis. If None, creates new figure.
    
    Returns:
        Tuple of (figure, axis).
    """
    if ax is None:
        # Fixed size for PDF report (fits 2 per page)
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    
    ax.set_title(feature_name, fontsize=10)
    ax.set_xlabel("feature value", fontsize=9)
    ax.set_ylabel("SHAP value", fontsize=9)
    ax.tick_params(labelsize=8)
    
    # Plot scatter
    ax.scatter(x, y, alpha=0.5, s=5, facecolors="none", edgecolors="black")
    
    # Plot fitted function if available
    if best_function and best_function != "None" and function_params is not None:
        x_grid = np.linspace(np.min(x), np.max(x), 100)
        y_pred = _evaluate_function(best_function, function_params, x_grid)
        ax.plot(x_grid, y_pred, alpha=1, c="red", label=best_function)
        ax.legend()
    
    plt.tight_layout()
    
    return fig, ax


def plot_tukey_hsd(
    tukey_matrix: np.ndarray,
    group_labels: List[str],
    p_bound: float = 0.05,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Tukey-HSD heatmap.
    
    Args:
        tukey_matrix: Matrix of p-values (converted to 0/1).
        group_labels: Labels for groups.
        p_bound: P-value threshold.
        ax: Optional matplotlib axis. If None, creates new figure.
    
    Returns:
        Tuple of (figure, axis).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    
    plt.rcParams["figure.figsize"] = (5, 5)
    sns.set_theme(style="white", font_scale=1.5)
    color_tukey = ["blue", "red"]
    cmap = LinearSegmentedColormap.from_list("Custom", colors=color_tukey, N=2)
    
    mask = np.triu(np.ones_like(tukey_matrix))
    ax = sns.heatmap(
        tukey_matrix,
        lw=1,
        linecolor="white",
        cmap=cmap,
        mask=mask,
        xticklabels=group_labels,
        yticklabels=group_labels,
        ax=ax
    )
    
    if ax.collections and ax.collections[0].colorbar:
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels([f"p > {p_bound}", f"p < {p_bound}"])
    
    ax.set_title("Tukey-HSD")
    ax.set_xlabel("feature value")
    ax.set_ylabel("feature value")
    
    _, labels = plt.yticks()
    plt.setp(labels, rotation=0)
    plt.tight_layout()
    
    return fig, ax


def plot_discrete_interaction(
    groups_by_interaction: Dict[str, Dict],
    target_feature: str,
    interaction_feature: str,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot discrete interaction analysis with clear color encoding.
    
    Args:
        groups_by_interaction: Dictionary mapping interaction values to groups.
        target_feature: Name of target feature.
        interaction_feature: Name of interaction feature.
        ax: Optional matplotlib axis. If None, creates new figure.
    
    Returns:
        Tuple of (figure, axis).
    """
    # Flatten all groups for combined plot
    all_groups = []
    all_labels = []
    colorset = ["red", "blue", "green", "dodgerblue", "olive", "orange"]
    
    for i, (inter_val, data) in enumerate(groups_by_interaction.items()):
        for j, group in enumerate(data['groups']):
            all_groups.append(group)
            # Use interaction value directly (already formatted like "Age > Mean")
            all_labels.append(f"{inter_val}\n{target_feature}={data['group_labels'][j]}")
    
    if ax is None:
        # Fixed size for PDF report (fits 2 per page)
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    
    # Clean title
    ax.set_title(f"{target_feature} × {interaction_feature}", fontsize=10)
    
    # Create boxplot with colors
    bp = ax.boxplot(
        all_groups,
        flierprops={"marker": "o", "markersize": 2},
        medianprops=dict(color="black"),
        patch_artist=True
    )
    
    # Color boxes by interaction value
    inter_val_idx = 0
    for i, (inter_val, data) in enumerate(groups_by_interaction.items()):
        color = colorset[i % len(colorset)]
        for j in range(len(data['groups'])):
            box_idx = inter_val_idx + j
            if box_idx < len(bp['boxes']):
                bp['boxes'][box_idx].set_facecolor(color)
                bp['boxes'][box_idx].set_alpha(0.6)
        inter_val_idx += len(data['groups'])
    
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=7)
    ax.set_xlabel(f"{target_feature} categories", fontsize=9)
    ax.set_ylabel("SHAP value", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    
    return fig, ax


def plot_continuous_interaction(
    x: np.ndarray,
    y: np.ndarray,
    interaction_mask: np.ndarray,
    target_feature: str,
    interaction_feature: str,
    upper_function: Optional[Dict] = None,
    lower_function: Optional[Dict] = None,
    statistics: Optional[Dict] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot continuous interaction analysis with trend change indicators.
    
    Args:
        x: Feature values.
        y: SHAP values.
        interaction_mask: Boolean mask for upper/lower split.
        target_feature: Name of target feature.
        interaction_feature: Name of interaction feature.
        upper_function: Dictionary with 'function', 'params' for upper group.
        lower_function: Dictionary with 'function', 'params' for lower group.
        statistics: Optional statistics dictionary with 'trend_change', 'location_shift' flags.
        ax: Optional matplotlib axis. If None, creates new figure.
    
    Returns:
        Tuple of (figure, axis).
    """
    colorset = ["red", "blue", "green", "dodgerblue", "olive"]
    
    if ax is None:
        # Fixed size for PDF report (fits 2 per page)
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    
    # Build clean title with color encoding information
    # interaction_mask is boolean, so we show Above/Below Mean
    title = f"{target_feature} × {interaction_feature}"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("feature value", fontsize=9)
    ax.set_ylabel("SHAP value", fontsize=9)
    ax.tick_params(labelsize=8)
    
    # Plot scatter for both groups with clear labels
    ax.scatter(
        x[interaction_mask],
        y[interaction_mask],
        alpha=0.5,
        s=5,
        facecolors="none",
        edgecolors=colorset[0],
        label=f"{interaction_feature} > Mean"
    )
    ax.scatter(
        x[~interaction_mask],
        y[~interaction_mask],
        alpha=0.5,
        s=5,
        facecolors="none",
        edgecolors=colorset[1],
        label=f"{interaction_feature} ≤ Mean"
    )
    
    # Plot fitted functions (no labels - color is sufficient)
    x_grid = np.linspace(np.min(x), np.max(x), 100)
    
    if upper_function is not None:
        func_name = upper_function.get('function')
        params = upper_function.get('params')
        if func_name and func_name != "None" and params is not None and len(params) > 0:
            y_upper = _evaluate_function(func_name, params, x_grid)
            ax.plot(x_grid, y_upper, alpha=1, c=colorset[0], linewidth=2)
    
    if lower_function is not None:
        func_name = lower_function.get('function')
        params = lower_function.get('params')
        if func_name and func_name != "None" and params is not None and len(params) > 0:
            y_lower = _evaluate_function(func_name, params, x_grid)
            ax.plot(x_grid, y_lower, alpha=1, c=colorset[1], linewidth=2)
    
    ax.legend(fontsize=8)
    plt.tight_layout()
    
    return fig, ax


def _evaluate_function(func_name: str, params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate a function with given parameters, with overflow protection."""
    if params is None or len(params) == 0:
        return np.zeros_like(x, dtype=float)
    
    if func_name == "linear":
        return params[0] + params[1] * x
    elif func_name == "quadratic":
        return params[0] + params[1] * x + params[2] * x**2
    elif func_name == "sigmoid":
        L, x0, k, b = params
        # Prevent overflow in exp with aggressive clipping
        exp_arg = -k * (x - x0)
        exp_arg = np.clip(exp_arg, -700, 700)
        return L / (1 + np.exp(exp_arg)) + b
    else:
        return np.zeros_like(x, dtype=float)
