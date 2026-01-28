"""
High-level Explainer class for CLE-SH analysis.

This class orchestrates analysis, visualization, and I/O operations.
It supports both path-based and in-memory APIs.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from clesh.config import CLEConfig
from clesh.analyzer import Analyzer, AnalysisResults
from clesh.visuals import (
    plot_shap_summary,
    plot_discrete_univariate,
    plot_continuous_univariate,
    plot_discrete_interaction,
    plot_continuous_interaction,
)

logger = logging.getLogger(__name__)


class Explainer:
    """
    High-level coordinator for CLE-SH analysis.
    
    Supports two initialization modes:
    1. Path-based mode: Explainer(path="...") - loads from file structure
    2. In-memory mode: Explainer(X=df, shap_values=array, config=config) - in-memory
    
    Example (Path-based):
        >>> explainer = Explainer(path="./my_project")
        >>> results = explainer.analyze()
    
    Example (Modern):
        >>> import pandas as pd
        >>> import numpy as np
        >>> X = pd.read_csv("features.csv")
        >>> shap_values = np.load("shap.npy")
        >>> explainer = Explainer(X=X, shap_values=shap_values)
        >>> results = explainer.analyze()
        >>> explainer.save_plots("./output")
    """
    
    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        X: Optional[pd.DataFrame] = None,
        shap_values: Optional[np.ndarray] = None,
        config: Optional[CLEConfig] = None
    ):
        """
        Initialize the Explainer.
        
        Args:
            path: Path to project directory (path-based mode). Must contain 'data/features.csv'
                and model directories with 'shap.npy' files.
            X: Feature DataFrame (in-memory mode). Required if path is None.
            shap_values: SHAP values array (in-memory mode). Required if path is None.
            config: CLEConfig instance. If None, uses default configuration.
        
        Raises:
            ValueError: If neither path nor (X, shap_values) are provided, or if
                dimensions don't match.
        """
        self.config = config if config is not None else CLEConfig()
        self._legacy_mode = path is not None
        self._analyzer: Optional[Analyzer] = None
        self._results: Optional[AnalysisResults] = None
        
        if self._legacy_mode:
            self.path = Path(path).resolve()
            self._validate_path()
            logger.info(f"Initialized in path-based mode with path: {self.path}")
        else:
            if X is None or shap_values is None:
                raise ValueError(
                    "Either 'path' or both 'X' and 'shap_values' must be provided"
                )
            self.X = X.copy()
            self.shap_values = shap_values.copy()
            self._analyzer = Analyzer(self.X, self.shap_values, self.config)
            logger.info(
                f"Initialized in modern mode with {X.shape[0]} samples and {X.shape[1]} features"
            )
    
    def _validate_path(self):
        """Validate path-based project structure."""
        if not self.path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.path}")
        
        path_data = self.path / "data"
        if not path_data.exists():
            raise FileNotFoundError(
                f"Data directory not found: {path_data}. "
                "Please ensure your project directory contains a 'data' subdirectory."
            )
    
    def analyze(self, model_name: Optional[str] = None) -> AnalysisResults:
        """
        Perform comprehensive CLE-SH analysis.
        
        Args:
            model_name: For path-based mode, name of model directory to analyze.
                If None, analyzes all models. Ignored in in-memory mode.
        
        Returns:
            AnalysisResults dataclass containing all analysis results.
        """
        if self._legacy_mode:
            return self._analyze_legacy(model_name)
        else:
            if self._analyzer is None:
                raise RuntimeError("Analyzer not initialized")
            self._results = self._analyzer.analyze()
            return self._results
    
    def _analyze_legacy(self, model_name: Optional[str] = None) -> Dict[str, AnalysisResults]:
        """
        Analyze in path-based mode (loads from file structure).
        
        Returns:
            Dictionary mapping model names to AnalysisResults.
        """
        path_data = self.path / "data"
        X = pd.read_csv(path_data / "features.csv")
        X = X.drop(X.columns[0], axis=1)
        
        # Find model directories
        if model_name:
            model_dirs = [model_name]
        else:
            model_dirs = [
                d.name
                for d in self.path.iterdir()
                if d.is_dir() and d.name != "data"
            ]
        
        all_results = {}
        
        for model in model_dirs:
            logger.info(f">> Processing Model: {model}")
            model_path = self.path / model
            shap_path = model_path / "shap.npy"
            
            if not shap_path.exists():
                logger.warning(f"!! No shap.npy found in {model_path}. Skipping...")
                continue
            
            shap_values = np.load(shap_path)
            analyzer = Analyzer(X, shap_values, self.config)
            results = analyzer.analyze()
            all_results[model] = results
            
            # Save results and plots
            output_dir = model_path / "clesh_results"
            temp_explainer = Explainer(X=X, shap_values=shap_values, config=self.config)
            temp_explainer._results = results
            temp_explainer.save_plots(output_dir)
        
        return all_results
    
    def univariate(self, feature_name: str) -> Dict:
        """
        Perform univariate analysis on a single feature.
        
        Args:
            feature_name: Name of the feature to analyze.
        
        Returns:
            Dictionary containing univariate analysis results.
        """
        if self._analyzer is None:
            raise RuntimeError("Analyzer not initialized. Use modern mode or call analyze() first.")
        
        return self._analyzer.univariate(feature_name)
    
    def inter(self, target_feature: str, interaction_feature: str) -> Dict:
        """
        Perform interaction analysis between two features.
        
        Args:
            target_feature: Name of the target feature.
            interaction_feature: Name of the interaction feature.
        
        Returns:
            Dictionary containing interaction analysis results.
        """
        if self._analyzer is None:
            raise RuntimeError("Analyzer not initialized. Use modern mode or call analyze() first.")
        
        return self._analyzer.inter(target_feature, interaction_feature)
    
    def save_plots(
        self,
        output_dir: Union[str, Path],
        results: Optional[AnalysisResults] = None,
        dpi: int = 300
    ) -> None:
        """
        Save all analysis plots to disk.
        
        Args:
            output_dir: Directory to save plots.
            results: AnalysisResults instance. If None, uses stored results.
            dpi: Resolution for saved plots.
        """
        if results is None:
            if self._results is None:
                raise ValueError("No results available. Call analyze() first.")
            results = self._results
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving plots to {output_dir}")
        
        # SHAP summary plot
        if self._analyzer:
            fig, ax = plot_shap_summary(
                self._analyzer.shap_values,
                self._analyzer.X,
                max_display=len(results.selected_features)
            )
            fig.savefig(output_dir / "shap_summary_plot.jpg", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        
        # Univariate plots
        univariate_dir = output_dir / "univariate_analysis"
        univariate_dir.mkdir(exist_ok=True)
        
        for feat_name, result in results.univariate_results.items():
            feat_type = result['feature_type']
            
            if feat_type == "continuous":
                fig, ax = plot_continuous_univariate(
                    result['x_clean'],
                    result['y_clean'],
                    feat_name,
                    result.get('best_function'),
                    result.get('function_params')
                )
            else:
                fig, ax = plot_discrete_univariate(
                    result['groups'],
                    result['group_labels'],
                    feat_name
                )
            
            fig.savefig(univariate_dir / f"{feat_name}.jpg", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        
        # Interaction plots (ALWAYS generate plots regardless of significance)
        interaction_dir = output_dir / "interaction_analysis"
        interaction_dir.mkdir(exist_ok=True)
        
        for key, result in results.interaction_results.items():
            target_feat = result.get('target_feature')
            inter_feat = result.get('interaction_feature')
            
            if not target_feat or not inter_feat:
                continue
            
            # Re-compute interaction to ensure we have fresh data
            inter_result = self.inter(target_feat, inter_feat)
            target_type = inter_result.get('target_type')
            inter_type = inter_result.get('interaction_type')
            
            if target_type == "continuous":
                # For continuous target, we can treat binary/discrete interaction as continuous
                # by splitting at the mean (upper/lower groups)
                if inter_type == "continuous" or (inter_type in ["discrete", "binary"] and len(np.unique(self._analyzer.X[inter_feat].values)) == 2):
                    # Both continuous, or binary treated as continuous
                    target_idx = list(self._analyzer.X.columns).index(target_feat)
                    inter_idx = list(self._analyzer.X.columns).index(inter_feat)
                    
                    x = self._analyzer.X[target_feat].values
                    y = self._analyzer.shap_values[:, target_idx]
                    inter_values = self._analyzer.X[inter_feat].values
                    avg_inter = np.mean(inter_values)
                    inter_mask = inter_values > avg_inter
                    
                    # Get function data - if inter_type was discrete, recompute with continuous treatment
                    if inter_type in ["discrete", "binary"]:
                        func_upper, params_upper, rmse_upper = self._analyzer._fit_best_function(
                            x[inter_mask], y[inter_mask]
                        )
                        func_lower, params_lower, rmse_lower = self._analyzer._fit_best_function(
                            x[~inter_mask], y[~inter_mask]
                        )
                        upper_func = {'function': func_upper, 'params': params_upper, 'rmse': rmse_upper}
                        lower_func = {'function': func_lower, 'params': params_lower, 'rmse': rmse_lower}
                    else:
                        upper_func = inter_result.get('upper_function')
                        lower_func = inter_result.get('lower_function')
                    
                    fig, ax = plot_continuous_interaction(
                        x, y, inter_mask, target_feat, inter_feat,
                        upper_func,
                        lower_func,
                        statistics=inter_result.get('statistics')
                    )
                else:
                    # Continuous target, discrete interaction
                    fig, ax = plt.subplots(figsize=(6, 4))
                    target_idx = list(self._analyzer.X.columns).index(target_feat)
                    x = self._analyzer.X[target_feat].values
                    y = self._analyzer.shap_values[:, target_idx]
                    inter_values = self._analyzer.X[inter_feat].values
                    
                    unique_inter_vals = sorted(np.unique(inter_values))
                    colorset = ["red", "blue", "green", "dodgerblue", "olive", "orange"]
                    interaction_functions = inter_result.get('interaction_functions', {})
                    
                    for i, inter_val in enumerate(unique_inter_vals):
                        mask = inter_values == inter_val
                        x_subset = x[mask]
                        y_subset = y[mask]
                        
                        if len(x_subset) == 0:
                            continue
                        
                        ax.scatter(x_subset, y_subset, alpha=0.5, s=5,
                                 facecolors="none", edgecolors=colorset[i % len(colorset)],
                                 label=str(inter_val))
                        
                        # Plot fitted function if available
                        inter_val_str = str(inter_val)
                        if inter_val_str in interaction_functions:
                            func_data = interaction_functions[inter_val_str]
                            func_name = func_data.get('function')
                            params = func_data.get('params')
                            
                            if func_name and func_name != "None" and params is not None and len(params) > 0:
                                x_min, x_max = np.min(x_subset), np.max(x_subset)
                                if x_max > x_min:
                                    x_grid = np.linspace(x_min, x_max, 100)
                                    
                                    if func_name == "linear":
                                        y_pred = params[0] + params[1] * x_grid
                                    elif func_name == "quadratic":
                                        y_pred = params[0] + params[1] * x_grid + params[2] * x_grid**2
                                    elif func_name == "sigmoid":
                                        L, x0, k, b = params
                                        exp_arg = np.clip(-k * (x_grid - x0), -700, 700)
                                        y_pred = L / (1 + np.exp(exp_arg)) + b
                                    else:
                                        y_pred = np.zeros_like(x_grid)
                                    
                                    ax.plot(x_grid, y_pred, c=colorset[i % len(colorset)], 
                                           linewidth=2)
                    
                    ax.set_title(f"{target_feat} × {inter_feat}", fontsize=10)
                    ax.set_xlabel("feature value", fontsize=9)
                    ax.set_ylabel("SHAP value", fontsize=9)
                    ax.tick_params(labelsize=8)
                    if len(unique_inter_vals) > 0:
                        ax.legend(fontsize=8)
                    plt.tight_layout()
            else:
                # Discrete target
                if 'interaction_groups' in inter_result:
                    fig, ax = plot_discrete_interaction(
                        inter_result['interaction_groups'],
                        target_feat,
                        inter_feat
                    )
                else:
                    # Upper/lower split - use plot_discrete_interaction format
                    upper_groups = inter_result.get('upper_groups', {})
                    lower_groups = inter_result.get('lower_groups', {})
                    
                    # Format as groups_by_interaction for plot_discrete_interaction
                    groups_by_interaction = {
                        f"{inter_feat} > Mean": upper_groups,
                        f"{inter_feat} ≤ Mean": lower_groups
                    }
                    
                    fig, ax = plot_discrete_interaction(
                        groups_by_interaction,
                        target_feat,
                        inter_feat
                    )
            
            plot_path = interaction_dir / f"{target_feat}_{inter_feat}.jpg"
            fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
    
    def generate_report(
        self,
        output_path: Union[str, Path],
        label: str = "Dataset",
        results: Optional[AnalysisResults] = None,
        plot_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Generate PDF report from analysis results.
        
        Args:
            output_path: Path to save PDF report.
            label: Label for the dataset.
            results: AnalysisResults instance. If None, uses stored results.
            plot_dir: Optional directory containing plot images. If None, tries to find plots
                in the same directory as the output PDF.
        """
        from clesh.report_generator import generate_pdf_report
        
        if results is None:
            if self._results is None:
                raise ValueError("No results available. Call analyze() first.")
            results = self._results
        
        # If plot_dir not provided, try to find it relative to output_path
        if plot_dir is None:
            # Resolve to avoid cwd-dependent relative path issues (common in notebooks)
            output_path_obj = Path(output_path).expanduser().resolve()
            # Check if plots are in the same directory or a subdirectory
            potential_dirs = [
                output_path_obj.parent,  # Same directory as PDF
                output_path_obj.parent / "clesh_results",  # Common subdirectory
            ]
            for pd in potential_dirs:
                if (pd / "univariate_analysis").exists():
                    plot_dir = pd
                    break
        
        generate_pdf_report(results, output_path, label, self.config, plot_dir=plot_dir)
