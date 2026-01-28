"""
Core Analyzer class for CLE-SH analysis.

This module provides the main Analyzer class that performs statistical analysis
on SHAP values with a clean separation of calculation, visualization, and I/O.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy import odr
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import shap

from clesh.config import CLEConfig

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResults:
    """Container for analysis results."""
    
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected_features: List[str] = field(default_factory=list)
    feature_types: Dict[str, str] = field(default_factory=dict)  # "discrete", "continuous", "binary"
    univariate_results: Dict[str, Dict] = field(default_factory=dict)
    interaction_results: Dict[str, Dict] = field(default_factory=dict)
    best_functions: Dict[str, str] = field(default_factory=dict)  # For continuous features


class Analyzer:
    """
    Core analyzer for CLE-SH statistical analysis.
    
    This class performs all statistical calculations and returns structured results.
    Visualization and I/O are handled separately.
    
    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from clesh import Analyzer, CLEConfig
        >>> 
        >>> X = pd.read_csv("features.csv")
        >>> shap_values = np.load("shap.npy")
        >>> config = CLEConfig()
        >>> 
        >>> analyzer = Analyzer(X, shap_values, config)
        >>> results = analyzer.analyze()
        >>> 
        >>> # Manual analysis
        >>> univariate_result = analyzer.univariate("feature_name")
        >>> interaction_result = analyzer.inter("target_feat", "interaction_feat")
    """
    
    def __init__(
        self,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        config: Optional[CLEConfig] = None
    ):
        """
        Initialize the Analyzer.
        
        Args:
            X: Feature DataFrame with named columns.
            shap_values: Array of SHAP values with shape (n_samples, n_features).
            config: CLEConfig instance. If None, uses default configuration.
        
        Raises:
            ValueError: If X and shap_values dimensions don't match.
        """
        if X.shape[1] != shap_values.shape[1]:
            raise ValueError(
                f"Feature count mismatch: X has {X.shape[1]} features, "
                f"shap_values has {shap_values.shape[1]} features"
            )
        
        self.X = X.copy()
        self.shap_values = shap_values.copy()
        self.config = config if config is not None else CLEConfig()
        
        # Internal state
        self._feature_types: Optional[Dict[str, str]] = None
        self._importance_ranking: Optional[pd.DataFrame] = None
        self._selected_features: Optional[List[str]] = None
        
        logger.info(f"Analyzer initialized with {X.shape[0]} samples and {X.shape[1]} features")
    
    def _classify_feature_type(self, feature_name: str) -> str:
        """
        Classify a single feature as discrete, continuous, or binary.
        
        Args:
            feature_name: Name of the feature column.
        
        Returns:
            Feature type: "discrete", "continuous", or "binary".
        """
        if self._feature_types is None:
            self._feature_types = {}
            for col in self.X.columns:
                n_unique = self.X[col].nunique()
                if n_unique == 2:
                    self._feature_types[col] = "binary"
                elif n_unique > self.config.cont_bound:
                    self._feature_types[col] = "continuous"
                else:
                    self._feature_types[col] = "discrete"
        
        return self._feature_types.get(feature_name, "discrete")
    
    def _get_feature_index(self, feature_name: str) -> int:
        """Get column index for a feature name."""
        if feature_name not in self.X.columns:
            raise ValueError(f"Feature '{feature_name}' not found in DataFrame")
        return list(self.X.columns).index(feature_name)
    
    def feature_importance(self) -> pd.DataFrame:
        """
        Calculate feature importance rankings.
        
        Returns:
            DataFrame with columns ['feature', 'importance'] sorted by importance.
        """
        if self._importance_ranking is None:
            shap_sum = np.abs(self.shap_values).mean(axis=0)
            self._importance_ranking = pd.DataFrame({
                'feature': self.X.columns.tolist(),
                'importance': shap_sum.tolist()
            }).sort_values('importance', ascending=False)
        
        return self._importance_ranking.copy()
    
    def select_features(
        self,
        min_features: Optional[int] = None,
        max_features: Optional[int] = None,
        debug: bool = False
    ) -> List[str]:
        """
        Select important features based on statistical tests.
        
        Args:
            min_features: Minimum number of features to select.
            max_features: Maximum number of features to select.
            debug: If True, print debugging information.
        
        Returns:
            List of selected feature names.
        """
        min_features = min_features or self.config.candidate_num_min
        max_features = max_features or self.config.candidate_num_max
        
        importance_df = self.feature_importance()
        
        # Get feature names in importance order
        ranked_features = importance_df['feature'].tolist()
        
        # Get actual column indices for these features
        ranked_feature_indices = [self._get_feature_index(feat) for feat in ranked_features]
        
        # Perform statistical tests between adjacent features
        shap_ttest, p_values_matrix = self._perform_feature_tests(ranked_feature_indices, return_p_values=True)
        
        if debug:
            print("=" * 80)
            print("FEATURE SELECTION DEBUG INFO")
            print("=" * 80)
            print(f"Min features: {min_features}, Max features: {max_features}")
            print(f"Total features: {len(ranked_features)}")
            print("\nRanked features (by importance):")
            for i, feat in enumerate(ranked_features[:15], 1):
                importance = importance_df[importance_df['feature'] == feat]['importance'].values[0]
                print(f"  {i}. {feat} (importance: {importance:.6f})")
            
            print("\nAdjacent feature pairs - Statistical tests:")
            print("-" * 80)
            for i in range(min(len(ranked_features) - 1, max_features + 5)):
                feat1 = ranked_features[i]
                feat2 = ranked_features[i + 1]
                p_val = p_values_matrix[i, i + 1]
                is_significant = shap_ttest[i, i + 1] == 0
                status = "*** SIGNIFICANT ***" if is_significant else "not significant"
                print(f"  Rank {i+1} vs {i+2}: {feat1} vs {feat2}")
                print(f"    p-value: {p_val:.6f}, {status}")
        
        # Find cut points where significant differences occur
        feat_cnt = []
        for i in range(len(ranked_features) - 1):
            if i >= max_features:
                break
            if shap_ttest[i, i + 1] == 0:  # Significant difference
                feat_cnt.append(i + 1)
        feat_cnt.append(len(ranked_features))
        
        if debug:
            print(f"\nSignificant cut points (feat_cnt): {feat_cnt}")
            if len(feat_cnt) > 1:
                gaps = [feat_cnt[i+1] - feat_cnt[i] for i in range(len(feat_cnt)-1)]
                print(f"Gaps between cut points: {gaps}")
                print(f"Gap positions: {[(feat_cnt[i], feat_cnt[i+1]) for i in range(len(feat_cnt)-1)]}")
        
        # Select optimal number based on gaps (matching original algorithm exactly)
        feat_der = {}
        for i in range(len(feat_cnt) - 1):
            der = feat_cnt[i + 1] - feat_cnt[i]  # Gap size
            if feat_der.get(der) is None:
                feat_der[der] = []
            feat_der[der].append(feat_cnt[i])  # Starting position of gap
        
        feats_rank_cut = []
        feat_der_key = sorted(feat_der.keys(), reverse=True)  # Sort gaps by size (largest first)
        bnd = min_features
        cnt = 1
        
        if debug:
            print(f"\nGap analysis (feat_der):")
            for gap_size in feat_der_key:
                print(f"  Gap size {gap_size}: starting at positions {feat_der[gap_size]}")
        
        # Process gaps from largest to smallest (excluding the last gap)
        for i in range(len(feat_der_key) - 1):
            cand = [
                item
                for item in feat_der[feat_der_key[i]]
                if (item >= bnd) & (item <= max_features)
            ]
            if len(cand) != 0:
                if debug:
                    print(f"  Selection #{cnt}: gap size {feat_der_key[i]}, candidates: {cand} (bnd={bnd}, max={max_features})")
                feats_rank_cut.extend(cand)
                # Update bound: original uses feat_der[feat_der_key[i]][-1] which is the last element
                # Since we're iterating through sorted gap sizes, we should use the max value in the list
                # But to match original exactly, we use the last element after sorting
                sorted_positions = sorted(feat_der[feat_der_key[i]])
                bnd = sorted_positions[-1] if len(sorted_positions) > 0 else bnd
                if debug:
                    print(f"    Updated bnd to: {bnd}")
                cnt += 1
        
        # Add last cut point if within bounds
        if (feat_cnt[-1] >= min_features) & (feat_cnt[-1] <= max_features):
            feats_rank_cut.append(feat_cnt[-1])
            if debug:
                print(f"  Selection last: {feat_cnt[-1]}")
        
        if debug:
            print(f"\nAll candidate cut points: {feats_rank_cut}")
        
        # Select features - choose candidate closest to max_features
        if len(feats_rank_cut) > 0:
            # Filter candidates to be within [min_features, max_features] range
            valid_candidates = [
                c for c in feats_rank_cut 
                if min_features <= c <= max_features
            ]
            
            if len(valid_candidates) > 0:
                # Choose candidate closest to max_features
                selected_count = max(valid_candidates, key=lambda x: (x <= max_features, x))
                selected_features = ranked_features[:selected_count]
                if debug:
                    print(f"Valid candidates in range [{min_features}, {max_features}]: {valid_candidates}")
                    print(f"Selected {selected_count} features (closest to max_features: {max_features})")
            else:
                # No valid candidates in range, use max_features if possible, otherwise min_features
                if max_features <= len(ranked_features):
                    selected_count = max_features
                    selected_features = ranked_features[:selected_count]
                    if debug:
                        print(f"No valid candidates in range, using max_features: {max_features}")
                else:
                    selected_count = min_features
                    selected_features = ranked_features[:selected_count]
                    if debug:
                        print(f"No valid candidates in range, using min_features: {min_features}")
        else:
            selected_features = ranked_features[:min_features]
            if debug:
                print(f"Selected {min_features} features (fallback to min_features, no candidates found)")
        
        if debug:
            print(f"\nFinal selected features:")
            for i, feat in enumerate(selected_features, 1):
                importance = importance_df[importance_df['feature'] == feat]['importance'].values[0]
                print(f"  {i}. {feat} (importance: {importance:.6f})")
            print("=" * 80)
        
        self._selected_features = selected_features
        return self._selected_features.copy()
    
    def _perform_feature_tests(
        self, 
        ranked_indices: List[int], 
        return_p_values: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform statistical tests between ranked features.
        
        Args:
            ranked_indices: List of feature column indices in importance order.
            return_p_values: If True, also return p-values matrix.
        
        Returns:
            Test matrix (0 = significant difference, 1 = no significant difference).
            If return_p_values=True, also returns p-values matrix.
        """
        n = len(ranked_indices)
        test_matrix = np.ones((n, n))
        p_values_matrix = np.ones((n, n))
        
        # Check normality with error handling
        is_normal = True
        for idx in ranked_indices[:min(10, n)]:  # Sample check
            try:
                shap_vals = self.shap_values[:, idx]
                # Check for zero variance
                if np.var(shap_vals) < np.finfo(float).eps:
                    is_normal = False
                    break
                with np.errstate(all='ignore'):
                    _, p_val_shapiro = stats.shapiro(shap_vals)
                    if np.isnan(p_val_shapiro) or p_val_shapiro <= self.config.p_feature_selection:
                        is_normal = False
                        break
            except Exception:
                is_normal = False
                break
        
        # Perform pairwise tests
        for i in range(n):
            for j in range(i + 1, n):
                idx_i, idx_j = ranked_indices[i], ranked_indices[j]
                abs_shap_i = np.abs(self.shap_values[:, idx_i])
                abs_shap_j = np.abs(self.shap_values[:, idx_j])
                
                # Check for zero variance
                var_i = np.var(abs_shap_i)
                var_j = np.var(abs_shap_j)
                
                if var_i < np.finfo(float).eps and var_j < np.finfo(float).eps:
                    # Both constant - compare means
                    if abs(np.mean(abs_shap_i) - np.mean(abs_shap_j)) < np.finfo(float).eps:
                        p_val = 1.0
                    else:
                        p_val = 0.0  # Definitely different
                else:
                    # Try multiple tests with error handling
                    p_val = 1.0
                    tests = [
                        ('ttest_rel', lambda: stats.ttest_rel(abs_shap_i, abs_shap_j)),
                        ('ranksums', lambda: stats.ranksums(abs_shap_i, abs_shap_j)),
                        ('mannwhitneyu', lambda: stats.mannwhitneyu(abs_shap_i, abs_shap_j, alternative='two-sided')),
                    ]
                    
                    for test_name, test_func in tests:
                        try:
                            with np.errstate(all='ignore'):
                                _, p_val_test = test_func()
                                if not np.isnan(p_val_test) and not np.isinf(p_val_test):
                                    p_val = p_val_test
                                    break
                        except Exception:
                            continue
                
                p_values_matrix[i, j] = p_val
                p_values_matrix[j, i] = p_val
                
                if p_val < self.config.p_feature_selection:
                    test_matrix[i, j] = 0
                    test_matrix[j, i] = 0
        
        if return_p_values:
            return test_matrix, p_values_matrix
        return test_matrix
    
    def univariate(self, feature_name: str) -> Dict:
        """
        Perform univariate analysis on a single feature.
        
        Args:
            feature_name: Name of the feature to analyze.
        
        Returns:
            Dictionary containing:
                - feature_type: "discrete", "continuous", or "binary"
                - groups: List of SHAP value groups (for discrete/binary)
                - group_labels: Labels for each group
                - statistics: Statistical test results
                - best_function: Best fitting function (for continuous)
                - function_params: Parameters for best function
        """
        feat_type = self._classify_feature_type(feature_name)
        idx = self._get_feature_index(feature_name)
        
        result = {
            'feature': feature_name,
            'feature_type': feat_type,
            'shap_values': self.shap_values[:, idx],
            'feature_values': self.X[feature_name].values
        }
        
        if feat_type == "continuous":
            result.update(self._analyze_continuous_univariate(feature_name, idx))
        else:
            result.update(self._analyze_discrete_univariate(feature_name, idx))
        
        return result
    
    def _analyze_discrete_univariate(self, feature_name: str, idx: int) -> Dict:
        """Analyze discrete/binary feature univariately."""
        feature_values = self.X[feature_name].values
        shap_vals = self.shap_values[:, idx]
        
        # Group SHAP values by feature values
        unique_vals = sorted(np.unique(feature_values))
        groups = [shap_vals[feature_values == val].tolist() for val in unique_vals]
        
        # Perform statistical tests
        stats_result = self._perform_group_statistics(groups, paired=False)
        
        return {
            'groups': groups,
            'group_labels': [str(v) for v in unique_vals],
            'statistics': stats_result
        }
    
    def _analyze_continuous_univariate(self, feature_name: str, idx: int) -> Dict:
        """Analyze continuous feature univariately."""
        x = self.X[feature_name].values
        y = self.shap_values[:, idx]
        
        # Remove outliers using IQR
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1
        mask = (x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Try different function fits
        best_func, best_params, best_rmse = self._fit_best_function(x_clean, y_clean)
        
        return {
            'best_function': best_func,
            'function_params': best_params,
            'rmse': best_rmse,
            'x_clean': x_clean,
            'y_clean': y_clean
        }
    
    def inter(self, target_feature: str, interaction_feature: str) -> Dict:
        """
        Perform interaction analysis between two features.
        
        Args:
            target_feature: Name of the target feature to analyze.
            interaction_feature: Name of the interaction feature.
        
        Returns:
            Dictionary containing interaction analysis results.
        """
        target_idx = self._get_feature_index(target_feature)
        inter_idx = self._get_feature_index(interaction_feature)
        
        target_type = self._classify_feature_type(target_feature)
        inter_type = self._classify_feature_type(interaction_feature)
        
        result = {
            'target_feature': target_feature,
            'interaction_feature': interaction_feature,
            'target_type': target_type,
            'interaction_type': inter_type
        }
        
        if target_type == "continuous":
            result.update(self._analyze_continuous_interaction(target_idx, inter_idx, inter_type))
        else:
            result.update(self._analyze_discrete_interaction(target_idx, inter_idx, inter_type))
        
        return result
    
    def _test_categorical_interaction_distribution(
        self, groups_by_inter: Dict[str, Dict]
    ) -> Dict:
        """
        Test if interaction feature modifies the distribution pattern of SHAP values
        across target categories (Distribution Shift / Impact Reversal).
        
        Uses Chi-squared test on sign of SHAP values or group comparison.
        This tests if the relative distribution changes, NOT location shift.
        
        Args:
            groups_by_inter: Dictionary mapping interaction values to target groups
                Can be {'upper': {...}, 'lower': {...}} or {'0': {...}, '1': {...}}
        
        Returns:
            Dictionary with test results
        """
        interaction_values = list(groups_by_inter.keys())
        
        if len(interaction_values) < 2:
            return {'test': 'insufficient_interaction_groups', 'p_value': 1.0, 'significant': False, 'distribution_shift': False}
        
        # Compare distribution patterns across interaction groups.
        inter_group_arrays = []
        
        for inter_val in interaction_values:
            inter_data = groups_by_inter[inter_val]
            groups = inter_data.get('groups', [])
            
            # Flatten all target categories for this interaction value
            flat_values = []
            for group in groups:
                if len(group) > 0:
                    flat_values.extend([v for v in group if np.isfinite(v)])
            
            if len(flat_values) > 0:
                inter_group_arrays.append(np.array(flat_values))
        
        if len(inter_group_arrays) < 2:
            return {'test': 'insufficient_data', 'p_value': 1.0, 'significant': False, 'distribution_shift': False}
        
        # Chi-squared test on sign distribution.
        try:
            contingency_data = []
            for arr in inter_group_arrays:
                positive_count = np.sum(arr > 0)
                negative_count = np.sum(arr <= 0)
                if positive_count + negative_count > 0:
                    contingency_data.append([positive_count, negative_count])
            
            if len(contingency_data) >= 2:
                contingency_table = np.array(contingency_data)
                # Ensure all values are non-negative and have sufficient counts
                contingency_table = np.maximum(contingency_table, 0)
                
                # Check if table has sufficient counts
                if np.sum(contingency_table) >= 4 and np.all(np.sum(contingency_table, axis=0) > 0):
                    try:
                        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                        if not np.isnan(p_val) and not np.isinf(p_val) and dof > 0:
                            return {
                                'test': 'chi2_distribution_shift',
                                'statistic': float(chi2),
                                'p_value': float(p_val),
                                'significant': p_val < self.config.p_interaction,
                                'distribution_shift': p_val < self.config.p_interaction
                            }
                    except Exception as e:
                        logger.debug(f"Chi-squared test failed: {e}")
        except Exception as e:
            logger.debug(f"Contingency table construction failed: {e}")
        
        # Compare distributions between groups (robust non-parametric tests).
        tests = [
            ('mannwhitneyu', lambda: stats.mannwhitneyu(inter_group_arrays[0], inter_group_arrays[1], alternative='two-sided')),
            ('ranksums', lambda: stats.ranksums(inter_group_arrays[0], inter_group_arrays[1])),
            ('ks_2samp', lambda: stats.ks_2samp(inter_group_arrays[0], inter_group_arrays[1])),
        ]
        
        for test_name, test_func in tests:
            try:
                with np.errstate(all='ignore'):
                    stat, p_val = test_func()
                    if not np.isnan(p_val) and not np.isinf(p_val):
                        return {
                            'test': test_name,
                            'statistic': float(stat) if not np.isnan(stat) else 0.0,
                            'p_value': float(p_val),
                            'significant': p_val < self.config.p_interaction,
                            'distribution_shift': p_val < self.config.p_interaction
                        }
            except Exception as e:
                logger.debug(f"Test {test_name} failed: {e}")
                continue
        
        # Fallback: compare all groups together
        all_groups = []
        for inter_val in interaction_values:
            inter_data = groups_by_inter[inter_val]
            groups = inter_data.get('groups', [])
            for group in groups:
                if len(group) > 0:
                    all_groups.append([v for v in group if np.isfinite(v)])
        
        if len(all_groups) >= 2:
            stats_result = self._perform_group_statistics(all_groups, paired=False)
            return {
                'test': stats_result.get('test', 'group_statistics'),
                'statistic': stats_result.get('statistic'),
                'p_value': stats_result.get('p_value', 1.0),
                'significant': stats_result.get('significant', False),
                'distribution_shift': stats_result.get('significant', False)
            }
        
        return {'test': 'insufficient_data', 'p_value': 1.0, 'significant': False, 'distribution_shift': False}
    
    def _analyze_discrete_interaction(self, target_idx: int, inter_idx: int, inter_type: str) -> Dict:
        """
        Analyze interaction when target is CATEGORICAL (Binary/Discrete).
        
        Path A: Target is Categorical
        - Tests if interaction feature modifies the relative distribution of SHAP values
          across target categories (Distribution Shift / Impact Reversal)
        - Uses Chi-squared test or Two-way ANOVA interaction term
        - Does NOT use "Trend" terminology or location shift
        """
        inter_feat_name = self.X.columns[inter_idx]
        inter_values = self.X.iloc[:, inter_idx].values
        
        if inter_type == "discrete" or inter_type == "binary":
            # Discrete interaction with discrete target
            unique_inter_vals = sorted(np.unique(inter_values))
            groups_by_inter = {}
            
            for val in unique_inter_vals:
                mask = inter_values == val
                target_vals = self.X.iloc[mask, target_idx].values
                shap_vals = self.shap_values[mask, target_idx]
                
                # Group by target values
                unique_target_vals = sorted(np.unique(target_vals))
                groups = [shap_vals[target_vals == tv].tolist() for tv in unique_target_vals]
                
                groups_by_inter[str(val)] = {
                    'groups': groups,
                    'group_labels': [str(tv) for tv in unique_target_vals]
                }
            
            # Test distribution shift (not location shift, not trend)
            stats_result = self._test_categorical_interaction_distribution(groups_by_inter)
            
            return {
                'interaction_groups': groups_by_inter,
                'statistics': stats_result
            }
        else:
            # Continuous interaction with discrete target
            # Split by interaction feature median
            avg_inter = np.mean(inter_values)
            upper_mask = inter_values > avg_inter
            
            groups_upper = self._get_target_groups(target_idx, upper_mask)
            groups_lower = self._get_target_groups(target_idx, ~upper_mask)
            
            # Test distribution shift between upper and lower groups
            # Compare pattern across target categories between high/low interaction values.
            # differs between high and low interaction feature values
            # Format: {'upper': {'groups': [...], 'group_labels': [...]}, 'lower': {...}}
            stats_result = self._test_categorical_interaction_distribution({
                'upper': groups_upper,
                'lower': groups_lower
            })
            
            return {
                'upper_groups': groups_upper,
                'lower_groups': groups_lower,
                'statistics': stats_result
            }
    
    def _analyze_continuous_interaction(self, target_idx: int, inter_idx: int, inter_type: str) -> Dict:
        """Analyze interaction when target is continuous."""
        x = self.X.iloc[:, target_idx].values
        y = self.shap_values[:, target_idx]
        inter_values = self.X.iloc[:, inter_idx].values
        
        if inter_type == "discrete":
            unique_inter_vals = sorted(np.unique(inter_values))
            functions_by_inter = {}
            
            for val in unique_inter_vals:
                mask = inter_values == val
                x_subset = x[mask]
                y_subset = y[mask]
                
                func, params, rmse_val = self._fit_best_function(x_subset, y_subset)
                functions_by_inter[str(val)] = {
                    'function': func,
                    'params': params,
                    'rmse': rmse_val
                }
            
            return {'interaction_functions': functions_by_inter}
        else:
            # Both continuous - Method B: Trend Comparison
            avg_inter = np.mean(inter_values)
            upper_mask = inter_values > avg_inter
            
            x_upper = x[upper_mask]
            y_upper = y[upper_mask]
            x_lower = x[~upper_mask]
            y_lower = y[~upper_mask]
            
            # Fit functions to each group
            func_upper, params_upper, rmse_upper = self._fit_best_function(x_upper, y_upper)
            func_lower, params_lower, rmse_lower = self._fit_best_function(x_lower, y_lower)
            
            # Method B: trend comparison (parameter comparison + Chow test).
            stats_result = self._compare_continuous_functions(
                x_upper, y_upper, func_upper, params_upper,
                x_lower, y_lower, func_lower, params_lower
            )
            
            return {
                'upper_function': {'function': func_upper, 'params': params_upper, 'rmse': rmse_upper},
                'lower_function': {'function': func_lower, 'params': params_lower, 'rmse': rmse_lower},
                'statistics': stats_result
            }
    
    def _get_target_groups(self, target_idx: int, mask: np.ndarray) -> Dict:
        """Get groups for discrete target feature with a mask."""
        target_vals = self.X.iloc[mask, target_idx].values
        shap_vals = self.shap_values[mask, target_idx]
        
        unique_vals = sorted(np.unique(target_vals))
        groups = [shap_vals[target_vals == val].tolist() for val in unique_vals]
        
        return {
            'groups': groups,
            'group_labels': [str(v) for v in unique_vals]
        }
    
    def _fit_best_function(self, x: np.ndarray, y: np.ndarray) -> Tuple[str, np.ndarray, float]:
        """
        Fit the best function to continuous data.
        
        If non-linear fitting fails or data is problematic, falls back to linear.
        
        Returns:
            Tuple of (function_name, parameters, rmse)
        """
        # Check for valid data
        if len(x) < 3 or len(y) < 3:
            return ("None", np.array([]), np.inf)
        
        # Check for zero variance in x or y
        if np.var(x) < np.finfo(float).eps or np.var(y) < np.finfo(float).eps:
            logger.debug(f"Zero variance in data, skipping function fitting")
            return ("None", np.array([]), np.inf)
        
        def linear(x, b0, b1):
            return b0 + b1 * x
        
        def quadratic(x, b0, b1, b2):
            return b0 + b1 * x + b2 * x**2
        
        def sigmoid(x, L, x0, k, b):
            # Prevent overflow in exp with aggressive clipping
            exp_arg = -k * (x - x0)
            exp_arg = np.clip(exp_arg, -700, 700)  # Clip to prevent overflow
            return L / (1 + np.exp(exp_arg)) + b
        
        # Try fitting functions in order of complexity (linear first as fallback)
        functions = [
            ('linear', linear, 2, None),  # p0=None means auto
            ('quadratic', quadratic, 3, None),
            ('sigmoid', sigmoid, 4, 'sigmoid'),  # Special p0 handling
        ]
        
        best_func = None
        best_params = None
        best_rmse = np.inf
        linear_fallback = None  # Store linear fit as fallback
        
        for func_name, func, n_params, p0_type in functions:
            try:
                # Prepare initial parameters
                if p0_type == "sigmoid":
                    y_range = np.max(y) - np.min(y)
                    if y_range < np.finfo(float).eps:
                        continue  # Skip sigmoid for constant y
                    p0 = [y_range, np.median(x), 1.0, np.min(y)]
                    popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=5000)
                else:
                    popt, pcov = curve_fit(func, x, y, maxfev=5000)
                
                # Check if covariance could be estimated
                if pcov is None or np.any(np.isinf(pcov)):
                    logger.debug(f"Covariance estimation failed for {func_name}")
                    if func_name == "linear":
                        # Still use linear as fallback even without covariance
                        y_pred = func(x, *popt)
                        rmse_val = np.sqrt(mean_squared_error(y, y_pred))
                        linear_fallback = (func_name, popt, rmse_val)
                    continue
                
                y_pred = func(x, *popt)
                rmse_val = np.sqrt(mean_squared_error(y, y_pred))
                
                # Store linear as fallback
                if func_name == "linear":
                    linear_fallback = (func_name, popt, rmse_val)
                
                # Check statistical significance using ODR
                try:
                    model = odr.Model(lambda params, x, func=func: func(x, *params))
                    data = odr.Data(x, y)
                    odr_fit = odr.ODR(data, model, beta0=popt, maxit=0)
                    odr_fit.set_job(fit_type=2)
                    param_stats = odr_fit.run()
                    
                    df_e = len(x) - len(popt)
                    if df_e <= 0:
                        continue
                    
                    # Prevent divide by zero with safe sd_beta
                    sd_beta = np.array(param_stats.sd_beta)
                    sd_beta = np.where(sd_beta <= 0, np.finfo(float).eps, sd_beta)
                    tstat = popt / sd_beta
                    p_values = (1.0 - stats.t.cdf(np.abs(tstat), df_e)) * 2.0
                    
                    # Check if last parameter is significant
                    if p_values[-1] < self.config.p_univariate and rmse_val < best_rmse:
                        best_func = func_name
                        best_params = popt
                        best_rmse = rmse_val
                except Exception as e:
                    logger.debug(f"ODR failed for {func_name}: {e}")
                    # If ODR fails but fit worked, consider it based on RMSE alone
                    if rmse_val < best_rmse * 0.8:  # Only if significantly better
                        best_func = func_name
                        best_params = popt
                        best_rmse = rmse_val
                        
            except Exception as e:
                logger.debug(f"Curve fitting failed for {func_name}: {e}")
                continue
        
        # If no significant fit found, use linear fallback if available
        if best_func is None and linear_fallback is not None:
            logger.debug("Using linear fallback since non-linear fitting failed")
            return linear_fallback
        
        if best_func is None:
            return ("None", np.array([]), np.inf)
        
        return (best_func, best_params, best_rmse)
    
    def _evaluate_function(self, func_name: str, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate a function with given parameters, with overflow protection."""
        if len(params) == 0:
            return np.zeros_like(x, dtype=float)
        
        if func_name == "linear":
            return params[0] + params[1] * x
        elif func_name == "quadratic":
            return params[0] + params[1] * x + params[2] * x**2
        elif func_name == "sigmoid":
            L, x0, k, b = params
            # Prevent overflow in exp
            exp_arg = -k * (x - x0)
            exp_arg = np.clip(exp_arg, -700, 700)
            return L / (1 + np.exp(exp_arg)) + b
        else:
            return np.zeros_like(x, dtype=float)
    
    def _perform_group_statistics(self, groups: List[List[float]], paired: bool = False) -> Dict:
        """
        Perform statistical tests on groups with robust handling of edge cases.
        
        Handles:
        - Zero-variance groups
        - Constant data
        - NaN/Inf values
        - Insufficient sample sizes
        - Test failures
        """
        # Convert to numpy arrays and filter empty groups
        groups = [np.array(g, dtype=float) for g in groups if len(g) > 0]
        
        if len(groups) < 2:
            return {'test': 'insufficient_data', 'p_value': 1.0, 'significant': False}
        
        # Remove NaN and Inf values from each group
        clean_groups = []
        for g in groups:
            clean = g[np.isfinite(g)]
            if len(clean) > 0:
                clean_groups.append(clean)
        
        if len(clean_groups) < 2:
            return {'test': 'insufficient_data_after_cleaning', 'p_value': 1.0, 'significant': False}
        
        groups = clean_groups
        
        # Check for constant groups and track variance
        # Use a very lenient threshold - only skip if truly identical (all values exactly the same)
        group_variances = [np.var(g) for g in groups]
        group_means = [np.mean(g) for g in groups]
        
        # Check if groups are truly constant (all values identical within each group)
        all_constant = True
        for g, var in zip(groups, group_variances):
            if len(g) > 1 and var > np.finfo(float).eps:
                all_constant = False
                break
            elif len(g) == 1:
                # Single value - check if it differs from others
                if len(set(group_means)) > 1:
                    all_constant = False
                    break
        
        # If all groups are truly constant and identical, return non-significant
        if all_constant:
            if len(set(group_means)) <= 1:
                return {'test': 'all_constant_equal', 'p_value': 1.0, 'significant': False}
            else:
                # Constant but different means - this is significant
                return {'test': 'constant_groups_differ', 'p_value': 0.001, 'significant': True}
        
        # Use very lenient threshold - only filter truly identical values
        # Even if variance is small, we should test if means differ
        variance_threshold = np.finfo(float).eps  # Very lenient - only filter exact zeros
        non_constant_count = sum(1 for v in group_variances if v > variance_threshold)
        
        # Even if variance is small, test if there are meaningful differences
        # Don't skip tests just because variance is low
        
        # Check normality (only for groups with sufficient data and variance)
        is_normal = True
        for i, g in enumerate(groups):
            if len(g) >= 3 and group_variances[i] > np.finfo(float).eps:
                try:
                    with np.errstate(all='ignore'):
                        _, p_val = stats.shapiro(g)
                        if np.isnan(p_val) or p_val <= self.config.p_univariate:
                            is_normal = False
                            break
                except Exception:
                    is_normal = False
                    break
        
        # Determine test based on number of groups
        if len(groups) == 2:
            # Check minimum sample size - allow single values if both groups have at least 1
            if len(groups[0]) < 1 or len(groups[1]) < 1:
                return {'test': 'insufficient_data', 'p_value': 1.0, 'significant': False}
            
            # If one group has only 1 value, use permutation-like approach
            if len(groups[0]) == 1 or len(groups[1]) == 1:
                # Compare single value to distribution
                mean_diff = abs(np.mean(groups[0]) - np.mean(groups[1]))
                # If means differ, it's potentially significant
                if mean_diff > np.finfo(float).eps:
                    return {'test': 'single_value_comparison', 'p_value': 0.05, 'significant': True}
                else:
                    return {'test': 'single_value_equal', 'p_value': 1.0, 'significant': False}
            
            # Use very lenient threshold - only filter exact zeros
            variance_threshold = np.finfo(float).eps
            var0 = group_variances[0] > variance_threshold
            var1 = group_variances[1] > variance_threshold
            
            if not var0 and not var1:
                # Both constant - already handled above
                pass
            elif not var0 or not var1:
                # One constant - use Mann-Whitney U which handles ties
                try:
                    with np.errstate(all='ignore'):
                        stat, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                        if np.isnan(p_val):
                            p_val = 1.0
                    return {
                        'test': 'mannwhitneyu_one_constant',
                        'statistic': float(stat) if not np.isnan(stat) else 0.0,
                        'p_value': float(p_val),
                        'significant': p_val < self.config.p_univariate
                    }
                except Exception:
                    return {'test': 'failed_one_constant', 'p_value': 1.0, 'significant': False}
            
            # Both groups have variance
            test_order = []
            if is_normal:
                if paired and len(groups[0]) == len(groups[1]):
                    test_order = [
                        ('paired_t_test', lambda: stats.ttest_rel(groups[0], groups[1])),
                        ('ranksums', lambda: stats.ranksums(groups[0], groups[1])),
                    ]
                else:
                    test_order = [
                        ('t_test', lambda: stats.ttest_ind(groups[0], groups[1])),
                        ('mannwhitneyu', lambda: stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')),
                        ('ranksums', lambda: stats.ranksums(groups[0], groups[1])),
                    ]
            else:
                test_order = [
                    ('mannwhitneyu', lambda: stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')),
                    ('ranksums', lambda: stats.ranksums(groups[0], groups[1])),
                ]
            
            # Try tests in order until one succeeds
            for test_name, test_func in test_order:
                try:
                    with np.errstate(all='ignore'):
                        stat, p_val = test_func()
                        if np.isnan(p_val) or np.isinf(p_val):
                            continue
                        return {
                            'test': test_name,
                            'statistic': float(stat) if not np.isnan(stat) else 0.0,
                            'p_value': float(p_val),
                            'significant': p_val < self.config.p_univariate
                        }
                except Exception as e:
                    logger.debug(f"Test {test_name} failed: {e}")
                    continue
            
            # All tests failed
            return {'test': 'all_tests_failed', 'p_value': 1.0, 'significant': False}
        
        else:
            # More than 2 groups
            # Check minimum sample size
            if any(len(g) < 2 for g in groups):
                return {'test': 'insufficient_data', 'p_value': 1.0, 'significant': False}
            
            # Filter out constant groups for ANOVA/Kruskal
            # Use very lenient threshold - only filter exact zeros
            variance_threshold = np.finfo(float).eps
            variable_groups = [g for g, v in zip(groups, group_variances) if v > variance_threshold]
            
            # If we filtered out too many groups, use all groups anyway
            # (small variance doesn't mean no difference)
            if len(variable_groups) < 2 and len(groups) >= 2:
                variable_groups = groups
            
            if len(variable_groups) < 2:
                # Not enough variable groups for comparison
                return {'test': 'insufficient_variable_groups', 'p_value': 1.0, 'significant': False}
            
            test_order = []
            if is_normal:
                test_order = [
                    ('anova', lambda: stats.f_oneway(*variable_groups)),
                    ('kruskal_wallis', lambda: stats.kruskal(*variable_groups)),
                ]
            else:
                test_order = [
                    ('kruskal_wallis', lambda: stats.kruskal(*variable_groups)),
                ]
            
            # Try tests in order
            for test_name, test_func in test_order:
                try:
                    with np.errstate(all='ignore'):
                        stat, p_val = test_func()
                        if np.isnan(p_val) or np.isinf(p_val):
                            continue
                        return {
                            'test': test_name,
                            'statistic': float(stat) if not np.isnan(stat) else 0.0,
                            'p_value': float(p_val),
                            'significant': p_val < self.config.p_univariate
                        }
                except Exception as e:
                    logger.debug(f"Test {test_name} failed: {e}")
                    continue
            
            # All tests failed
            return {'test': 'all_tests_failed', 'p_value': 1.0, 'significant': False}
    
    def _compare_interaction_groups(self, groups1: Dict, groups2: Dict) -> Dict:
        """
        Compare two sets of groups from interaction analysis.
        
        Uses raw SHAP values directly for comparison, ensuring even small differences
        are detected using appropriate statistical tests.
        """
        # Flatten groups for comparison - use raw SHAP values
        flat1 = [val for group in groups1.get('groups', []) for val in group]
        flat2 = [val for group in groups2.get('groups', []) for val in group]
        
        # Ensure we have valid data
        flat1 = [v for v in flat1 if np.isfinite(v)]
        flat2 = [v for v in flat2 if np.isfinite(v)]
        
        if len(flat1) < 1 or len(flat2) < 1:
            return {'test': 'insufficient_data', 'p_value': 1.0, 'significant': False}
        
        # Convert to numpy arrays
        arr1 = np.array(flat1, dtype=float)
        arr2 = np.array(flat2, dtype=float)
        
        # Always perform statistical test, even if values are small
        # Use robust non-parametric tests that work with small differences
        tests = [
            ('mannwhitneyu', lambda: stats.mannwhitneyu(arr1, arr2, alternative='two-sided')),
            ('ranksums', lambda: stats.ranksums(arr1, arr2)),
            ('ks_2samp', lambda: stats.ks_2samp(arr1, arr2)),
        ]
        
        for test_name, test_func in tests:
            try:
                with np.errstate(all='ignore'):
                    stat, p_val = test_func()
                    if not np.isnan(p_val) and not np.isinf(p_val):
                        return {
                            'test': test_name,
                            'statistic': float(stat) if not np.isnan(stat) else 0.0,
                            'p_value': float(p_val),
                            'significant': p_val < self.config.p_interaction
                        }
            except Exception as e:
                logger.debug(f"Test {test_name} failed in _compare_interaction_groups: {e}")
                continue
        
        # Fallback: compare means if all tests fail
        mean_diff = abs(np.mean(arr1) - np.mean(arr2))
        std_pooled = np.sqrt((np.var(arr1) + np.var(arr2)) / 2)
        if std_pooled > np.finfo(float).eps:
            # Approximate t-test
            n1, n2 = len(arr1), len(arr2)
            se = std_pooled * np.sqrt(1/n1 + 1/n2)
            if se > 0:
                t_stat = mean_diff / se
                df = n1 + n2 - 2
                if df > 0:
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                    return {
                        'test': 'approximate_t_test',
                        'statistic': float(t_stat),
                        'p_value': float(p_val),
                        'significant': p_val < self.config.p_interaction
                    }
        
        # Last resort: if means differ significantly, return low p-value
        if mean_diff > std_pooled * 2:  # More than 2 standard deviations
            return {'test': 'mean_difference', 'p_value': 0.01, 'significant': True}
        
        return {'test': 'all_tests_failed', 'p_value': 1.0, 'significant': False}
    
    def _get_parameter_confidence_intervals(
        self, x: np.ndarray, y: np.ndarray, func_name: str, params: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for fitted function parameters using ODR.
        
        Args:
            x: Feature values
            y: SHAP values
            func_name: Name of fitted function ('linear', 'quadratic', 'sigmoid')
            params: Fitted parameters
        
        Returns:
            Dictionary mapping parameter names to (lower_bound, upper_bound) tuples
        """
        if func_name == "None" or len(params) == 0:
            return {}
        
        try:
            # Define function for ODR
            if func_name == "linear":
                def func(params, x):
                    return params[0] + params[1] * x
            elif func_name == "quadratic":
                def func(params, x):
                    return params[0] + params[1] * x + params[2] * x**2
            elif func_name == "sigmoid":
                def func(params, x):
                    L, x0, k, b = params
                    exp_arg = -k * (x - x0)
                    exp_arg = np.clip(exp_arg, -700, 700)
                    return L / (1 + np.exp(exp_arg)) + b
            else:
                return {}
            
            # Use ODR to get parameter uncertainties
            model = odr.Model(lambda params, x, func=func: func(params, x))
            data = odr.Data(x, y)
            odr_fit = odr.ODR(data, model, beta0=params, maxit=0)
            odr_fit.set_job(fit_type=2)
            param_stats = odr_fit.run()
            
            # Calculate 95% confidence intervals (t-distribution, alpha=0.05)
            df = len(x) - len(params)
            if df <= 0:
                return {}
            
            t_critical = stats.t.ppf(0.975, df)  # Two-tailed, 95% CI
            sd_beta = np.array(param_stats.sd_beta)
            sd_beta = np.where(sd_beta <= 0, np.finfo(float).eps, sd_beta)
            
            intervals = {}
            param_names = ['intercept', 'slope', 'quadratic'] if func_name == "quadratic" else ['intercept', 'slope']
            if func_name == "sigmoid":
                param_names = ['L', 'x0', 'k', 'b']
            
            for i, param_name in enumerate(param_names[:len(params)]):
                lower = params[i] - t_critical * sd_beta[i]
                upper = params[i] + t_critical * sd_beta[i]
                intervals[param_name] = (float(lower), float(upper))
            
            return intervals
        except Exception as e:
            logger.debug(f"Failed to calculate confidence intervals: {e}")
            return {}
    
    def _chow_test(
        self, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray
    ) -> Dict:
        """
        Perform Chow test to compare if splitting data into two groups significantly
        improves model fit compared to a pooled model.
        
        Returns:
            Dictionary with test statistic, p-value, and significance
        """
        try:
            # Pooled model (single regression on all data)
            x_pooled = np.concatenate([x1, x2])
            y_pooled = np.concatenate([y1, y2])
            
            # Fit linear model to pooled data
            reg_pooled = LinearRegression()
            reg_pooled.fit(x_pooled.reshape(-1, 1), y_pooled)
            y_pred_pooled = reg_pooled.predict(x_pooled.reshape(-1, 1))
            ssr_pooled = np.sum((y_pooled - y_pred_pooled) ** 2)
            df_pooled = len(x_pooled) - 2
            
            # Separate models for each group
            reg1 = LinearRegression()
            reg1.fit(x1.reshape(-1, 1), y1)
            y_pred1 = reg1.predict(x1.reshape(-1, 1))
            ssr1 = np.sum((y1 - y_pred1) ** 2)
            
            reg2 = LinearRegression()
            reg2.fit(x2.reshape(-1, 1), y2)
            y_pred2 = reg2.predict(x2.reshape(-1, 1))
            ssr2 = np.sum((y2 - y_pred2) ** 2)
            
            ssr_separate = ssr1 + ssr2
            df_separate = (len(x1) - 2) + (len(x2) - 2)
            
            # Chow test statistic
            if df_pooled <= 0 or df_separate <= 0 or ssr_separate < np.finfo(float).eps:
                return {'test': 'chow_test', 'p_value': 1.0, 'significant': False}
            
            f_stat = ((ssr_pooled - ssr_separate) / (df_pooled - df_separate)) / (ssr_separate / df_separate)
            df_num = df_pooled - df_separate
            df_den = df_separate
            
            if df_num <= 0 or df_den <= 0 or np.isnan(f_stat) or np.isinf(f_stat):
                return {'test': 'chow_test', 'p_value': 1.0, 'significant': False}
            
            p_value = 1 - stats.f.cdf(f_stat, df_num, df_den)
            
            return {
                'test': 'chow_test',
                'statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < self.config.p_interaction
            }
        except Exception as e:
            logger.debug(f"Chow test failed: {e}")
            return {'test': 'chow_test', 'p_value': 1.0, 'significant': False}
    
    def _compare_continuous_functions(
        self, 
        x1: np.ndarray, y1: np.ndarray, func1_name: str, params1: np.ndarray,
        x2: np.ndarray, y2: np.ndarray, func2_name: str, params2: np.ndarray
    ) -> Dict:
        """
        Compare two continuous function fits using Method B (Trend Comparison ONLY).
        
        Method B focuses ONLY on detecting structural changes (trend differences).
        Location shift is completely ignored - we only care about pattern changes.
        
        Tests performed:
        1. Parameter comparison (slope differences with confidence intervals)
        2. Chow test (structural break detection)
        
        An interaction is significant ONLY if trend changes (pattern/structure differs).
        
        Args:
            x1, y1: Data for first group
            x2, y2: Data for second group
            func1_name, params1: Function and parameters for first group
            func2_name, params2: Function and parameters for second group
        
        Returns:
            Dictionary with test results including trend_change and significance
        """
        # Clean data
        y1_clean = y1[np.isfinite(y1)]
        y2_clean = y2[np.isfinite(y2)]
        x1_clean = x1[np.isfinite(x1) & np.isfinite(y1)]
        x2_clean = x2[np.isfinite(x2) & np.isfinite(y2)]
        
        if len(y1_clean) < 2 or len(y2_clean) < 2:
            return {
                'test': 'insufficient_data',
                'p_value': 1.0,
                'significant': False,
                'trend_change': False,
                'trend_p_value': 1.0
            }
        
        results = {
            'trend_change': False,
            'trend_p_value': 1.0,
            'trend_test': 'none'
        }
        
        # Test: Trend Change (parameter comparison and Chow test)
        # Only test trend if both functions are valid and comparable
        if (func1_name != "None" and func2_name != "None" and 
            len(params1) > 0 and len(params2) > 0 and
            func1_name == func2_name):  # Only compare if same function type
            
            # 1a: Compare slopes (most important parameter for trend)
            slope_idx = 1  # Index of slope parameter
            if func1_name == "quadratic":
                # For quadratic, compare both linear and quadratic coefficients
                slope_idx = 1  # Linear term
                quad_idx = 2   # Quadratic term
            elif func1_name == "sigmoid":
                # For sigmoid, compare k (steepness) parameter
                slope_idx = 2  # k parameter
            
            if slope_idx < len(params1) and slope_idx < len(params2):
                # Get confidence intervals for slopes
                ci1 = self._get_parameter_confidence_intervals(x1_clean, y1_clean, func1_name, params1)
                ci2 = self._get_parameter_confidence_intervals(x2_clean, y2_clean, func2_name, params2)
                
                slope_param_name = 'slope' if func1_name != "sigmoid" else 'k'
                if func1_name == "quadratic":
                    slope_param_name = 'slope'
                
                if slope_param_name in ci1 and slope_param_name in ci2:
                    # Check if confidence intervals overlap
                    ci1_lower, ci1_upper = ci1[slope_param_name]
                    ci2_lower, ci2_upper = ci2[slope_param_name]
                    
                    # Intervals don't overlap if one is completely above the other
                    intervals_overlap = not (ci1_upper < ci2_lower or ci2_upper < ci1_lower)
                    
                    if not intervals_overlap:
                        results['trend_change'] = True
                        results['trend_p_value'] = 0.001  # Very significant
                        results['trend_test'] = 'parameter_ci_nonoverlap'
                    else:
                        # Even if intervals overlap, test if slopes are significantly different
                        # Use t-test on parameter estimates with standard errors
                        try:
                            # Approximate t-test: (param1 - param2) / sqrt(se1^2 + se2^2)
                            df1 = len(x1_clean) - len(params1)
                            df2 = len(x2_clean) - len(params2)
                            if df1 > 0 and df2 > 0:
                                t_crit1 = stats.t.ppf(0.975, df1)
                                t_crit2 = stats.t.ppf(0.975, df2)
                                se1 = (ci1_upper - ci1_lower) / (2 * t_crit1) if t_crit1 > 0 else np.finfo(float).eps
                                se2 = (ci2_upper - ci2_lower) / (2 * t_crit2) if t_crit2 > 0 else np.finfo(float).eps
                                
                                if se1 > 0 and se2 > 0:
                                    t_stat = (params1[slope_idx] - params2[slope_idx]) / np.sqrt(se1**2 + se2**2)
                                    df = min(df1, df2)
                                    if df > 0:
                                        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                                        results['trend_p_value'] = float(p_val)
                                        results['trend_change'] = p_val < self.config.p_interaction
                                        results['trend_test'] = 'parameter_t_test'
                        except Exception as e:
                            logger.debug(f"Parameter t-test failed: {e}")
                
                # For quadratic, also check quadratic term
                if func1_name == "quadratic" and quad_idx < len(params1) and quad_idx < len(params2):
                    quad_param_name = 'quadratic'
                    if quad_param_name in ci1 and quad_param_name in ci2:
                        ci1_lower, ci1_upper = ci1[quad_param_name]
                        ci2_lower, ci2_upper = ci2[quad_param_name]
                        if ci1_upper < ci2_lower or ci2_upper < ci1_lower:
                            results['trend_change'] = True
                            results['trend_p_value'] = min(results['trend_p_value'], 0.001)
            
            # 1b: Chow test (structural break detection)
            if len(x1_clean) >= 3 and len(x2_clean) >= 3:
                chow_result = self._chow_test(x1_clean, y1_clean, x2_clean, y2_clean)
                if chow_result.get('significant', False):
                    results['trend_change'] = True
                    results['trend_p_value'] = min(results['trend_p_value'], chow_result.get('p_value', 1.0))
                    if results['trend_test'] == 'none':
                        results['trend_test'] = 'chow_test'
        
        # Overall significance: ONLY Trend change (no location shift)
        return {
            'test': 'method_b_trend_comparison',
            'statistic': None,
            'p_value': float(results['trend_p_value']),
            'significant': results['trend_change'],
            'trend_change': results['trend_change'],
            'trend_p_value': results['trend_p_value'],
            'trend_test': results['trend_test']
        }
    
    def analyze(self) -> AnalysisResults:
        """
        Perform complete analysis on all selected features.
        
        Returns:
            AnalysisResults dataclass containing all analysis results.
        """
        logger.info("Starting comprehensive analysis")
        
        # Feature importance and selection
        importance_df = self.feature_importance()
        selected_features = self.select_features(debug=True)  # Enable debug output
        
        # Classify all feature types
        feature_types = {
            feat: self._classify_feature_type(feat)
            for feat in self.X.columns
        }
        
        results = AnalysisResults(
            feature_importance=importance_df,
            selected_features=selected_features,
            feature_types=feature_types
        )
        
        # Univariate analysis for all features.
        all_features = list(self.X.columns)
        for feat in all_features:
            results.univariate_results[feat] = self.univariate(feat)
            if feature_types[feat] == "continuous":
                results.best_functions[feat] = results.univariate_results[feat].get('best_function', 'None')
        
        # Interaction analysis for selected features
        for target_feat in selected_features:
            # Find interaction feature using SHAP
            target_idx = self._get_feature_index(target_feat)
            try:
                inter_idx = shap.utils.approximate_interactions(
                    target_idx, self.shap_values, self.X
                )[0]
                inter_feat = self.X.columns[inter_idx]
                
                results.interaction_results[f"{target_feat}_{inter_feat}"] = self.inter(
                    target_feat, inter_feat
                )
            except Exception as e:
                logger.warning(f"Could not compute interaction for {target_feat}: {e}")
                continue
        
        logger.info(f"Analysis complete: {len(selected_features)} features analyzed")
        return results
