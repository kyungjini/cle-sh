"""
Configuration classes for CLE-SH package.

This module defines the configuration structure for CLE-SH analysis.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CLEConfig:
    """
    Configuration class for CLE-SH analysis parameters.

    This class holds all analysis-related parameters with sensible defaults.
    Training and data-specific parameters (like PATH, LABEL) are handled separately.

    Attributes:
        cont_bound: Threshold for determining continuous features.
            Features with more unique values than this are considered continuous.
            Default: 10
        candidate_num_min: Minimum number of features to select during
            feature selection. Default: 10
        candidate_num_max: Maximum number of features to select during
            feature selection. Default: 20
        p_feature_selection: P-value threshold for feature selection
            statistical tests. Default: 0.05
        manual_num: Manual override for number of features to select.
            Set to 0 for automatic selection. Default: 0
        p_univariate: P-value threshold for univariate analysis
            statistical tests. Default: 0.05
        p_interaction: P-value threshold for interaction analysis
            statistical tests. Default: 0.05
    """

    cont_bound: int = 10
    candidate_num_min: int = 10
    candidate_num_max: int = 20
    p_feature_selection: float = 0.05
    manual_num: int = 0
    p_univariate: float = 0.05
    p_interaction: float = 0.05

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.cont_bound < 1:
            raise ValueError("cont_bound must be >= 1")
        if self.candidate_num_min < 1:
            raise ValueError("candidate_num_min must be >= 1")
        if self.candidate_num_max < self.candidate_num_min:
            raise ValueError("candidate_num_max must be >= candidate_num_min")
        if not 0 < self.p_feature_selection <= 1:
            raise ValueError("p_feature_selection must be in (0, 1]")
        if self.manual_num < 0:
            raise ValueError("manual_num must be >= 0")
        if not 0 < self.p_univariate <= 1:
            raise ValueError("p_univariate must be in (0, 1]")
        if not 0 < self.p_interaction <= 1:
            raise ValueError("p_interaction must be in (0, 1]")
