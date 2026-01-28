"""
CLE-SH: Comprehensive Literal Explanation Package for SHapley Values by Statistical Validity.

A Python package for comprehensive statistical analysis of SHAP values.
"""

__version__ = "1.0.0"

from clesh.config import CLEConfig
from clesh.analyzer import Analyzer, AnalysisResults
from clesh.explainer import Explainer

__all__ = [
    "CLEConfig",
    "Analyzer",
    "AnalysisResults",
    "Explainer",
]
