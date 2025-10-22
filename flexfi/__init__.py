"""
flexfi: Flexible Feature Importance for Deep Learning Models.

This package provides:
- Permutation-based feature importance scoring
- Support for structured/group-wise penalties
- Visualization utilities for feature importance and model diagnostics
"""

from .core.importance import feature_importance
from .visualization.plots import (
    plot_fi_bar,
    plot_fi_violin,
    plot_score_vs_pval,
)

__all__ = [
    "feature_importance",
    "plot_fi_bar",
    "plot_fi_violin",
    "plot_score_vs_pval",
]


__version__ = "0.1.0"
__author__ = "David Gabrielli"
__email__ = "gabrielli.d@ufl.edu"