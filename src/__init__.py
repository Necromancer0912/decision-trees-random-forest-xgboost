"""
Decision Trees & Boosted Ensembles From Scratch (`src/`).

Pure NumPy implementation of CART Decision Trees (Classification & Regression),
Cost-Complexity Pruning, Random Forests (Out-of-Bag validation & Feature Subsampling),
and Second-Order Taylor XGBoost (exact gradients & Hessians).
"""

from .trees_engine import (
    DecisionTreeClassifierScratch,
    DecisionTreeRegressorScratch,
    OneHotEncoderScratch,
    RandomForestScratch,
    XGBoostScratch,
    entropy_impurity,
    gini_impurity,
    variance_impurity,
)
from .plot_theme import (
    AMBER,
    BG_AXES,
    BG_CANVAS,
    CLASS_COLORS,
    CMAP_DECISION,
    CMAP_MULTICLASS,
    CMAP_SURFACE,
    CORAL,
    CYAN,
    EMERALD,
    MAGENTA,
    PALETTE,
    PURPLE,
    SLATE_BLUE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    apply_github_dark_style,
    plot_decision_boundary,
    plot_loss_landscape,
    save_fig,
    style_3d_axis,
)

__all__ = [
    "OneHotEncoderScratch",
    "entropy_impurity",
    "gini_impurity",
    "variance_impurity",
    "DecisionTreeClassifierScratch",
    "DecisionTreeRegressorScratch",
    "RandomForestScratch",
    "XGBoostScratch",
    "apply_github_dark_style",
    "style_3d_axis",
    "plot_decision_boundary",
    "plot_loss_landscape",
    "save_fig",
    "PALETTE",
    "CLASS_COLORS",
    "BG_CANVAS",
    "BG_AXES",
    "TEXT_PRIMARY",
    "TEXT_MUTED",
    "CYAN",
    "CORAL",
    "AMBER",
    "EMERALD",
    "PURPLE",
    "SLATE_BLUE",
    "MAGENTA",
    "CMAP_SURFACE",
    "CMAP_DECISION",
    "CMAP_MULTICLASS",
]

