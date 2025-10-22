"""
Tests for feature importance functions.
"""

import torch
import pytest
import numpy as np
from flexfi import feature_importance


def test_feature_importance_binary(binary_sim_data):
    """Test feature importance with binary outcome using simulated data."""
    X = binary_sim_data["X"]
    y = binary_sim_data["y"]
    signal_features = binary_sim_data["signal_features"]
    
    # Run feature importance (fast test settings)
    results = feature_importance(
        X=X,
        y=y,
        outcome="binary",
        iterations=2,
        kfolds=3,
        epochs=50,
        print_level=0
    )
    
    # Check results structure
    assert "fi_summary" in results
    assert len(results["fi_summary"]) == X.shape[1]
    assert "Feature/Group" in results["fi_summary"].columns
    assert "Importance Score" in results["fi_summary"].columns
    
    # Check that signal features are ranked highly
    # Get importance scores
    importance_scores = results["fi_summary"]["Importance Score"].values
    
    # Features 0, 1, 2 should have more negative scores (more important)
    # Get indices of top 3 most important features
    top_3_indices = np.argsort(importance_scores)[:3]
    
    # At least 2 of the top 3 should be signal features
    signal_in_top3 = sum(idx in signal_features for idx in top_3_indices)
    assert signal_in_top3 >= 2, f"Expected at least 2 signal features in top 3, got {signal_in_top3}"


def test_feature_importance_continuous(continuous_sim_data):
    """Test feature importance with continuous outcome."""
    X = continuous_sim_data["X"]
    y = continuous_sim_data["y"]
    signal_features = continuous_sim_data["signal_features"]
    
    results = feature_importance(
        X=X,
        y=y,
        outcome="continuous",
        iterations=2,
        kfolds=3,
        epochs=50,
        print_level=0
    )
    
    assert "fi_summary" in results
    assert len(results["fi_summary"]) == X.shape[1]
    
    # Check signal features are identified
    importance_scores = results["fi_summary"]["Importance Score"].values
    top_3_indices = np.argsort(importance_scores)[:3]
    signal_in_top3 = sum(idx in signal_features for idx in top_3_indices)
    assert signal_in_top3 >= 2


def test_feature_importance_multiclass(multiclass_sim_data):
    """Test feature importance with multiclass outcome."""
    X = multiclass_sim_data["X"]
    y = multiclass_sim_data["y"].squeeze()  # Multiclass needs 1D labels
    
    results = feature_importance(
        X=X,
        y=y,
        outcome="multiclass",
        iterations=2,
        kfolds=3,
        epochs=50,
        print_level=0
    )
    
    assert "fi_summary" in results
    assert len(results["fi_summary"]) == X.shape[1]


def test_feature_importance_groups(binary_sim_data):
    """Test group-wise feature importance."""
    X = binary_sim_data["X"]
    y = binary_sim_data["y"]
    
    # Define groups - Group1 contains the signal features
    groups = {
        "Group1_Signal": [0, 1, 2],  # Signal features
        "Group2_Noise": [3, 4, 5],
        "Group3_Noise": [6, 7, 8, 9]
    }
    
    results = feature_importance(
        X=X,
        y=y,
        outcome="binary",
        iterations=2,
        groups=groups,
        kfolds=3,
        epochs=50,
        print_level=0
    )
    
    # Should have 3 groups
    assert len(results["fi_summary"]) == 3
    assert set(results["fi_summary"]["Feature/Group"]) == {"Group1_Signal", "Group2_Noise", "Group3_Noise"}
    
    # Group1 should be most important (most negative score)
    group_scores = results["fi_summary"].set_index("Feature/Group")["Importance Score"]
    most_important = group_scores.idxmin()
    assert most_important == "Group1_Signal", f"Expected Group1_Signal to be most important, got {most_important}"


def test_nonlinear_relationships(nonlinear_sim_data):
    """Test that feature importance works with nonlinear relationships."""
    X = nonlinear_sim_data["X"]
    y = nonlinear_sim_data["y"]
    
    results = feature_importance(
        X=X,
        y=y,
        outcome="continuous",
        iterations=2,
        kfolds=3,
        epochs=100,  # More epochs for nonlinear
        hidden_activation="relu",  # Good for nonlinear
        print_level=0
    )
    
    assert "fi_summary" in results
    # Signal features should still be identified
    signal_features = nonlinear_sim_data["signal_features"]
    importance_scores = results["fi_summary"]["Importance Score"].values
    top_4_indices = np.argsort(importance_scores)[:4]
    signal_in_top4 = sum(idx in signal_features for idx in top_4_indices)
    assert signal_in_top4 >= 3  # At least 3 of 4 signal features in top 4


def test_correlated_features(binary_sim_data):
    """Test behavior with correlated features."""
    # Features 0 and 4 are correlated in this simulation
    # Both should show some importance if feature 0 is a signal
    X = binary_sim_data["X"]
    y = binary_sim_data["y"]
    
    results = feature_importance(
        X=X,
        y=y,
        outcome="binary",
        iterations=3,
        kfolds=3,
        epochs=50,
        print_level=0
    )
    
    # Get scores for features 0 and 4
    fi_df = results["fi_summary"]
    score_0 = fi_df[fi_df["Feature/Group"] == "Feature_0"]["Importance Score"].values[0]
    score_4 = fi_df[fi_df["Feature/Group"] == "Feature_4"]["Importance Score"].values[0]
    
    # Both should be negative (important), but 0 should be more important
    assert score_0 < 0, "Feature 0 should be important (negative score)"
    assert score_0 < score_4, "Feature 0 should be more important than correlated feature 4"


def test_input_validation():
    """Test that invalid inputs raise appropriate errors."""
    X = torch.randn(100, 5)
    y = torch.randn(100)
    
    # Test invalid outcome
    with pytest.raises(ValueError, match="outcome must be"):
        feature_importance(X, y, "invalid_outcome", iterations=1)
    
    # Test mismatched shapes
    with pytest.raises(ValueError, match="same number of samples"):
        feature_importance(X, torch.randn(50), "binary", iterations=1)
    
    # Test invalid iterations
    with pytest.raises(ValueError, match="iterations must be"):
        feature_importance(X, y, "binary", iterations=0)


def test_feature_names(binary_sim_data):
    """Test that feature names are used correctly."""
    X = binary_sim_data["X"][:, :5]  # Use first 5 features
    y = binary_sim_data["y"]
    
    feature_names = ["age", "BMI", "glucose", "BP_sys", "cholesterol"]
    
    results = feature_importance(
        X=X,
        y=y,
        outcome="binary",
        iterations=2,
        feature_names=feature_names,
        kfolds=3,
        epochs=50,
        print_level=0
    )
    
    # Check that feature names appear in results
    assert set(results["fi_summary"]["Feature/Group"]) == set(feature_names)