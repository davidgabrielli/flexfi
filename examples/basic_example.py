"""
Basic example of using flexfi for feature importance analysis.

This example demonstrates:
- Creating synthetic data
- Running feature importance analysis
- Interpreting results
- Visualizing importance scores
"""

import torch
from flexfi import feature_importance, plot_fi_bar

print("="*60)
print("flexfi Basic Example: Feature Importance Analysis")
print("="*60)

# Set random seed for reproducibility
torch.manual_seed(42)

# ============================================================================
# 1. Create Synthetic Data
# ============================================================================

n_samples = 500
n_features = 10

print(f"\nGenerating synthetic data...")
print(f"  Samples: {n_samples}")
print(f"  Features: {n_features}")

# Create feature matrix
X = torch.randn(n_samples, n_features)

# Create binary outcome where first 3 features are truly important
# y = sigmoid(X_0 + 0.5*X_1 - 0.8*X_2) + noise
y_logits = X[:, 0] + 0.5 * X[:, 1] - 0.8 * X[:, 2]
y_prob = torch.sigmoid(y_logits)
y = torch.bernoulli(y_prob).long()

print(f"\nTrue important features: 0, 1, 2")
print(f"  (Features 3-9 are noise)")

# Feature names for interpretability
feature_names = [f"Feature_{i}" for i in range(n_features)]

# ============================================================================
# 2. Run Feature Importance Analysis
# ============================================================================

print("\n" + "="*60)
print("Running Feature Importance Analysis...")
print("="*60)

results = feature_importance(
    X=X,
    y=y,
    outcome="binary",
    iterations=3,              # Multiple iterations for stability
    feature_names=feature_names,
    kfolds=5,                  # 5-fold cross-validation
    perm_count=100,             # 100 permutations per feature
    epochs=100,                # Training epochs
    print_level=0              # Suppress training output
)

# ============================================================================
# 3. Display Results
# ============================================================================

print("\n" + "="*60)
print("Results: Top 5 Most Important Features")
print("="*60)
print("\nNote: More negative scores = More important")
print(results["fi_summary"].head())

print("\n" + "="*60)
print("Interpretation:")
print("="*60)
print("Features 0, 1, and 2 should have the most negative scores")
print("(indicating they are most important for prediction)")
print("\nFeatures 3-9 should have scores near zero")
print("(indicating they contribute little to prediction)")

# ============================================================================
# 4. Visualize Results
# ============================================================================

print("\nGenerating visualization...")
plot_fi_bar(results["fi_summary"], top_k=10, 
           title="Feature Importance - Basic Example")

plot_score_vs_pval(
    results["fi_summary"], top_k=10, 
    title = "Importance vs. P-Value - Basic Example")

print("\n" + "="*60)
print("Example Complete!")
print("="*60)