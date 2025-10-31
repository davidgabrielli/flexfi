"""
Group-wise feature importance example.

This example demonstrates:
- Grouping related features together
- Testing importance of feature groups
- Comparing individual vs. group-wise analysis
"""

import torch
from flexfi import feature_importance, plot_fi_bar

print("="*60)
print("flexfi Group-wise Feature Importance Example")
print("="*60)

# Set random seed
torch.manual_seed(42)

# ============================================================================
# 1. Create Synthetic Data with Natural Groupings
# ============================================================================

n_samples = 500
n_features = 12

print(f"\nGenerating synthetic data with grouped features...")
print(f"  Samples: {n_samples}")
print(f"  Features: {n_features}")

X = torch.randn(n_samples, n_features)

# Create outcome where one GROUP of features is important
# Group 1 (Demographics): features 0, 1, 2 - IMPORTANT
# Group 2 (Vitals): features 3, 4, 5 - SOMEWHAT IMPORTANT
# Group 3 (Labs): features 6, 7, 8 - NOT IMPORTANT
# Group 4 (Other): features 9, 10, 11 - NOT IMPORTANT

y_logits = (X[:, 0] + 0.8 * X[:, 1] + 0.6 * X[:, 2] +  # Demographics
            0.3 * X[:, 3] + 0.2 * X[:, 4])              # Vitals (weaker)

y_prob = torch.sigmoid(y_logits)
y = torch.bernoulli(y_prob).long()

print("\nTrue structure:")
print("  Demographics (features 0-2): IMPORTANT")
print("  Vitals (features 3-5): SOMEWHAT IMPORTANT")
print("  Labs (features 6-8): NOT IMPORTANT")
print("  Other (features 9-11): NOT IMPORTANT")

# ============================================================================
# 2. Define Feature Groups
# ============================================================================

groups = {
    "Demographics": [0, 1, 2],
    "Vitals": [3, 4, 5],
    "Labs": [6, 7, 8],
    "Other": [9, 10, 11]
}

print("\n" + "="*60)
print("Defined Feature Groups:")
print("="*60)
for group_name, indices in groups.items():
    print(f"  {group_name}: features {indices}")

# ============================================================================
# 3. Run Group-wise Feature Importance
# ============================================================================

print("\n" + "="*60)
print("Running Group-wise Feature Importance...")
print("="*60)

results = feature_importance(
    X=X,
    y=y,
    outcome="binary",
    iterations=3,
    groups=groups,            # Specify groups
    kfolds=5,
    perm_count=50,
    epochs=100,
    print_level=0
)

# ============================================================================
# 4. Display Results
# ============================================================================

print("\n" + "="*60)
print("Group Importance Results:")
print("="*60)
print(results["fi_summary"])

print("\n" + "="*60)
print("Interpretation:")
print("="*60)
print("Demographics should be most important (most negative)")
print("Vitals should be moderately important")
print("Labs and Other should be near zero (not important)")

# ============================================================================
# 5. Visualize
# ============================================================================

print("\nGenerating visualization...")
plot_fi_bar(results["fi_summary"], top_k=4,
           title="Group-wise Feature Importance")

print("\n" + "="*60)
print("Example Complete!")
print("="*60)
print("\nKey Takeaway:")
print("Group-wise analysis lets you test importance of")
print("conceptual units (demographics, vitals, etc.) rather")
print("than just individual features.")