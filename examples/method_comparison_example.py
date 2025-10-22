# ============================================================================
# examples/compare_methods_example.py
# ============================================================================

"""
Comparison of individual vs. group-wise feature importance.

This example demonstrates:
- Running both individual and group-wise analysis
- Comparing the results
- Understanding when to use each approach
"""

import torch
from flexfi import feature_importance

print("="*60)
print("Comparing Individual vs. Group-wise Analysis")
print("="*60)

torch.manual_seed(42)

# ============================================================================
# 1. Create Data
# ============================================================================

n_samples = 500
n_features = 9

X = torch.randn(n_samples, n_features)

# Group 1 (features 0,1,2) is important as a GROUP
# Each individual feature has weak signal, but together they're strong
y_logits = 0.4 * X[:, 0] + 0.4 * X[:, 1] + 0.4 * X[:, 2]
y_prob = torch.sigmoid(y_logits)
y = torch.bernoulli(y_prob).long()

print("\nData structure:")
print("  Features 0-2: Each has WEAK individual signal")
print("  But TOGETHER they create STRONG group signal")
print("  Features 3-8: Pure noise")

# ============================================================================
# 2. Individual Feature Analysis
# ============================================================================

print("\n" + "="*60)
print("Method 1: Individual Feature Importance")
print("="*60)

results_individual = feature_importance(
    X=X,
    y=y,
    outcome="binary",
    iterations=3,
    kfolds=5,
    epochs=100,
    print_level=0
)

print("\nTop 5 features:")
print(results_individual["fi_summary"].head())

# ============================================================================
# 3. Group-wise Analysis
# ============================================================================

print("\n" + "="*60)
print("Method 2: Group-wise Feature Importance")
print("="*60)

groups = {
    "Important_Group": [0, 1, 2],
    "Noise_Group1": [3, 4, 5],
    "Noise_Group2": [6, 7, 8]
}

results_groupwise = feature_importance(
    X=X,
    y=y,
    outcome="binary",
    iterations=3,
    groups=groups,
    kfolds=5,
    epochs=100,
    print_level=0
)

print("\nGroup importance:")
print(results_groupwise["fi_summary"])

# ============================================================================
# 4. Comparison
# ============================================================================

print("\n" + "="*60)
print("Comparison of Results:")
print("="*60)

# Individual scores for features 0, 1, 2
indiv_scores = results_individual["fi_summary"].iloc[:3]["Importance Score"].values
group_score = results_groupwise["fi_summary"].iloc[0]["Importance Score"]

print("\nIndividual feature scores (0, 1, 2):")
for i, score in enumerate(indiv_scores):
    print(f"  Feature_{i}: {score:.4f}")

print(f"\nGroup score (features 0-2 together): {group_score:.4f}")

print("\n" + "="*60)
print("Key Insight:")
print("="*60)
print("Individual features may show weak importance,")
print("but when grouped together, their collective importance")
print("becomes more apparent!")
print("\nThis is common in:")
print("  - One-hot encoded categorical variables")
print("  - Gene sets with redundant variants")
print("  - Correlated feature clusters")

print("\n" + "="*60)
print("Example Complete!")
print("="*60)
