"""
Advanced usage example with custom hyperparameters.

This example demonstrates:
- Manual hyperparameter tuning
- Different activation functions
- Regularization options
- GPU usage (if available)
"""

import torch
from flexfi import feature_importance

print("="*60)
print("flexfi Advanced Usage Example")
print("="*60)

torch.manual_seed(42)

# ============================================================================
# 1. Create Complex Data
# ============================================================================

n_samples = 1000
n_features = 20

X = torch.randn(n_samples, n_features)

# Nonlinear relationship
y_logits = torch.sin(X[:, 0]) + X[:, 1]**2 + 0.5 * X[:, 2] * X[:, 3]
y_prob = torch.sigmoid(y_logits)
y = torch.bernoulli(y_prob).long()

print(f"\nData: {n_samples} samples, {n_features} features")
print("Relationship: Nonlinear (sin, square, interaction)")

# ============================================================================
# 2. Run with Custom Hyperparameters
# ============================================================================

print("\n" + "="*60)
print("Running with Custom Hyperparameters...")
print("="*60)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

results = feature_importance(
    X=X,
    y=y,
    outcome="binary",
    iterations=5,
    
    # Model architecture
    nh1=64,                      # First hidden layer: 64 neurons
    nh2=128,                     # Second hidden layer: 128 neurons
    hidden_activation="relu",    # ReLU works well for nonlinear
    
    # Training parameters
    epochs=200,                  # More epochs for complex relationships
    learning_rate=0.001,         # Manual learning rate
    optimizer_type="Adam",       # Adam optimizer
    scheduler_type="Step",       # Learning rate scheduler
    batch_size=64,               # Batch size
    
    # Regularization
    penalty_type="L1",           # L1 penalty for sparsity
    
    # Cross-validation
    kfolds=5,
    perm_count=100,              # More permutations for stability
    
    # Output
    print_level=0,
    results_per_iteration=False
)

# ============================================================================
# 3. Display Results
# ============================================================================

print("\n" + "="*60)
print("Top 10 Most Important Features:")
print("="*60)
print(results["fi_summary"].head(10))

print("\n" + "="*60)
print("Advanced Features Demonstrated:")
print("="*60)
print("Custom network architecture (64 → 128)")
print("Manual hyperparameter specification")
print("L1 regularization for sparsity")
print("Learning rate scheduling")
print("GPU support (if available)")
print("Increased permutations for stability")

print("\n" + "="*60)
print("Example Complete!")
print("="*60)
print("\nTip: For most use cases, the automatic hyperparameter")
print("selection works well. Only customize when you have")
print("specific requirements or domain knowledge!")