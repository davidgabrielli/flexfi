# flexfi: Flexible Feature Importance for Deep Learning

A Python package for individual and group-wise permutation feature importance analysis for PyTorch neural networks with cross-validation support.

## Features

- **Group-wise analysis**: Analyze importance of feature groups as well as individual features
- **Deep learning compatible**: Works with PyTorch neural networks
- **Statistical testing**: Provides p-values and confidence intervals for importance scores
- **Cross-validation**: Robust estimates using k-fold cross-validation
- **Multiple outcomes**: Supports binary, continuous, and multiclass outcomes

## Installation
```bash
pip install flexfi
```

## Quick Start
```python
import torch
from flexfi import feature_importance, plot_fi_bar

# Prepare your data
X = torch.randn(1000, 20)  # 1000 samples, 20 features
y = torch.randint(0, 2, (1000,))  # Binary classification

# Compute feature importance
results = feature_importance(
    X=X, 
    y=y, 
    outcome="binary",
    iterations=3,
    kfolds=5
)

# Visualize results
plot_fi_bar(results["fi_summary"], top_k=10)
```

## Citation

If you use this package in your research, please cite...