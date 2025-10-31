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
## Compatibility and Requirements
flexfi supports Python 3.8 - 3.12.
The default installation includes a CPU-only build of PyTorch.
For GPU acceleration, install a CUDA-enabled PyTorch build that matches your system.
See https://pytorch.org/get-started/locally/ for installation details.

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

If you use this package in your research, please cite:

David Gabrielli (2025).
flexfi: Flexible Feature Importance for Neural Networks.
Version 0.1.0.
URL: https://github.com/davidgabrielli/flexfi

@software{gabrielli2025flexfi,
  author       = {David Gabrielli},
  title        = {flexfi: Flexible Feature Importance for Neural Networks},
  year         = {2025},
  version      = {0.1.0},
  url          = {https://github.com/davidgabrielli/flexfi}

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.