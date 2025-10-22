"""
Tests for utility functions.
"""

import torch
import pytest
from torch import nn
from flexfi.core.utils import (
    permute_input_tensor,
    init_weights,
    create_optimizer,
    make_loss_function
)


def test_permute_single_feature():
    """Test permuting a single feature."""
    torch.manual_seed(42)
    X = torch.randn(100, 5)
    X_original = X.clone()
    
    X_permuted = permute_input_tensor(X, [2])
    
    # Column 2 should be different
    assert not torch.allclose(X_permuted[:, 2], X_original[:, 2])
    
    # Other columns should be unchanged
    assert torch.allclose(X_permuted[:, 0], X_original[:, 0])
    assert torch.allclose(X_permuted[:, 1], X_original[:, 1])
    assert torch.allclose(X_permuted[:, 3], X_original[:, 3])
    assert torch.allclose(X_permuted[:, 4], X_original[:, 4])


def test_permute_multiple_features():
    """Test permuting multiple features together."""
    torch.manual_seed(42)
    X = torch.randn(100, 5)
    X_original = X.clone()
    
    X_permuted = permute_input_tensor(X, [1, 2, 3])
    
    # Columns 1, 2, 3 should be different
    assert not torch.allclose(X_permuted[:, 1], X_original[:, 1])
    assert not torch.allclose(X_permuted[:, 2], X_original[:, 2])
    assert not torch.allclose(X_permuted[:, 3], X_original[:, 3])
    
    # Columns 0, 4 should be unchanged
    assert torch.allclose(X_permuted[:, 0], X_original[:, 0])
    assert torch.allclose(X_permuted[:, 4], X_original[:, 4])


def test_init_weights():
    """Test weight initialization methods."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Test xavier_uniform
    model_xavier = init_weights(model, "xavier_uniform")
    assert model_xavier is not None
    
    # Test he_normal
    model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 1))
    model_he = init_weights(model, "he_normal")
    assert model_he is not None
    
    # Test invalid method
    with pytest.raises(ValueError):
        init_weights(model, "invalid_method")


def test_create_optimizer():
    """Test optimizer creation."""
    model = nn.Linear(10, 1)
    
    # Test Adam
    opt_adam = create_optimizer(model, "adam", 0.001, 0.01)
    assert isinstance(opt_adam, torch.optim.Adam)
    
    # Test SGD
    opt_sgd = create_optimizer(model, "sgd", 0.01, 0.01)
    assert isinstance(opt_sgd, torch.optim.SGD)
    
    # Test RMSprop
    opt_rms = create_optimizer(model, "rmsprop", 0.001, 0.01)
    assert isinstance(opt_rms, torch.optim.RMSprop)
    
    # Test invalid optimizer
    with pytest.raises(ValueError):
        create_optimizer(model, "invalid", 0.001, 0.01)


def test_loss_functions():
    """Test loss function creation."""
    # Test BCE with logits
    loss_bce = make_loss_function("BCEWithLogitsLoss")
    assert isinstance(loss_bce, nn.BCEWithLogitsLoss)
    
    # Test MSE
    loss_mse = make_loss_function("MSELoss")
    assert isinstance(loss_mse, nn.MSELoss)
    
    # Test CrossEntropy
    loss_ce = make_loss_function("CrossEntropyLoss")
    assert isinstance(loss_ce, nn.CrossEntropyLoss)
    
    # Test invalid loss
    with pytest.raises(ValueError):
        make_loss_function("InvalidLoss")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])