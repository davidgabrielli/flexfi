"""
Tests for neural network models.
"""

import torch
import pytest
from flexfi.core.models import DeepNeuralNetwork


def test_model_creation():
    """Test that model can be created."""
    model = DeepNeuralNetwork(
        input_size=10,
        hidden_size1=32,
        hidden_size2=64,
        output_size=1,
        hidden_activation="relu"
    )
    
    assert model is not None
    assert len(model.W) == 3  # 3 weight matrices


def test_model_forward():
    """Test forward pass."""
    model = DeepNeuralNetwork(
        input_size=10,
        hidden_size1=32,
        hidden_size2=64,
        output_size=1,
        hidden_activation="relu"
    )
    
    X = torch.randn(5, 10)
    output = model(X)
    
    assert output.shape == (5, 1)


def test_activation_functions():
    """Test all activation functions work."""
    for activation in ["relu", "sigmoid", "tanh"]:
        model = DeepNeuralNetwork(
            input_size=10,
            hidden_size1=16,
            hidden_size2=32,
            output_size=1,
            hidden_activation=activation
        )
        
        X = torch.randn(5, 10)
        output = model(X)
        assert output.shape == (5, 1)


def test_invalid_activation():
    """Test that invalid activation raises error."""
    with pytest.raises(ValueError, match="Unsupported activation"):
        DeepNeuralNetwork(
            input_size=10,
            hidden_size1=16,
            hidden_size2=32,
            output_size=1,
            hidden_activation="invalid"
        )


def test_sgl_penalty():
    """Test sparse group lasso penalty computation."""
    model = DeepNeuralNetwork(
        input_size=10,
        hidden_size1=16,
        hidden_size2=32,
        output_size=1,
        hidden_activation="relu"
    )
    
    # Test 'all' penalty
    penalty_all = model.SGLpenalty(SGL_type='all', lambda_best=0.01, thres=0.001)
    assert penalty_all.item() >= 0
    
    # Test 'input' penalty
    penalty_input = model.SGLpenalty(SGL_type='input', lambda_best=0.01, thres=0.001)
    assert penalty_input.item() >= 0
    assert penalty_input.item() <= penalty_all.item()  # Input only should be less


if __name__ == "__main__":
    pytest.main([__file__, "-v"])