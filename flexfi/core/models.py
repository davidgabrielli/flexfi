"""
Deep Neural Network models for feature importance analysis.

This module provides a two-hidden-layer fully connected neural network
with support for Sparse Group Lasso regularization.
"""

import torch
from torch import nn
from typing import List


class DeepNeuralNetwork(nn.Module):
    """
    A two-hidden-layer fully connected neural network.
    
    This architecture is designed for feature importance analysis with support
    for Sparse Group Lasso penalties on layer connections to encourage feature
    selection and group sparsity.
    
    Parameters:
    -----------
    input_size : int
        Number of input features.
    hidden_size1 : int
        Number of neurons in the first hidden layer.
    hidden_size2 : int
        Number of neurons in the second hidden layer.
    output_size : int
        Number of output neurons (typically 1 for regression/binary classification).
    hidden_activation : str
        Activation function for hidden layers. Options: 'relu', 'sigmoid', 'tanh'.
        
    Attributes:
    -----------
    linear_activation_stack : nn.Sequential
        Sequential container of all layers and activation functions.
    W : list of torch.nn.Parameter
        Weight matrices for each linear layer, used for penalty computation.
        
    Raises:
    -------
    ValueError
        If hidden_activation is not one of 'relu', 'sigmoid', or 'tanh'.
        
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size1: int, 
                 hidden_size2: int, 
                 output_size: int,
                 hidden_activation: str) -> None:
        """
        Initialize the neural network with specified architecture.
        
        Parameters:
        -----------
        input_size : int
            Number of input features.
        hidden_size1 : int
            Number of neurons in the first hidden layer.
        hidden_size2 : int
            Number of neurons in the second hidden layer.
        output_size : int
            Number of output neurons.
        hidden_activation : str
            Activation function for hidden layers ('relu', 'sigmoid', or 'tanh').
        """
        
        super(DeepNeuralNetwork, self).__init__()
        
        # Define activation functions
        hidden_activation = hidden_activation.lower()

        if hidden_activation == "sigmoid":
            hidden_activation_fn = nn.Sigmoid()
        elif hidden_activation == "relu":
            hidden_activation_fn = nn.ReLU()
        elif hidden_activation == "tanh":
            hidden_activation_fn = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {hidden_activation}")

        # Build network architecture
        self.linear_activation_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            hidden_activation_fn,
            nn.Linear(hidden_size1, hidden_size2),
            hidden_activation_fn,
            nn.Linear(hidden_size2, output_size)
        )    

        # Store weight matrices for regularization penalty computation
        self.W: List[nn.Parameter] = [
            self.linear_activation_stack[0].weight, 
            self.linear_activation_stack[2].weight, 
            self.linear_activation_stack[4].weight
        ]

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        input_tensor : torch.Tensor
            Input features of shape (batch_size, input_size).
            
        Returns:
        --------
        torch.Tensor
            Network predictions of shape (batch_size, output_size).
        """
        output_tensor = self.linear_activation_stack(input_tensor)
        return output_tensor
    
    def SGLpenalty(self, SGL_type: str, lambda_best: float, thres: float) -> torch.Tensor:
        """
        Compute Sparse Group Lasso (SGL) penalty for regularization.
        
        This penalty encourages both individual weight sparsity and group-level
        sparsity (entire neurons), which is useful for feature selection in
        neural networks. The penalty uses a smooth approximation of the L1 norm
        below a threshold to maintain differentiability.
        
        Parameters:
        -----------
        SGL_type : str
            Type of penalty application:
            - 'all': Apply penalty to all layers
            - 'input': Apply penalty only to the input layer (first layer)
        lambda_best : float
            Regularization strength parameter. Higher values increase sparsity.
        thres : float
            Threshold for smooth approximation of L1 norm. Below this threshold,
            a quadratic approximation is used to avoid non-differentiability at zero.
            
        Returns:
        --------
        torch.Tensor
            Scalar penalty value to be added to the loss function.
            
        Notes:
        ------
        The penalty uses L2 norm across neuron weights (columns) to encourage
        group sparsity, where entire neurons can be pruned. The smooth approximation
        for small norms is: h(x) = x²/(2*thres) + thres/2 when x < thres, else x.
        
        Examples:
        ---------
        >>> model = DeepNeuralNetwork(10, 32, 64, 1, "relu")
        >>> penalty = model.SGLpenalty(SGL_type='input', lambda_best=0.01, thres=0.001)
        >>> print(penalty.item())  # Scalar penalty value
        """
        sgl_value = torch.tensor(0.0, dtype=torch.float32)

        if SGL_type == 'all':
            # Apply penalty to all layers
            for W1 in self.W:
                # Compute L2 norm for each neuron (column-wise)
                W_p_norm = torch.norm(W1, dim=0, p=2)
                # Smooth approximation of L1 norm
                h_W_p = torch.where(
                    W_p_norm < thres, 
                    torch.pow(W_p_norm, 2) / (2 * thres) + thres / 2, 
                    W_p_norm
                ).sum()
                sgl_value += lambda_best * h_W_p
        
        elif SGL_type == 'input':
            # Apply penalty only to input layer
            W1 = self.W[0]
            W_p_norm = torch.norm(W1, dim=0, p=2)
            h_W_p = torch.where(
                W_p_norm < thres, 
                torch.pow(W_p_norm, 2) / (2 * thres) + thres / 2, 
                W_p_norm
            ).sum()
            sgl_value += lambda_best * h_W_p
            
        return sgl_value