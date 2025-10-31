"""
Dataset wrapper for feature importance analysis.

This module provides a simple PyTorch Dataset class that wraps tensor data
for use with DataLoader during model training and evaluation.
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple


class SimpleDataset(Dataset):
    """
    A simple PyTorch Dataset for wrapping feature and target tensors.
    
    This dataset creates clones of the input tensors to ensure data independence
    and converts them to float tensors when accessed.
    
    Parameters:
    -----------
    x : torch.Tensor
        Input features of shape (n_samples, n_features).
    y : torch.Tensor
        Target values of shape (n_samples,) or (n_samples, n_outputs).
        
    Attributes:
    -----------
    x : torch.Tensor
        Cloned copy of input features.
    y : torch.Tensor
        Cloned copy of target values.
        
    """
    
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Initialize the dataset with feature and target tensors.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features.
        y : torch.Tensor
            Target values.
        """
        super().__init__()
        self.x = x.clone()
        self.y = y.clone()
        
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
        --------
        int
            Number of samples.
        """
        return len(self.x)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample from the dataset.
        
        Parameters:
        -----------
        idx : int
            Index of the sample to retrieve.
            
        Returns:
        --------
        tuple of torch.Tensor
            A tuple containing (features, target), both converted to float tensors.
        """
        return self.x[idx].float(), self.y[idx].float()