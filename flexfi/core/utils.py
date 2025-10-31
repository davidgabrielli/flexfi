"""
Utility functions for model training and feature importance estimation.

Includes:
- Model performance tracking
- Tensor permutation for permutation-based feature importance
- Weight initialization utilities (Xavier, He, Uniform)
- Optimizer and scheduler creation
- Common loss functions
- Regularization penalty application (L1, Group Lasso)
"""

import torch
import numpy as np
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_squared_error


# ---------------------------------------------------------------------
# Compute Model Metrics
# ---------------------------------------------------------------------

def compute_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    outcome: str
) -> Dict[str, float]:
    """
    Computes minimal, task-appropriate performance metrics for evaluating
    model quality in the context of feature importance analysis.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model to evaluate.
    data_loader : DataLoader
        DataLoader containing the dataset for evaluation (shuffle=False).
    outcome : str
        Type of prediction task: "binary", "multiclass", or "continuous".

    Returns
    -------
    dict
        Dictionary of key performance metrics:
        - Binary: {'AUC', 'Accuracy'}
        - Multiclass: {'Accuracy'}
        - Continuous: {'R2', 'MSE'}
    """

    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for X_batch, y_batch in data_loader:
            inputs = X_batch.float()

            # Multiclass classification
            if outcome == "multiclass":
                labels = y_batch.long()
                preds = model(inputs).softmax(dim=1).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

            # Binary classification
            elif outcome == "binary":
                labels = y_batch.float().reshape(-1, 1)
                probs = torch.sigmoid(model(inputs)).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy().flatten())

            # Continuous regression
            elif outcome == "continuous":
                labels = y_batch.cpu().numpy().flatten()
                preds = model(inputs).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(labels)

            else:
                raise ValueError("Outcome must be 'binary', 'multiclass', or 'continuous'.")

    metrics: Dict[str, float] = {}

    # ------------------------
    # Binary classification
    # ------------------------
    if outcome == "binary":
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        metrics["AUC"] = roc_auc_score(all_labels, all_preds)
        metrics["Accuracy"] = accuracy_score(all_labels, (all_preds > 0.5).astype(int))

    # ------------------------
    # Multiclass classification
    # ------------------------
    elif outcome == "multiclass":
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        metrics["Accuracy"] = accuracy_score(all_labels, all_preds)

    # ------------------------
    # Continuous regression
    # ------------------------
    elif outcome == "continuous":
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        metrics["R2"] = r2_score(all_labels, all_preds)
        metrics["MSE"] = mean_squared_error(all_labels, all_preds)

    return metrics


# ---------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------
def permute_input_tensor(
    input_tensor: torch.Tensor, 
    feature_indices: List[int]
) -> torch.Tensor:
    """
    Returns a copy of the input tensor with selected columns randomly permuted.
    
    Parameters:
    -----------
    input_tensor : torch.Tensor
        Input tensor of shape (n_samples, n_features).
    feature_indices : list of int
        Indices of columns/features to permute together. Can be a single feature [0]
        or a group of features [0, 1, 2].
        
    Returns:
    --------
    torch.Tensor
        Copy of input tensor with specified columns permuted (same permutation
        applied to all features in the group).
        
    Notes:
    ------
    All features in the group receive the same permutation.
    """

    input_copy = input_tensor.clone()

    # Generate single permutation to apply to all features in group
    n_samples = input_copy.shape[0]
    permute_idx = torch.randperm(n_samples)
    
    # Apply same permutation to all features in the group
    for idx in feature_indices:
        input_copy[:, idx] = input_copy[permute_idx, idx]
    
    return input_copy


# ---------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------
def init_weights_xavier_uniform(m: nn.Module) -> None:
    """Apply Xavier uniform initialization to linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_weights_he_normal(m: nn.Module) -> None:
    """Apply He normal initialization to linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_weights_uniform(m: nn.Module) -> None:
    """Apply uniform initialization to linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-1, b=1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_weights(model: nn.Module, init_method: str) -> nn.Module:
    """
    Apply the specified weight initialization to all linear layers in the model.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model to initialize.
    init_method : str
        Initialization method: "xavier_uniform" or "he_normal".
        
    Returns:
    --------
    nn.Module
        Model with initialized weights.
        
    Raises:
    -------
    ValueError
        If init_method is not supported.
    """
    init_method_lower = init_method.lower()
    if init_method_lower == "xavier_uniform":
        model.apply(init_weights_xavier_uniform)
    elif init_method_lower == "he_normal":
        model.apply(init_weights_he_normal)
    else:
        raise ValueError(f"Unsupported init_method: {init_method}")
    return model


# ---------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------
def create_optimizer(
    model: nn.Module,
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float,
) -> Optimizer:
    """
    Create and return a PyTorch optimizer for the given model.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model to optimize.
    optimizer_type : str
        Type of optimizer: "adam", "sgd", or "rmsprop".
    learning_rate : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay (L2 penalty) coefficient.
        
    Returns:
    --------
    Optimizer
        Configured PyTorch optimizer.
        
    Raises:
    -------
    ValueError
        If optimizer_type is not supported.
    """
    opt = optimizer_type.lower()
    if opt == "adam":
        return torch.optim.Adam(model.parameters(), 
                                lr=learning_rate, 
                                weight_decay=weight_decay)
    elif opt == "sgd":
        return torch.optim.SGD(model.parameters(), 
                               lr=learning_rate, 
                               weight_decay=weight_decay)
    elif opt == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), 
                                   lr=learning_rate, 
                                   weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")


# ---------------------------------------------------------------------
# Learning rate schedulers
# ---------------------------------------------------------------------
def create_scheduler(
    optimizer: Optimizer, 
    scheduler_type: Optional[str], 
    epochs: int
) -> Optional[LRScheduler]:
    """
    Create a learning rate scheduler for the given optimizer.
    
    Parameters:
    -----------
    optimizer : Optimizer
        PyTorch optimizer to schedule.
    scheduler_type : str or None
        Type of scheduler: "lambda", "step", "exponential", or None.
    epochs : int
        Total number of training epochs.
        
    Returns:
    --------
    LRScheduler or None
        Configured learning rate scheduler, or None if scheduler_type is None.
        
    Raises:
    -------
    ValueError
        If scheduler_type is not supported.
    """
    if scheduler_type is None:
        return None

    sched = scheduler_type.lower()
    if sched == "lambda":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1 / (1 + np.log(epoch + 2))
        )
    elif sched == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(10, epochs // 4), gamma=0.5
        )
    elif sched == "exponential":
        gamma = 0.9 if epochs > 100 else 0.95
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif sched in ("none",):
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")


# ---------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------
def make_loss_function(loss_fn: str, reduction: str = "mean") -> nn.Module:
    """
    Return a PyTorch loss function based on the provided name.
    
    Parameters:
    -----------
    loss_fn : str
        Loss function name: "bcewithlogitsloss", "crossentropyloss", or "mseloss".
    reduction : str, default="mean"
        Reduction mode: "mean", "sum", or "none".
        
    Returns:
    --------
    nn.Module
        Configured PyTorch loss function.
        
    Raises:
    -------
    ValueError
        If loss_fn is not supported.
    """
    lf = loss_fn.lower()
    if lf == "bcewithlogitsloss":
        return nn.BCEWithLogitsLoss(reduction=reduction)
    elif lf == "crossentropyloss":
        return nn.CrossEntropyLoss(reduction=reduction)
    elif lf == "mseloss":
        return nn.MSELoss(reduction=reduction)
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")


# ---------------------------------------------------------------------
# Regularization penalties
# ---------------------------------------------------------------------
def apply_penalty(
    model: nn.Module,
    input_size: int,
    penalty_type: Optional[str],
    loss: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the specified penalty (L1 or Group Lasso) to the loss value.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model (must have SGLpenalty method if using grouplasso).
    input_size : int
        Number of input features.
    penalty_type : str or None
        Type of penalty: "l1", "grouplasso", "l2", "none", or None.
    loss : torch.Tensor
        Base loss value to which penalty will be added.
        
    Returns:
    --------
    torch.Tensor
        Loss value with penalty applied (if applicable).
        
    Raises:
    -------
    ValueError
        If penalty_type is not supported.
    """
    if penalty_type is None:
        return loss

    ptype = penalty_type.lower()

    if ptype == "l1":
        l1_penalty = 1e-4 if input_size > 100 else 1e-5
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_penalty * l1_norm

    elif ptype == "grouplasso":
        sgl_lambda = 1e-3
        sgl_type = "all"
        sgl_thres = 0.001
        group_lasso_penalty = model.SGLpenalty(
            SGL_type=sgl_type, lambda_best=sgl_lambda, thres=sgl_thres
        )
        loss = loss + group_lasso_penalty

    # L2 is handled by optimizer weight_decay
    elif ptype in {"l2", "none"}:
        pass
    else:
        raise ValueError(f"Unsupported penalty: {penalty_type}")

    return loss