"""
Feature importance computation using permutation-based methods.
"""

import torch
import pandas as pd
import numpy as np
import time
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from statistics import NormalDist
from typing import Optional, Tuple, Dict, List, Callable

from .utils import (
    make_loss_function, 
    create_optimizer, 
    create_scheduler, 
    init_weights, 
    apply_penalty,
    permute_input_tensor
)
from .models import DeepNeuralNetwork
from .data import SimpleDataset


def feature_importance_statistics(
    returnscore: torch.Tensor, 
    returnscorelist: torch.Tensor, 
    n_samples: int, 
    feature_indices: List[int],
    feature_names: Optional[List[str]] = None,
    group_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregates feature importance scores across samples and computes summary statistics 
    for a single iteration.

    This function takes the raw per-sample feature importance scores and permutation losses,
    computes the overall mean importance score, its standard error, and a p-value assuming
    normality.

    Parameters:
    -----------
    returnscore : torch.Tensor
        Tensor of shape (n_samples,) with importance scores for each sample.
    returnscorelist : torch.Tensor  
        Tensor of shape (n_samples, perm_count) with permutation losses for each sample
        across all permutations.
    n_samples : int
        Total number of samples in the dataset.
    feature_indices : list of int
        Indices of feature(s) being analyzed.
    feature_names : list of str, optional
        Names of all features in dataset.
    group_name : str, optional
        Name for this group (if analyzing a group). If None, creates name
        from feature indices.

    Returns:
    --------
    pd.DataFrame
        Single-row DataFrame with columns:
        ['Covariate/Group', 'Importance Score', 'Importance Score STD', 'P-Value']
    """
    
    # Extract importance scores
    importance_score_all = returnscore
    importance_score_list_all = returnscorelist
    
    # Compute mean importance score across all samples
    stt = torch.mean(importance_score_all).item()
    
    # Compute standard error: sqrt(sum of variances across permutations) / n_samples
    stt_std = torch.sqrt(
        torch.sum(torch.var(importance_score_list_all, dim=1, unbiased=False))
        ).item() / n_samples
    
    # Calculate p-value using normal distribution
    standard_stt = stt / stt_std
    rv = NormalDist(mu=0., sigma=1.)
    pval = rv.cdf(standard_stt)
    
    # Create name for feature/group
    if group_name is not None:
        # User provided a group name
        name = group_name
    elif len(feature_indices) == 1:
        # Single feature
        if feature_names is not None:
            name = feature_names[feature_indices[0]]
        else:
            name = f"Feature_{feature_indices[0]}"
    else:
        # Multiple features - create descriptive name
        if feature_names is not None:
            feature_list = [feature_names[i] for i in feature_indices]
            name = f"Group({', '.join(feature_list)})"
        else:
            name = f"Group({', '.join(map(str, feature_indices))})"
    
    # Create results DataFrame
    fi_statistics_per_iteration = pd.DataFrame(
        [[name, stt, stt_std, pval]],
        columns=['Feature/Group', 'Importance Score', 'Importance Score STD', 'P-Value']
    )
    
    return fi_statistics_per_iteration


def feature_importance_scores(
    model: nn.Module, 
    outcome: str, 
    data_loader: DataLoader, 
    perm_count: int, 
    feature_indices: List[int],
    loss_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes per-sample feature importance scores using permutation-based approach.

    For a given feature/covariate or group of features, this function compares the model's prediction loss on 
    the original dataset versus the loss after permuting those feature's values across 
    samples. The difference in loss indicates how important those features are for the 
    model's predictions, with greater negative values indicating greater importance.

    Parameters:
    -----------
    model : torch.nn.Module
        Trained PyTorch model to evaluate.
    outcome : str
        Type of prediction task: "binary", "multiclass", or "continuous".
        Used to format target labels appropriately.
    data_loader : torch.utils.data.DataLoader
        DataLoader containing the evaluation dataset.
    perm_count : int
        Number of permutations to perform for each sample.
    feature_indices : list of int
        Indices of feature to permute. Single feature [0] or group [0, 1, 2].
    loss_fn : callable
        Loss function with reduction='none' to get per-sample losses.

    Returns:
    --------
    tuple of torch.Tensor
        - returnscore: Tensor of shape (n_samples,) with importance scores 
          (original_loss - mean_permuted_loss) for each sample
        - returnscorelist: Tensor of shape (n_samples, perm_count) with importance 
          scores for each sample across all permutations
    """
    returnscore = []
    returnscorelist = []
    
    model.eval()  # Set model to evaluation mode
    
    for X_batch, y_batch in data_loader:
        inputs = X_batch.float()
        
        # Format labels based on outcome type
        if outcome == "multiclass":
            labels = y_batch.long()
        else:
            labels = y_batch.float().reshape(-1, 1)   
                
        with torch.inference_mode():
            # Get original predictions without permutation
            predictors = model(inputs)
            pred_loss = loss_fn(predictors, labels)
        
            # Initialize storage for permutation results
            pred_loss_perm = torch.zeros((inputs.shape[0], perm_count))
            
            # Permute the selected feature multiple times
            for i in range(perm_count):
                inputs_perm = permute_input_tensor(inputs, feature_indices)
                predictors_perm = model(inputs_perm)
                perm_loss = loss_fn(predictors_perm, labels)
                pred_loss_perm[:, i] = perm_loss.squeeze()
        
        # Calculate average permutation loss for each sample
        pred_loss_perm_avg = pred_loss_perm.mean(dim=1, keepdim=True)
        
        # Feature importance = original_loss - average_permuted_loss
        score_k = pred_loss - pred_loss_perm_avg
        scorelist_k = pred_loss - pred_loss_perm
        
        returnscore.append(score_k)
        returnscorelist.append(scorelist_k)
    
    # Combine results from all batches
    returnscore = torch.cat(returnscore, dim=0)
    returnscorelist = torch.cat(returnscorelist, dim=0)
    
    return returnscore, returnscorelist


def feat_imp_per_unit(
    X: torch.Tensor, 
    y: torch.Tensor, 
    outcome: str, 
    model: Optional[nn.Module], 
    feature_indices: List[int],
    feature_names: Optional[List[str]],
    group_name: Optional[str],
    model_seed: int, 
    iteration: int, 
    kfolds: int, 
    perm_count: int, 
    epochs: int, 
    loss_function: str, 
    nh1: int, 
    nh2: int, 
    hidden_activation: str, 
    init_method: str, 
    learning_rate: float, 
    optimizer_type: str, 
    scheduler_type: str, 
    penalty_type: Optional[str], 
    batch_size: int, 
    weight_decay: float, 
    print_level: int
) -> pd.DataFrame:
    """
    Computes feature importance for a single feature/covariate or group of features using cross-validation.

    This function trains a fresh model on each fold of the data and evaluates 
    the importance of the specified feature or group using permutation-based feature 
    importance on the held-out test set.

    Parameters:
    -----------
    X : torch.Tensor
        Input feature matrix of shape (n_samples, n_features).
    y : torch.Tensor
        Target values of shape (n_samples, 1) or (n_samples,).
    outcome : str
        Type of prediction task: "binary", "multiclass", or "continuous".
    model : nn.Module or None
        Placeholder parameter for potential future model input (currently unused).
    feature_indices : list of int
        Indices of feature(s) to evaluate. [0] for single, [0,1,2] for group.
    feature_names : list of str or None
        Names of all features in the dataset.
    group_name : str or None
        Optional name for this group/feature.
    model_seed : int
        Random seed for reproducible cross-validation splits.
    iteration : int
        Current iteration number for tracking across multiple runs.
    kfolds : int
        Number of cross-validation folds.
    perm_count : int
        Number of permutations for feature importance calculation.
    epochs : int
        Number of training epochs per fold.
    loss_function : str
        Loss function name (e.g., "BCEWithLogitsLoss", "MSELoss").
    nh1 : int
        Number of neurons in first hidden layer.
    nh2 : int
        Number of neurons in second hidden layer.
    hidden_activation : str
        Activation function for hidden layers ("relu", "sigmoid", "tanh").
    init_method : str
        Weight initialization method ("xavier_uniform", "he_normal").
    learning_rate : float
        Learning rate for optimizer.
    optimizer_type : str
        Optimizer type ("Adam", "SGD", "RMSprop").
    scheduler_type : str
        Learning rate scheduler type ("Step", "Exponential", etc.).
    penalty_type : str or None
        Regularization penalty type ("L1", "L2", "GroupLasso", None).
    batch_size : int
        Batch size for training.
    weight_decay : float
        Weight decay (L2 regularization) parameter.
    print_level : int
        Print training loss every N epochs (0 to disable).

    Returns:
    --------
    pd.DataFrame
        Single-row DataFrame containing feature importance statistics with columns:
        ['Feature/Group', 'Importance Score', 'Importance Score STD', 'P-Value', 'Iteration']
    """

    # extract device from X
    device = X.device

    # Initialize loss functions for training and evaluation
    loss_fn_mean = make_loss_function(loss_function, reduction='mean')
    loss_fn_none = make_loss_function(loss_function, reduction='none')
    
    # Set up cross-validation
    kfold = KFold(n_splits=kfolds, shuffle=True, random_state=model_seed)
    full_dataset = SimpleDataset(X, y)

    returnscore = []
    returnscorelist = []
    
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(full_dataset)):
        print(f'Running feature importance for fold {fold + 1}/{kfolds}')
        print('-----------------------------')

        # Create data loaders for current fold
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)
        train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_subsampler)
        valid_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=valid_subsampler)
        
        # Create fresh model for this fold
        input_size = X.shape[-1]
        if model is None:  # Always true currently
            model = DeepNeuralNetwork(
                input_size=input_size,
                hidden_size1=nh1,
                hidden_size2=nh2,
                output_size=1, 
                hidden_activation=hidden_activation
            )

        # Move model to device
        model = model.to(device) 

        # Initialize model weights
        model = init_weights(model, init_method)
        
        # Set up optimizer and scheduler
        optimizer = create_optimizer(model, optimizer_type, learning_rate, weight_decay)
        scheduler = create_scheduler(optimizer, scheduler_type, epochs)
                    
        # Get training data (single batch since we use full fold)
        X_train, y_train = next(iter(train_loader))
        inputs_train = X_train.float()
        labels_train = y_train.float().reshape(-1, 1)
        
        # Training loop
        for epoch in range(epochs):
            model.train()

            optimizer.zero_grad()
            outputs = model(inputs_train)
            loss = loss_fn_mean(outputs, labels_train)
            
            # Apply regularization penalty if specified
            loss = apply_penalty(model, input_size, penalty_type, loss)
            
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # Print progress
            if print_level > 0 and epoch % print_level == 0:
                print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}")

        # Compute feature importance on validation set
        fold_returnscore, fold_returnscorelist = feature_importance_scores(
            model=model,
            outcome=outcome,
            data_loader=valid_loader,
            perm_count=perm_count,
            feature_indices=feature_indices,
            loss_fn=loss_fn_none
        )
        
        # Accumulate results across folds
        returnscore.append(fold_returnscore)
        returnscorelist.append(fold_returnscorelist)
    
    # Combine results from all folds
    returnscore = torch.cat(returnscore, dim=0)
    returnscorelist = torch.cat(returnscorelist, dim=0)
    
    # Calculate summary statistics
    fi_statistics_per_iteration = feature_importance_statistics(
        returnscore=returnscore,
        returnscorelist=returnscorelist,
        n_samples=len(X),
        feature_indices=feature_indices,
        feature_names=feature_names,
        group_name=group_name
    )   
    
    # Add iteration tracking
    fi_statistics_per_iteration["Iteration"] = iteration
    
    return fi_statistics_per_iteration


def feature_importance(
    X: torch.Tensor, 
    y: torch.Tensor, 
    outcome: str, 
    iterations: int, 
    feature_names: Optional[List[str]] = None,
    groups: Optional[Dict[str, List[int]]] = None,
    perm_count: int = 100, 
    kfolds: int = 5, 
    epochs: Optional[int] = None, 
    nh1: Optional[int] = None, 
    nh2: Optional[int] = None, 
    learning_rate: Optional[float] = None,
    hidden_activation: str = "relu", 
    init_method: str = "xavier_uniform", 
    optimizer_type: Optional[str] = None, 
    scheduler_type: str = "Step", 
    penalty_type: Optional[str] = None,
    batch_size: Optional[int] = None, 
    model: Optional[nn.Module] = None, 
    random_state: str = "varied", 
    print_level: int = 50, 
    results_per_iteration: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Computes permutation-based feature importance for all features or specified groups 
    of features using cross-validation.

    This function supports both individual feature importance and group-wise
    importance analysis. When groups are specified, it evaluates the importance
    of feature groups by permuting all features in a group together.

    Parameters:
    -----------
    X : torch.Tensor
        Input feature matrix of shape (n_samples, n_features).
    y : torch.Tensor
        Target values. Shape depends on outcome type.
    outcome : str
        Type of prediction task: "binary", "multiclass", or "continuous".
    iterations : int
        Number of iterations to run with different random seeds.
    feature_names : list of str, optional
        Names for each feature. If None, uses "Feature_0", "Feature_1", etc.
    groups : dict, optional
        Dictionary mapping group names to lists of feature indices.
        Example: {"Vitals": [0, 1, 2], "Labs": [3, 4, 5]}
        If None, evaluates each feature individually.
    perm_count : int, default=100
        Number of permutations for each feature importance calculation.
    kfolds : int, default=5
        Number of cross-validation folds.
    epochs : int, optional
        Number of training epochs. Auto-determined based on dataset size if None.
    nh1 : int, optional
        First hidden layer size. Auto-determined based on feature count if None.
    nh2 : int, optional
        Second hidden layer size. Auto-determined based on feature count if None.
    learning_rate : float, optional
        Learning rate. Auto-determined based on model size and optimizer if None.
    hidden_activation : str, default="relu"
        Activation function for hidden layers.
    init_method : str, default="xavier_uniform"
        Weight initialization method.
    optimizer_type : str, optional
        Optimizer type. Auto-determined based on dataset size if None.
    scheduler_type : str, default="Step"
        Learning rate scheduler type.
    penalty_type : str, optional
        Regularization type ("L1", "L2", "GroupLasso", None).
    batch_size : int, optional
        Training batch size. Auto-determined based on dataset size if None.
    model : nn.Module, optional
        Placeholder for potential custom model input (currently unused).
    random_state : str, default="varied"
        Random seed strategy: "varied" for different seeds, "fixed" for reproducibility.
    print_level : int, default=50
        Print training progress every N epochs.
    results_per_iteration : bool, default=False
        Whether to return per-iteration results in addition to summary.

    Returns:
    --------
    dict
        Dictionary containing:
        - "fi_summary": DataFrame with mean importance scores across iterations
        - "fi_per_iteration": DataFrame with per-iteration results (if requested)

    Raises:
    -------
    TypeError
        If X or y are not torch.Tensor.
    ValueError
        If parameters are invalid or incompatible.
    """
    
    # Input validation
    if not isinstance(X, torch.Tensor):
        raise TypeError("X must be a torch.Tensor")
    if not isinstance(y, torch.Tensor):
        raise TypeError("y must be a torch.Tensor")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples. Got {X.shape[0]} and {y.shape[0]}")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if outcome not in ["binary", "continuous", "multiclass"]:
        raise ValueError("outcome must be 'binary', 'continuous', or 'multiclass'")

    start_time = time.time()

    # Detect and use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    X = X.to(device)
    y = y.to(device)

    # Extract dataset characteristics
    n_samples, n_features = X.shape

    # Validate feature_names
    if feature_names is not None and len(feature_names) != n_features:
        raise ValueError(f"feature_names must have length {n_features}")
    
    # Set up evaluation units (individual features or groups)
    if groups is None:
        # Individual feature mode
        eval_units = [(f"Feature_{i}" if feature_names is None else feature_names[i], [i]) 
                      for i in range(n_features)]
    else:
        # Group mode - validate groups
        for group_name, indices in groups.items():
            if not all(0 <= idx < n_features for idx in indices):
                raise ValueError(f"Group '{group_name}' contains invalid feature indices")
        eval_units = [(name, indices) for name, indices in groups.items()]
    
    # Set loss function based on outcome type
    if outcome == "binary":
        loss_function = "BCEWithLogitsLoss"
    elif outcome == "continuous":
        loss_function = "MSELoss"
    elif outcome == "multiclass":
        loss_function = "CrossEntropyLoss"

    # Auto-determine training parameters if not provided
    if epochs is None:
        epochs = 100 if n_samples < 1000 else 150 if n_samples < 10_000 else 200
            
    if nh1 is None or nh2 is None:
        if n_features <= 10:
            nh1, nh2 = 16, 32
        elif n_features <= 50:
            nh1, nh2 = 32, 64
        elif n_features <= 200:
            nh1, nh2 = 64, 128
        else:
            nh1, nh2 = 128, 256

    model_size = nh1 + nh2

    if optimizer_type is None:
        optimizer_type = "Adam" if n_samples < 5000 else "SGD"
    
    if learning_rate is None:
        if optimizer_type == "Adam":
            if model_size < 64:
                learning_rate = 1e-3
            elif model_size < 128:
                learning_rate = 3e-3 if n_samples > 1000 else 1e-3
            else:
                learning_rate = 5e-3 if n_samples > 5000 else 3e-3
        else:
            learning_rate = 1e-2 if n_samples < 10_000 else 5e-2

    if batch_size is None:
        batch_size = 32 if n_samples < 1000 else 64 if n_samples < 10_000 else 128

    # Weight decay based on model size and outcome
    weight_decay = 1e-2 if model_size < 64 else 1e-3 if model_size < 128 else 1e-4
    if outcome == "binary" and n_samples < 1000:
        weight_decay = max(weight_decay, 1e-2)

    all_results: List[pd.DataFrame] = []

    # Main computation loop: iterations x features
    for iteration in range(iterations):
        print(f"\n==== Running for iteration {iteration} ====")
                    
        # Set random seed for this iteration
        if random_state == "varied":            
            model_seed = np.random.randint(0, 1_000_000)
        elif random_state == "fixed":
            model_seed = 47
        else:
            model_seed = 47  # Default to fixed if invalid option
        
        # Apply random seeds
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(model_seed)
            torch.cuda.manual_seed_all(model_seed)  
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Compute importance for each feature
        for unit_name, feature_indices in eval_units:
            print(f"Running Feature Importance for {unit_name}")
            result_df = feat_imp_per_unit(        
                X=X, y=y, outcome=outcome, model=model, covariate=cov_idx,
                feature_indices=feature_indices,
                feature_names=feature_names,
                group_name=unit_name if groups is not None else None,                
                model_seed=model_seed, iteration=iteration, kfolds=kfolds,
                perm_count=perm_count, epochs=epochs, loss_function=loss_function, 
                nh1=nh1, nh2=nh2, hidden_activation=hidden_activation,
                init_method=init_method, learning_rate=learning_rate,
                optimizer_type=optimizer_type, scheduler_type=scheduler_type,
                penalty_type=penalty_type, batch_size=batch_size,
                weight_decay=weight_decay, print_level=print_level
            )
            all_results.append(result_df)
            
    # Process and format results
    fi_per_iteration = pd.concat(all_results, ignore_index=True)
    fi_per_iteration.sort_values(by=["Feature/Group", "Iteration"], inplace=True)

    def floor_format(x: float) -> str:
        """Custom formatting to avoid printing very small numbers as 0."""
        if 0 <= x < 1e-8:
            return "1.00e-8"
        else:
            return f"{x:.4g}"
    
    # Optional: print and save per-iteration results
    if results_per_iteration:
        with pd.option_context('display.float_format', floor_format):
            print("Feature Importance Results - Per Iteration Statistics:\n")
            print(fi_per_iteration.to_string(index=False))
        
        fi_per_iteration.to_csv("feature_importance_per_iteration.csv", 
                               index=False, float_format="%.4g")
        
    # Calculate summary statistics across iterations
    fi_summary = (fi_per_iteration
        .groupby("Feature/Group")[["Importance Score", "Importance Score STD", "P-Value"]]
        .mean()
        .reset_index()
    )

    # Print and save summary results
    with pd.option_context('display.float_format', floor_format):
        print("Feature Importance Results:\n")
        print(fi_summary.to_string(index=False))
    
    fi_summary.to_csv("feature_importance_summary.csv", 
                     index=False, float_format="%.4g")
    
    # Print timing information
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds.")

    # Return results
    if results_per_iteration:
        results: Dict[str, pd.DataFrame] = {
            "fi_per_iteration": fi_per_iteration,
            "fi_summary": fi_summary
        }
    else:
        results = {"fi_summary": fi_summary}
        
    return results