"""
Data simulation utilities for testing.
"""

import torch
import numpy as np


def simulate_data(
    n=5000,
    p=10,
    outcome_type="continuous",
    sim_function="linear",
    K=3,
    signal_features=[0, 1, 2],
    correlated_features=None,
    corr_param=0.5,
    noise_sd=0.2,
    heteroskedastic_features=None,
    interaction_pairs=None,
    outlier_fraction=0.0,
    outlier_features=None,
    seed=0
):
    """
    Predictable synthetic dataset for FI benchmarking.
    
    Parameters:
    -----------
    n : int
        Number of samples.
    p : int
        Number of features.
    outcome_type : str
        'continuous', 'binary', or 'multiclass'.
    sim_function : str
        'linear' or 'nonlinear'.
    K : int
        Number of classes for multiclass.
    signal_features : list
        Indices of truly informative features.
    correlated_features : list, optional
        Features to correlate with each other.
    corr_param : float
        Correlation strength (rho).
    noise_sd : float
        Base noise level for continuous outcomes.
    heteroskedastic_features : list, optional
        Features that drive heteroskedasticity.
    interaction_pairs : list, optional
        List of (i,j) tuples for interaction terms.
    outlier_fraction : float
        Fraction of samples with outliers.
    outlier_features : list, optional
        Features that drive outlier placement.
    seed : int
        Random seed for reproducibility.
        
    Returns:
    --------
    dict
        Dictionary with keys: X, y, and simulation parameters.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build covariance matrix
    Sigma = np.eye(p)
    if correlated_features is not None and len(correlated_features) > 1:
        for i in correlated_features:
            for j in correlated_features:
                if i != j:
                    Sigma[i, j] = corr_param

    # Generate predictors
    L = np.linalg.cholesky(Sigma)
    Z = np.random.normal(0, 1, size=(n, p))
    X = torch.tensor(Z @ L.T, dtype=torch.float32)

    # True signal
    if sim_function == "linear":
        y0 = sum(X[:, j] for j in signal_features)
    elif sim_function == "nonlinear":
        y0 = torch.zeros(n)
        for idx, j in enumerate(signal_features):
            if idx % 3 == 0:
                y0 = y0 + torch.sigmoid(X[:, j])
            elif idx % 3 == 1:
                y0 = y0 + X[:, j] ** 2
            else:
                y0 = y0 + torch.sin(X[:, j])
    else:
        raise ValueError("sim_function must be 'linear' or 'nonlinear'")
    
    y0 = y0.reshape(n, 1)

    # Interactions
    if interaction_pairs is not None:
        for (i, j) in interaction_pairs:
            y0 = y0 + 0.5 * (X[:, i] * X[:, j]).reshape(-1, 1)

    # Outcomes
    if outcome_type == "continuous":
        noise_scale = noise_sd * torch.ones(n, 1)
        if heteroskedastic_features is not None:
            for j in heteroskedastic_features:
                noise_scale = noise_scale * (1 + 0.5 * torch.abs(X[:, j].reshape(-1, 1)))

        noise = torch.normal(0., noise_scale)

        # Outliers
        if outlier_fraction > 0:
            mask = torch.rand(n, 1) < outlier_fraction
            noise = noise + mask * torch.normal(0., 5.0, size=(n, 1))

        if outlier_features is not None:
            for j in outlier_features:
                mask = torch.abs(X[:, j]) > 2
                noise = noise + mask.reshape(-1, 1) * torch.normal(0., 5.0, size=(n, 1))

        y = y0 + noise

    elif outcome_type == "binary":
        probs = torch.sigmoid(2 * y0)
        y = torch.bernoulli(probs)

    elif outcome_type == "multiclass":
        logits = torch.cat(
            [y0 + torch.normal(0., 0.5, size=(n, 1)) + k for k in range(K)],
            dim=1
        )
        probs = torch.softmax(logits, dim=1)
        y = torch.multinomial(probs, num_samples=1)

    else:
        raise ValueError("outcome_type must be 'continuous','binary','multiclass'")

    # Collect results
    sim_dict = {
        "X": X,
        "y": y,
        "signal_features": signal_features,
        "correlated_features": correlated_features,
        "corr_param": corr_param if correlated_features else None,
        "interaction_pairs": interaction_pairs,
        "heteroskedastic_features": heteroskedastic_features,
        "outlier_features": outlier_features,
        "outlier_fraction": outlier_fraction,
        "Sigma": Sigma
    }

    return sim_dict