"""
Test suite for flexfi package with shared fixtures.
"""

import pytest
import torch
from tests.test_data import simulate_data


@pytest.fixture(scope="session")
def binary_sim_data():
    """Fixture providing binary classification simulated data."""
    sim_dict = simulate_data(
        n=500,  # Smaller for faster tests
        p=10,
        outcome_type="binary",
        sim_function="linear",
        signal_features=[0, 1, 2],
        correlated_features=[0, 4],
        corr_param=0.5,
        seed=42
    )
    return sim_dict


@pytest.fixture(scope="session")
def continuous_sim_data():
    """Fixture providing continuous outcome simulated data."""
    sim_dict = simulate_data(
        n=500,
        p=10,
        outcome_type="continuous",
        sim_function="linear",
        signal_features=[0, 1, 2],
        correlated_features=[0, 4],
        corr_param=0.5,
        noise_sd=0.2,
        seed=42
    )
    return sim_dict


@pytest.fixture(scope="session")
def multiclass_sim_data():
    """Fixture providing multiclass simulated data."""
    sim_dict = simulate_data(
        n=500,
        p=10,
        outcome_type="multiclass",
        sim_function="linear",
        K=3,
        signal_features=[0, 1, 2],
        seed=42
    )
    return sim_dict


@pytest.fixture(scope="session")
def nonlinear_sim_data():
    """Fixture providing nonlinear continuous data."""
    sim_dict = simulate_data(
        n=500,
        p=10,
        outcome_type="continuous",
        sim_function="nonlinear",
        signal_features=[0, 1, 2, 3],
        seed=42
    )
    return sim_dict
