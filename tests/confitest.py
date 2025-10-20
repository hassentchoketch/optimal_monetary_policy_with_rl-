"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import torch
import yaml
import os
import sys
from typing import Dict
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.svar_economy import SVAREconomy
from src.environment.ann_economy import ANNEconomy, EconomyNetwork
from src.agents.ddpg_agent import DDPGAgent
from src.policies.baseline_policies import BaselinePolicy
from src.data.data_loader import DataLoader


@pytest.fixture
def config():
    """Load test configuration."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'hyperparameters.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def set_seeds(seed):
    """Set all random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


@pytest.fixture
def svar_params():
    """Sample SVAR parameters for testing."""
    return {
        'coefficients': {
            'output_gap': {
                'const': 0.1,
                'y_lag1': 0.7,
                'pi_lag1': -0.05,
                'i_lag1': 0.15,
                'i_lag2': 0.18
            },
            'inflation': {
                'const': 0.2,
                'y_lag0': 0.12,
                'y_lag1': -0.10,
                'pi_lag1': 1.55,
                'pi_lag2': -0.63,
                'i_lag1': -0.004
            }
        },
        'shock_std': {
            'output_gap': 1.0,
            'inflation': 0.3
        }
    }


@pytest.fixture
def svar_economy(svar_params, seed):
    """Create SVAR economy for testing."""
    return SVAREconomy(
        params=svar_params['coefficients'],
        shock_std=svar_params['shock_std'],
        target_inflation=2.0,
        target_output_gap=0.0,
        seed=seed
    )


@pytest.fixture
def ann_networks():
    """Create simple ANN networks for testing."""
    network_y = EconomyNetwork(input_dim=4, hidden_units=2)
    network_pi = EconomyNetwork(input_dim=5, hidden_units=2)
    return network_y, network_pi


@pytest.fixture
def ann_economy(ann_networks, seed):
    """Create ANN economy for testing."""
    network_y, network_pi = ann_networks
    return ANNEconomy(
        network_y=network_y,
        network_pi=network_pi,
        shock_std={'output_gap': 1.0, 'inflation': 0.3},
        target_inflation=2.0,
        target_output_gap=0.0,
        seed=seed
    )


@pytest.fixture
def ddpg_agent(seed):
    """Create DDPG agent for testing."""
    return DDPGAgent(
        state_dim=2,
        action_dim=1,
        critic_hidden=2,
        actor_hidden=None,
        linear_policy=True,
        lr_actor=1e-4,
        lr_critic=1e-4,
        buffer_size=1000,
        batch_size=32,
        seed=seed
    )


@pytest.fixture
def baseline_policy():
    """Create baseline policy for testing."""
    return BaselinePolicy(rule_type='TR93', r_star=2.0, pi_star=2.0)


@pytest.fixture
def sample_state():
    """Sample economic state."""
    return np.array([
        0.5,   # y_t
        0.3,   # y_{t-1}
        2.5,   # π_t
        2.2,   # π_{t-1}
        3.0,   # i_{t-1}
        2.8    # i_{t-2}
    ], dtype=np.float32)


@pytest.fixture
def sample_data():
    """Sample time series data for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('1987-07-01', periods=n, freq='Q')
    
    data = pd.DataFrame({
        'inflation': 2.0 + np.random.randn(n) * 0.5,
        'output_gap': 0.0 + np.random.randn(n) * 1.0,
        'interest_rate': 3.0 + np.random.randn(n) * 0.5
    }, index=dates)
    
    return data