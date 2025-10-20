"""
RL-based policy wrapper.

Provides a consistent interface for using trained RL agents as policies.
"""

import numpy as np
import torch
from typing import Optional
from src.agents.ddpg_agent import DDPGAgent


class RLPolicy:
    """
    Wrapper for RL-trained policy.
    
    Provides the same interface as BaselinePolicy for
    easy comparison and evaluation.
    """
    
    def __init__(
        self,
        agent: DDPGAgent,
        name: str = "RL Policy"
    ):
        """
        Initialize RL policy.
        
        Args:
            agent: Trained DDPG agent
            name: Descriptive name for this policy
        """
        self.agent = agent
        self.name = name
        
        # Set agent to evaluation mode
        self.agent.actor.eval()
    
    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = True
    ) -> float:
        """
        Compute interest rate using RL policy.
        
        Args:
            state: State vector (full state with lags)
                  Extract relevant observations based on agent's state_dim
            deterministic: If True, don't add exploration noise
            
        Returns:
            Nominal interest rate i_t
        """
        # Extract observations based on agent's expected state dimension
        if self.agent.state_dim == 2:
            # No lags: [y_t, π_t]
            obs = state[:4:2] if len(state) > 2 else state  # [y_t, π_t]
        elif self.agent.state_dim == 4:
            # One lag: [y_t, y_{t-1}, π_t, π_{t-1}]
            obs = state[:4]
        else:
            obs = state
        
        # Get action from agent
        action = self.agent.select_action(
            obs,
            add_noise=not deterministic,
            noise_scale=0.0 if deterministic else 1.0
        )
        
        return action
    
    def get_parameters(self):
        """
        Get policy parameters (if linear).
        
        Returns:
            Dictionary with parameters or note if nonlinear
        """
        return self.agent.get_policy_parameters()
    
    def __str__(self) -> str:
        """String representation."""
        params = self.get_parameters()
        if 'note' in params:
            return f"{self.name}: {params['note']}"
        else:
            return f"{self.name}: {params}"