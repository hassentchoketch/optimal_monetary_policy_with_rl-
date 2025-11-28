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
        deterministic: bool = False
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
        # Deduce lags p from state length (state has 3*p elements)
        p = len(state) // 3
        
        # Extract observations based on agent's expected state dimension
        if self.agent.state_dim == 2:
            # No lags: [y_t, π_t]
            # y_t is at index 0, π_t is at index p
            if len(state) >= p + 1:
                obs = np.array([state[0], state[p]])
            else:
                obs = state  # Fallback
        elif self.agent.state_dim == 4:
            # One lag: [y_t, y_{t-1}, π_t, π_{t-1}]
            if p >= 2:
                # Normal case with sufficient history
                obs = np.array([state[0], state[1], state[p], state[p+1]])
            elif p == 1:
                # Fallback for p=1 (not enough history)
                # Matches train_agent.py behavior: [y_t, π_t, π_t, i_{t-1}]
                obs = np.array([state[0], state[1], state[1], state[2]])
            else:
                obs = np.zeros(4)  # Should not happen
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