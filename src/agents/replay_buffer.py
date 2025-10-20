"""
Experience replay buffer for DDPG agent.

Implements step h) from Table 1: Store transitions in buffer B.
"""

import numpy as np
from typing import Tuple


class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.
    
    Stores transitions (x_t, i_t, r_t, x_{t+1}) and samples mini-batches
    uniformly for training (step i in Table 1).
    
    Using experience replay provides:
    1. Sample efficiency: Each experience can be used multiple times
    2. Stability: Breaks temporal correlations in sequential data
    3. Off-policy learning: Can learn from past policies
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store (10000 in paper)
            state_dim: Dimension of state vector
            action_dim: Dimension of action vector
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Preallocate arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        # Buffer management
        self.position = 0
        self.size = 0
    
    def add(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add transition to buffer.
        
        Implements step h) from Table 1:
        Store transition (x_t, i_t, r_t, x_{t+1}) in B
        
        Args:
            state: Current state x_t
            action: Action taken i_t
            reward: Reward received r_t
            next_state: Next state x_{t+1}
            done: Episode termination flag
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Circular buffer: overwrite oldest when full
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample random mini-batch from buffer.
        
        Implements step i) from Table 1:
        Sample a random minibatch of N transitions (x_j, i_j, r_j, x_{j+1}) from B
        
        Args:
            batch_size: Number of transitions to sample (64 in paper)
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            Each with shape [batch_size, ...]
        """
        # Sample indices uniformly
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= batch_size