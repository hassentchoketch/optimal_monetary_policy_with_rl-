"""
Neural network architectures for DDPG agent.

Implements actor and critic networks as specified in Table A.2 of the paper.
"""

import torch
import torch.nn as nn
from typing import Optional


class ActorNetwork(nn.Module):
    """
    Actor network for DDPG agent.
    
    Linear version (Equation 11 with q=1, purelin activation):
        i_t = α_0 + β^0_π π_t + β^0_y y_t [+ β^1_π π_{t-1} + β^1_y y_{t-1}]
    
    Nonlinear version (full Equation 11 with tanh):
        i_t = α_0 + Σ_{j=1}^q δ_j tanh(β_j · x^z_t + α_j)
    
    where:
        x^1_t = [y_t, π_t]                    (no lags)
        x^2_t = [y_t, y_{t-1}, π_t, π_{t-1}]  (one lag)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_units: Optional[int] = None,
        linear: bool = True
    ):
        """
        Initialize actor network.
        
        Args:
            state_dim: Dimension of observation space (2 or 4)
            hidden_units: Number of hidden units for nonlinear policy
                         (None for linear, 8-10 for nonlinear from Table A.1)
            linear: If True, use linear policy (purelin activation)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.linear = linear
        
        if linear:
            # Linear policy: single layer with no activation
            # Equivalent to q=1, purelin in Equation 11
            self.network = nn.Linear(state_dim, 1)
        else:
            # Nonlinear policy: hidden layer with tanh
            assert hidden_units is not None, "hidden_units required for nonlinear policy"
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_units),
                nn.Tanh(),
                nn.Linear(hidden_units, 1)
            )
        
        # He initialization for actor (Table A.2)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute action (interest rate).
        
        Args:
            state: State tensor [batch_size, state_dim]
                  Contains [y_t, π_t] or [y_t, y_{t-1}, π_t, π_{t-1}]
            
        Returns:
            Action tensor [batch_size, 1] representing i_t
        """
        return self.network(state)


class CriticNetwork(nn.Module):
    """
    Critic network approximating Q^P(x_t, i_t) (Equation 12).
    
    Architecture from Table A.2:
    - Observation path: state → FC(n) → tanh → FC(n)
    - Action path: action → FC(n) → tanh → FC(n)
    - Common path: concatenate → FC(1)
    
    where n is the number of hidden units (1-10, optimized during training).
    
    This architecture allows the critic to learn complex relationships
    between states and actions for Q-value estimation.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: int
    ):
        """
        Initialize critic network.
        
        Args:
            state_dim: Dimension of state space (2 or 4)
            action_dim: Dimension of action space (1)
            hidden_units: Number of hidden units per path (1-10)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        
        # Observation path
        self.obs_fc1 = nn.Linear(state_dim, hidden_units)
        self.obs_fc2 = nn.Linear(hidden_units, hidden_units)
        
        # Action path
        self.action_fc1 = nn.Linear(action_dim, hidden_units)
        self.action_fc2 = nn.Linear(hidden_units, hidden_units)
        
        # Common path (concatenated features → Q-value)
        self.common_fc = nn.Linear(2 * hidden_units, 1)
        
        # Glorot initialization for critic (Table A.2)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Glorot uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to compute Q-value.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            Q-value tensor [batch_size, 1]
        """
        # Observation path
        obs = torch.tanh(self.obs_fc1(state))
        obs = torch.tanh(self.obs_fc2(obs))
        
        # Action path
        act = torch.tanh(self.action_fc1(action))
        act = torch.tanh(self.action_fc2(act))
        
        # Concatenate and compute Q-value
        combined = torch.cat([obs, act], dim=1)
        q_value = self.common_fc(combined)
        
        return q_value