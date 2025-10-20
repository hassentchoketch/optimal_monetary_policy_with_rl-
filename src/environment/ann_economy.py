"""
Nonlinear ANN economy implementation.

Implements the artificial neural network representation of the economy
as described in Equation (8) of the paper.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from src.environment.base_economy import BaseEconomy


class ANNEconomy(BaseEconomy):
    """
    Nonlinear ANN economy with feed-forward neural networks.
    
    Implements Equation (8):
    
    f^m = b^m_0 + Σ_{j=1}^h ν^m_j G(ω^m_j · s^m_t + b^m_j)
    
    where:
        m ∈ {y, π}: output gap or inflation
        G(·): hyperbolic tangent activation (tanh)
        h: number of hidden units (2 for y, 8 for π)
        s^m_t: input state vector
    
    The paper shows this nonlinear representation improves fit by 21%
    compared to the SVAR, especially for inflation dynamics post-COVID.
    """
    
    def __init__(
        self,
        network_y: nn.Module,
        network_pi: nn.Module,
        shock_std: Dict[str, float],
        target_inflation: float = 2.0,
        target_output_gap: float = 0.0,
        reward_weights: Optional[Dict[str, float]] = None,
        penalty_threshold: float = 2.0,
        penalty_multiplier: float = 10.0,
        seed: Optional[int] = None,
        device: str = 'cpu'
    ):
        """
        Initialize ANN economy.
        
        Args:
            network_y: Trained neural network for output gap equation
                      Input: [y_{t-1}, π_{t-1}, i_{t-1}, i_{t-2}]
                      Output: y_t (before shock)
            network_pi: Trained neural network for inflation equation
                       Input: [y_t, y_{t-1}, π_{t-1}, π_{t-2}, i_{t-1}]
                       Output: π_t (before shock)
            shock_std: Standard deviations for structural shocks
            Other args: See BaseEconomy
        """
        super().__init__(
            shock_std=shock_std,
            target_inflation=target_inflation,
            target_output_gap=target_output_gap,
            reward_weights=reward_weights,
            penalty_threshold=penalty_threshold,
            penalty_multiplier=penalty_multiplier,
            seed=seed
        )
        
        self.device = torch.device(device)
        self.network_y = network_y.to(self.device)
        self.network_pi = network_pi.to(self.device)
        
        # Set networks to evaluation mode
        self.network_y.eval()
        self.network_pi.eval()
    
    def step(
        self,
        state: np.ndarray,
        action: float
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Compute next state using ANN equations.
        
        State vector format: [y_t, y_{t-1}, π_t, π_{t-1}, i_{t-1}, i_{t-2}]
        
        Recursive structure maintained:
        1. First compute y_{t+1} = f^y(y_t, π_t, i_t, i_{t-1}) + ε^y
        2. Then compute π_{t+1} = f^π(y_{t+1}, y_t, π_t, π_{t-1}, i_t) + ε^π
        
        Args:
            state: Current state vector (6 elements)
            action: Nominal interest rate i_t
            
        Returns:
            next_state, reward, done, info
        """
        # Unpack current state
        y_t, y_lag1, pi_t, pi_lag1, i_lag1, i_lag2 = state
        i_t = action
        
        # Generate structural shocks
        shock_y = self.sample_shock('output_gap')
        shock_pi = self.sample_shock('inflation')
        
        # Step 1: Compute output gap y_{t+1}
        # Input: [y_t, π_t, i_t, i_{t-1}]
        input_y = torch.tensor(
            [y_t, pi_t, i_t, i_lag1],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            y_next_pred = self.network_y(input_y).item()
        
        y_next = y_next_pred + shock_y
        
        # Step 2: Compute inflation π_{t+1}
        # Input: [y_{t+1}, y_t, π_t, π_{t-1}, i_t]
        # Recursive: uses contemporaneous y_{t+1}
        input_pi = torch.tensor(
            [y_next, y_t, pi_t, pi_lag1, i_t],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            pi_next_pred = self.network_pi(input_pi).item()
        
        pi_next = pi_next_pred + shock_pi
        
        # Construct next state vector
        next_state = np.array([
            y_next,      # y_{t+1}
            y_t,         # y_t (becomes lag)
            pi_next,     # π_{t+1}
            pi_t,        # π_t (becomes lag)
            i_t,         # i_t (becomes lag)
            i_lag1       # i_{t-1} (becomes second lag)
        ], dtype=np.float32)
        
        # Compute reward (Equation 14)
        reward = self.compute_reward(pi_next, y_next)
        
        # Episode termination
        done = False
        
        # Diagnostic information
        info = {
            'inflation': pi_next,
            'output_gap': y_next,
            'interest_rate': i_t,
            'shock_y': shock_y,
            'shock_pi': shock_pi,
            'predictions': {
                'y_pred': y_next_pred,
                'pi_pred': pi_next_pred
            },
            'reward_components': {
                'inflation_loss': self.reward_weights['inflation'] * (pi_next - self.target_inflation) ** 2,
                'output_gap_loss': self.reward_weights['output_gap'] * y_next ** 2
            }
        }
        
        return next_state, reward, done, info
    
    def reset(
        self,
        initial_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Args:
            initial_state: Optional initial state
                          If None, initialize near steady state
        
        Returns:
            Initial state vector
        """
        if initial_state is not None:
            self.current_state = initial_state.copy()
        else:
            # Initialize near steady state with small noise
            self.current_state = np.array([
                self.target_output_gap + self.rng.normal(0, 0.1),
                self.target_output_gap + self.rng.normal(0, 0.1),
                self.target_inflation + self.rng.normal(0, 0.1),
                self.target_inflation + self.rng.normal(0, 0.1),
                self.target_inflation + self.rng.normal(0, 0.5),
                self.target_inflation + self.rng.normal(0, 0.5)
            ], dtype=np.float32)
        
        self.episode_step = 0
        return self.current_state
    
    def predict_deterministic(
        self,
        state: np.ndarray,
        action: float
    ) -> Tuple[float, float]:
        """
        Predict next inflation and output gap without shocks.
        
        Used for evaluation and counterfactual analysis.
        
        Args:
            state: Current state vector
            action: Nominal interest rate
            
        Returns:
            (y_{t+1}, π_{t+1}) without stochastic shocks
        """
        y_t, y_lag1, pi_t, pi_lag1, i_lag1, i_lag2 = state
        i_t = action
        
        # Output gap prediction (no shock)
        input_y = torch.tensor(
            [y_t, pi_t, i_t, i_lag1],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            y_next = self.network_y(input_y).item()
        
        # Inflation prediction (no shock)
        input_pi = torch.tensor(
            [y_next, y_t, pi_t, pi_lag1, i_t],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            pi_next = self.network_pi(input_pi).item()
        
        return y_next, pi_next


class EconomyNetwork(nn.Module):
    """
    Feed-forward neural network for economy equation.
    
    Architecture (Equation 8):
    - Input layer
    - Hidden layer with tanh activation
    - Output layer (linear)
    
    Weights initialized using Glorot initialization (Table A.2).
    """
    
    def __init__(self, input_dim: int, hidden_units: int):
        """
        Initialize network.
        
        Args:
            input_dim: Number of input features (4 for y, 5 for π)
            hidden_units: Number of hidden units (2 for y, 8 for π)
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1)
        )
        
        # Glorot initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Glorot uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Output tensor [batch_size, 1]
        """
        return self.network(x).squeeze(-1)