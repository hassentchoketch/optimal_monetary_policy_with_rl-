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
        device: str = 'cpu',
        lags: int = 2
    ):
        """
        Initialize ANN economy.
        
        Args:
            network_y: Trained neural network for output gap equation
            network_pi: Trained neural network for inflation equation
            shock_std: Standard deviations for structural shocks
            lags: Number of lags p
            Other args: See BaseEconomy
        """
        super().__init__(
            shock_std=shock_std,
            target_inflation=target_inflation,
            target_output_gap=target_output_gap,
            reward_weights=reward_weights,
            penalty_threshold=penalty_threshold,
            penalty_multiplier=penalty_multiplier,
            seed=seed,
            lags=lags
        )
        
        self.device = torch.device(device)
        self.network_y = network_y.to(self.device)
        self.network_pi = network_pi.to(self.device)
        
        # Set networks to evaluation mode
        self.network_y.eval()
        self.network_pi.eval()
    
    def _construct_input_y(self, state: np.ndarray, action: float) -> torch.Tensor:
        """Construct input for output gap network."""
        p = self.lags
        y_lags = state[0:p]      # y_t, ..., y_{t-p+1}
        pi_lags = state[p:2*p]   # π_t, ..., π_{t-p+1}
        i_lags = state[2*p:3*p]  # i_{t-1}, ..., i_{t-p}
        i_t = action
        
        # Input features: lags 1 to p of y, π, i
        # lag 1: y_t, π_t, i_t
        # lag k: y_{t-k+1}, π_{t-k+1}, i_{t-k+1}
        
        features = []
        for k in range(1, p + 1):
            features.append(y_lags[k-1])
            features.append(pi_lags[k-1])
            if k == 1:
                features.append(i_t)
            else:
                features.append(i_lags[k-2])
                
        return torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _construct_input_pi(self, state: np.ndarray, action: float, y_next: float) -> torch.Tensor:
        """Construct input for inflation network."""
        # Input features: y_{t+1} + lags 1 to p of y, π, i
        
        input_y = self._construct_input_y(state, action)
        # Prepend y_next
        y_next_tensor = torch.tensor([[y_next]], dtype=torch.float32, device=self.device)
        return torch.cat([y_next_tensor, input_y], dim=1)

    def step(
        self,
        state: np.ndarray,
        action: float
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Compute next state using ANN equations.
        
        State vector format: [y_t, ..., y_{t-p+1}, π_t, ..., π_{t-p+1}, i_{t-1}, ..., i_{t-p}]
        """
        # Unpack current state for shifting later
        p = self.lags
        y_lags = state[0:p]
        pi_lags = state[p:2*p]
        i_lags = state[2*p:3*p]
        i_t = action
        
        # Generate structural shocks
        shock_y = self.sample_shock('output_gap')
        shock_pi = self.sample_shock('inflation')
        
        # Step 1: Compute output gap y_{t+1}
        input_y = self._construct_input_y(state, action)
        
        with torch.no_grad():
            y_next_pred = self.network_y(input_y).item()
        
        y_next = y_next_pred + shock_y
        
        # Step 2: Compute inflation π_{t+1}
        input_pi = self._construct_input_pi(state, action, y_next)
        
        with torch.no_grad():
            pi_next_pred = self.network_pi(input_pi).item()
        
        pi_next = pi_next_pred + shock_pi
        
        # Construct next state vector
        next_y_lags = np.r_[y_next, y_lags[:-1]]
        next_pi_lags = np.r_[pi_next, pi_lags[:-1]]
        next_i_lags = np.r_[i_t, i_lags[:-1]]
        
        next_state = np.concatenate([next_y_lags, next_pi_lags, next_i_lags]).astype(np.float32)
        
        # Compute reward
        reward = self.compute_reward(pi_next, y_next)
        done = False
        
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
        """Reset environment to initial state."""
        if initial_state is not None:
            self.current_state = initial_state.copy()
        else:
            # Initialize near steady state
            p = self.lags
            y_init = self.target_output_gap + self.rng.normal(0, 0.1, size=p)
            pi_init = self.target_inflation + self.rng.normal(0, 0.1, size=p)
            i_init = self.target_inflation + self.rng.normal(0, 0.5, size=p)
            
            self.current_state = np.concatenate([y_init, pi_init, i_init]).astype(np.float32)
        
        self.episode_step = 0
        return self.current_state
    
    def predict_deterministic(
        self,
        state: np.ndarray,
        action: float
    ) -> Tuple[float, float]:
        """Predict next inflation and output gap without shocks."""
        input_y = self._construct_input_y(state, action)
        with torch.no_grad():
            y_next = self.network_y(input_y).item()
            
        input_pi = self._construct_input_pi(state, action, y_next)
        with torch.no_grad():
            pi_next = self.network_pi(input_pi).item()
        
        return y_next, pi_next


class EconomyNetwork(nn.Module):
    """
    Feed-forward neural network for economy equation.
    """
    
    def __init__(self, input_dim: int, hidden_units: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)