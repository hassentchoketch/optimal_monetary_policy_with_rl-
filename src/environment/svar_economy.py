"""
Linear SVAR economy implementation.

Implements the structural vector autoregression with recursive structure
as described in Equations (6) and (7) of the paper.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from src.environment.base_economy import BaseEconomy


class SVAREconomy(BaseEconomy):
    """
    Linear SVAR(2) economy with recursive structure.
    
    Equations (6) and (7):
    
    y_t = C^y + a^y_{y,1}y_{t-1} + a^y_{π,1}π_{t-1} + a^y_{i,1}i_{t-1} + a^y_{i,2}i_{t-2} + ε^y_t
    
    π_t = C^π + a^π_{y,0}y_t + a^π_{y,1}y_{t-1} + a^π_{π,1}π_{t-1} + a^π_{π,2}π_{t-2} + a^π_{i,1}i_{t-1} + ε^π_t
    
    Recursive structure: π_t depends on y_t contemporaneously, but not vice versa.
    This reflects demand pressures affecting inflation within the same period.
    """
    
    def __init__(
        self,
        params: Dict[str, Dict[str, float]],
        shock_std: Dict[str, float],
        target_inflation: float = 2.0,
        target_output_gap: float = 0.0,
        reward_weights: Optional[Dict[str, float]] = None,
        penalty_threshold: float = 2.0,
        penalty_multiplier: float = 10.0,
        seed: Optional[int] = None
    ):
        """
        Initialize SVAR economy.
        
        Args:
            params: Dictionary with structure:
                   {'output_gap': {'const': C^y, 'y_lag1': a^y_{y,1}, ...},
                    'inflation': {'const': C^π, 'y_lag0': a^π_{y,0}, ...}}
            shock_std: Standard deviations {'output_gap': σ_y, 'inflation': σ_π}
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
        
        self.params = params
        
        # Extract parameters for easier access
        self.y_params = params['output_gap']
        self.pi_params = params['inflation']
        
    def step(
        self,
        state: np.ndarray,
        action: float
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Compute next state using SVAR equations.
        
        State vector format: [y_t, y_{t-1}, π_t, π_{t-1}, i_{t-1}, i_{t-2}]
        
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
        
        # Step 1: Compute output gap y_{t+1} (Equation 6)
        # y_{t+1} = C^y + a^y_{y,1}y_t + a^y_{π,1}π_t + a^y_{i,1}i_t + a^y_{i,2}i_{t-1} + ε^y_{t+1}
        y_next = (
            self.y_params['const'] +
            self.y_params['y_lag1'] * y_t +
            self.y_params['pi_lag1'] * pi_t +
            self.y_params['i_lag1'] * i_t +
            self.y_params['i_lag2'] * i_lag1 +
            shock_y
        )
        
        # Step 2: Compute inflation π_{t+1} (Equation 7)
        # Recursive: π_{t+1} depends on y_{t+1} contemporaneously
        # π_{t+1} = C^π + a^π_{y,0}y_{t+1} + a^π_{y,1}y_t + a^π_{π,1}π_t + a^π_{π,2}π_{t-1} + a^π_{i,1}i_t + ε^π_{t+1}
        pi_next = (
            self.pi_params['const'] +
            self.pi_params['y_lag0'] * y_next +  # Contemporaneous effect
            self.pi_params['y_lag1'] * y_t +
            self.pi_params['pi_lag1'] * pi_t +
            self.pi_params['pi_lag2'] * pi_lag1 +
            self.pi_params['i_lag1'] * i_t +
            shock_pi
        )
        
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
        
        # Episode termination (not used during training, only for evaluation)
        done = False
        
        # Diagnostic information
        info = {
            'inflation': pi_next,
            'output_gap': y_next,
            'interest_rate': i_t,
            'shock_y': shock_y,
            'shock_pi': shock_pi,
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
            initial_state: Optional [y_0, y_{-1}, π_0, π_{-1}, i_{-1}, i_{-2}]
                          If None, initialize at steady state with small noise
        
        Returns:
            Initial state vector
        """
        if initial_state is not None:
            self.current_state = initial_state.copy()
        else:
            # Initialize near steady state (targets) with small noise
            self.current_state = np.array([
                self.target_output_gap + self.rng.normal(0, 0.1),  # y_0
                self.target_output_gap + self.rng.normal(0, 0.1),  # y_{-1}
                self.target_inflation + self.rng.normal(0, 0.1),   # π_0
                self.target_inflation + self.rng.normal(0, 0.1),   # π_{-1}
                self.target_inflation + self.rng.normal(0, 0.5),   # i_{-1} ≈ π* + r*
                self.target_inflation + self.rng.normal(0, 0.5)    # i_{-2}
            ], dtype=np.float32)
        
        self.episode_step = 0
        return self.current_state
    
    def predict_deterministic(
        self,
        state: np.ndarray,
        action: float
    ) -> Tuple[float, float]:
        """
        Predict next inflation and output gap without shocks (for evaluation).
        
        Args:
            state: Current state vector
            action: Nominal interest rate
            
        Returns:
            (y_{t+1}, π_{t+1}) without stochastic shocks
        """
        y_t, y_lag1, pi_t, pi_lag1, i_lag1, i_lag2 = state
        i_t = action
        
        # Output gap (no shock)
        y_next = (
            self.y_params['const'] +
            self.y_params['y_lag1'] * y_t +
            self.y_params['pi_lag1'] * pi_t +
            self.y_params['i_lag1'] * i_t +
            self.y_params['i_lag2'] * i_lag1
        )
        
        # Inflation (no shock)
        pi_next = (
            self.pi_params['const'] +
            self.pi_params['y_lag0'] * y_next +
            self.pi_params['y_lag1'] * y_t +
            self.pi_params['pi_lag1'] * pi_t +
            self.pi_params['pi_lag2'] * pi_lag1 +
            self.pi_params['i_lag1'] * i_t
        )
        
        return y_next, pi_next