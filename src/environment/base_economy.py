"""
Base class for macroeconomic environments.

This module implements the abstract interface for economy representations
as described in Equations (1) and (2) of Hinterlang & Tänzer (2024).
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import numpy as np


class BaseEconomy(ABC):
    """
    Abstract base class for macroeconomic environments.
    
    Represents the transition equations for inflation (π_t) and output gap (y_t):
    
    y_t = f^y(y_{t-1}, y_{t-2}, π_t, π_{t-1}, π_{t-2}, i_t, i_{t-1}, i_{t-2}) + ε^y_t  (Eq. 1)
    π_t = f^π(y_t, y_{t-1}, y_{t-2}, π_{t-1}, π_{t-2}, i_t, i_{t-1}, i_{t-2}) + ε^π_t  (Eq. 2)
    
    where:
        y_t: Output gap (percent deviation from potential)
        π_t: Inflation rate (year-over-year percent change)
        i_t: Nominal interest rate (percent)
        ε^y_t, ε^π_t: Structural shocks
    """
    
    def __init__(
        self,
        shock_std: Dict[str, float],
        target_inflation: float = 2.0,
        target_output_gap: float = 0.0,
        reward_weights: Optional[Dict[str, float]] = None,
        penalty_threshold: float = 2.0,
        penalty_multiplier: float = 10.0,
        seed: Optional[int] = None
    ):
        """
        Initialize base economy.
        
        Args:
            shock_std: Standard deviations for structural shocks
                      {'output_gap': σ_y, 'inflation': σ_π}
            target_inflation: Inflation target π* (default: 2.0%)
            target_output_gap: Output gap target y* (default: 0.0%)
            reward_weights: Weights for reward function
                          {'inflation': ω_π, 'output_gap': ω_y}
                          Default: {0.5, 0.5} for equal weighting
            penalty_threshold: Threshold for extreme deviation penalty (pp)
            penalty_multiplier: Multiplier for penalty term
            seed: Random seed for shock generation
        """
        self.shock_std = shock_std
        self.target_inflation = target_inflation
        self.target_output_gap = target_output_gap
        
        # Default equal weights (Table 1 discussion, footnote 6)
        if reward_weights is None:
            reward_weights = {'inflation': 0.5, 'output_gap': 0.5}
        self.reward_weights = reward_weights
        
        self.penalty_threshold = penalty_threshold
        self.penalty_multiplier = penalty_multiplier
        
        # Random number generator for shocks
        self.rng = np.random.default_rng(seed)
        
        # State tracking
        self.current_state: Optional[np.ndarray] = None
        self.episode_step: int = 0
        
    @abstractmethod
    def step(
        self,
        state: np.ndarray,
        action: float
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Compute next state given current state and policy action.
        
        This method implements the environment dynamics (transition equations)
        and must be implemented by subclasses (SVAR or ANN).
        
        Args:
            state: Current economic state vector
                  [y_t, y_{t-1}, π_t, π_{t-1}, i_{t-1}, i_{t-2}]
            action: Nominal interest rate i_t set by policy
            
        Returns:
            next_state: Updated state vector for t+1
            reward: Immediate reward r_t (negative loss)
            done: Boolean indicating episode termination
            info: Dictionary with diagnostic information
                  {'inflation': π_{t+1}, 'output_gap': y_{t+1}, 
                   'shocks': {ε^y, ε^π}, ...}
        """
        pass
    
    @abstractmethod
    def reset(
        self,
        initial_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Args:
            initial_state: Optional initial state. If None, sample from data.
            
        Returns:
            Initial state vector
        """
        pass
    
    def compute_reward(
        self,
        inflation: float,
        output_gap: float
    ) -> float:
        """
        Compute quadratic loss function (Equation 14).
        
        The reward is the negative of the central bank loss:
        
        r_t = -ω_π(π_{t+1} - π*)² - ω_y(y_{t+1})²
        
        With penalty for extreme deviations (> penalty_threshold):
        
        r^penalty_π = 10 · r^π_t  if |π_{t+1} - π*| > 2
        r^penalty_y = 10 · r^y_t  if |y_{t+1}| > 2
        
        See footnote 7 of the paper for penalty justification.
        
        Args:
            inflation: Inflation rate π_{t+1}
            output_gap: Output gap y_{t+1}
            
        Returns:
            Reward value (higher is better)
        """
        # Compute squared deviations
        inflation_dev_sq = (inflation - self.target_inflation) ** 2
        output_gap_dev_sq = (output_gap - self.target_output_gap) ** 2
        
        # Base reward (negative loss)
        base_reward = -(
            self.reward_weights['inflation'] * inflation_dev_sq +
            self.reward_weights['output_gap'] * output_gap_dev_sq
        )
        
        # Penalty for extreme deviations (computational stability)
        # This encourages the agent to avoid extreme states during training
        penalty = 0.0
        
        if inflation_dev_sq > self.penalty_threshold ** 2:
            penalty += self.penalty_multiplier * inflation_dev_sq
            
        if output_gap_dev_sq > self.penalty_threshold ** 2:
            penalty += self.penalty_multiplier * output_gap_dev_sq
        
        return base_reward - penalty
    
    def check_termination(
        self,
        inflation: float,
        output_gap: float,
        step: int,
        max_steps: int,
        tolerance: Dict[str, float]
    ) -> bool:
        """
        Check if episode should terminate.
        
        Termination occurs if:
        1. Target reached: |π_{t+1} - π*| < tolerance AND |y_{t+1}| < tolerance
        2. Max steps reached: step >= max_steps
        
        Args:
            inflation: Current inflation rate
            output_gap: Current output gap
            step: Current episode step
            max_steps: Maximum steps per episode
            tolerance: Tolerance thresholds
                      {'inflation': tol_π, 'output_gap': tol_y}
            
        Returns:
            True if episode should terminate
        """
        # Check if targets reached
        inflation_close = abs(inflation - self.target_inflation) < tolerance['inflation']
        output_gap_close = abs(output_gap - self.target_output_gap) < tolerance['output_gap']
        
        targets_reached = inflation_close and output_gap_close
        max_steps_reached = step >= max_steps
        
        return targets_reached or max_steps_reached
    
    def sample_shock(self, variable: str) -> float:
        """
        Sample structural shock from normal distribution.
        
        Args:
            variable: Either 'output_gap' or 'inflation'
            
        Returns:
            Shock value ε ~ N(0, σ²)
        """
        return self.rng.normal(0, self.shock_std[variable])
    
    @property
    def state_dim(self) -> int:
        """Dimension of state vector."""
        return 6  # [y_t, y_{t-1}, π_t, π_{t-1}, i_{t-1}, i_{t-2}]
    
    @property
    def action_dim(self) -> int:
        """Dimension of action vector."""
        return 1  # Nominal interest rate i_t