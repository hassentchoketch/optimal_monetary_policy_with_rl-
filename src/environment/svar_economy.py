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
    Linear SVAR(p) economy with recursive structure.
    
    Equations (6) and (7) generalized for p lags:
    
    y_t = C^y + Σ a^y_{y,k}y_{t-k} + Σ a^y_{π,k}π_{t-k} + Σ a^y_{i,k}i_{t-k} + ε^y_t
    
    π_t = C^π + a^π_{y,0}y_t + Σ a^π_{y,k}y_{t-k} + Σ a^π_{π,k}π_{t-k} + Σ a^π_{i,k}i_{t-k} + ε^π_t
    
    Recursive structure: π_t depends on y_t contemporaneously, but not vice versa.
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
        seed: Optional[int] = None,
        lags: int = 2
    ):
        """
        Initialize SVAR economy.
        
        Args:
            params: Dictionary with structure:
                   {'output_gap': {'const': C^y, 'y_lag1': a^y_{y,1}, ...},
                    'inflation': {'const': C^π, 'y_lag0': a^π_{y,0}, ...}}
            shock_std: Standard deviations {'output_gap': σ_y, 'inflation': σ_π}
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
        
        self.params = params
        self.y_params = params['output_gap']
        self.pi_params = params['inflation']
        
    def step(
        self,
        state: np.ndarray,
        action: float
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Compute next state using SVAR equations.
        
        State vector format: 
        [y_t, ..., y_{t-p+1}, π_t, ..., π_{t-p+1}, i_{t-1}, ..., i_{t-p}]
        """
        # Unpack current state components
        p = self.lags
        y_lags = state[0:p]      # y_t, ..., y_{t-p+1}
        pi_lags = state[p:2*p]   # π_t, ..., π_{t-p+1}
        i_lags = state[2*p:3*p]  # i_{t-1}, ..., i_{t-p}
        
        i_t = action
        
        # Generate structural shocks
        shock_y = self.sample_shock('output_gap')
        shock_pi = self.sample_shock('inflation')
        
        # Step 1: Compute output gap y_{t+1}
        y_next = self.y_params.get('const', 0.0)
        
        # Add lag contributions
        for k in range(1, p + 1):
            # y_{t+1} depends on y_{t-k+1}, π_{t-k+1}, i_{t-k+1}
            # In our notation relative to t+1:
            # lag 1 corresponds to t
            # lag k corresponds to t-k+1
            
            # y lags: y_t is at index 0 (lag 1 relative to t+1)
            y_val = y_lags[k-1]
            y_next += self.y_params.get(f'y_lag{k}', 0.0) * y_val
            
            # π lags: π_t is at index 0
            pi_val = pi_lags[k-1]
            y_next += self.y_params.get(f'pi_lag{k}', 0.0) * pi_val
            
            # i lags: i_t is action (lag 1), i_{t-1} is at index 0 (lag 2)
            if k == 1:
                i_val = i_t
            else:
                i_val = i_lags[k-2]
            y_next += self.y_params.get(f'i_lag{k}', 0.0) * i_val
            
        y_next += shock_y
        
        # Step 2: Compute inflation π_{t+1}
        pi_next = self.pi_params.get('const', 0.0)
        
        # Contemporaneous y_{t+1}
        pi_next += self.pi_params.get('y_lag0', 0.0) * y_next
        
        # Add lag contributions
        for k in range(1, p + 1):
            y_val = y_lags[k-1]
            pi_next += self.pi_params.get(f'y_lag{k}', 0.0) * y_val
            
            pi_val = pi_lags[k-1]
            pi_next += self.pi_params.get(f'pi_lag{k}', 0.0) * pi_val
            
            if k == 1:
                i_val = i_t
            else:
                i_val = i_lags[k-2]
            pi_next += self.pi_params.get(f'i_lag{k}', 0.0) * i_val
            
        pi_next += shock_pi
        
        # Construct next state vector
        # Shift lags and insert new values
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
            
            # y lags
            y_init = self.target_output_gap + self.rng.normal(0, 0.1, size=p)
            
            # π lags
            pi_init = self.target_inflation + self.rng.normal(0, 0.1, size=p)
            
            # i lags
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
        p = self.lags
        y_lags = state[0:p]
        pi_lags = state[p:2*p]
        i_lags = state[2*p:3*p]
        i_t = action
        
        # Output gap
        y_next = self.y_params.get('const', 0.0)
        for k in range(1, p + 1):
            y_val = y_lags[k-1]
            pi_val = pi_lags[k-1]
            if k == 1:
                i_val = i_t
            else:
                i_val = i_lags[k-2]
                
            y_next += self.y_params.get(f'y_lag{k}', 0.0) * y_val
            y_next += self.y_params.get(f'pi_lag{k}', 0.0) * pi_val
            y_next += self.y_params.get(f'i_lag{k}', 0.0) * i_val
            
        # Inflation
        pi_next = self.pi_params.get('const', 0.0)
        pi_next += self.pi_params.get('y_lag0', 0.0) * y_next
        
        for k in range(1, p + 1):
            y_val = y_lags[k-1]
            pi_val = pi_lags[k-1]
            if k == 1:
                i_val = i_t
            else:
                i_val = i_lags[k-2]
                
            pi_next += self.pi_params.get(f'y_lag{k}', 0.0) * y_val
            pi_next += self.pi_params.get(f'pi_lag{k}', 0.0) * pi_val
            pi_next += self.pi_params.get(f'i_lag{k}', 0.0) * i_val
            
        return y_next, pi_next