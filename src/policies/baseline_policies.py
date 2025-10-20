"""
Baseline monetary policy rules from literature.

Implements common rules for comparison:
- Taylor (1993)
- NPP (Nikolsko-Rzhevskyy et al., 2018)
- Balanced Approach (Federal Reserve MPR)
"""

import numpy as np
from typing import Dict, Optional


class BaselinePolicy:
    """
    Baseline monetary policy rules.
    
    General form:
    i_t = r* + β_π(π_t - π*) + β_y y_t
    
    where:
        i_t: Nominal interest rate
        r*: Equilibrium real interest rate
        π*: Inflation target
        β_π: Inflation response coefficient
        β_y: Output gap response coefficient
    """
    
    def __init__(
        self,
        rule_type: str,
        r_star: float = 2.0,
        pi_star: float = 2.0
    ):
        """
        Initialize baseline policy.
        
        Args:
            rule_type: One of {'TR93', 'NPP', 'BA'}
            r_star: Equilibrium real interest rate (default: 2%)
            pi_star: Inflation target (default: 2%)
        """
        self.rule_type = rule_type.upper()
        self.r_star = r_star
        self.pi_star = pi_star
        
        # Coefficients from Table 3
        self.coefficients = {
            'TR93': {  # Taylor (1993)
                'alpha_0': 1.0,
                'beta_pi': 1.5,
                'beta_y': 0.5
            },
            'NPP': {  # Nikolsko-Rzhevskyy et al. (2018)
                'alpha_0': 0.0,
                'beta_pi': 2.0,
                'beta_y': 0.5
            },
            'BA': {  # Balanced Approach
                'alpha_0': 1.0,
                'beta_pi': 1.5,
                'beta_y': 1.0
            }
        }
        
        if self.rule_type not in self.coefficients:
            raise ValueError(
                f"Unknown rule type: {rule_type}. "
                f"Choose from {list(self.coefficients.keys())}"
            )
        
        self.params = self.coefficients[self.rule_type]
    
    def get_action(
        self,
        state: np.ndarray,
        return_components: bool = False
    ) -> float:
        """
        Compute interest rate given state.
        
        Args:
            state: State vector [y_t, π_t] or [y_t, y_{t-1}, π_t, π_{t-1}]
                  Only uses contemporaneous values [y_t, π_t]
            return_components: If True, return breakdown of rate components
            
        Returns:
            Nominal interest rate i_t
            Or tuple (i_t, components_dict) if return_components=True
        """
        # Extract contemporaneous values
        if len(state) == 2:
            y_t, pi_t = state
        else:
            y_t, _, pi_t, _ = state[:4]
        
        # Compute interest rate
        # i_t = r* + β_π(π_t - π*) + β_y y_t
        # Note: α_0 = r* - (β_π - 1)π* in paper notation
        
        inflation_gap = pi_t - self.pi_star
        
        interest_rate = (
            self.r_star +
            self.params['beta_pi'] * inflation_gap +
            self.params['beta_y'] * y_t
        )
        
        if return_components:
            components = {
                'r_star': self.r_star,
                'inflation_component': self.params['beta_pi'] * inflation_gap,
                'output_gap_component': self.params['beta_y'] * y_t,
                'total': interest_rate
            }
            return interest_rate, components
        
        return interest_rate
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"{self.rule_type} Rule: "
            f"i_t = {self.r_star:.1f} + {self.params['beta_pi']:.1f}(π_t - {self.pi_star:.1f}) "
            f"+ {self.params['beta_y']:.1f}y_t"
        )
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get policy parameters.
        
        Returns:
            Dictionary with all parameters
        """
        return {
            'rule_type': self.rule_type,
            'r_star': self.r_star,
            'pi_star': self.pi_star,
            **self.params
        }


class CustomLinearPolicy:
    """
    Custom linear policy with specified coefficients.
    
    Useful for implementing RL-optimized linear policies or
    custom variations of baseline rules.
    """
    
    def __init__(
        self,
        alpha_0: float,
        beta_pi_0: float,
        beta_y_0: float,
        beta_pi_1: Optional[float] = None,
        beta_y_1: Optional[float] = None
    ):
        """
        Initialize custom linear policy.
        
        Without lags:
            i_t = α_0 + β^0_π π_t + β^0_y y_t
        
        With lags:
            i_t = α_0 + β^0_π π_t + β^0_y y_t + β^1_π π_{t-1} + β^1_y y_{t-1}
        
        Args:
            alpha_0: Intercept
            beta_pi_0: Contemporaneous inflation coefficient
            beta_y_0: Contemporaneous output gap coefficient
            beta_pi_1: Lagged inflation coefficient (optional)
            beta_y_1: Lagged output gap coefficient (optional)
        """
        self.alpha_0 = alpha_0
        self.beta_pi_0 = beta_pi_0
        self.beta_y_0 = beta_y_0
        self.beta_pi_1 = beta_pi_1
        self.beta_y_1 = beta_y_1
        
        self.has_lags = (beta_pi_1 is not None) and (beta_y_1 is not None)
    
    def get_action(self, state: np.ndarray) -> float:
        """
        Compute interest rate.
        
        Args:
            state: State vector [y_t, π_t] or [y_t, y_{t-1}, π_t, π_{t-1}]
            
        Returns:
            Nominal interest rate i_t
        """
        if not self.has_lags:
            # No lags: use only contemporaneous values
            if len(state) == 2:
                y_t, pi_t = state
            else:
                y_t, _, pi_t, _ = state[:4]
            
            interest_rate = (
                self.alpha_0 +
                self.beta_pi_0 * pi_t +
                self.beta_y_0 * y_t
            )
        else:
            # With lags
            if len(state) == 2:
                raise ValueError("State must include lags for this policy")
            
            y_t, y_lag1, pi_t, pi_lag1 = state[:4]
            
            interest_rate = (
                self.alpha_0 +
                self.beta_pi_0 * pi_t +
                self.beta_y_0 * y_t +
                self.beta_pi_1 * pi_lag1 +
                self.beta_y_1 * y_lag1
            )
        
        return interest_rate
    
    def get_parameters(self) -> Dict[str, float]:
        """Get all parameters."""
        params = {
            'alpha_0': self.alpha_0,
            'beta_pi_0': self.beta_pi_0,
            'beta_y_0': self.beta_y_0
        }
        
        if self.has_lags:
            params['beta_pi_1'] = self.beta_pi_1
            params['beta_y_1'] = self.beta_y_1
        
        return params