"""
Evaluation metrics for policy performance.

Implements loss functions and performance measures from the paper.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def compute_loss(
    inflation: np.ndarray,
    output_gap: np.ndarray,
    target_inflation: float = 2.0,
    target_output_gap: float = 0.0,
    weight_inflation: float = 0.5,
    weight_output_gap: float = 0.5
) -> Dict[str, float]:
    """
    Compute central bank loss function (Equation 14).
    
    Loss = ω_π * (π_t - π*)² + ω_y * y_t²
    
    Args:
        inflation: Array of inflation values
        output_gap: Array of output gap values
        target_inflation: Inflation target π* (default: 2%)
        target_output_gap: Output gap target y* (default: 0%)
        weight_inflation: Weight on inflation ω_π
        weight_output_gap: Weight on output gap ω_y
    
    Returns:
        Dictionary with loss components and total loss
    """
    # Compute squared deviations
    inflation_dev_sq = (inflation - target_inflation) ** 2
    output_gap_dev_sq = (output_gap - target_output_gap) ** 2
    
    # Mean squared deviations (Table 4 notation: Δ²)
    mse_inflation = np.mean(inflation_dev_sq)
    mse_output_gap = np.mean(output_gap_dev_sq)
    
    # Total loss
    total_loss = weight_inflation * mse_inflation + weight_output_gap * mse_output_gap
    
    return {
        'mse_inflation': mse_inflation,
        'mse_output_gap': mse_output_gap,
        'total_loss': total_loss,
        'rmse_inflation': np.sqrt(mse_inflation),
        'rmse_output_gap': np.sqrt(mse_output_gap)
    }


def compute_metrics(
    inflation: np.ndarray,
    output_gap: np.ndarray,
    interest_rate: np.ndarray,
    target_inflation: float = 2.0,
    target_output_gap: float = 0.0
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        inflation: Array of inflation values
        output_gap: Array of output gap values
        interest_rate: Array of interest rate values
        target_inflation: Inflation target
        target_output_gap: Output gap target
    
    Returns:
        Dictionary with various performance metrics
    """
    # Loss metrics
    loss_metrics = compute_loss(
        inflation, output_gap,
        target_inflation, target_output_gap
    )
    
    # Mean absolute errors
    mae_inflation = np.mean(np.abs(inflation - target_inflation))
    mae_output_gap = np.mean(np.abs(output_gap - target_output_gap))
    
    # Volatility metrics (unconditional variances)
    var_inflation = np.var(inflation)
    var_output_gap = np.var(output_gap)
    var_interest_rate = np.var(interest_rate)
    
    # Interest rate changes (for policy smoothness)
    interest_rate_changes = np.diff(interest_rate)
    var_interest_rate_change = np.var(interest_rate_changes)
    mean_abs_change = np.mean(np.abs(interest_rate_changes))
    
    # Max deviations
    max_inflation_dev = np.max(np.abs(inflation - target_inflation))
    max_output_gap_dev = np.max(np.abs(output_gap - target_output_gap))
    
    # Percentage of time near target
    inflation_tolerance = 0.5  # within ±0.5 pp
    output_gap_tolerance = 1.0  # within ±1.0 pp
    
    pct_inflation_near_target = np.mean(
        np.abs(inflation - target_inflation) < inflation_tolerance
    ) * 100
    
    pct_output_gap_near_target = np.mean(
        np.abs(output_gap - target_output_gap) < output_gap_tolerance
    ) * 100
    
    return {
        **loss_metrics,
        'mae_inflation': mae_inflation,
        'mae_output_gap': mae_output_gap,
        'var_inflation': var_inflation,
        'var_output_gap': var_output_gap,
        'var_interest_rate': var_interest_rate,
        'var_interest_rate_change': var_interest_rate_change,
        'mean_abs_interest_rate_change': mean_abs_change,
        'max_inflation_deviation': max_inflation_dev,
        'max_output_gap_deviation': max_output_gap_dev,
        'pct_inflation_near_target': pct_inflation_near_target,
        'pct_output_gap_near_target': pct_output_gap_near_target
    }


def compare_policies(
    baseline_metrics: Dict[str, float],
    policy_metrics: Dict[str, float],
    baseline_name: str = "Actual",
    policy_name: str = "RL Policy"
) -> Dict[str, float]:
    """
    Compare policy performance relative to baseline.
    
    Args:
        baseline_metrics: Metrics for baseline (e.g., actual Fed policy)
        policy_metrics: Metrics for policy to evaluate
        baseline_name: Name of baseline
        policy_name: Name of policy
    
    Returns:
        Dictionary with relative improvements (negative = worse)
    """
    improvements = {}
    
    # Loss improvement (lower is better)
    loss_improvement = (
        (baseline_metrics['total_loss'] - policy_metrics['total_loss']) /
        baseline_metrics['total_loss'] * 100
    )
    improvements['loss_improvement_pct'] = loss_improvement
    
    # Component improvements
    inflation_improvement = (
        (baseline_metrics['mse_inflation'] - policy_metrics['mse_inflation']) /
        baseline_metrics['mse_inflation'] * 100
    )
    improvements['inflation_improvement_pct'] = inflation_improvement
    
    output_gap_improvement = (
        (baseline_metrics['mse_output_gap'] - policy_metrics['mse_output_gap']) /
        baseline_metrics['mse_output_gap'] * 100
    )
    improvements['output_gap_improvement_pct'] = output_gap_improvement
    
    # Volatility comparison
    var_ratio_inflation = policy_metrics['var_inflation'] / baseline_metrics['var_inflation']
    var_ratio_output_gap = policy_metrics['var_output_gap'] / baseline_metrics['var_output_gap']
    
    improvements['variance_ratio_inflation'] = var_ratio_inflation
    improvements['variance_ratio_output_gap'] = var_ratio_output_gap
    
    return improvements


def compute_steady_state_reward(
    policy,
    economy,
    n_simulations: int = 1000,
    n_steps: int = 100,
    seed: Optional[int] = None
) -> float:
    """
    Compute steady-state reward by simulating policy in economy.
    
    Used for agent selection criteria (Section 2.2).
    
    Args:
        policy: Policy function (agent or baseline)
        economy: Economic environment
        n_simulations: Number of simulation runs
        n_steps: Steps per simulation
        seed: Random seed
    
    Returns:
        Mean steady-state reward
    """
    if seed is not None:
        np.random.seed(seed)
    
    rewards = []
    
    for _ in range(n_simulations):
        state = economy.reset()
        episode_rewards = []
        
        for _ in range(n_steps):
            if hasattr(policy, 'get_action'):
                action = policy.get_action(state)
            else:
                action = policy.select_action(state, add_noise=False)
            
            state, reward, _, _ = economy.step(state, action)
            episode_rewards.append(reward)
        
        # Use later rewards (after convergence)
        steady_state_reward = np.mean(episode_rewards[-20:])
        rewards.append(steady_state_reward)
    
    return np.mean(rewards)