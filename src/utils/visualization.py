"""
Visualization utilities for publication-quality plots.

All plots use LaTeX formatting and 300 DPI resolution as specified in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Configure matplotlib for publication quality
rc('text', usetex=True)
rc('font', family='serif', size=12)
rc('figure', dpi=300)
sns.set_style("whitegrid")
sns.set_palette("deep")


def plot_economy_fit(
    actual_y: np.ndarray,
    actual_pi: np.ndarray,
    fitted_y_svar: np.ndarray,
    fitted_pi_svar: np.ndarray,
    fitted_y_ann: np.ndarray,
    fitted_pi_ann: np.ndarray,
    dates: np.ndarray,
    save_path: str
):
    """
    Reproduce Figure 2: Economy fit comparison (SVAR vs ANN).
    
    Shows squared errors for output gap and inflation across time.
    
    Args:
        actual_y: Actual output gap values
        actual_pi: Actual inflation values
        fitted_y_svar: SVAR fitted output gap
        fitted_pi_svar: SVAR fitted inflation
        fitted_y_ann: ANN fitted output gap
        fitted_pi_ann: ANN fitted inflation
        dates: Time index
        save_path: Path to save figure
    """
    # Compute squared errors
    se_y_svar = (actual_y - fitted_y_svar) ** 2
    se_pi_svar = (actual_pi - fitted_pi_svar) ** 2
    se_y_ann = (actual_y - fitted_y_ann) ** 2
    se_pi_ann = (actual_pi - fitted_pi_ann) ** 2
    
    # Split pre/post COVID (2019:Q2 = index 127 for 1987:Q3 start)
    covid_idx = 127
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Output gap - Pre COVID
    axes[0, 0].plot(dates[:covid_idx], se_y_svar[:covid_idx], 
                    color='tab:red', label='SVAR', linewidth=1.5)
    axes[0, 0].plot(dates[:covid_idx], se_y_ann[:covid_idx], 
                    color='tab:blue', label='ANN', linewidth=1.5)
    axes[0, 0].set_ylabel(r'Squared Error', fontsize=12)
    axes[0, 0].set_title(r'Output Gap (Pre-COVID)', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Output gap - Post COVID
    axes[0, 1].plot(dates[covid_idx:], se_y_svar[covid_idx:], 
                    color='tab:red', label='SVAR', linewidth=1.5)
    axes[0, 1].plot(dates[covid_idx:], se_y_ann[covid_idx:], 
                    color='tab:blue', label='ANN', linewidth=1.5)
    axes[0, 1].set_title(r'Output Gap (Post-COVID)', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Inflation - Pre COVID
    axes[1, 0].plot(dates[:covid_idx], se_pi_svar[:covid_idx], 
                    color='tab:red', label='SVAR', linewidth=1.5)
    axes[1, 0].plot(dates[:covid_idx], se_pi_ann[:covid_idx], 
                    color='tab:blue', label='ANN', linewidth=1.5)
    axes[1, 0].set_xlabel(r'Year', fontsize=12)
    axes[1, 0].set_ylabel(r'Squared Error', fontsize=12)
    axes[1, 0].set_title(r'Inflation (Pre-COVID)', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Inflation - Post COVID
    axes[1, 1].plot(dates[covid_idx:], se_pi_svar[covid_idx:], 
                    color='tab:red', label='SVAR', linewidth=1.5)
    axes[1, 1].plot(dates[covid_idx:], se_pi_ann[covid_idx:], 
                    color='tab:blue', label='ANN', linewidth=1.5)
    axes[1, 1].set_xlabel(r'Year', fontsize=12)
    axes[1, 1].set_title(r'Inflation (Post-COVID)', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_partial_dependence(
    network,
    variable: str,
    input_ranges: Dict[str, Tuple[float, float]],
    grid_resolution: int = 50,
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Create 3D partial dependence plot for ANN economy or policy.
    
    Reproduces Figures 3, 4, 5 showing how outputs depend on inputs.
    
    Args:
        network: Neural network (economy or policy)
        variable: Variable to plot ('inflation', 'output_gap', 'interest_rate')
        input_ranges: Dictionary with ranges for grid
        grid_resolution: Number of grid points per dimension
        save_path: Path to save figure
        title: Plot title
    """
    import torch
    
    # Create grid for two main variables
    if variable == 'inflation':
        # Determine lags from input dim
        # Input: [y_{t+1}, y_t, π_t, i_t, y_{t-1}, π_{t-1}, i_{t-1}, ...]
        # Dim = 1 + 3 * lags
        input_dim = network.network[0].weight.shape[1]
        lags = (input_dim - 1) // 3
        
        # π_t vs (y_t, i_{t-1}) if lags >= 2
        # π_t vs (y_t, i_t) if lags == 1
        
        y_vals = np.linspace(input_ranges['output_gap'][0], 
                            input_ranges['output_gap'][1], 
                            grid_resolution)
        i_vals = np.linspace(input_ranges['interest_rate'][0], 
                            input_ranges['interest_rate'][1], 
                            grid_resolution)
        Y, I = np.meshgrid(y_vals, i_vals)
        
        # Compute network outputs
        Z = np.zeros_like(Y)
        
        # Base input vector (steady state)
        # y=0, pi=2, i=2
        base_input = np.zeros(input_dim)
        # y_{t+1} (0) = 0
        base_input[0] = 0.0
        for k in range(1, lags + 1):
            # y_{t-k+1} = 0
            base_input[1 + (k-1)*3] = 0.0
            # pi_{t-k+1} = 2.0
            base_input[2 + (k-1)*3] = 2.0
            # i_{t-k+1} = 2.0
            base_input[3 + (k-1)*3] = 2.0
            
        for idx_i in range(grid_resolution):
            for idx_j in range(grid_resolution):
                input_vec = torch.tensor(base_input, dtype=torch.float32).clone()
                
                # Set y_t (index 1)
                input_vec[1] = Y[idx_i, idx_j]
                
                if lags >= 2:
                    # Set i_{t-1} (index 6)
                    input_vec[6] = I[idx_i, idx_j]
                    ylabel = r'Interest Rate $i_{t-1}$ (\%)'
                else:
                    # Set i_t (index 3)
                    input_vec[3] = I[idx_i, idx_j]
                    ylabel = r'Interest Rate $i_t$ (\%)'
                
                with torch.no_grad():
                    Z[idx_i, idx_j] = network(input_vec.unsqueeze(0)).item()
        
        xlabel = r'Output Gap $y_t$ (\%)'
        zlabel = r'Inflation $\pi_t$ (\%)'
        
    elif variable == 'output_gap':
        # Determine lags from input dim
        # Input: [y_t, π_t, i_t, y_{t-1}, π_{t-1}, i_{t-1}, ...]
        # Dim = 3 * lags
        input_dim = network.network[0].weight.shape[1]
        lags = input_dim // 3
        
        # y_t vs (π_{t-1}, i_{t-1}) if lags >= 2
        # y_t vs (π_t, i_t) if lags == 1
        
        pi_vals = np.linspace(input_ranges['inflation'][0], 
                             input_ranges['inflation'][1], 
                             grid_resolution)
        i_vals = np.linspace(input_ranges['interest_rate'][0], 
                            input_ranges['interest_rate'][1], 
                            grid_resolution)
        Pi, I = np.meshgrid(pi_vals, i_vals)
        
        Z = np.zeros_like(Pi)
        
        # Base input vector (steady state)
        base_input = np.zeros(input_dim)
        for k in range(1, lags + 1):
            # y_{t-k+1} = 0
            base_input[(k-1)*3] = 0.0
            # pi_{t-k+1} = 2.0
            base_input[1 + (k-1)*3] = 2.0
            # i_{t-k+1} = 2.0
            base_input[2 + (k-1)*3] = 2.0
            
        for idx_i in range(grid_resolution):
            for idx_j in range(grid_resolution):
                input_vec = torch.tensor(base_input, dtype=torch.float32).clone()
                
                if lags >= 2:
                    # Set π_{t-1} (index 4)
                    input_vec[4] = Pi[idx_i, idx_j]
                    # Set i_{t-1} (index 5)
                    input_vec[5] = I[idx_i, idx_j]
                    xlabel = r'Inflation $\pi_{t-1}$ (\%)'
                    ylabel = r'Interest Rate $i_{t-1}$ (\%)'
                else:
                    # Set π_t (index 1)
                    input_vec[1] = Pi[idx_i, idx_j]
                    # Set i_t (index 2)
                    input_vec[2] = I[idx_i, idx_j]
                    xlabel = r'Inflation $\pi_t$ (\%)'
                    ylabel = r'Interest Rate $i_t$ (\%)'
                
                with torch.no_grad():
                    Z[idx_i, idx_j] = network(input_vec.unsqueeze(0)).item()
        
        zlabel = r'Output Gap $y_t$ (\%)'
        Pi, I = Pi, I  # For consistency
        Y, I = Pi, I   # Rename for plotting
        
    else:  # interest_rate (policy)
        # i_t vs (π_t, y_t)
        pi_vals = np.linspace(input_ranges['inflation'][0], 
                             input_ranges['inflation'][1], 
                             grid_resolution)
        y_vals = np.linspace(input_ranges['output_gap'][0], 
                            input_ranges['output_gap'][1], 
                            grid_resolution)
        Pi, Y = np.meshgrid(pi_vals, y_vals)
        
        Z = np.zeros_like(Pi)
        for idx_i in range(grid_resolution):
            for idx_j in range(grid_resolution):
                # Input depends on policy structure
                input_vec = torch.tensor([
                    Y[idx_i, idx_j],   # y_t
                    Pi[idx_i, idx_j]   # π_t
                ], dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    Z[idx_i, idx_j] = network(input_vec).item()
        
        xlabel = r'Inflation $\pi_t$ (\%)'
        ylabel = r'Output Gap $y_t$ (\%)'
        zlabel = r'Interest Rate $i_t$ (\%)'
        Y, Pi = Y, Pi  # For consistency
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(Y if variable != 'output_gap' else Pi, 
                          I if variable != 'interest_rate' else (Pi if variable == 'output_gap' else Y),
                          Z, 
                          cmap='viridis', 
                          alpha=0.9,
                          edgecolor='none')
    
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.set_zlabel(zlabel, fontsize=12, labelpad=10)
    
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Adjust viewing angle
    ax.view_init(elev=20, azim=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_counterfactual(
    dates: np.ndarray,
    actual_ffr: np.ndarray,
    actual_inflation: np.ndarray,
    actual_output_gap: np.ndarray,
    counterfactual_data: Dict[str, Dict[str, np.ndarray]],
    save_path: str,
    title: str = "Historical Counterfactual Analysis"
):
    """
    Reproduce Figures 6, 7: Historical counterfactual analysis.
    
    Shows FFR, inflation, and output gap under different policies.
    
    Args:
        dates: Time index
        actual_ffr: Actual federal funds rate
        actual_inflation: Actual inflation
        actual_output_gap: Actual output gap
        counterfactual_data: Dictionary with policy names as keys
                            and dicts of {'ffr', 'inflation', 'output_gap'} as values
        save_path: Path to save figure
        title: Figure title
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Define colors for policies
    colors = {
        'Actual': 'black',
        'TR93': 'tab:red',
        'NPP': 'tab:orange',
        'BA': 'tab:green',
        'RL_SVAR_no_lag': 'tab:pink',
        'RL_SVAR_one_lag': 'tab:olive',
        'RL_SVAR_no_lag_nonlin': 'tab:brown',
        'RL_SVAR_one_lag_nonlin': 'tab:gray',
        'RL_ANN_no_lag': 'tab:blue',
        'RL_ANN_one_lag': 'tab:purple',
        'RL_ANN_no_lag_nonlin': 'tab:cyan',
        'RL_ANN_one_lag_nonlin': 'tab:pink'
    }
    stylses = {
        'Actual': '-',
        'TR93': '--',
        'NPP': '-.',
        'BA': ':',
        'RL_SVAR_no_lag': '-',
        'RL_SVAR_one_lag': '--',
        'RL_SVAR_no_lag_nonlin': '-.',
        'RL_SVAR_one_lag_nonlin': ':',
        'RL_ANN_no_lag': '-',
        'RL_ANN_one_lag': '--',
        'RL_ANN_no_lag_nonlin': '-.',
        'RL_ANN_one_lag_nonlin': ':'
    }
    
    # Plot FFR
    axes[0].plot(dates, actual_ffr, color=colors['Actual'], 
                label='Actual', linewidth=2)
    for policy_name, data in counterfactual_data.items():
        color = colors.get(policy_name, 'gray')
        style = stylses.get(policy_name, '-')
        axes[0].plot(dates, data['ffr'], color=color, linestyle=style,
                    label=policy_name, linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel(r'Interest Rate (\%)', fontsize=12)
    axes[0].set_title(title, fontsize=14)
    axes[0].legend(loc='best', ncol=2, fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot inflation
    axes[1].plot(dates, actual_inflation, color=colors['Actual'], 
                label='Actual', linewidth=2)
    for policy_name, data in counterfactual_data.items():
        color = colors.get(policy_name, 'gray')
        style = stylses.get(policy_name, '-')
        axes[1].plot(dates, data['inflation'], color=color, linestyle=style,
                    label=policy_name, linewidth=1.5, alpha=0.8)
    axes[1].axhline(y=2.0, color='red', linestyle='--', 
                   label=r'Target ($\pi^* = 2\%$)', alpha=0.7)
    axes[1].set_ylabel(r'Inflation (\%)', fontsize=12)
    axes[1].legend(loc='best', ncol=2, fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Plot output gap
    axes[2].plot(dates, actual_output_gap, color=colors['Actual'], 
                label='Actual', linewidth=2)
    for policy_name, data in counterfactual_data.items():
        color = colors.get(policy_name, 'gray')
        style = stylses.get(policy_name, '-')
        axes[2].plot(dates, data['output_gap'], color=color, linestyle=style,
                    label=policy_name, linewidth=1.5, alpha=0.8)
    axes[2].axhline(y=0, color='red', linestyle='--', 
                   label=r'Target ($y^* = 0\%$)', alpha=0.7)
    axes[2].set_xlabel(r'Year', fontsize=12)
    axes[2].set_ylabel(r'Output Gap (\%)', fontsize=12)
    axes[2].legend(loc='best', ncol=2, fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(
    episode_rewards: List[float],
    critic_losses: List[float],
    actor_losses: List[float],
    save_path: str
):
    """
    Plot training progress curves.
    
    Args:
        episode_rewards: Reward per episode
        critic_losses: Critic loss per update
        actor_losses: Actor loss per update
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    episodes = np.arange(len(episode_rewards))
    updates = np.arange(len(critic_losses))
    
    # Episode rewards
    axes[0].plot(episodes, episode_rewards, color='tab:blue', alpha=0.6)
    # Moving average
    window = 50
    if len(episode_rewards) >= window:
        ma_rewards = pd.Series(episode_rewards).rolling(window=window).mean()
        axes[0].plot(episodes, ma_rewards, color='darkblue', linewidth=2, 
                    label=f'{window}-episode MA')
    axes[0].set_xlabel(r'Episode', fontsize=12)
    axes[0].set_ylabel(r'Episode Reward', fontsize=12)
    axes[0].set_title(r'Training Progress', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Critic loss
    axes[1].plot(updates, critic_losses, color='tab:red', alpha=0.4)
    if len(critic_losses) >= window:
        ma_critic = pd.Series(critic_losses).rolling(window=window).mean()
        axes[1].plot(updates, ma_critic, color='darkred', linewidth=2, 
                    label=f'{window}-update MA')
    axes[1].set_xlabel(r'Update Step', fontsize=12)
    axes[1].set_ylabel(r'Critic Loss', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    # Actor loss
    axes[2].plot(updates, actor_losses, color='tab:green', alpha=0.4)
    if len(actor_losses) >= window:
        ma_actor = pd.Series(actor_losses).rolling(window=window).mean()
        axes[2].plot(updates, ma_actor, color='darkgreen', linewidth=2, 
                    label=f'{window}-update MA')
    axes[2].set_xlabel(r'Update Step', fontsize=12)
    axes[2].set_ylabel(r'Actor Loss', fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_results_table(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: str,
    format: str = 'latex'
):
    """
    Create formatted results table (e.g., Table 4).
    
    Args:
        metrics_dict: Dictionary with policy names and their metrics
        save_path: Path to save table
        format: Output format ('latex' or 'csv')
    """
    df = pd.DataFrame(metrics_dict).T
    
    # Select key columns
    columns = ['mse_inflation', 'mse_output_gap', 'total_loss']
    df_display = df[columns].copy()
    
    # Rename for display
    df_display.columns = [r'$\Delta^2(\pi^*, \pi_t)$', 
                         r'$\Delta^2(y^*, y_t)$', 
                         'Loss']
    
    if format == 'latex':
        latex_str = df_display.to_latex(
            float_format="%.2f",
            escape=False,
            caption="Counterfactual Target Deviation and Loss",
            label="tab:counterfactual_loss"
        )
        with open(save_path, 'w') as f:
            f.write(latex_str)
    else:
        df_display.to_csv(save_path)