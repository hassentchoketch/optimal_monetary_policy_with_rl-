#!/usr/bin/env python3
"""
Generate all figures from the paper.

Usage:
    python scripts/generate_figures.py --all
    python scripts/generate_figures.py --figure 2
    python scripts/generate_figures.py --learning_curve
"""

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import yaml
import pickle
import torch
import numpy as np
import pandas as pd
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader
from src.environment.ann_economy import EconomyNetwork
from src.utils.visualization import (
    plot_economy_fit,
    plot_partial_dependence,
    plot_counterfactual,
    create_results_table
)
from src.utils.logger import setup_logger


def generate_figure2(config: dict, checkpoint_dir: str, output_dir: str, logger):
    """Generate Figure 2: Economy fit comparison."""
    logger.info("\nGenerating Figure 2: Economy Fit Comparison...")
    
    # Create subfolder for paper figures
    paper_fig_dir = os.path.join(output_dir, 'paper_figures')
    os.makedirs(paper_fig_dir, exist_ok=True)
    
    # Load data
    data_loader = DataLoader(
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    data = data_loader.get_data()
    
    # Load SVAR fitted values
    with open(os.path.join(checkpoint_dir, 'svar_params.pkl'), 'rb') as f:
        svar_results = pickle.load(f)
    
    fitted_y_svar = svar_results['fitted_values']['output_gap']
    fitted_pi_svar = svar_results['fitted_values']['inflation']
    
    # Load tuned parameters if available
    params_path = os.path.join(checkpoint_dir, 'ann_params.pkl')
    tuned_params = None
    if os.path.exists(params_path):
        with open(params_path, 'rb') as f:
            tuned_params = pickle.load(f)
    
    # Determine hyperparameters
    if tuned_params:
        hidden_units_y = tuned_params['output_gap']['hidden_units']
        hidden_units_pi = tuned_params['inflation']['hidden_units']
        lags_y = tuned_params['output_gap'].get('lags', 2)
        lags_pi = tuned_params['inflation'].get('lags', 2)
        lags = max(lags_y, lags_pi)
    else:
        hidden_units_y = config['economy']['ann']['hidden_units_y']
        hidden_units_pi = config['economy']['ann']['hidden_units_pi']
        lags = 2
        lags_y = 2
        lags_pi = 2
    
    # Calculate input dimensions
    input_dim_y = 3 * lags_y
    input_dim_pi = 1 + 3 * lags_pi
    
    # Load ANN and compute fitted values
    network_y = EconomyNetwork(input_dim_y, hidden_units_y)
    network_pi = EconomyNetwork(input_dim_pi, hidden_units_pi)
    
    network_y.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, 'ann_y_network.pth'),
        map_location='cpu'
    ))
    network_pi.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, 'ann_pi_network.pth'),
        map_location='cpu'
    ))
    
    # Prepare inputs
    from src.data.data_loader import create_lagged_features
    lagged_data = create_lagged_features(data, lags=lags)
    
    # Construct inputs dynamically based on lags
    # For Y network: lags 1..p of y, pi, i
    X_y_list = []
    for k in range(1, lags_y + 1):
        X_y_list.append(lagged_data[f'output_gap_lag{k}'].values)
        X_y_list.append(lagged_data[f'inflation_lag{k}'].values)
        X_y_list.append(lagged_data[f'interest_rate_lag{k}'].values)
    X_y = np.column_stack(X_y_list)
    
    # For Pi network: y_t + lags 1..p of y, pi, i
    # Note: y_t here is the ACTUAL y_t (or fitted y_t? usually actual in one-step ahead)
    # The original code used 'output_gap' which is y_t.
    X_pi_list = [lagged_data['output_gap'].values]
    for k in range(1, lags_pi + 1):
        X_pi_list.append(lagged_data[f'output_gap_lag{k}'].values)
        X_pi_list.append(lagged_data[f'inflation_lag{k}'].values)
        X_pi_list.append(lagged_data[f'interest_rate_lag{k}'].values)
    X_pi = np.column_stack(X_pi_list)
    
    network_y.eval()
    network_pi.eval()
    
    with torch.no_grad():
        fitted_y_ann = network_y(torch.FloatTensor(X_y)).numpy().flatten()
        fitted_pi_ann = network_pi(torch.FloatTensor(X_pi)).numpy().flatten()
    
    # Align array lengths - all arrays must have the same length
    # SVAR may have fewer observations due to lag structure
    min_len = min(
        len(fitted_y_svar),
        len(fitted_pi_svar),
        len(fitted_y_ann),
        len(fitted_pi_ann),
        len(lagged_data)
    )
    
    # Trim all arrays to the same length (use the last min_len observations)
    actual_y = lagged_data['output_gap'].values[-min_len:]
    actual_pi = lagged_data['inflation'].values[-min_len:]
    fitted_y_svar_aligned = fitted_y_svar[-min_len:]
    fitted_pi_svar_aligned = fitted_pi_svar[-min_len:]
    fitted_y_ann_aligned = fitted_y_ann[-min_len:]
    fitted_pi_ann_aligned = fitted_pi_ann[-min_len:]
    dates_aligned = lagged_data.index[-min_len:]
    
    # Plot
    plot_economy_fit(
        actual_y=actual_y,
        actual_pi=actual_pi,
        fitted_y_svar=fitted_y_svar_aligned,
        fitted_pi_svar=fitted_pi_svar_aligned,
        fitted_y_ann=fitted_y_ann_aligned,
        fitted_pi_ann=fitted_pi_ann_aligned,
        dates=dates_aligned,
        save_path=os.path.join(paper_fig_dir, 'figure2_economy_fit.pdf')
    )
    
    logger.info(f"  Saved to: {paper_fig_dir}/figure2_economy_fit.pdf")


def generate_figure3(config: dict, checkpoint_dir: str, output_dir: str, logger):
    """Generate Figure 3: Partial dependence for ANN economy."""
    logger.info("\nGenerating Figure 3: ANN Economy Partial Dependence...")
    
    # Create subfolder for paper figures
    paper_fig_dir = os.path.join(output_dir, 'paper_figures')
    os.makedirs(paper_fig_dir, exist_ok=True)
    
    # Load tuned parameters if available
    params_path = os.path.join(checkpoint_dir, 'ann_params.pkl')
    tuned_params = None
    if os.path.exists(params_path):
        with open(params_path, 'rb') as f:
            tuned_params = pickle.load(f)
    
    # Determine hyperparameters
    if tuned_params:
        hidden_units_y = tuned_params['output_gap']['hidden_units']
        hidden_units_pi = tuned_params['inflation']['hidden_units']
        lags_y = tuned_params['output_gap'].get('lags', 2)
        lags_pi = tuned_params['inflation'].get('lags', 2)
    else:
        hidden_units_y = config['economy']['ann']['hidden_units_y']
        hidden_units_pi = config['economy']['ann']['hidden_units_pi']
        lags_y = 2
        lags_pi = 2
        
    # Calculate input dimensions
    input_dim_y = 3 * lags_y
    input_dim_pi = 1 + 3 * lags_pi
    
    # Load networks
    network_y = EconomyNetwork(input_dim_y, hidden_units_y)
    network_pi = EconomyNetwork(input_dim_pi, hidden_units_pi)
    
    network_y.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, 'ann_y_network.pth'),
        map_location='cpu'
    ))
    network_pi.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, 'ann_pi_network.pth'),
        map_location='cpu'
    ))
    
    # Plot inflation partial dependence
    plot_partial_dependence(
        network=network_pi,
        variable='inflation',
        input_ranges={
            'inflation': (0, 6),
            'output_gap': (-5, 3),
            'interest_rate': (-3, 7)
        },
        grid_resolution=50,
        save_path=os.path.join(paper_fig_dir, 'figure3_pd_inflation.pdf'),
        title=r'Partial Dependence: Inflation $\pi_t$'
    )
    
    # Plot output gap partial dependence
    plot_partial_dependence(
        network=network_y,
        variable='output_gap',
        input_ranges={
            'inflation': (0, 6),
            'output_gap': (-5, 3),
            'interest_rate': (-3, 7)
        },
        grid_resolution=50,
        save_path=os.path.join(paper_fig_dir, 'figure3_pd_output_gap.pdf'),
        title=r'Partial Dependence: Output Gap $y_t$'
    )
    
    logger.info(f"  Saved to: {paper_fig_dir}/figure3_pd_*.pdf")


def generate_figures_6_7_8_9(config: dict, output_dir: str, logger):
    """Generate Figures 6-9: Counterfactual analysis."""
    logger.info("\nGenerating Figures 6-9: Counterfactual Analysis...")
    
    # Create subfolder for paper figures
    paper_fig_dir = os.path.join(output_dir, 'paper_figures')
    os.makedirs(paper_fig_dir, exist_ok=True)
    
    # Load data
    data_loader = DataLoader(
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    data = data_loader.get_data()
    
    # Load counterfactual results
    tables_dir = os.path.join(output_dir, '..', 'tables')
    
    # Figure 6: Historical counterfactual with linear policies
    linear_policies = ['TR93', 'NPP', 'BA', 'RL_SVAR_no_lag','RL_SVAR_one_lag','RL_ANN_no_lag', 'RL_ANN_one_lag']
    cf_data_linear = {}
    
    for policy in linear_policies:
        try:
            df = pd.read_csv(os.path.join(tables_dir, f'historical_cf_{policy}.csv'))
            cf_data_linear[policy] = {
                'ffr': df['ffr'].values,
                'inflation': df['inflation'].values,
                'output_gap': df['output_gap'].values
            }
        except FileNotFoundError:
            logger.warning(f"  Could not find data for {policy}")
    
    if cf_data_linear:
        plot_counterfactual(
            dates=data.index,
            actual_ffr=data['interest_rate'].values,
            actual_inflation=data['inflation'].values,
            actual_output_gap=data['output_gap'].values,
            counterfactual_data=cf_data_linear,
            save_path=os.path.join(paper_fig_dir, 'figure6_counterfactual_linear.pdf'),
            title='Historical Counterfactual: Linear Policies'
        )
        logger.info(f"  Saved Figure 6")
    
    # Figure 7: Historical counterfactual with RL policies
    rl_policies = [
        # 'TR93', 'NPP', 'BA',
                   'RL_SVAR_no_lag','RL_SVAR_one_lag','RL_ANN_no_lag', 'RL_ANN_one_lag',
                   'RL_ANN_no_lag_nonlin', 'RL_ANN_one_lag_nonlin']
    cf_data_rl = {}
    
    for policy in rl_policies:
        try:
            df = pd.read_csv(os.path.join(tables_dir, f'historical_cf_{policy}.csv'))
            cf_data_rl[policy] = {
                'ffr': df['ffr'].values,
                'inflation': df['inflation'].values,
                'output_gap': df['output_gap'].values
            }
        except FileNotFoundError:
            logger.warning(f"  Could not find data for {policy}")
    
    if cf_data_rl:
        plot_counterfactual(
            dates=data.index,
            actual_ffr=data['interest_rate'].values,
            actual_inflation=data['inflation'].values,
            actual_output_gap=data['output_gap'].values,
            counterfactual_data=cf_data_rl,
            save_path=os.path.join(paper_fig_dir, 'figure7_counterfactual_rl.pdf'),
            title='Historical Counterfactual: RL Policies'
        )
        logger.info(f"  Saved Figure 7")
    
    # Figure 8: Static counterfactual linear policies
    static_policies = ['TR93', 'NPP', 'BA','RL_SVAR_no_lag','RL_SVAR_one_lag','RL_ANN_no_lag', 'RL_ANN_one_lag',
                       ]
    cf_data_static = {}
    
    for policy in static_policies:
        try:
            df = pd.read_csv(os.path.join(tables_dir, f'static_cf_{policy}.csv'))
            cf_data_static[policy] = {
                'ffr': df['ffr'].values,
                'inflation': df['inflation'].values,
                'output_gap': df['output_gap'].values
            }
        except FileNotFoundError:
            logger.warning(f"  Could not find data for {policy}")
    
    if cf_data_static:
        plot_counterfactual(
            dates=data.index,
            actual_ffr=data['interest_rate'].values,
            actual_inflation=data['inflation'].values,
            actual_output_gap=data['output_gap'].values,
            counterfactual_data=cf_data_static,
            save_path=os.path.join(paper_fig_dir, 'figure8_static_counterfactual_linear.pdf'),
            title='Static Counterfactual: Linear Policies'
        )
        logger.info(f"  Saved Figure 8")

    # Figure 9: Static counterfactual RL policies
    static_policies = ['RL_SVAR_no_lag','RL_SVAR_one_lag','RL_ANN_no_lag', 'RL_ANN_one_lag',
                       'RL_SVAR_one_lag_nonlin','RL_SVAR_no_lag_nonlin', 'RL_ANN_one_lag_nonlin','RL_ANN_no_lag_nonlin']
    cf_data_static = {}
    
    for policy in static_policies:
        try:
            df = pd.read_csv(os.path.join(tables_dir, f'static_cf_{policy}.csv'))
            cf_data_static[policy] = {
                'ffr': df['ffr'].values,
                'inflation': df['inflation'].values,
                'output_gap': df['output_gap'].values
            }
        except FileNotFoundError:
            logger.warning(f"  Could not find data for {policy}")
    
    if cf_data_static:
        plot_counterfactual(
            dates=data.index,
            actual_ffr=data['interest_rate'].values,
            actual_inflation=data['inflation'].values,
            actual_output_gap=data['output_gap'].values,
            counterfactual_data=cf_data_static,
            save_path=os.path.join(paper_fig_dir, 'figure9_static_counterfactual_rl.pdf'),
            title='Static Counterfactual: RL Policies'
        )
        logger.info(f"  Saved Figure 9")


def generate_tables(config: dict, output_dir: str, logger):
    """Generate LaTeX tables."""
    logger.info("\nGenerating tables...")
    
    tables_dir = os.path.join(output_dir, '..', 'tables')
    
    # Table 4: Counterfactual loss
    try:
        metrics_df = pd.read_csv(os.path.join(tables_dir, 'table4_counterfactual_loss.csv'),
                                index_col=0)
        
        # Create LaTeX table
        create_results_table(
            metrics_dict=metrics_df.to_dict('index'),
            save_path=os.path.join(tables_dir, 'table4_counterfactual_loss.tex'),
            format='latex'
        )
        logger.info(f"  Saved Table 4 (LaTeX)")
    except FileNotFoundError:
        logger.warning("  Could not find counterfactual loss data")


def generate_learning_curves(output_dir: str, logger):
    """Generate learning curves from training logs."""
    logger.info("\nGenerating Learning Curves...")
    
    from src.utils.visualization import plot_training_curves
    
    # Create subfolder
    lc_dir = os.path.join(output_dir, 'learning_curves')
    os.makedirs(lc_dir, exist_ok=True)
    
    log_dir = os.path.join(output_dir, '..', 'logs')
    if not os.path.exists(log_dir):
        logger.warning(f"  Log directory not found: {log_dir}")
        return
        
    # Find all training logs (CSV files starting with ddpg_)
    log_files = [f for f in os.listdir(log_dir) if f.startswith('ddpg_') and f.endswith('.csv')]
    
    if not log_files:
        logger.warning("  No training logs found")
        return
        
    for log_file in log_files:
        try:
            # Read log file
            df = pd.read_csv(os.path.join(log_dir, log_file))
            
            # Extract metrics
            episode_rewards = df['total_reward'].values
            critic_losses = df['critic_loss'].values
            actor_losses = df['actor_loss'].values
            
            # Generate plot name from log filename
            # Remove timestamp and extension
            base_name = log_file.rsplit('_', 2)[0]
            plot_path = os.path.join(lc_dir, f'{base_name}_learning_curve.pdf')
            
            plot_training_curves(episode_rewards, critic_losses, actor_losses, plot_path)
            logger.info(f"  Saved learning curve: {plot_path}")
            
        except Exception as e:
            logger.error(f"  Failed to generate curve for {log_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate figures from paper')
    parser.add_argument('--all', action='store_true', help='Generate all figures')
    parser.add_argument('--figure', type=int, choices=[2, 3, 6, 7, 8, 9],
                       help='Generate specific figure')
    parser.add_argument('--tables', action='store_true', help='Generate tables')
    parser.add_argument('--learning_curve', action='store_true', help='Generate learning curves')
    parser.add_argument('--config', type=str, default='configs/hyperparameters.yaml')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints')
    parser.add_argument('--output_dir', type=str, default='results/figures')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    logger = setup_logger('figure_generation', None)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("FIGURE GENERATION")
    logger.info("="*60)
    
    # Generate figures
    if args.all or args.figure == 2:
        generate_figure2(config, args.checkpoint_dir, args.output_dir, logger)
    
    if args.all or args.figure == 3:
        generate_figure3(config, args.checkpoint_dir, args.output_dir, logger)
    
    if args.all or args.figure in [6, 7, 8, 9]:
        generate_figures_6_7_8_9(config, args.output_dir, logger)
    
    if args.all or args.tables:
        generate_tables(config, args.output_dir, logger)
        
    if args.all or args.learning_curve:
        generate_learning_curves(args.output_dir, logger)
    
    logger.info("\nFigure generation complete!")


if __name__ == "__main__":
    main()