#!/usr/bin/env python3
"""
Generate all figures from the paper.

Usage:
    python scripts/generate_figures.py --all
    python scripts/generate_figures.py --figure 2
"""

import argparse
import os
import sys
import yaml
import pickle
import torch
import numpy as np
import pandas as pd

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
    
    # Load ANN and compute fitted values
    network_y = EconomyNetwork(4, config['economy']['ann']['hidden_units_y'])
    network_pi = EconomyNetwork(5, config['economy']['ann']['hidden_units_pi'])
    
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
    lagged_data = create_lagged_features(data, lags=2)
    
    X_y = lagged_data[['output_gap_lag1', 'inflation_lag1',
                       'interest_rate_lag1', 'interest_rate_lag2']].values
    X_pi = lagged_data[['output_gap', 'output_gap_lag1', 'inflation_lag1',
                        'inflation_lag2', 'interest_rate_lag1']].values
    
    network_y.eval()
    network_pi.eval()
    
    with torch.no_grad():
        fitted_y_ann = network_y(torch.FloatTensor(X_y)).numpy()
        fitted_pi_ann = network_pi(torch.FloatTensor(X_pi)).numpy()
    
    # Plot
    plot_economy_fit(
        actual_y=lagged_data['output_gap'].values,
        actual_pi=lagged_data['inflation'].values,
        fitted_y_svar=fitted_y_svar,
        fitted_pi_svar=fitted_pi_svar,
        fitted_y_ann=fitted_y_ann,
        fitted_pi_ann=fitted_pi_ann,
        dates=lagged_data.index,
        save_path=os.path.join(output_dir, 'figure2_economy_fit.pdf')
    )
    
    logger.info(f"  Saved to: {output_dir}/figure2_economy_fit.pdf")


def generate_figure3(config: dict, checkpoint_dir: str, output_dir: str, logger):
    """Generate Figure 3: Partial dependence for ANN economy."""
    logger.info("\nGenerating Figure 3: ANN Economy Partial Dependence...")
    
    # Load networks
    network_y = EconomyNetwork(4, config['economy']['ann']['hidden_units_y'])
    network_pi = EconomyNetwork(5, config['economy']['ann']['hidden_units_pi'])
    
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
        save_path=os.path.join(output_dir, 'figure3_pd_inflation.pdf'),
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
        save_path=os.path.join(output_dir, 'figure3_pd_output_gap.pdf'),
        title=r'Partial Dependence: Output Gap $y_t$'
    )
    
    logger.info(f"  Saved to: {output_dir}/figure3_pd_*.pdf")


def generate_figures_6_7_8_9(config: dict, output_dir: str, logger):
    """Generate Figures 6-9: Counterfactual analysis."""
    logger.info("\nGenerating Figures 6-9: Counterfactual Analysis...")
    
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
            save_path=os.path.join(output_dir, 'figure6_counterfactual_linear.pdf'),
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
            save_path=os.path.join(output_dir, 'figure7_counterfactual_rl.pdf'),
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
            save_path=os.path.join(output_dir, 'figure8_static_counterfactual_linear.pdf'),
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
                save_path=os.path.join(output_dir, 'figure9_static_counterfactual_rl.pdf'),
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


def main():
    parser = argparse.ArgumentParser(description='Generate figures from paper')
    parser.add_argument('--all', action='store_true', help='Generate all figures')
    parser.add_argument('--figure', type=int, choices=[2, 3, 6, 7, 8, 9],
                       help='Generate specific figure')
    parser.add_argument('--tables', action='store_true', help='Generate tables')
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
    
    logger.info("\nFigure generation complete!")


if __name__ == "__main__":
    main()