#!/usr/bin/env python3
"""
Evaluate policies using counterfactual analysis.

Usage:
    python scripts/evaluate_policy.py --mode historical
    python scripts/evaluate_policy.py --mode static
"""

import argparse
import os
import sys
import yaml
import pickle
import torch
import numpy as np
import pandas as pd
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.svar_economy import SVAREconomy
from src.environment.ann_economy import ANNEconomy, EconomyNetwork
from src.agents.ddpg_agent import DDPGAgent
from src.policies.baseline_policies import BaselinePolicy
from src.policies.rl_policy import RLPolicy
from src.data.data_loader import DataLoader
from src.utils.metrics import compute_metrics, compare_policies, compute_loss
from src.utils.logger import setup_logger

def load_all_policies(checkpoint_dir: str, config: dict, device: str = 'cpu') -> Dict:
    """Load all trained policies and baselines."""
    policies = {}
    
    # Baseline policies
    for rule_type in ['TR93', 'NPP', 'BA']:
        policies[rule_type] = BaselinePolicy(
            rule_type=rule_type,
            r_star=config['baseline_policies'][rule_type.lower()]['r_star'],
            pi_star=config['reward']['target_inflation']
        )
    
    # RL policies
    for economy in ['svar', 'ann']:
        for policy_type in ['linear', 'nonlinear']:
            for lags in [0, 1]:
                if policy_type == 'nonlinear' and economy == 'svar':
                    continue
                
                name = f"RL_{economy.upper()}_{'no' if lags == 0 else 'one'}_lag"
                if policy_type == 'nonlinear':
                    name += "_nonlin"
                
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"ddpg_{economy}_{policy_type}_lags{lags}.pth"
                )
                
                if os.path.exists(checkpoint_path):
                    try:
                        # Load checkpoint first to determine architecture
                        checkpoint = torch.load(checkpoint_path, map_location=device)
                        
                        state_dim = 2 if lags == 0 else 4
                        action_dim = 1
                        
                        # Detect critic hidden size
                        critic_hidden = None
                        critic_state_dict = checkpoint.get('critic_state_dict', {})
                        for key, value in critic_state_dict.items():
                            if any(layer in key for layer in ['obs_fc1.weight', 'fc1.weight', 'network.0.weight']):
                                critic_hidden = value.shape[0]
                                break
                        
                        # Detect actor hidden size
                        actor_hidden = None
                        actor_state_dict = checkpoint.get('actor_state_dict', {})
                        for key, value in actor_state_dict.items():
                            if any(layer in key for layer in ['fc1.weight', 'network.0.weight', '0.weight']):
                                actor_hidden = value.shape[0]
                                break
                        
                        # Use detected values or fallbacks
                        if critic_hidden is None:
                            critic_hidden = 5  # Based on your earlier errors
                        if actor_hidden is None:
                            actor_hidden = 3 if policy_type == 'linear' else 8
                        
                        print(f"Loading {name}: state_dim={state_dim}, critic_hidden={critic_hidden}, actor_hidden={actor_hidden}")
                        
                        # Create agent with detected architecture
                        agent = DDPGAgent(
                            state_dim=state_dim,
                            action_dim=action_dim,
                            critic_hidden=critic_hidden,
                            actor_hidden=actor_hidden,
                            linear_policy=(policy_type == 'linear'),
                            device=device
                        )
                        agent.load(checkpoint_path)
                        policies[name] = RLPolicy(agent, name)
                        
                    except Exception as e:
                        print(f"Error loading {checkpoint_path}: {e}")
                        continue
                else:
                    print(f"Warning: Checkpoint not found: {checkpoint_path}")
    
    return policies

def run_historical_counterfactual(
    economy,
    policies: Dict,
    data: pd.DataFrame,
    logger
) -> Dict:
    """
    Run historical counterfactual analysis (Figures 6-7).
    
    Simulates economy under different policies using estimated shocks.
    """
    logger.info("\n" + "="*60)
    logger.info("HISTORICAL COUNTERFACTUAL ANALYSIS")
    logger.info("="*60)
    
    n_periods = len(data)
    results = {}
    
    # Load estimated shocks (from economy estimation)
    # For simplicity, we'll simulate with random shocks
    # In full implementation, use actual estimated structural shocks
    
    for policy_name, policy in policies.items():
        logger.info(f"\nSimulating with {policy_name}...")
        
        # Initialize state
        initial_state = np.array([
            data['output_gap'].iloc[0],
            data['output_gap'].iloc[0],
            data['inflation'].iloc[0],
            data['inflation'].iloc[0],
            data['interest_rate'].iloc[0],
            data['interest_rate'].iloc[0]
        ])
        
        state = economy.reset(initial_state)
        
        # Storage
        ffr_series = []
        inflation_series = []
        output_gap_series = []
        
        for t in range(n_periods):
            # Get action from policy
            action = policy.get_action(state)
            
            # Step environment
            state, reward, done, info = economy.step(state, action)
            
            # Store
            ffr_series.append(action)
            inflation_series.append(info['inflation'])
            output_gap_series.append(info['output_gap'])
        
        results[policy_name] = {
            'ffr': np.array(ffr_series),
            'inflation': np.array(inflation_series),
            'output_gap': np.array(output_gap_series)
        }
    
    return results

def run_static_counterfactual(
    policies: Dict,
    data: pd.DataFrame,
    logger
) -> Dict:
    """
    Run static counterfactual analysis (Figure 8).
    
    Applies policies to actual data without feedback.
    """
    logger.info("\n" + "="*60)
    logger.info("STATIC COUNTERFACTUAL ANALYSIS")
    logger.info("="*60)
    
    results = {}
    
    for policy_name, policy in policies.items():
        logger.info(f"\nEvaluating {policy_name}...")
        
        ffr_series = []
        
        for idx in range(len(data)):
            # Create state from actual data
            if idx == 0:
                state = np.array([
                    data['output_gap'].iloc[idx],
                    data['output_gap'].iloc[idx],
                    data['inflation'].iloc[idx],
                    data['inflation'].iloc[idx],
                    data['interest_rate'].iloc[idx],
                    data['interest_rate'].iloc[idx]
                ])
            else:
                state = np.array([
                    data['output_gap'].iloc[idx],
                    data['output_gap'].iloc[idx-1],
                    data['inflation'].iloc[idx],
                    data['inflation'].iloc[idx-1],
                    data['interest_rate'].iloc[idx-1],
                    data['interest_rate'].iloc[max(0, idx-2)]
                ])
            
            # Get action
            action = policy.get_action(state)
            ffr_series.append(action)
        
        results[policy_name] = {
            'ffr': np.array(ffr_series),
            'inflation': data['inflation'].values,
            'output_gap': data['output_gap'].values
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate monetary policies')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['historical', 'static', 'both'],
                       help='Evaluation mode')
    parser.add_argument('--economy', type=str, default='ann',
                       choices=['svar', 'ann'],
                       help='Economy model to use')
    parser.add_argument('--config', type=str, default='configs/hyperparameters.yaml')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    logger = setup_logger(
        'policy_evaluation',
        os.path.join(args.output_dir, 'logs', 'policy_evaluation.log')
    )
    
    logger.info("="*60)
    logger.info("POLICY EVALUATION")
    logger.info("="*60)
    
    # Load data
    logger.info("\nLoading data...")
    data_loader = DataLoader(
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    data = data_loader.get_data()
    
    # Load economy
    logger.info(f"\nLoading {args.economy.upper()} economy...")
    if args.economy == 'svar':
        with open(os.path.join(args.checkpoint_dir, 'svar_params.pkl'), 'rb') as f:
            params = pickle.load(f)
        economy = SVAREconomy(
            params=params['coefficients'],shock_std=params['shock_std'],
            target_inflation=config['reward']['target_inflation'],
            reward_weights={
                'inflation': config['reward']['weight_inflation'],
                'output_gap': config['reward']['weight_output_gap']
            },
            seed=config['seed']
        )
    else:  # ann
        # Load networks
        network_y = EconomyNetwork(4, config['economy']['ann']['hidden_units_y'])
        network_pi = EconomyNetwork(5, config['economy']['ann']['hidden_units_pi'])
        
        network_y.load_state_dict(torch.load(
            os.path.join(args.checkpoint_dir, 'ann_y_network.pth'),
            map_location=args.device
        ))
        network_pi.load_state_dict(torch.load(
            os.path.join(args.checkpoint_dir, 'ann_pi_network.pth'),
            map_location=args.device
        ))
        
        with open(os.path.join(args.checkpoint_dir, 'ann_shock_std.pkl'), 'rb') as f:
            shock_std = pickle.load(f)
        
        economy = ANNEconomy(
            network_y=network_y,
            network_pi=network_pi,
            shock_std=shock_std,
            target_inflation=config['reward']['target_inflation'],
            reward_weights={
                'inflation': config['reward']['weight_inflation'],
                'output_gap': config['reward']['weight_output_gap'],
            
            },
            seed=config['seed'],
            device=args.device
        )

    # Load policies
    logger.info("\nLoading policies...")
    policies = load_all_policies(args.checkpoint_dir, config, args.device)
    logger.info(f"Loaded {len(policies)} policies")
    
    # Run evaluations
    if args.mode in ['historical', 'both']:
        hist_results = run_historical_counterfactual(economy, policies, data, logger)
        
        # Compute metrics
        logger.info("\n" + "="*60)
        logger.info("HISTORICAL COUNTERFACTUAL METRICS (Table 4)")
        logger.info("="*60)
        
        all_metrics = {}
        
        # Actual data metrics
        actual_metrics = compute_metrics(
            data['inflation'].values,
            data['output_gap'].values,
            data['interest_rate'].values
        )
        all_metrics['Actual'] = actual_metrics
        
        logger.info(f"\nActual:")
        logger.info(f"  MSE Inflation:  {actual_metrics['mse_inflation']:.4f}")
        logger.info(f"  MSE Output Gap: {actual_metrics['mse_output_gap']:.4f}")
        logger.info(f"  Total Loss:     {actual_metrics['total_loss']:.4f}")
        
        # Policy metrics
        for policy_name, result in hist_results.items():
            metrics = compute_metrics(
                result['inflation'],
                result['output_gap'],
                result['ffr']
            )
            all_metrics[policy_name] = metrics
            
            improvement = compare_policies(actual_metrics, metrics)
            
            logger.info(f"\n{policy_name}:")
            logger.info(f"  MSE Inflation:  {metrics['mse_inflation']:.4f}")
            logger.info(f"  MSE Output Gap: {metrics['mse_output_gap']:.4f}")
            logger.info(f"  Total Loss:     {metrics['total_loss']:.4f}")
            logger.info(f"  Improvement:    {improvement['loss_improvement_pct']:.1f}%")
        
        # Save results
        results_df = pd.DataFrame(all_metrics).T
        results_df.to_csv(os.path.join(args.output_dir, 'tables', 
                                       'table4_counterfactual_loss.csv'))
        
        # Save time series
        for policy_name, result in hist_results.items():
            series_df = pd.DataFrame({
                'date': data.index,
                'ffr': result['ffr'],
                'inflation': result['inflation'],
                'output_gap': result['output_gap']
            })
            series_df.to_csv(os.path.join(args.output_dir, 'tables',
                                          f'historical_cf_{policy_name}.csv'),
                            index=False)
    
    if args.mode in ['static', 'both']:
        static_results = run_static_counterfactual(policies, data, logger)
        
        # Save static counterfactual results
        for policy_name, result in static_results.items():
            series_df = pd.DataFrame({
                'date': data.index,
                'ffr': result['ffr'],
                'inflation': result['inflation'],
                'output_gap': result['output_gap']
            })
            series_df.to_csv(os.path.join(args.output_dir, 'tables',
                                          f'static_cf_{policy_name}.csv'),
                            index=False)
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()