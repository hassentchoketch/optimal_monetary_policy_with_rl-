#!/usr/bin/env python3
"""
Estimate economy transition equations (SVAR and ANN).

Usage:
    python scripts/estimate_economy.py --model svar
    python scripts/estimate_economy.py --model ann
"""

import argparse
import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import random
from typing import Dict, Any, Tuple

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader, create_lagged_features
from src.environment.ann_economy import EconomyNetwork
from src.utils.logger import setup_logger


def select_optimal_lags(
    raw_data: pd.DataFrame,
    max_lags: int,
    criterion: str = 'bic'
) -> Tuple[int, Dict]:
    """
    Select optimal lag length using information criteria.
    
    Args:
        raw_data: DataFrame with raw time series
        max_lags: Maximum number of lags to test
        criterion: 'aic' or 'bic'
        
    Returns:
        Tuple of (optimal_lags, results_summary)
    """
    best_ic = np.inf
    best_lags = 0
    results_summary = []
    
    for p in range(1, max_lags + 1):
        # Create lagged data for this lag length
        data_p = create_lagged_features(raw_data, lags=p)
        
        # Output Gap Equation
        features_y = []
        for l in range(1, p + 1):
            features_y.extend([f'output_gap_lag{l}', f'inflation_lag{l}', f'interest_rate_lag{l}'])
            
        X_y = data_p[features_y].values
        y_y = data_p['output_gap'].values
        X_y = add_constant(X_y)
        model_y = OLS(y_y, X_y).fit()
        
        # Inflation Equation (includes contemporaneous output gap)
        features_pi = ['output_gap']
        for l in range(1, p + 1):
            features_pi.extend([f'output_gap_lag{l}', f'inflation_lag{l}', f'interest_rate_lag{l}'])
            
        X_pi = data_p[features_pi].values
        y_pi = data_p['inflation'].values
        X_pi = add_constant(X_pi)
        model_pi = OLS(y_pi, X_pi).fit()
        
        # Calculate System Information Criteria
        if criterion == 'aic':
            current_ic = model_y.aic + model_pi.aic
        else: # bic
            current_ic = model_y.bic + model_pi.bic
            
        results_summary.append({
            'lags': p,
            'ic': current_ic,
            'mse_total': (model_y.mse_resid + model_pi.mse_resid) / 2
        })
        
        if current_ic < best_ic:
            best_ic = current_ic
            best_lags = p
            
    return best_lags, results_summary

def estimate_svar(
    raw_data: pd.DataFrame,
    config: dict,
    logger
) -> dict:
    """
    Estimate SVAR model with optimal lag selection (BIC).
    """
    logger.info("\n" + "="*60)
    logger.info("ESTIMATING SVAR ECONOMY (OPTIMAL LAGS)")
    logger.info("="*60)
    
    max_lags = config.get('economy', {}).get('svar', {}).get('max_lags', 8)
    logger.info(f"Searching for optimal lags (1 to {max_lags}) using BIC...")
    
    # Select optimal lags
    best_lags, results_summary = select_optimal_lags(raw_data, max_lags, criterion='bic')
    
    # Log selection results
    logger.info("\nLag Selection Results:")
    logger.info(f"{'Lags':<5} {'BIC':<12} {'MSE':<12}")
    logger.info("-" * 35)
    for res in results_summary:
        mark = "*" if res['lags'] == best_lags else ""
        logger.info(f"{res['lags']:<5} {res['ic']:<12.2f} {res['mse_total']:<12.6f} {mark}")
        
    logger.info(f"\nOptimal lag length selected: {best_lags}")
    
    # Re-estimate optimal model
    data_p = create_lagged_features(raw_data, lags=best_lags)
    
    # Output Gap
    features_y = []
    for l in range(1, best_lags + 1):
        features_y.extend([f'output_gap_lag{l}', f'inflation_lag{l}', f'interest_rate_lag{l}'])
    X_y = add_constant(data_p[features_y].values)
    best_model_y = OLS(data_p['output_gap'].values, X_y).fit()
    
    # Inflation
    features_pi = ['output_gap']
    for l in range(1, best_lags + 1):
        features_pi.extend([f'output_gap_lag{l}', f'inflation_lag{l}', f'interest_rate_lag{l}'])
    X_pi = add_constant(data_p[features_pi].values)
    best_model_pi = OLS(data_p['inflation'].values, X_pi).fit()
    
    # Log summaries
    logger.info("\nOutput Gap Equation (Optimal Lags):")
    logger.info(best_model_y.summary())
    logger.info("\nInflation Equation (Optimal Lags):")
    logger.info(best_model_pi.summary())
    
    # Construct params dictionaries
    params_y = {'const': best_model_y.params[0]}
    features_y_names = []
    for l in range(1, best_lags + 1):
        features_y_names.extend([f'output_gap_lag{l}', f'inflation_lag{l}', f'interest_rate_lag{l}'])
        
    for idx, name in enumerate(features_y_names):
        short_name = name.replace('output_gap', 'y').replace('inflation', 'pi').replace('interest_rate', 'i')
        params_y[short_name] = best_model_y.params[idx + 1]

    params_pi = {'const': best_model_pi.params[0]}
    features_pi_names = ['output_gap']
    for l in range(1, best_lags + 1):
        features_pi_names.extend([f'output_gap_lag{l}', f'inflation_lag{l}', f'interest_rate_lag{l}'])
        
    for idx, name in enumerate(features_pi_names):
        if name == 'output_gap':
            short_name = 'y_lag0'
        else:
            short_name = name.replace('output_gap', 'y').replace('inflation', 'pi').replace('interest_rate', 'i')
        params_pi[short_name] = best_model_pi.params[idx + 1]
    
    # Stats
    residuals_y = best_model_y.resid
    residuals_pi = best_model_pi.resid
    mse_y = best_model_y.mse_resid
    mse_pi = best_model_pi.mse_resid
    mse_total = (mse_y + mse_pi) / 2
    
    logger.info("\n" + "-"*60)
    logger.info("FIT STATISTICS (OPTIMAL MODEL)")
    logger.info("-"*60)
    logger.info(f"Output Gap MSE: {mse_y:.6f}")
    logger.info(f"Inflation MSE:  {mse_pi:.6f}")
    logger.info(f"Total MSE:      {mse_total:.6f}")
    
    return {
        'coefficients': {'output_gap': params_y, 'inflation': params_pi},
        'shock_std': {'output_gap': np.std(residuals_y), 'inflation': np.std(residuals_pi)},
        'lags': best_lags,
        'fit_statistics': {
            'mse_output_gap': mse_y,
            'mse_inflation': mse_pi,
            'mse_total': mse_total,
            'r2_output_gap': best_model_y.rsquared,
            'r2_inflation': best_model_pi.rsquared
        },
        'residuals': {'output_gap': residuals_y, 'inflation': residuals_pi},
        'fitted_values': {'output_gap': best_model_y.fittedvalues, 'inflation': best_model_pi.fittedvalues}
    }

def train_ann_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_units: int,
    config: dict,
    logger,
    variable_name: str
) -> EconomyNetwork:
    """Train ANN for single equation using early stopping."""
    input_dim = X_train.shape[1]
    network = EconomyNetwork(input_dim, hidden_units)
    optimizer = optim.Adam(network.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    best_val_loss = np.inf
    patience_counter = 0
    patience = config['patience']
    best_state_dict = None
    
    for epoch in range(config['max_epochs']):
        network.train()
        optimizer.zero_grad()
        predictions = network(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()
        
        network.eval()
        with torch.no_grad():
            val_predictions = network(X_val_t)
            val_loss = criterion(val_predictions, y_val_t)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = network.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 100 == 0:
            logger.info(f"  Epoch {epoch+1}: Val Loss={val_loss.item():.6f}")
    
    if best_state_dict:
        network.load_state_dict(best_state_dict)
    logger.info(f"  Best validation loss: {best_val_loss:.6f}")
    return network

def train_evaluate_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    epochs: int
) -> float:
    """Train network and return validation loss."""
    input_dim = X_train.shape[1]
    hidden_units = params['hidden_units']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    
    network = EconomyNetwork(input_dim, hidden_units)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = np.inf
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        network.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = network(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
        
        network.eval()
        with torch.no_grad():
            val_predictions = network(X_val_t)
            val_loss = criterion(val_predictions, y_val_t).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    return best_val_loss

def sample_hyperparameters() -> Dict[str, Any]:
    """Sample random hyperparameters including lags."""
    return {
        'hidden_units': random.choice([2, 4, 8, 16, 32]),
        'learning_rate': random.choice([0.01, 0.005, 0.001, 0.0005, 0.0001]),
        'batch_size': random.choice([16, 32, 64]),
        'lags': random.choice([1, 2, 3, 4])
    }

def tune_hyperparameters(
    data: pd.DataFrame,
    config: dict,
    logger,
    trials: int,
    epochs: int
) -> Dict[str, Dict[str, Any]]:
    """Tune hyperparameters for ANN economy."""
    logger.info("\n" + "="*60)
    logger.info("TUNING HYPERPARAMETERS")
    logger.info("="*60)
    
    from src.data.data_loader import prepare_training_data, create_lagged_features
    train_data, val_data = prepare_training_data(data, config['data']['validation_split'])
    
    best_y_params = None
    best_y_loss = np.inf
    best_pi_params = None
    best_pi_loss = np.inf
    
    logger.info(f"\nStarting {trials} trials...")
    
    for i in range(trials):
        params = sample_hyperparameters()
        logger.info(f"\nTrial {i+1}/{trials}: {params}")
        
        lags = params['lags']
        train_lagged = create_lagged_features(train_data, lags=lags)
        val_lagged = create_lagged_features(val_data, lags=lags)
        
        # Output Gap Features
        features_y = []
        for l in range(1, lags + 1):
            features_y.extend([f'output_gap_lag{l}', f'inflation_lag{l}', f'interest_rate_lag{l}'])
            
        X_train_y = train_lagged[features_y].values
        y_train_y = train_lagged['output_gap'].values
        X_val_y = val_lagged[features_y].values
        y_val_y = val_lagged['output_gap'].values
        
        # Inflation Features
        features_pi = ['output_gap']
        for l in range(1, lags + 1):
            features_pi.extend([f'output_gap_lag{l}', f'inflation_lag{l}', f'interest_rate_lag{l}'])
            
        X_train_pi = train_lagged[features_pi].values
        y_train_pi = train_lagged['inflation'].values
        X_val_pi = val_lagged[features_pi].values
        y_val_pi = val_lagged['inflation'].values
        
        # Tune
        loss_y = train_evaluate_network(X_train_y, y_train_y, X_val_y, y_val_y, params, epochs)
        if loss_y < best_y_loss:
            best_y_loss = loss_y
            best_y_params = params
            logger.info(f"  New best Output Gap loss: {loss_y:.6f}")
            
        loss_pi = train_evaluate_network(X_train_pi, y_train_pi, X_val_pi, y_val_pi, params, epochs)
        if loss_pi < best_pi_loss:
            best_pi_loss = loss_pi
            best_pi_params = params
            logger.info(f"  New best Inflation loss: {loss_pi:.6f}")
            
    return {'output_gap': best_y_params, 'inflation': best_pi_params}

def estimate_ann(
    data: pd.DataFrame,
    config: dict,
    logger,
    tuned_params: Dict[str, Dict[str, Any]] = None
) -> dict:
    """Estimate ANN economy model."""
    logger.info("\n" + "="*60)
    logger.info("ESTIMATING ANN ECONOMY")
    logger.info("="*60)
    
    ann_config = config['economy']['ann'].copy()
    
    # Determine lags
    lags_y = tuned_params['output_gap'].get('lags', 2) if tuned_params else 2
    lags_pi = tuned_params['inflation'].get('lags', 2) if tuned_params else 2
    
    if tuned_params:
        logger.info("\nUsing tuned hyperparameters:")
        logger.info(f"  Output Gap (Lags={lags_y}): {tuned_params['output_gap']}")
        logger.info(f"  Inflation (Lags={lags_pi}): {tuned_params['inflation']}")
        ann_config['hidden_units_y'] = tuned_params['output_gap']['hidden_units']
        ann_config['hidden_units_pi'] = tuned_params['inflation']['hidden_units']

    from src.data.data_loader import prepare_training_data, create_lagged_features
    train_data, val_data = prepare_training_data(data, config['data']['validation_split'])
    
    # --- Output Gap Network ---
    logger.info(f"\nTraining Output Gap Network (Lags: {lags_y})")
    train_lagged_y = create_lagged_features(train_data, lags=lags_y)
    val_lagged_y = create_lagged_features(val_data, lags=lags_y)
    
    features_y = []
    for l in range(1, lags_y + 1):
        features_y.extend([f'output_gap_lag{l}', f'inflation_lag{l}', f'interest_rate_lag{l}'])
        
    X_train_y = train_lagged_y[features_y].values
    y_train_y = train_lagged_y['output_gap'].values
    X_val_y = val_lagged_y[features_y].values
    y_val_y = val_lagged_y['output_gap'].values
    
    y_config = ann_config.copy()
    if tuned_params:
        y_config['learning_rate'] = tuned_params['output_gap']['learning_rate']
        y_config['batch_size'] = tuned_params['output_gap']['batch_size']
        
    network_y = train_ann_network(
        X_train_y, y_train_y, X_val_y, y_val_y,
        ann_config['hidden_units_y'], y_config, logger, 'output_gap'
    )
    
    # --- Inflation Network ---
    logger.info(f"\nTraining Inflation Network (Lags: {lags_pi})")
    train_lagged_pi = create_lagged_features(train_data, lags=lags_pi)
    val_lagged_pi = create_lagged_features(val_data, lags=lags_pi)
    
    features_pi = ['output_gap']
    for l in range(1, lags_pi + 1):
        features_pi.extend([f'output_gap_lag{l}', f'inflation_lag{l}', f'interest_rate_lag{l}'])
        
    X_train_pi = train_lagged_pi[features_pi].values
    y_train_pi = train_lagged_pi['inflation'].values
    X_val_pi = val_lagged_pi[features_pi].values
    y_val_pi = val_lagged_pi['inflation'].values
    
    pi_config = ann_config.copy()
    if tuned_params:
        pi_config['learning_rate'] = tuned_params['inflation']['learning_rate']
        pi_config['batch_size'] = tuned_params['inflation']['batch_size']
        
    network_pi = train_ann_network(
        X_train_pi, y_train_pi, X_val_pi, y_val_pi,
        ann_config['hidden_units_pi'], pi_config, logger, 'inflation'
    )
    
    # --- Evaluation ---
    full_lagged_y = create_lagged_features(data, lags=lags_y)
    full_lagged_pi = create_lagged_features(data, lags=lags_pi)
    
    X_full_y = full_lagged_y[features_y].values
    X_full_pi = full_lagged_pi[features_pi].values
    
    network_y.eval()
    network_pi.eval()
    with torch.no_grad():
        fitted_y = network_y(torch.FloatTensor(X_full_y)).numpy()
        fitted_pi = network_pi(torch.FloatTensor(X_full_pi)).numpy()
        
    actual_y = full_lagged_y['output_gap'].values
    actual_pi = full_lagged_pi['inflation'].values
    
    residuals_y = actual_y - fitted_y
    residuals_pi = actual_pi - fitted_pi
    
    mse_y = mean_squared_error(actual_y, fitted_y)
    mse_pi = mean_squared_error(actual_pi, fitted_pi)
    mse_total = (mse_y + mse_pi) / 2
    
    logger.info("\n" + "-"*60)
    logger.info("FIT STATISTICS")
    logger.info("-"*60)
    logger.info(f"Output Gap MSE: {mse_y:.6f}")
    logger.info(f"Inflation MSE:  {mse_pi:.6f}")
    logger.info(f"Total MSE:      {mse_total:.6f}")
    
    return {
        'network_y': network_y, 'network_pi': network_pi,
        'shock_std': {'output_gap': np.std(residuals_y), 'inflation': np.std(residuals_pi)},
        'fit_statistics': {'mse_output_gap': mse_y, 'mse_inflation': mse_pi, 'mse_total': mse_total},
        'lags': {'output_gap': lags_y, 'inflation': lags_pi}
    }

def main():
    parser = argparse.ArgumentParser(description='Estimate economy models')
    parser.add_argument('--model', type=str, required=True, choices=['svar', 'ann', 'both'], default='both')
    parser.add_argument('--config', type=str, default='configs/hyperparameters.yaml')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='results/checkpoints')
    parser.add_argument('--tune', action='store_true', help='Tune hyperparameters')
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=500)
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    os.makedirs('results/logs', exist_ok=True)
    logger = setup_logger('economy_estimation', 'results/logs/economy_estimation.log')
    
    logger.info("="*60)
    logger.info("ECONOMY ESTIMATION")
    logger.info("="*60)
    
    data_loader = DataLoader(start_date=config['data']['start_date'], end_date=config['data']['end_date'])
    data = data_loader.get_data()
    logger.info(f"Data loaded: {len(data)} observations")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    svar_results = None
    ann_results = None
    
    if args.model in ['svar', 'both']:
        svar_results = estimate_svar(data, config, logger)
        with open(os.path.join(args.output_dir, 'svar_params.pkl'), 'wb') as f:
            pickle.dump(svar_results, f)
            
    if args.model in ['ann', 'both']:
        tuned_params = None
        if args.tune:
            tuned_params = tune_hyperparameters(data, config, logger, args.trials, args.epochs)
            
        ann_results = estimate_ann(data, config, logger, tuned_params)
        
        torch.save(ann_results['network_y'].state_dict(), os.path.join(args.output_dir, 'ann_y_network.pth'))
        torch.save(ann_results['network_pi'].state_dict(), os.path.join(args.output_dir, 'ann_pi_network.pth'))
        with open(os.path.join(args.output_dir, 'ann_shock_std.pkl'), 'wb') as f:
            pickle.dump(ann_results['shock_std'], f)
        
        if tuned_params:
            with open(os.path.join(args.output_dir, 'ann_params.pkl'), 'wb') as f:
                pickle.dump(tuned_params, f)

    if args.model == 'both' and svar_results and ann_results:
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        
        comparison_df = pd.DataFrame({
            'Model': ['SVAR', 'ANN'],
            'MSE_Output_Gap': [svar_results['fit_statistics']['mse_output_gap'], ann_results['fit_statistics']['mse_output_gap']],
            'MSE_Inflation': [svar_results['fit_statistics']['mse_inflation'], ann_results['fit_statistics']['mse_inflation']],
            'MSE_Total': [svar_results['fit_statistics']['mse_total'], ann_results['fit_statistics']['mse_total']]
        })
        logger.info("\n" + str(comparison_df))
        
        improvement = (svar_results['fit_statistics']['mse_total'] - ann_results['fit_statistics']['mse_total']) / svar_results['fit_statistics']['mse_total'] * 100
        logger.info(f"\nANN improvement over SVAR: {improvement:.1f}%")

if __name__ == "__main__":
    main()