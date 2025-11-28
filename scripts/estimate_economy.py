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


def estimate_svar(
    raw_data: pd.DataFrame,
    config: dict,
    logger
) -> dict:
    """
    Estimate SVAR model with optimal lag selection (AIC).
    
    Process:
    1. Iterate over lag lengths (1 to max_lags).
    2. For each lag, estimate full SVAR model.
    3. Calculate AIC/BIC.
    4. Select optimal lag length based on AIC.
    5. Return parameters for the optimal model (keeping all variables).
    
    Args:
        raw_data: DataFrame with raw time series (not lagged)
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        Dictionary with estimated parameters and statistics
    """
    logger.info("\n" + "="*60)
    logger.info("ESTIMATING SVAR ECONOMY (OPTIMAL LAGS)")
    logger.info("="*60)
    
    max_lags = config.get('economy', {}).get('svar', {}).get('max_lags', 8)
    logger.info(f"Searching for optimal lags (1 to {max_lags})...")
    
    best_aic = np.inf
    best_lags = 0
    best_model_y = None
    best_model_pi = None
    best_data = None
    
    results_summary = []
    
    for p in range(1, max_lags + 1):
        # Create lagged data for this lag length
        data_p = create_lagged_features(raw_data, lags=p)
        
        # ---------------------------------------------------------
        # Output Gap Equation
        # y_t = C + sum(y_{t-k}) + sum(π_{t-k}) + sum(i_{t-k})
        # ---------------------------------------------------------
        features_y = []
        for l in range(1, p + 1):
            features_y.extend([f'output_gap_lag{l}', f'inflation_lag{l}', f'interest_rate_lag{l}'])
            
        X_y = data_p[features_y].values
        y_y = data_p['output_gap'].values
        X_y = add_constant(X_y)
        
        model_y = OLS(y_y, X_y).fit()
        
        # ---------------------------------------------------------
        # Inflation Equation
        # π_t = C + y_t + sum(y_{t-k}) + sum(π_{t-k}) + sum(i_{t-k})
        # ---------------------------------------------------------
        features_pi = ['output_gap'] # Contemporaneous output gap
        for l in range(1, p + 1):
            features_pi.extend([f'output_gap_lag{l}', f'inflation_lag{l}', f'interest_rate_lag{l}'])
            
        X_pi = data_p[features_pi].values
        y_pi = data_p['inflation'].values
        X_pi = add_constant(X_pi)
        
        model_pi = OLS(y_pi, X_pi).fit()
        
        # ---------------------------------------------------------
        # Calculate Information Criteria
        # ---------------------------------------------------------
        # Sum of AICs for both equations (System AIC)
        current_aic = model_y.aic + model_pi.aic
        current_bic = model_y.bic + model_pi.bic
        
        results_summary.append({
            'lags': p,
            'aic': current_aic,
            'bic': current_bic,
            'mse_total': (model_y.mse_resid + model_pi.mse_resid) / 2
        })
        
        if current_aic < best_aic:
            best_aic = current_aic
            best_lags = p
            best_model_y = model_y
            best_model_pi = model_pi
            best_data = data_p
            
    # Log selection results
    logger.info("\nLag Selection Results:")
    logger.info(f"{'Lags':<5} {'AIC':<12} {'BIC':<12} {'MSE':<12}")
    logger.info("-" * 45)
    for res in results_summary:
        mark = "*" if res['lags'] == best_lags else ""
        logger.info(f"{res['lags']:<5} {res['aic']:<12.2f} {res['bic']:<12.2f} {res['mse_total']:<12.6f} {mark}")
        
    logger.info(f"\nOptimal lag length selected: {best_lags}")
    
    # ---------------------------------------------------------
    # Final Model Summary
    # ---------------------------------------------------------
    logger.info("\nOutput Gap Equation (Optimal Lags):")
    logger.info(best_model_y.summary())
    
    logger.info("\nInflation Equation (Optimal Lags):")
    logger.info(best_model_pi.summary())
    
    # ---------------------------------------------------------
    # Map Coefficients
    # ---------------------------------------------------------
    
    # Helper to map feature names to param keys
    # We need to handle dynamic lag names
    
    # Construct params_y
    params_y = {}
    params_y['const'] = best_model_y.params[0]
    
    # Features in order: [const, output_gap_lag1, inflation_lag1, interest_rate_lag1, output_gap_lag2, ...]
    # Note: create_lagged_features might order them differently, need to be careful.
    # Let's reconstruct the feature list used for training to map correctly
    features_y_names = []
    for l in range(1, best_lags + 1):
        features_y_names.extend([f'output_gap_lag{l}', f'inflation_lag{l}', f'interest_rate_lag{l}'])
    
    # Map coefficients
    # params[0] is const
    for idx, name in enumerate(features_y_names):
        # Map standard names to internal param names
        # e.g. output_gap_lag1 -> y_lag1
        # e.g. inflation_lag1 -> pi_lag1
        # e.g. interest_rate_lag1 -> i_lag1
        
        short_name = name.replace('output_gap', 'y').replace('inflation', 'pi').replace('interest_rate', 'i')
        params_y[short_name] = best_model_y.params[idx + 1]

    # Construct params_pi
    params_pi = {}
    params_pi['const'] = best_model_pi.params[0]
    
    features_pi_names = ['output_gap']
    for l in range(1, best_lags + 1):
        features_pi_names.extend([f'output_gap_lag{l}', f'inflation_lag{l}', f'interest_rate_lag{l}'])
        
    for idx, name in enumerate(features_pi_names):
        if name == 'output_gap':
            short_name = 'y_lag0'
        else:
            short_name = name.replace('output_gap', 'y').replace('inflation', 'pi').replace('interest_rate', 'i')
        
        params_pi[short_name] = best_model_pi.params[idx + 1]
    
    # Compute residuals and stats
    residuals_y = best_model_y.resid
    residuals_pi = best_model_pi.resid
    
    shock_std = {
        'output_gap': np.std(residuals_y),
        'inflation': np.std(residuals_pi)
    }
    
    fitted_y = best_model_y.fittedvalues
    fitted_pi = best_model_pi.fittedvalues
    
    # We need to be careful with MSE calculation because different lags mean different sample sizes
    # But here we just report the MSE of the fitted model on its training data
    mse_y = best_model_y.mse_resid
    mse_pi = best_model_pi.mse_resid
    mse_total = (mse_y + mse_pi) / 2
    
    logger.info("\n" + "-"*60)
    logger.info("FIT STATISTICS (OPTIMAL MODEL)")
    logger.info("-"*60)
    logger.info(f"Output Gap MSE: {mse_y:.6f}")
    logger.info(f"Inflation MSE:  {mse_pi:.6f}")
    logger.info(f"Total MSE:      {mse_total:.6f}")
    logger.info(f"Output Gap R²:  {best_model_y.rsquared:.4f}")
    logger.info(f"Inflation R²:   {best_model_pi.rsquared:.4f}")
    
    return {
        'coefficients': {
            'output_gap': params_y,
            'inflation': params_pi
        },
        'shock_std': shock_std,
        'lags': best_lags,
        'fit_statistics': {
            'mse_output_gap': mse_y,
            'mse_inflation': mse_pi,
            'mse_total': mse_total,
            'r2_output_gap': best_model_y.rsquared,
            'r2_inflation': best_model_pi.rsquared
        },
        'residuals': {
            'output_gap': residuals_y,
            'inflation': residuals_pi
        },
        'fitted_values': {
            'output_gap': fitted_y,
            'inflation': fitted_pi
        }
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
    """
    Train ANN for single equation using early stopping.
    
    Args:
        X_train: Training inputs
        y_train: Training targets
        X_val: Validation inputs
        y_val: Validation targets
        hidden_units: Number of hidden units
        config: ANN configuration
        logger: Logger instance
        variable_name: Name of variable being predicted
    
    Returns:
        Trained network
    """
    input_dim = X_train.shape[1]
    
    # Initialize network
    network = EconomyNetwork(input_dim, hidden_units)
    
    # Optimizer and loss
    optimizer = optim.Adam(network.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    # Training loop with early stopping
    best_val_loss = np.inf
    patience_counter = 0
    patience = config['patience']
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config['max_epochs']):
        # Training
        network.train()
        optimizer.zero_grad()
        
        predictions = network(X_train_t)
        loss = criterion(predictions, y_train_t)
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        network.eval()
        with torch.no_grad():
            val_predictions = network(X_val_t)
            val_loss = criterion(val_predictions, y_val_t)
        
        val_losses.append(val_loss.item())
        
        # Early stopping check
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
            logger.info(f"  Epoch {epoch+1}: Train Loss={loss.item():.6f}, "
                       f"Val Loss={val_loss.item():.6f}")
    
    # Restore best model
    network.load_state_dict(best_state_dict)
    
    logger.info(f"  Best validation loss: {best_val_loss:.6f}")
    
    return network

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_evaluate_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    epochs: int
) -> float:
    """
    Train network and return validation loss.
    """
    input_dim = X_train.shape[1]
    hidden_units = params['hidden_units']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    
    # Initialize network
    network = EconomyNetwork(input_dim, hidden_units)
    
    # Optimizer and loss
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    # Create data loader for batching
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
        
        # Validation
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
    """Sample random hyperparameters."""
    return {
        'hidden_units': random.choice([2, 4, 8, 16, 32]),
        'learning_rate': random.choice([0.01, 0.005, 0.001, 0.0005, 0.0001]),
        'batch_size': random.choice([16, 32, 64])
    }

def tune_hyperparameters(
    data: pd.DataFrame,
    config: dict,
    logger,
    trials: int,
    epochs: int
) -> Dict[str, Dict[str, Any]]:
    """
    Tune hyperparameters for ANN economy.
    """
    logger.info("\n" + "="*60)
    logger.info("TUNING HYPERPARAMETERS")
    logger.info("="*60)
    
    # Prepare data
    from src.data.data_loader import prepare_training_data, create_lagged_features
    train_data, val_data = prepare_training_data(data, config['data']['validation_split'])
    train_lagged = create_lagged_features(train_data, lags=2)
    val_lagged = create_lagged_features(val_data, lags=2)
    
    # Prepare datasets for Output Gap
    X_train_y = train_lagged[['output_gap_lag1', 'inflation_lag1', 'interest_rate_lag1', 'interest_rate_lag2']].values
    y_train_y = train_lagged['output_gap'].values
    X_val_y = val_lagged[['output_gap_lag1', 'inflation_lag1', 'interest_rate_lag1', 'interest_rate_lag2']].values
    y_val_y = val_lagged['output_gap'].values
    
    # Prepare datasets for Inflation
    X_train_pi = train_lagged[['output_gap', 'output_gap_lag1', 'inflation_lag1', 'inflation_lag2', 'interest_rate_lag1']].values
    y_train_pi = train_lagged['inflation'].values
    X_val_pi = val_lagged[['output_gap', 'output_gap_lag1', 'inflation_lag1', 'inflation_lag2', 'interest_rate_lag1']].values
    y_val_pi = val_lagged['inflation'].values
    
    best_y_params = None
    best_y_loss = np.inf
    
    best_pi_params = None
    best_pi_loss = np.inf
    
    logger.info(f"\nStarting {trials} trials...")
    
    for i in range(trials):
        params = sample_hyperparameters()
        logger.info(f"\nTrial {i+1}/{trials}: {params}")
        
        # Tune Output Gap
        loss_y = train_evaluate_network(X_train_y, y_train_y, X_val_y, y_val_y, params, epochs)
        if loss_y < best_y_loss:
            best_y_loss = loss_y
            best_y_params = params
            logger.info(f"  New best Output Gap loss: {loss_y:.6f}")
            
        # Tune Inflation
        loss_pi = train_evaluate_network(X_train_pi, y_train_pi, X_val_pi, y_val_pi, params, epochs)
        if loss_pi < best_pi_loss:
            best_pi_loss = loss_pi
            best_pi_params = params
            logger.info(f"  New best Inflation loss: {loss_pi:.6f}")
            
    return {
        'output_gap': best_y_params,
        'inflation': best_pi_params
    }

def estimate_ann(
    data: pd.DataFrame,
    config: dict,
    logger,
    tuned_params: Dict[str, Dict[str, Any]] = None
) -> dict:
    """
    Estimate ANN economy model (Equation 8).
    
    Args:
        data: Full dataset
        config: Configuration dictionary
        logger: Logger instance
        tuned_params: Optional tuned hyperparameters
    
    Returns:
        Dictionary with trained networks and statistics
    """
    logger.info("\n" + "="*60)
    logger.info("ESTIMATING ANN ECONOMY")
    logger.info("="*60)
    
    ann_config = config['economy']['ann'].copy()
    
    # Apply tuned parameters if available
    if tuned_params:
        logger.info("\nUsing tuned hyperparameters:")
        
        # Output gap params
        y_params = tuned_params['output_gap']
        ann_config['hidden_units_y'] = y_params['hidden_units']
        # Note: learning_rate and batch_size are shared in config but can be specific here if we modify train_ann_network
        # For now, we'll use the tuned values for the respective networks
        
        logger.info("  Output Gap:")
        logger.info(f"    Hidden Units: {y_params['hidden_units']}")
        logger.info(f"    Learning Rate: {y_params['learning_rate']}")
        logger.info(f"    Batch Size: {y_params['batch_size']}")
        
        # Inflation params
        pi_params = tuned_params['inflation']
        ann_config['hidden_units_pi'] = pi_params['hidden_units']
        
        logger.info("  Inflation:")
        logger.info(f"    Hidden Units: {pi_params['hidden_units']}")
        logger.info(f"    Learning Rate: {pi_params['learning_rate']}")
        logger.info(f"    Batch Size: {pi_params['batch_size']}")
    
    # Split into train/validation
    from src.data.data_loader import prepare_training_data
    train_data, val_data = prepare_training_data(
        data,
        config['data']['validation_split'],
        # config['seed']
    )
    
    logger.info(f"\nDataset sizes:")
    logger.info(f"  Training:   {len(train_data)} observations")
    logger.info(f"  Validation: {len(val_data)} observations")
    
    # Create lagged features
    from src.data.data_loader import create_lagged_features
    train_lagged = create_lagged_features(train_data, lags=2)
    val_lagged = create_lagged_features(val_data, lags=2)
    
    # Output gap network
    # Input: [y_t, π_t, i_t, i_{t-1}]
    logger.info("\n" + "-"*60)
    logger.info("Training Output Gap Network")
    logger.info("-"*60)
    
    X_train_y = train_lagged[['output_gap_lag1', 'inflation_lag1', 
                               'interest_rate_lag1', 'interest_rate_lag2']].values
    y_train_y = train_lagged['output_gap'].values
    
    X_val_y = val_lagged[['output_gap_lag1', 'inflation_lag1',
                          'interest_rate_lag1', 'interest_rate_lag2']].values
    y_val_y = val_lagged['output_gap'].values
    
    # Use tuned config or default
    y_config = ann_config.copy()
    if tuned_params:
        y_config['learning_rate'] = tuned_params['output_gap']['learning_rate']
        y_config['batch_size'] = tuned_params['output_gap']['batch_size']
    
    network_y = train_ann_network(
        X_train_y, y_train_y,
        X_val_y, y_val_y,
        ann_config['hidden_units_y'],
        y_config,
        logger,
        'output_gap'
    )
    
    # Inflation network
    # Input: [y_t, y_{t-1}, π_{t-1}, π_{t-2}, i_{t-1}]
    logger.info("\n" + "-"*60)
    logger.info("Training Inflation Network")
    logger.info("-"*60)
    
    X_train_pi = train_lagged[['output_gap', 'output_gap_lag1', 'inflation_lag1',
                                'inflation_lag2', 'interest_rate_lag1']].values
    y_train_pi = train_lagged['inflation'].values
    
    X_val_pi = val_lagged[['output_gap', 'output_gap_lag1', 'inflation_lag1',
                           'interest_rate_lag1', 'interest_rate_lag2']].values
    y_val_pi = val_lagged['inflation'].values
    
    # Use tuned config or default
    pi_config = ann_config.copy()
    if tuned_params:
        pi_config['learning_rate'] = tuned_params['inflation']['learning_rate']
        pi_config['batch_size'] = tuned_params['inflation']['batch_size']
    
    network_pi = train_ann_network(
        X_train_pi, y_train_pi,
        X_val_pi, y_val_pi,
        ann_config['hidden_units_pi'],
        pi_config,
        logger,
        'inflation'
    )
    
    # Compute predictions on full dataset (for fit statistics)
    full_lagged = create_lagged_features(data, lags=2)
    
    X_full_y = full_lagged[['output_gap_lag1', 'inflation_lag1',
                            'interest_rate_lag1', 'interest_rate_lag2']].values
    X_full_pi = full_lagged[['output_gap', 'output_gap_lag1', 'inflation_lag1',
                             'inflation_lag2', 'interest_rate_lag1']].values
    
    network_y.eval()
    network_pi.eval()
    
    with torch.no_grad():
        fitted_y = network_y(torch.FloatTensor(X_full_y)).numpy()
        fitted_pi = network_pi(torch.FloatTensor(X_full_pi)).numpy()
    
    actual_y = full_lagged['output_gap'].values
    actual_pi = full_lagged['inflation'].values
    
    # Compute residuals
    residuals_y = actual_y - fitted_y
    residuals_pi = actual_pi - fitted_pi
    
    shock_std = {
        'output_gap': np.std(residuals_y),
        'inflation': np.std(residuals_pi)
    }
    
    # Fit statistics
    mse_y = mean_squared_error(actual_y, fitted_y)
    mse_pi = mean_squared_error(actual_pi, fitted_pi)
    mse_total = (mse_y + mse_pi) / 2
    
    # R² calculation
    ss_res_y = np.sum(residuals_y ** 2)
    ss_tot_y = np.sum((actual_y - np.mean(actual_y)) ** 2)
    r2_y = 1 - (ss_res_y / ss_tot_y)
    
    ss_res_pi = np.sum(residuals_pi ** 2)
    ss_tot_pi = np.sum((actual_pi - np.mean(actual_pi)) ** 2)
    r2_pi = 1 - (ss_res_pi / ss_tot_pi)
    
    logger.info("\n" + "-"*60)
    logger.info("FIT STATISTICS")
    logger.info("-"*60)
    logger.info(f"Output Gap MSE: {mse_y:.6f}")
    logger.info(f"Inflation MSE:  {mse_pi:.6f}")
    logger.info(f"Total MSE:      {mse_total:.6f}")
    logger.info(f"Output Gap R²:  {r2_y:.4f}")
    logger.info(f"Inflation R²:   {r2_pi:.4f}")
    
    return {
        'network_y': network_y,
        'network_pi': network_pi,
        'shock_std': shock_std,
        'fit_statistics': {
            'mse_output_gap': mse_y,
            'mse_inflation': mse_pi,
            'mse_total': mse_total,
            'r2_output_gap': r2_y,
            'r2_inflation': r2_pi
        },
        'residuals': {
            'output_gap': residuals_y,
            'inflation': residuals_pi
        },
        'fitted_values': {
            'output_gap': fitted_y,
            'inflation': fitted_pi
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Estimate economy models')
    parser.add_argument('--model', type=str, required=True, choices=['svar', 'ann', 'both'],
                       help='Model to estimate',default= 'both'
                       )
    parser.add_argument('--config', type=str, default='configs/hyperparameters.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results/checkpoints',
                       help='Output directory')
    parser.add_argument('--tune', action='store_true', help='Tune hyperparameters before estimation')
    parser.add_argument('--trials', type=int, default=20, help='Number of tuning trials')
    parser.add_argument('--epochs', type=int, default=500, help='Max epochs for tuning')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    os.makedirs('results/logs', exist_ok=True)
    logger = setup_logger(
        'economy_estimation',
        'results/logs/economy_estimation.log'
    )
    
    logger.info("="*60)
    logger.info("ECONOMY ESTIMATION")
    logger.info("="*60)
    
    # Load data
    logger.info("\nLoading data...")
    data_loader = DataLoader(
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    data = data_loader.get_data()
    # lagged_data = data_loader.get_lagged_data(lags=2) # No longer needed for SVAR here
    
    logger.info(f"Data loaded: {len(data)} observations")
    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Estimate models
    if args.model in ['svar', 'both']:
        # Pass raw data to estimate_svar so it can generate lags dynamically
        svar_results = estimate_svar(data, config, logger)
        
        # Save SVAR results
        svar_path = os.path.join(args.output_dir, 'svar_params.pkl')
        with open(svar_path, 'wb') as f:
            pickle.dump(svar_results, f)
        logger.info(f"\nSVAR parameters saved to: {svar_path}")
    
    if args.model in ['ann', 'both']:
        tuned_params = None
        if args.tune:
            tuned_params = tune_hyperparameters(data, config, logger, args.trials, args.epochs)
            
        ann_results = estimate_ann(data, config, logger, tuned_params)
        
        # Save ANN results
        ann_y_path = os.path.join(args.output_dir, 'ann_y_network.pth')
        ann_pi_path = os.path.join(args.output_dir, 'ann_pi_network.pth')
        ann_shock_path = os.path.join(args.output_dir, 'ann_shock_std.pkl')
        
        torch.save(ann_results['network_y'].state_dict(), ann_y_path)
        torch.save(ann_results['network_pi'].state_dict(), ann_pi_path)
        
        with open(ann_shock_path, 'wb') as f:
            pickle.dump(ann_results['shock_std'], f)
        
        logger.info(f"\nANN networks saved to:")
        logger.info(f"  Output gap:  {ann_y_path}")
        logger.info(f"  Inflation:   {ann_pi_path}")
        logger.info(f"  Shock std:   {ann_shock_path}")
    
    # Create comparison table (Table 2)
    if args.model == 'both':
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON (Table 2)")
        logger.info("="*60)
        
        comparison_df = pd.DataFrame({
            'Model': ['SVAR', 'ANN'],
            'MSE_Output_Gap': [
                svar_results['fit_statistics']['mse_output_gap'],
                ann_results['fit_statistics']['mse_output_gap']
            ],
            'MSE_Inflation': [
                svar_results['fit_statistics']['mse_inflation'],
                ann_results['fit_statistics']['mse_inflation']
            ],
            'MSE_Total': [
                svar_results['fit_statistics']['mse_total'],
                ann_results['fit_statistics']['mse_total']
            ]
        })
        
        logger.info("\n" + str(comparison_df))
        
        # Save to CSV
        table_dir = 'results/tables'
        os.makedirs(table_dir, exist_ok=True)
        comparison_df.to_csv(os.path.join(table_dir, 'table2_economy_mse.csv'), index=False)
        
        # Calculate improvement
        improvement = (svar_results['fit_statistics']['mse_total'] - 
                      ann_results['fit_statistics']['mse_total']) / \
                      svar_results['fit_statistics']['mse_total'] * 100
        
        logger.info(f"\nANN improvement over SVAR: {improvement:.1f}%")
    
    logger.info("\nEstimation complete!")


if __name__ == "__main__":
    main()