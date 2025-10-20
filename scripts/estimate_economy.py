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

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader
from src.environment.ann_economy import EconomyNetwork
from src.utils.logger import setup_logger


def estimate_svar(
    data: pd.DataFrame,
    config: dict,
    logger
) -> dict:
    """
    Estimate SVAR model with recursive structure (Equations 6-7).
    
    Args:
        data: DataFrame with lagged features
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        Dictionary with estimated parameters and statistics
    """
    logger.info("\n" + "="*60)
    logger.info("ESTIMATING SVAR ECONOMY")
    logger.info("="*60)
    
    # Prepare data
    # Output gap equation: y_t = C^y + a^y_{y,1} y_{t-1} + a^y_{π,1} π_{t-1} + 
    #                              a^y_{i,1} i_{t-1} + a^y_{i,2} i_{t-2} + ε^y_t
    
    X_y = data[['output_gap_lag1', 'inflation_lag1', 
                'interest_rate_lag1', 'interest_rate_lag2']].values
    y_y = data['output_gap'].values
    
    # Add constant
    X_y = add_constant(X_y)
    
    # OLS estimation
    model_y = OLS(y_y, X_y).fit()
    
    logger.info("\nOutput Gap Equation:")
    logger.info(model_y.summary())
    
    # Inflation equation: π_t = C^π + a^π_{y,0} y_t + a^π_{y,1} y_{t-1} + 
    #                            a^π_{π,1} π_{t-1} + a^π_{π,2} π_{t-2} + a^π_{i,1} i_{t-1} + ε^π_t
    
    X_pi = data[['output_gap', 'output_gap_lag1', 'inflation_lag1', 
                 'inflation_lag2', 'interest_rate_lag1']].values
    y_pi = data['inflation'].values
    
    X_pi = add_constant(X_pi)
    model_pi = OLS(y_pi, X_pi).fit()
    
    logger.info("\nInflation Equation:")
    logger.info(model_pi.summary())
    
    # Extract parameters
    params_y = {
        'const': model_y.params[0],
        'y_lag1': model_y.params[1],
        'pi_lag1': model_y.params[2],
        'i_lag1': model_y.params[3],
        'i_lag2': model_y.params[4]
    }
    
    params_pi = {
        'const': model_pi.params[0],
        'y_lag0': model_pi.params[1],
        'y_lag1': model_pi.params[2],
        'pi_lag1': model_pi.params[3],
        'pi_lag2': model_pi.params[4],
        'i_lag1': model_pi.params[5]
    }
    
    # Compute residuals (structural shocks)
    residuals_y = model_y.resid
    residuals_pi = model_pi.resid
    
    shock_std = {
        'output_gap': np.std(residuals_y),
        'inflation': np.std(residuals_pi)
    }
    
    # Compute fit statistics
    fitted_y = model_y.fittedvalues
    fitted_pi = model_pi.fittedvalues
    
    mse_y = mean_squared_error(y_y, fitted_y)
    mse_pi = mean_squared_error(y_pi, fitted_pi)
    mse_total = (mse_y + mse_pi) / 2
    
    logger.info("\n" + "-"*60)
    logger.info("FIT STATISTICS")
    logger.info("-"*60)
    logger.info(f"Output Gap MSE: {mse_y:.6f}")
    logger.info(f"Inflation MSE:  {mse_pi:.6f}")
    logger.info(f"Total MSE:      {mse_total:.6f}")
    logger.info(f"Output Gap R²:  {model_y.rsquared:.4f}")
    logger.info(f"Inflation R²:   {model_pi.rsquared:.4f}")
    
    return {
        'coefficients': {
            'output_gap': params_y,
            'inflation': params_pi
        },
        'shock_std': shock_std,
        'fit_statistics': {
            'mse_output_gap': mse_y,
            'mse_inflation': mse_pi,
            'mse_total': mse_total,
            'r2_output_gap': model_y.rsquared,
            'r2_inflation': model_pi.rsquared
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


def estimate_ann(
    data: pd.DataFrame,
    config: dict,
    logger
) -> dict:
    """
    Estimate ANN economy model (Equation 8).
    
    Args:
        data: Full dataset
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        Dictionary with trained networks and statistics
    """
    logger.info("\n" + "="*60)
    logger.info("ESTIMATING ANN ECONOMY")
    logger.info("="*60)
    
    ann_config = config['economy']['ann']
    
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
    
    network_y = train_ann_network(
        X_train_y, y_train_y,
        X_val_y, y_val_y,
        ann_config['hidden_units_y'],
        ann_config,
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
                           'inflation_lag2', 'interest_rate_lag1']].values
    y_val_pi = val_lagged['inflation'].values
    
    network_pi = train_ann_network(
        X_train_pi, y_train_pi,
        X_val_pi, y_val_pi,
        ann_config['hidden_units_pi'],
        ann_config,
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
    lagged_data = data_loader.get_lagged_data(lags=2)
    
    logger.info(f"Data loaded: {len(data)} observations")
    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Estimate models
    if args.model in ['svar', 'both']:
        svar_results = estimate_svar(lagged_data, config, logger)
        
        # Save SVAR results
        svar_path = os.path.join(args.output_dir, 'svar_params.pkl')
        with open(svar_path, 'wb') as f:
            pickle.dump(svar_results, f)
        logger.info(f"\nSVAR parameters saved to: {svar_path}")
    
    if args.model in ['ann', 'both']:
        ann_results = estimate_ann(data, config, logger)
        
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