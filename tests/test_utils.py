"""
Tests for utility functions.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

from src.utils.visualization import create_results_table
from src.utils.logger import setup_logger, TrainingLogger


class TestLogger:
    """Tests for logging utilities."""
    
    def test_setup_logger(self, tmp_path):
        """Test logger setup."""
        log_file = tmp_path / "test.log"
        logger = setup_logger('test_logger', str(log_file))
        
        assert logger.name == 'test_logger'
        logger.info("Test message")
        
        # Check log file created
        assert log_file.exists()
    
    def test_training_logger(self, tmp_path):
        """Test training logger."""
        logger = TrainingLogger(
            log_dir=str(tmp_path),
            experiment_name='test_experiment'
        )
        
        # Log episode
        logger.log_episode(
            episode=0,
            total_reward=-10.5,
            episode_steps=5,
            inflation_history=[2.5, 2.3, 2.1, 2.0, 2.0],
            output_gap_history=[1.0, 0.8, 0.5, 0.2, 0.1],
            interest_rate_history=[3.5, 3.3, 3.0, 2.8, 2.5],
            critic_loss=0.05,
            actor_loss=-1.2
        )
        
        # Check CSV created
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 1
        
        # Check episode logged
        assert len(logger.episode_logs) == 1
        
        logger.close()
    
    def test_training_logger_save_summary(self, tmp_path):
        """Test saving training summary."""
        logger = TrainingLogger(
            log_dir=str(tmp_path),
            experiment_name='test_experiment'
        )
        
        # Log some episodes
        for i in range(3):
            logger.log_episode(
                episode=i,
                total_reward=-10.0,
                episode_steps=5,
                inflation_history=[2.0] * 5,
                output_gap_history=[0.0] * 5,
                interest_rate_history=[3.0] * 5
            )
        
        # Save summary
        logger.save_summary(
            final_metrics={'total_loss': 1.5},
            best_agent_info={'episode': 2, 'reward': -8.0}
        )
        
        # Check JSON created
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1
        
        logger.close()


class TestVisualization:
    """Tests for visualization utilities."""
    
    def test_create_results_table_csv(self, tmp_path):
        """Test creating results table in CSV format."""
        metrics = {
            'Policy1': {
                'mse_inflation': 1.0,
                'mse_output_gap': 2.0,
                'total_loss': 1.5
            },
            'Policy2': {
                'mse_inflation': 0.8,
                'mse_output_gap': 1.5,
                'total_loss': 1.15
            }
        }
        
        output_path = tmp_path / "results.csv"
        
        create_results_table(
            metrics_dict=metrics,
            save_path=str(output_path),
            format='csv'
        )
        
        # Check file created
        assert output_path.exists()
        
        # Check content
        df = pd.read_csv(output_path, index_col=0)
        assert 'Policy1' in df.index
        assert 'Policy2' in df.index
    
    def test_create_results_table_latex(self, tmp_path):
        """Test creating results table in LaTeX format."""
        metrics = {
            'Policy1': {
                'mse_inflation': 1.0,
                'mse_output_gap': 2.0,
                'total_loss': 1.5
            }
        }
        
        output_path = tmp_path / "results.tex"
        
        create_results_table(
            metrics_dict=metrics,
            save_path=str(output_path),
            format='latex'
        )
        
        # Check file created
        assert output_path.exists()
        
        # Check it's LaTeX
        with open(output_path, 'r') as f:
            content = f.read()
            assert '\\begin{tabular}' in content or '\\begin{table}' in content