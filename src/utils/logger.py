"""
Logging utilities for training and evaluation.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
import json
import csv
import numpy as np


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """
    Logger for RL training metrics.
    
    Tracks and saves training progress to CSV and JSON files.
    """
    def __init__(
        self,
        log_dir: str,
        experiment_name: str
    ):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # File paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(
            log_dir, 
            f"{experiment_name}_{timestamp}.csv"
        )
        self.json_path = os.path.join(
            log_dir, 
            f"{experiment_name}_{timestamp}.json"
        )
        
        # Storage
        self.episode_logs = []
        self.update_logs = []
        
        # CSV file setup
        self.csv_file = None
        self.csv_writer = None
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        self.csv_file = open(self.csv_path, 'w', newline='')
        fieldnames = [
            'episode', 'total_reward', 'episode_steps', 
            'mean_inflation', 'mean_output_gap', 'mean_interest_rate',
            'final_inflation', 'final_output_gap',
            'critic_loss', 'actor_loss'
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()
    
    def log_episode(
        self,
        episode: int,
        total_reward: float,
        episode_steps: int,
        inflation_history: List[float],
        output_gap_history: List[float],
        interest_rate_history: List[float],
        critic_loss: float = 0.0,
        actor_loss: float = 0.0
    ):
        """
        Log episode metrics.
        
        Args:
            episode: Episode number
            total_reward: Total reward for episode
            episode_steps: Number of steps in episode
            inflation_history: Inflation values during episode
            output_gap_history: Output gap values during episode
            interest_rate_history: Interest rate values during episode
            critic_loss: Mean critic loss
            actor_loss: Mean actor loss
        """
        log_entry = {
            'episode': episode,
            'total_reward': total_reward,
            'episode_steps': episode_steps,
            'mean_inflation': np.mean(inflation_history),
            'mean_output_gap': np.mean(output_gap_history),
            'mean_interest_rate': np.mean(interest_rate_history),
            'final_inflation': inflation_history[-1] if inflation_history else 0,
            'final_output_gap': output_gap_history[-1] if output_gap_history else 0,
            'critic_loss': critic_loss,
            'actor_loss': actor_loss
        }
        
        self.episode_logs.append(log_entry)
        
        # Write to CSV
        self.csv_writer.writerow(log_entry)
        self.csv_file.flush()
    
    def log_update(
        self,
        update_step: int,
        critic_loss: float,
        actor_loss: float
    ):
        """
        Log update step metrics.
        
        Args:
            update_step: Update step number
            critic_loss: Critic loss
            actor_loss: Actor loss
        """
        self.update_logs.append({
            'update_step': update_step,
            'critic_loss': critic_loss,
            'actor_loss': actor_loss
        })
    
    def save_summary(
        self,
        final_metrics: Dict,
        best_agent_info: Optional[Dict] = None
    ):
        """
        Save training summary to JSON.
        
        Args:
            final_metrics: Final evaluation metrics
            best_agent_info: Information about best agent
        """
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'total_episodes': len(self.episode_logs),
            'total_updates': len(self.update_logs),
            'final_metrics': final_metrics,
            'best_agent': best_agent_info,
            'episode_logs': self.episode_logs[-10:],  # Last 10 episodes
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def close(self):
        """Close CSV file."""
        if self.csv_file:
            self.csv_file.close()
    
    def __del__(self):
        """Ensure CSV file is closed."""
        self.close()