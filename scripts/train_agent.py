#!/usr/bin/env python3
"""
Training script for DDPG agents.

Implements the full training procedure from Table 1 of the paper.

Usage:
    python scripts/train_agent.py --economy ann --policy linear --lags 0
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
import pickle

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.svar_economy import SVAREconomy
from src.environment.ann_economy import ANNEconomy, EconomyNetwork
from src.agents.ddpg_agent import DDPGAgent
from src.utils.logger import setup_logger, TrainingLogger
from src.utils.metrics import compute_steady_state_reward


def load_config(config_path: str = "configs/hyperparameters.yaml") -> dict:
    """Load hyperparameters from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_economy(
    economy_type: str,
    checkpoint_dir: str,
    config: dict,
    device: str = 'cpu'
):
    """
    Load estimated economy model.
    
    Args:
        economy_type: 'svar' or 'ann'
        checkpoint_dir: Directory with saved models
        config: Configuration dictionary
        device: Computation device
    
    Returns:
        Economy instance
    """
    if economy_type == 'svar':
        # Load SVAR parameters
        params_path = os.path.join(checkpoint_dir, 'svar_params.pkl')
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
        
        economy = SVAREconomy(
            params=params['coefficients'],
            shock_std=params['shock_std'],
            target_inflation=config['reward']['target_inflation'],
            target_output_gap=config['reward']['target_output_gap'],
            reward_weights={
                'inflation': config['reward']['weight_inflation'],
                'output_gap': config['reward']['weight_output_gap']
            },
            penalty_threshold=config['reward']['penalty_threshold'],
            penalty_multiplier=config['reward']['penalty_multiplier'],
            seed=config['seed']
        )
        
    else:  # ann
        # Load ANN networks
        network_y_path = os.path.join(checkpoint_dir, 'ann_y_network.pth')
        network_pi_path = os.path.join(checkpoint_dir, 'ann_pi_network.pth')
        
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
            # Use max lags for economy simulation if they differ
            lags = max(lags_y, lags_pi)
        else:
            hidden_units_y = config['economy']['ann']['hidden_units_y']
            hidden_units_pi = config['economy']['ann']['hidden_units_pi']
            lags = 2 # Default
            lags_y = 2
            lags_pi = 2
            
        # Calculate input dimensions based on lags
        input_dim_y = 3 * lags_y
        input_dim_pi = 1 + 3 * lags_pi
        
        network_y = EconomyNetwork(input_dim=input_dim_y, hidden_units=hidden_units_y)
        network_pi = EconomyNetwork(input_dim=input_dim_pi, hidden_units=hidden_units_pi)
        
        network_y.load_state_dict(torch.load(network_y_path, map_location=device))
        network_pi.load_state_dict(torch.load(network_pi_path, map_location=device))
        
        # Load shock standard deviations
        with open(os.path.join(checkpoint_dir, 'ann_shock_std.pkl'), 'rb') as f:
            shock_std = pickle.load(f)
        
        economy = ANNEconomy(
            network_y=network_y,
            network_pi=network_pi,
            shock_std=shock_std,
            target_inflation=config['reward']['target_inflation'],
            target_output_gap=config['reward']['target_output_gap'],
            reward_weights={
                'inflation': config['reward']['weight_inflation'],
                'output_gap': config['reward']['weight_output_gap']
            },
            penalty_threshold=config['reward']['penalty_threshold'],
            penalty_multiplier=config['reward']['penalty_multiplier'],
            seed=config['seed'],
            device=device,
            lags=lags
        )
    
    return economy


def train_agent(
    economy,
    config: dict,
    state_dim: int,
    critic_hidden: int,
    actor_hidden: int,
    linear_policy: bool,
    logger: TrainingLogger,
    device: str = 'cpu'
) -> DDPGAgent:
    """
    Train DDPG agent following Table 1 algorithm.
    
    Args:
        economy: Economic environment
        config: Configuration dictionary
        state_dim: Observation space dimension (2 or 4)
        critic_hidden: Number of critic hidden units
        actor_hidden: Number of actor hidden units (None for linear)
        linear_policy: Whether to use linear policy
        logger: Training logger
        device: Computation device
    
    Returns:
        Trained agent
    """
    rl_config = config['rl_training']
    
    # Initialize agent
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=1,
        critic_hidden=critic_hidden,
        actor_hidden=actor_hidden,
        linear_policy=linear_policy,
        lr_actor=rl_config['learning_rate_actor'],
        lr_critic=rl_config['learning_rate_critic'],
        gamma=rl_config['discount_factor'],
        tau=rl_config['tau'],
        buffer_size=rl_config['buffer_size'],
        batch_size=rl_config['batch_size'],
        noise_std=rl_config['noise_std'],
        noise_theta=rl_config.get('noise_theta', 0.15),
        noise_sigma=rl_config.get('noise_sigma', 0.2),
        gradient_clip=rl_config['gradient_clip'],
        device=device,
        seed=config['seed']
    )
    
    # Training parameters
    num_episodes = rl_config['num_episodes']
    max_steps = rl_config['max_steps_per_episode']
    stopping_tolerance = rl_config['stopping_tolerance']
    
    # Training loop
    best_reward = -np.inf
    best_agent_state = None
    saved_agents = []
    
    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Configuration: state_dim={state_dim}, critic_hidden={critic_hidden}, "
          f"actor_hidden={actor_hidden}, linear={linear_policy}")
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Reset environment and noise
        state = economy.reset()
        agent.reset_noise()
        
        # Episode tracking
        episode_reward = 0
        episode_steps = 0
        inflation_history = []
        output_gap_history = []
        interest_rate_history = []
        critic_losses = []
        actor_losses = []
        
        for step in range(max_steps):
            # Extract observation based on state_dim and lags
            # State: [y_t, ..., y_{t-p+1}, π_t, ..., π_{t-p+1}, i_{t-1}, ..., i_{t-p}]
            p = economy.lags
            
            if state_dim == 2:
                # No lags: [y_t, π_t]
                # y_t is at index 0
                # π_t is at index p
                obs = np.array([state[0], state[p]])
            else:  # state_dim == 4
                # One lag: [y_t, y_{t-1}, π_t, π_{t-1}]
                # y_t is at 0, y_{t-1} is at 1
                # π_t is at p, π_{t-1} is at p+1
                obs = np.array([state[0], state[1], state[p], state[p+1]])
            
            # Select action (step f)
            action = agent.select_action(obs, add_noise=True)
            
            # Execute action (step g)
            next_state, reward, done, info = economy.step(state, action)
            
            # Store transition (step h)
            if state_dim == 2:
                next_obs = np.array([next_state[0], next_state[p]])
            else:
                next_obs = np.array([next_state[0], next_state[1], 
                                    next_state[p], next_state[p+1]])
            
            agent.replay_buffer.add(obs, action, reward, next_obs, done)
            
            # Update agent (steps i-m)
            if agent.replay_buffer.is_ready(agent.batch_size):
                losses = agent.update()
                critic_losses.append(losses['critic_loss'])
                actor_losses.append(losses['actor_loss'])
            
            # Track metrics
            episode_reward += reward
            episode_steps += 1
            inflation_history.append(info['inflation'])
            output_gap_history.append(info['output_gap'])
            interest_rate_history.append(action)
            
            # Check stopping criteria
            terminal = economy.check_termination(
                info['inflation'],
                info['output_gap'],
                step,
                max_steps,
                stopping_tolerance
            )
            
            if terminal:
                break
            
            state = next_state
        
        # Log episode
        mean_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        mean_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        
        logger.log_episode(
            episode=episode,
            total_reward=episode_reward,
            episode_steps=episode_steps,
            inflation_history=inflation_history,
            output_gap_history=output_gap_history,
            interest_rate_history=interest_rate_history,
            critic_loss=mean_critic_loss,
            actor_loss=mean_actor_loss
        )
        
        # Save agent if it meets criteria
        avg_reward_per_step = episode_reward / episode_steps if episode_steps > 0 else -np.inf
        
        if (avg_reward_per_step > rl_config['selection_criteria']['min_episode_reward_per_step'] and
            rl_config['selection_criteria']['min_episode_steps'] < episode_steps < 
            rl_config['selection_criteria']['max_episode_steps']):
            
            saved_agents.append({
                'episode': episode,
                'episode_reward': episode_reward,
                'episode_steps': episode_steps,
                'avg_reward_per_step': avg_reward_per_step,
                'final_inflation': inflation_history[-1],
                'final_output_gap': output_gap_history[-1]
            })
            
            # Update best agent
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_agent_state = {
                    'actor_state_dict': agent.actor.state_dict().copy(),
                    'critic_state_dict': agent.critic.state_dict().copy()
                }
    
    # Select best agent based on steady-state reward
    print("\nSelecting best agent based on steady-state reward...")
    
    if len(saved_agents) > 0 and best_agent_state is not None:
        agent.actor.load_state_dict(best_agent_state['actor_state_dict'])
        agent.critic.load_state_dict(best_agent_state['critic_state_dict'])
    
    return agent, saved_agents


def main():
    parser = argparse.ArgumentParser(description='Train DDPG agent for monetary policy')
    parser.add_argument('--economy', type=str, required=True, choices=['svar', 'ann'],
                       help='Economy type')
    parser.add_argument('--policy', type=str, required=True, choices=['linear', 'nonlinear'],
                       help='Policy function type')
    parser.add_argument('--lags', type=int, required=True, choices=[0, 1],
                       help='Include lagged observations (0=no, 1=yes)')
    parser.add_argument('--config', type=str, default='configs/hyperparameters.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints',
                       help='Directory with economy checkpoints')
    parser.add_argument('--output_dir', type=str, default='results/checkpoints',
                       help='Output directory for trained agent')
    parser.add_argument('--log_dir', type=str, default='results/logs',
                       help='Log directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Setup logging
    experiment_name = f"ddpg_{args.economy}_{args.policy}_lags{args.lags}"
    logger_main = setup_logger(
        experiment_name,
        os.path.join(args.log_dir, f"{experiment_name}.log")
    )
    logger_main.info(f"Starting training: {experiment_name}")
    
    training_logger = TrainingLogger(args.log_dir, experiment_name)
    
    # Load economy
    logger_main.info("Loading economy model...")
    economy = load_economy(args.economy, args.checkpoint_dir, config, args.device)
    
    # Determine state dimension
    state_dim = 2 if args.lags == 0 else 4
    linear_policy = (args.policy == 'linear')
    
    # Search over critic nodes
    critic_nodes_list = config['rl_training']['critic_nodes_search']
    
    if linear_policy:
        actor_hidden = None
        actor_nodes_list = [None]
    else:
        actor_nodes_list = config['rl_training']['actor_nodes_nonlinear_search']
    
    best_agent = None
    best_steady_state_reward = -np.inf
    best_config = None
    
    # Grid search over network architectures
    for critic_nodes in critic_nodes_list:
        for actor_nodes in actor_nodes_list:
            logger_main.info(f"\nTraining with critic_nodes={critic_nodes}, actor_nodes={actor_nodes}")
            
            # Train agent
            agent, saved_agents = train_agent(
                economy=economy,
                config=config,
                state_dim=state_dim,
                critic_hidden=critic_nodes,
                actor_hidden=actor_nodes,
                linear_policy=linear_policy,
                logger=training_logger,
                device=args.device
            )
            
            # Evaluate steady-state reward
            from src.policies.rl_policy import RLPolicy
            rl_policy = RLPolicy(agent)
            
            ss_reward = compute_steady_state_reward(
                policy=rl_policy,
                economy=economy,
                n_simulations=100,
                n_steps=50,
                seed=config['seed']
            )
            
            logger_main.info(f"Steady-state reward: {ss_reward:.4f}")
            
            # Update best agent
            if ss_reward > best_steady_state_reward:
                best_steady_state_reward = ss_reward
                best_agent = agent
                best_config = {
                    'critic_nodes': critic_nodes,
                    'actor_nodes': actor_nodes,
                    'steady_state_reward': ss_reward
                }
                logger_main.info(f"New best agent found!")
    
    # Save best agent
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{experiment_name}.pth")
    best_agent.save(save_path)
    logger_main.info(f"\nBest agent saved to: {save_path}")
    logger_main.info(f"Best configuration: {best_config}")
    
    # Save policy parameters (if linear)
    params = best_agent.get_policy_parameters()
    params_path = os.path.join(args.output_dir, f"{experiment_name}_params.pkl")
    with open(params_path, 'wb') as f:
        pickle.dump({
            'parameters': params,
            'config': best_config,
            'experiment_name': experiment_name
        }, f)
    logger_main.info(f"Policy parameters saved to: {params_path}")
    
    if 'alpha_0' in params:
        logger_main.info("\nOptimized Policy Parameters:")
        for key, value in params.items():
            logger_main.info(f"  {key}: {value:.4f}")
    
    # Close logger
    training_logger.close()
    logger_main.info("Training complete!")


if __name__ == "__main__":
    main()