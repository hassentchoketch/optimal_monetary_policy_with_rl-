"""
Integration tests for complete workflows.
"""

import pytest
import numpy as np
import torch
from src.environment.svar_economy import SVAREconomy
from src.agents.ddpg_agent import DDPGAgent
from src.policies.baseline_policies import BaselinePolicy
from src.utils.metrics import compute_metrics, compare_policies


class TestTrainingWorkflow:
    """Test complete training workflow."""
    
    def test_train_one_episode(self, svar_economy, seed):
        """Test training for one complete episode."""
        # Create agent
        agent = DDPGAgent(
            state_dim=2,
            action_dim=1,
            critic_hidden=2,
            actor_hidden=None,
            linear_policy=True,
            buffer_size=1000,
            batch_size=32,
            seed=seed
        )
        
        # Reset environment
        state = svar_economy.reset()
        
        # Run episode
        for step in range(10):
            # Get observation
            obs = np.array([state[0], state[2]])
            
            # Select action
            action = agent.select_action(obs, add_noise=True)
            
            # Step environment
            next_state, reward, done, info = svar_economy.step(state, action)
            
            # Store experience
            next_obs = np.array([next_state[0], next_state[2]])
            agent.replay_buffer.add(obs, action, reward, next_obs, done)
            
            # Update (if enough data)
            if agent.replay_buffer.is_ready(32):
                losses = agent.update()
                assert 'critic_loss' in losses
                assert 'actor_loss' in losses
            
            state = next_state
        
        # Check that we completed episode
        assert step == 9


class TestEvaluationWorkflow:
    """Test complete evaluation workflow."""
    
    def test_policy_comparison(self, svar_economy):
        """Test comparing multiple policies."""
        # Create policies
        taylor = BaselinePolicy('TR93')
        npp = BaselinePolicy('NPP')
        
        # Simulate both policies
        n_steps = 50
        results = {}
        
        for name, policy in [('Taylor', taylor), ('NPP', npp)]:
            state = svar_economy.reset()
            
            inflation_series = []
            output_gap_series = []
            interest_rate_series = []
            
            for _ in range(n_steps):
                obs = np.array([state[0], state[2]])
                action = policy.get_action(obs)
                
                state, reward, done, info = svar_economy.step(state, action)
                
                inflation_series.append(info['inflation'])
                output_gap_series.append(info['output_gap'])
                interest_rate_series.append(action)
            
            # Compute metrics
            metrics = compute_metrics(
                np.array(inflation_series),
                np.array(output_gap_series),
                np.array(interest_rate_series)
            )
            
            results[name] = metrics
        
        # Compare
        comparison = compare_policies(results['Taylor'], results['NPP'])
        
        assert 'loss_improvement_pct' in comparison


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_estimate_train_evaluate(self, svar_params, seed):
        """Test complete pipeline: estimate → train → evaluate."""
        # 1. Create economy (simulating estimation)
        economy = SVAREconomy(
            params=svar_params['coefficients'],
            shock_std=svar_params['shock_std'],
            seed=seed
        )
        
        # 2. Train agent
        agent = DDPGAgent(
            state_dim=2,
            action_dim=1,
            critic_hidden=2,
            actor_hidden=None,
            linear_policy=True,
            buffer_size=500,
            batch_size=32,
            seed=seed
        )
        
        # Quick training (2 episodes)
        for episode in range(2):
            state = economy.reset()
            
            for step in range(10):
                obs = np.array([state[0], state[2]])
                action = agent.select_action(obs, add_noise=True)
                next_state, reward, done, info = economy.step(state, action)
                next_obs = np.array([next_state[0], next_state[2]])
                
                agent.replay_buffer.add(obs, action, reward, next_obs, done)
                
                if agent.replay_buffer.is_ready(32):
                    agent.update()
                
                state = next_state
        
        # 3. Evaluate
        state = economy.reset()
        total_reward = 0
        state = economy.reset()
        total_reward = 0
        
        for step in range(20):
            obs = np.array([state[0], state[2]])
            action = agent.select_action(obs, add_noise=False)
            state, reward, done, info = economy.step(state, action)
            total_reward += reward
        
        # Should complete without errors
        assert isinstance(total_reward, (float, np.floating))
        
        # 4. Extract policy parameters
        params = agent.get_policy_parameters()
        assert 'alpha_0' in params
        assert 'beta_pi_0' in params
        assert 'beta_y_0' in params


class TestSaveLoadWorkflow:
    """Test save/load workflow."""
    
    def test_save_load_agent(self, ddpg_agent, svar_economy, tmp_path, seed):
        """Test saving and loading trained agent."""
        # Train briefly
        state = svar_economy.reset()
        for _ in range(50):
            obs = np.array([state[0], state[2]])
            action = ddpg_agent.select_action(obs, add_noise=True)
            next_state, reward, done, info = svar_economy.step(state, action)
            next_obs = np.array([next_state[0], next_state[2]])
            ddpg_agent.replay_buffer.add(obs, action, reward, next_obs, done)
            
            if ddpg_agent.replay_buffer.is_ready(32):
                ddpg_agent.update()
            
            state = next_state
        
        # Get action before saving
        test_state = np.array([1.0, 2.5])
        action_before = ddpg_agent.select_action(test_state, add_noise=False)
        
        # Save
        save_path = tmp_path / "agent.pth"
        ddpg_agent.save(str(save_path))
        
        # Create new agent and load
        new_agent = DDPGAgent(
            state_dim=2,
            action_dim=1,
            critic_hidden=2,
            actor_hidden=None,
            linear_policy=True,
            seed=seed
        )
        new_agent.load(str(save_path))
        
        # Get action after loading
        action_after = new_agent.select_action(test_state, add_noise=False)
        
        # Should be the same
        np.testing.assert_almost_equal(action_before, action_after)


class TestMultipleEconomies:
    """Test workflows with multiple economy types."""
    
    def test_same_policy_different_economies(self, svar_economy, ann_economy):
        """Test applying same policy to different economies."""
        policy = BaselinePolicy('TR93')
        
        results = {}
        
        for name, economy in [('SVAR', svar_economy), ('ANN', ann_economy)]:
            state = economy.reset()
            
            total_reward = 0
            for _ in range(20):
                obs = np.array([state[0], state[2]])
                action = policy.get_action(obs)
                state, reward, done, info = economy.step(state, action)
                total_reward += reward
            
            results[name] = total_reward
        
        # Both should complete
        assert isinstance(results['SVAR'], (float, np.floating))
        assert isinstance(results['ANN'], (float, np.floating))
        
        # Results will differ due to different dynamics
        # Just check they're reasonable
        assert -1000 < results['SVAR'] < 0
        assert -1000 < results['ANN'] < 0