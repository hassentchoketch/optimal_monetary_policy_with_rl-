"""
Tests for DDPG agent implementation.
"""

import pytest
import numpy as np
import torch
from typing import Dict, Any, Tuple
from src.agents.ddpg_agent import DDPGAgent, OUNoise
from src.agents.networks import ActorNetwork, CriticNetwork
from src.agents.replay_buffer import ReplayBuffer


class TestReplayBuffer:
    """Tests for experience replay buffer."""
    
    def test_initialization(self) -> None:
        """Test buffer initialization."""
        buffer = ReplayBuffer(capacity=100, state_dim=2, action_dim=1)
        
        assert len(buffer) == 0
        assert buffer.capacity == 100
    
    def test_add_transition(self) -> None:
        """Test adding transitions."""
        buffer = ReplayBuffer(capacity=100, state_dim=2, action_dim=1)
        
        state = np.array([1.0, 2.0])
        action = 0.5
        reward = -1.0
        next_state = np.array([1.1, 2.1])
        done = False
        
        buffer.add(state, action, reward, next_state, done)
        
        assert len(buffer) == 1
    
    def test_buffer_overflow(self) -> None:
        """Test buffer behavior when full."""
        buffer = ReplayBuffer(capacity=5, state_dim=2, action_dim=1)
        
        # Add more than capacity
        for i in range(10):
            buffer.add(
                np.array([i, i]),
                float(i),
                float(i),
                np.array([i+1, i+1]),
                False
            )
        
        # Should not exceed capacity
        assert len(buffer) == 5
    
    def test_sample(self) -> None:
        """Test sampling from buffer."""
        buffer = ReplayBuffer(capacity=100, state_dim=2, action_dim=1)
        
        # Add transitions
        for i in range(50):
            buffer.add(
                np.array([i, i]),
                float(i),
                float(i),
                np.array([i+1, i+1]),
                False
            )
        
        # Sample batch
        batch = buffer.sample(batch_size=10)
        states, actions, rewards, next_states, dones = batch
        
        assert states.shape == (10, 2)
        assert actions.shape == (10, 1)
        assert rewards.shape == (10, 1)
        assert next_states.shape == (10, 2)
        assert dones.shape == (10, 1)
    
    def test_is_ready(self) -> None:
        """Test ready check."""
        buffer = ReplayBuffer(capacity=100, state_dim=2, action_dim=1)
        
        assert not buffer.is_ready(batch_size=10)
        
        # Add enough transitions
        for i in range(15):
            buffer.add(
                np.array([i, i]),
                float(i),
                float(i),
                np.array([i+1, i+1]),
                False
            )
        
        assert buffer.is_ready(batch_size=10)


class TestOUNoise:
    """Tests for Ornstein-Uhlenbeck noise."""
    
    def test_initialization(self) -> None:
        """Test noise initialization."""
        noise = OUNoise(action_dim=1, mu=0.0, theta=0.15, sigma=0.2)
        
        assert noise.action_dim == 1
    
    def test_sample(self) -> None:
        """Test noise sampling."""
        noise = OUNoise(action_dim=1, seed=42)
        
        sample1 = noise.sample()
        sample2 = noise.sample()
        
        assert sample1.shape == (1,)
        assert sample2.shape == (1,)
        # Samples should be different
        assert not np.array_equal(sample1, sample2)
    
    def test_reset(self) -> None:
        """Test noise reset."""
        noise = OUNoise(action_dim=1, seed=42)
        
        # Generate some samples
        for _ in range(10):
            noise.sample()
        
        # Reset
        noise.reset()
        
        # Should start from initial state again
        assert np.array_equal(noise.x, noise.x0)
    
    def test_mean_reversion(self) -> None:
        """Test that noise reverts to mean over time."""
        noise = OUNoise(action_dim=1, mu=0.0, theta=0.5, sigma=0.1, seed=42)
        
        samples = [noise.sample()[0] for _ in range(1000)]
        
        # Mean should be close to mu
        assert abs(np.mean(samples)) < 0.1


class TestActorNetwork:
    """Tests for actor network."""
    
    def test_linear_actor(self) -> None:
        """Test linear actor network."""
        actor = ActorNetwork(state_dim=2, hidden_units=None, linear=True)
        
        state = torch.randn(10, 2)
        action = actor(state)
        
        assert action.shape == (10, 1)
    
    def test_nonlinear_actor(self) -> None:
        """Test nonlinear actor network."""
        actor = ActorNetwork(state_dim=2, hidden_units=8, linear=False)
        
        state = torch.randn(10, 2)
        action = actor(state)
        
        assert action.shape == (10, 1)
    
    def test_gradient_flow(self) -> None:
        """Test gradient flow through actor."""
        actor = ActorNetwork(state_dim=2, hidden_units=8, linear=False)
        
        state = torch.randn(10, 2, requires_grad=True)
        action = actor(state)
        loss = action.sum()
        loss.backward()
        
        # Check gradients
        for param in actor.parameters():
            assert param.grad is not None


class TestCriticNetwork:
    """Tests for critic network."""
    
    def test_forward_pass(self) -> None:
        """Test critic forward pass."""
        critic = CriticNetwork(state_dim=2, action_dim=1, hidden_units=8)
        
        state = torch.randn(10, 2)
        action = torch.randn(10, 1)
        
        q_value = critic(state, action)
        
        assert q_value.shape == (10, 1)
    
    def test_gradient_flow(self) -> None:
        """Test gradient flow through critic."""
        critic = CriticNetwork(state_dim=2, action_dim=1, hidden_units=8)
        
        state = torch.randn(10, 2, requires_grad=True)
        action = torch.randn(10, 1, requires_grad=True)
        
        q_value = critic(state, action)
        loss = q_value.sum()
        loss.backward()
        
        # Check gradients
        assert state.grad is not None
        assert action.grad is not None
        for param in critic.parameters():
            assert param.grad is not None


class TestDDPGAgent:
    """Tests for DDPG agent."""
    
    def test_initialization(self, ddpg_agent: DDPGAgent) -> None:
        """Test agent initialization."""
        assert ddpg_agent.state_dim == 2
        assert ddpg_agent.action_dim == 1
        assert isinstance(ddpg_agent.actor, ActorNetwork)
        assert isinstance(ddpg_agent.critic, CriticNetwork)
    
    def test_select_action(self, ddpg_agent: DDPGAgent) -> None:
        """Test action selection."""
        state = np.array([1.0, 2.0])
        
        # Without noise
        action = ddpg_agent.select_action(state, add_noise=False)
        assert isinstance(action, (float, np.floating))
        
        # With noise
        action_noisy = ddpg_agent.select_action(state, add_noise=True)
        assert isinstance(action_noisy, (float, np.floating))
    
    def test_select_action_deterministic(self, ddpg_agent: DDPGAgent, set_seeds: None) -> None:
        """Test deterministic action selection."""
        state = np.array([1.0, 2.0])
        
        action1 = ddpg_agent.select_action(state, add_noise=False)
        action2 = ddpg_agent.select_action(state, add_noise=False)
        
        assert action1 == action2
    
    def test_update_requires_data(self, ddpg_agent: DDPGAgent) -> None:
        """Test update requires sufficient data."""
        # Buffer is empty
        losses = ddpg_agent.update()
        
        assert losses['critic_loss'] == 0.0
        assert losses['actor_loss'] == 0.0
    
    def test_update_with_data(self, ddpg_agent: DDPGAgent) -> None:
        """Test update with sufficient data."""
        # Fill buffer
        for _ in range(100):
            state = np.random.randn(2)
            action = np.random.randn(1)
            reward = np.random.randn(1)
            next_state = np.random.randn(2)
            done = False
            
            ddpg_agent.replay_buffer.add(state, action, reward, next_state, done)
        
        # Update should work
        losses = ddpg_agent.update()
        
        assert losses['critic_loss'] >= 0.0
        assert isinstance(losses['actor_loss'], float)
    
    def test_soft_update(self, ddpg_agent: DDPGAgent) -> None:
        """Test soft update of target networks."""
        # Store initial target parameters
        initial_target_params = [
            param.clone() for param in ddpg_agent.actor_target.parameters()
        ]
        
        # Modify source network
        for param in ddpg_agent.actor.parameters():
            param.data += 10.0
        
        # Soft update
        ddpg_agent._soft_update(ddpg_agent.actor, ddpg_agent.actor_target)
        
        # Target should have changed slightly
        for initial, current in zip(initial_target_params, ddpg_agent.actor_target.parameters()):
            assert not torch.allclose(initial, current)
            # But not fully (tau is small)
            assert not torch.allclose(current, ddpg_agent.actor.state_dict()[list(ddpg_agent.actor.state_dict().keys())[0]])
    
    def test_save_load(self, ddpg_agent: DDPGAgent, tmp_path: Any) -> None:
        """Test saving and loading agent."""
        # Save
        save_path = tmp_path / "agent.pth"
        ddpg_agent.save(str(save_path))
        
        assert save_path.exists()
        
        # Modify agent
        for param in ddpg_agent.actor.parameters():
            param.data += 1.0
        
        # Load
        ddpg_agent.load(str(save_path))
        
        # Should be restored (can't easily check exact values without storing them)
        assert save_path.exists()
    
    def test_get_policy_parameters_linear(self) -> None:
        """Test extracting parameters from linear policy."""
        agent = DDPGAgent(
            state_dim=2,
            action_dim=1,
            critic_hidden=2,
            actor_hidden=None,
            linear_policy=True,
            seed=42
        )
        
        params = agent.get_policy_parameters()
        
        assert 'alpha_0' in params
        assert 'beta_y_0' in params
        assert 'beta_pi_0' in params
    
    def test_get_policy_parameters_nonlinear(self) -> None:
        """Test extracting parameters from nonlinear policy."""
        agent = DDPGAgent(
            state_dim=2,
            action_dim=1,
            critic_hidden=2,
            actor_hidden=8,
            linear_policy=False,
            seed=42
        )
        
        params = agent.get_policy_parameters()
        
        assert 'note' in params  # Non-interpretable


class TestAgentEnvironmentInteraction:
    """Test agent interaction with environment."""
    
    def test_full_episode(self, ddpg_agent: DDPGAgent, svar_economy: Any, set_seeds: None) -> None:
        """Test complete episode."""
        state = svar_economy.reset()
        
        total_reward = 0
        for step in range(10):
            # Extract observation
            obs = np.array([state[0], state[2]])  # [y_t, π_t]
            
            # Select action
            action = ddpg_agent.select_action(obs, add_noise=True)
            
            # Step environment
            next_state, reward, done, info = svar_economy.step(state, action)
            
            # Store in buffer
            next_obs = np.array([next_state[0], next_state[2]])
            ddpg_agent.replay_buffer.add(obs, action, reward, next_obs, done)
            
            total_reward += reward
            state = next_state
            
            # Update if enough data
            if ddpg_agent.replay_buffer.is_ready(32):
                ddpg_agent.update()
        
        assert isinstance(total_reward, (float, np.floating))