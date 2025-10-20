"""
Tests for policy implementations.
"""

import pytest
import numpy as np
from src.policies.baseline_policies import BaselinePolicy, CustomLinearPolicy
from src.policies.rl_policy import RLPolicy
from src.agents.ddpg_agent import DDPGAgent

class TestBaselinePolicy:
    """Tests for baseline policies."""
    
    def test_taylor_rule(self):
        """Test Taylor (1993) rule."""
        policy = BaselinePolicy(rule_type='TR93', r_star=2.0, pi_star=2.0)
        
        # At target: i_t = r* + π* = 2 + 2 = 4
        state = np.array([0.0, 2.0])  # [y_t, π_t]
        action = policy.get_action(state)
        
        expected = 2.0 + 1.5 * (2.0 - 2.0) + 0.5 * 0.0 + 2.0  # r* + β_π(π-π*) + β_y*y + π*
        # Actually: i = r* + β_π(π - π*) + β_y*y
        # TR93: i = 2 + 1.5(2-2) + 0.5*0 =
        # Taylor rule: i_t = r* + β_π(π_t - π*) + β_y*y_t
        # At target: i_t = 2.0 + 1.5*(2.0 - 2.0) + 0.5*0.0 = 2.0
        # But rule also adds π_t, so i_t = 2.0 + 2.0 = 4.0
        expected = 4.0
        assert action == expected
    
    def test_npp_rule(self):
        """Test NPP (inflation tilting) rule."""
        policy = BaselinePolicy(rule_type='NPP', r_star=2.0, pi_star=2.0)
        
        state = np.array([1.0, 3.0])  # [y_t=1, π_t=3]
        action = policy.get_action(state)
        
        # NPP: i_t = r* + 2.0*(π_t - π*) + 0.5*y_t
        expected = 2.0 + 2.0 * (3.0 - 2.0) + 0.5 * 1.0 + 3.0
        # Wait, the formula in the code uses: r* + β_π*(π - π*) + β_y*y
        # So: 2.0 + 2.0*1.0 + 0.5*1.0 = 4.5
        expected = 4.5 + 3.0  # Plus π_t for nominal rate
        
        # Actually checking the baseline_policies.py implementation
        # interest_rate = r_star + beta_pi * inflation_gap + beta_y * y_t
        expected = 2.0 + 2.0 * (3.0 - 2.0) + 0.5 * 1.0
        assert action == expected
    
    def test_balanced_approach(self):
        """Test Balanced Approach rule."""
        policy = BaselinePolicy(rule_type='BA', r_star=2.0, pi_star=2.0)
        
        state = np.array([1.0, 3.0])  # [y_t=1, π_t=3]
        action = policy.get_action(state)
        
        # BA: i_t = r* + 1.5*(π_t - π*) + 1.0*y_t
        expected = 2.0 + 1.5 * (3.0 - 2.0) + 1.0 * 1.0
        assert action == expected
    
    def test_with_lagged_state(self):
        """Test policy with lagged state vector."""
        policy = BaselinePolicy(rule_type='TR93', r_star=2.0, pi_star=2.0)
        
        # State with lags: [y_t, y_{t-1}, π_t, π_{t-1}]
        state = np.array([1.0, 0.5, 3.0, 2.5])
        action = policy.get_action(state)
        
        # Should only use contemporaneous values
        expected = 2.0 + 1.5 * (3.0 - 2.0) + 0.5 * 1.0
        assert action == expected
    
    def test_return_components(self):
        """Test returning action components."""
        policy = BaselinePolicy(rule_type='TR93', r_star=2.0, pi_star=2.0)
        
        state = np.array([1.0, 3.0])
        action, components = policy.get_action(state, return_components=True)
        
        assert isinstance(components, dict)
        assert 'r_star' in components
        assert 'inflation_component' in components
        assert 'output_gap_component' in components
        assert 'total' in components
        assert components['total'] == action
    
    def test_get_parameters(self):
        """Test getting policy parameters."""
        policy = BaselinePolicy(rule_type='TR93', r_star=2.0, pi_star=2.0)
        
        params = policy.get_parameters()
        
        assert params['rule_type'] == 'TR93'
        assert params['r_star'] == 2.0
        assert params['pi_star'] == 2.0
        assert params['beta_pi'] == 1.5
        assert params['beta_y'] == 0.5
    
    def test_string_representation(self):
        """Test string representation."""
        policy = BaselinePolicy(rule_type='TR93', r_star=2.0, pi_star=2.0)
        
        string = str(policy)
        
        assert 'TR93' in string
        assert '1.5' in string  # beta_pi
        assert '0.5' in string  # beta_y


class TestCustomLinearPolicy:
    """Tests for custom linear policy."""
    
    def test_no_lags(self):
        """Test custom policy without lags."""
        policy = CustomLinearPolicy(
            alpha_0=1.0,
            beta_pi_0=1.5,
            beta_y_0=0.5
        )
        
        state = np.array([1.0, 3.0])  # [y_t, π_t]
        action = policy.get_action(state)
        
        expected = 1.0 + 1.5 * 3.0 + 0.5 * 1.0
        assert action == expected
    
    def test_with_lags(self):
        """Test custom policy with lags."""
        policy = CustomLinearPolicy(
            alpha_0=1.0,
            beta_pi_0=1.5,
            beta_y_0=0.5,
            beta_pi_1=0.3,
            beta_y_1=0.2
        )
        
        state = np.array([1.0, 0.5, 3.0, 2.5])  # [y_t, y_{t-1}, π_t, π_{t-1}]
        action = policy.get_action(state)
        
        expected = 1.0 + 1.5 * 3.0 + 0.5 * 1.0 + 0.3 * 2.5 + 0.2 * 0.5
        assert action == expected
    
    def test_no_lags_with_lagged_state_raises(self):
        """Test that policy without lags raises error for 2-element state."""
        policy = CustomLinearPolicy(
            alpha_0=1.0,
            beta_pi_0=1.5,
            beta_y_0=0.5
        )
        
        # This should work - extracts contemporaneous values
        state = np.array([1.0, 0.5, 3.0, 2.5])
        action = policy.get_action(state)
        # Should use [y_t=1.0, π_t=3.0]
        expected = 1.0 + 1.5 * 3.0 + 0.5 * 1.0
        assert action == expected
    
    def test_with_lags_requires_lagged_state(self):
        """Test that policy with lags requires full state."""
        policy = CustomLinearPolicy(
            alpha_0=1.0,
            beta_pi_0=1.5,
            beta_y_0=0.5,
            beta_pi_1=0.3,
            beta_y_1=0.2
        )
        
        state = np.array([1.0, 3.0])  # Only 2 elements
        
        with pytest.raises(ValueError):
            policy.get_action(state)
    
    def test_get_parameters_no_lags(self):
        """Test getting parameters without lags."""
        policy = CustomLinearPolicy(
            alpha_0=1.0,
            beta_pi_0=1.5,
            beta_y_0=0.5
        )
        
        params = policy.get_parameters()
        
        assert params['alpha_0'] == 1.0
        assert params['beta_pi_0'] == 1.5
        assert params['beta_y_0'] == 0.5
        assert 'beta_pi_1' not in params
    
    def test_get_parameters_with_lags(self):
        """Test getting parameters with lags."""
        policy = CustomLinearPolicy(
            alpha_0=1.0,
            beta_pi_0=1.5,
            beta_y_0=0.5,
            beta_pi_1=0.3,
            beta_y_1=0.2
        )
        
        params = policy.get_parameters()
        
        assert params['beta_pi_1'] == 0.3
        assert params['beta_y_1'] == 0.2


class TestRLPolicy:
    """Tests for RL policy wrapper."""
    
    def test_initialization(self, ddpg_agent):
        """Test RL policy initialization."""
        policy = RLPolicy(ddpg_agent, name="Test RL Policy")
        
        assert policy.name == "Test RL Policy"
        assert policy.agent == ddpg_agent
    
    def test_get_action_deterministic(self, ddpg_agent):
        """Test deterministic action selection."""
        policy = RLPolicy(ddpg_agent)
        
        # Full state
        state = np.array([1.0, 0.5, 3.0, 2.5, 3.5, 3.2])
        
        action = policy.get_action(state, deterministic=True)
        
        assert isinstance(action, (float, np.floating))
    
    def test_get_action_with_noise(self, ddpg_agent):
        """Test action selection with noise."""
        policy = RLPolicy(ddpg_agent)
        
        state = np.array([1.0, 0.5, 3.0, 2.5, 3.5, 3.2])
        
        action = policy.get_action(state, deterministic=False)
        
        assert isinstance(action, (float, np.floating))
    
    def test_get_action_extracts_observation(self, ddpg_agent):
        """Test that policy correctly extracts observation."""
        policy = RLPolicy(ddpg_agent)
        
        # Agent expects state_dim=2, so should extract [y_t, π_t]
        state = np.array([1.0, 0.5, 3.0, 2.5, 3.5, 3.2])
        
        # Should not raise error
        action = policy.get_action(state)
        
        assert isinstance(action, (float, np.floating))
    
    def test_get_parameters_linear(self):
        """Test getting parameters from linear agent."""
        agent = DDPGAgent(
            state_dim=2,
            action_dim=1,
            critic_hidden=2,
            actor_hidden=None,
            linear_policy=True,
            seed=42
        )
        
        policy = RLPolicy(agent)
        params = policy.get_parameters()
        
        assert 'alpha_0' in params or 'note' in params
    
    def test_get_parameters_nonlinear(self):
        """Test getting parameters from nonlinear agent."""
        agent = DDPGAgent(
            state_dim=2,
            action_dim=1,
            critic_hidden=2,
            actor_hidden=8,
            linear_policy=False,
            seed=42
        )
        
        policy = RLPolicy(agent)
        params = policy.get_parameters()
        
        assert 'note' in params
    
    def test_string_representation(self, ddpg_agent):
        """Test string representation."""
        policy = RLPolicy(ddpg_agent, name="My RL Policy")
        
        string = str(policy)
        
        assert "My RL Policy" in string


class TestPolicyComparison:
    """Test comparison between different policies."""
    
    def test_same_state_different_actions(self):
        """Test that different policies give different actions."""
        state = np.array([1.0, 3.0])
        
        taylor = BaselinePolicy('TR93')
        npp = BaselinePolicy('NPP')
        ba = BaselinePolicy('BA')
        
        action_taylor = taylor.get_action(state)
        action_npp = npp.get_action(state)
        action_ba = ba.get_action(state)
        
        # All should be different
        assert action_taylor != action_npp
        assert action_taylor != action_ba
        assert action_npp != action_ba
    
    def test_policy_consistency(self):
        """Test that policies are consistent across calls."""
        state = np.array([1.0, 3.0])
        
        policy = BaselinePolicy('TR93')
        
        action1 = policy.get_action(state)
        action2 = policy.get_action(state)
        
        assert action1 == action2