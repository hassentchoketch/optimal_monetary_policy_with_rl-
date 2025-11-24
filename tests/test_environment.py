"""
Tests for economic environment implementations.
"""

import pytest
import numpy as np
import torch
from typing import Dict, Any, Tuple
from src.environment.base_economy import BaseEconomy
from src.environment.svar_economy import SVAREconomy
from src.environment.ann_economy import ANNEconomy, EconomyNetwork


class TestBaseEconomy:
    """Tests for BaseEconomy abstract class."""
    
    def test_compute_reward_at_target(self) -> None:
        """Test reward when at target values."""
        # Create concrete implementation for testing
        class DummyEconomy(BaseEconomy):
            def step(self, state, action):
                pass
            def reset(self, initial_state=None):
                pass
        
        economy = DummyEconomy(
            shock_std={'output_gap': 1.0, 'inflation': 0.3},
            target_inflation=2.0,
            target_output_gap=0.0
        )
        
        # At target: should give 0 reward
        reward = economy.compute_reward(inflation=2.0, output_gap=0.0)
        assert reward == 0.0, "Reward at target should be 0.0"
    
    def test_compute_reward_with_deviations(self) -> None:
        """Test reward with deviations from target."""
        class DummyEconomy(BaseEconomy):
            def step(self, state, action):
                pass
            def reset(self, initial_state=None):
                pass
        
        economy = DummyEconomy(
            shock_std={'output_gap': 1.0, 'inflation': 0.3},
            target_inflation=2.0,
            target_output_gap=0.0
        )
        
        # Deviation should give negative reward
        reward = economy.compute_reward(inflation=3.0, output_gap=1.0)
        expected = -0.5 * (1.0**2 + 1.0**2)  # Equal weights
        assert reward == expected, f"Expected reward {expected}, got {reward}"
    
    def test_compute_reward_with_penalty(self) -> None:
        """Test penalty for extreme deviations."""
        class DummyEconomy(BaseEconomy):
            def step(self, state, action):
                pass
            def reset(self, initial_state=None):
                pass
        
        economy = DummyEconomy(
            shock_std={'output_gap': 1.0, 'inflation': 0.3},
            penalty_threshold=2.0,
            penalty_multiplier=10.0
        )
        
        # Extreme deviation (> 2 pp) should trigger penalty
        reward = economy.compute_reward(inflation=5.0, output_gap=0.0)
        base_reward = -0.5 * (3.0**2)
        penalty = 10.0 * (3.0**2)
        expected = base_reward - penalty
        assert reward == expected, "Penalty calculation is incorrect"
    
    def test_check_termination_target_reached(self) -> None:
        """Test termination when target is reached."""
        class DummyEconomy(BaseEconomy):
            def step(self, state, action):
                pass
            def reset(self, initial_state=None):
                pass
        
        economy = DummyEconomy(shock_std={'output_gap': 1.0, 'inflation': 0.3})
        
        # Within tolerance: should terminate
        done = economy.check_termination(
            inflation=2.1,
            output_gap=0.1,
            step=5,
            max_steps=10,
            tolerance={'inflation': 0.2, 'output_gap': 0.2}
        )
        assert done is True, "Should terminate when within tolerance"
    
    def test_check_termination_max_steps(self) -> None:
        """Test termination at max steps."""
        class DummyEconomy(BaseEconomy):
            def step(self, state, action):
                pass
            def reset(self, initial_state=None):
                pass
        
        economy = DummyEconomy(shock_std={'output_gap': 1.0, 'inflation': 0.3})
        
        # Max steps reached: should terminate
        done = economy.check_termination(
            inflation=3.0,
            output_gap=1.0,
            step=10,
            max_steps=10,
            tolerance={'inflation': 0.2, 'output_gap': 0.2}
        )
        assert done is True, "Should terminate when max steps reached"


class TestSVAREconomy:
    """Tests for SVAR economy."""
    
    def test_initialization(self, svar_economy: SVAREconomy) -> None:
        """Test SVAR economy initialization."""
        assert svar_economy.state_dim == 6
        assert svar_economy.action_dim == 1
        assert svar_economy.target_inflation == 2.0
    
    def test_reset(self, svar_economy: SVAREconomy) -> None:
        """Test environment reset."""
        state = svar_economy.reset()
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (6,)
        assert not np.any(np.isnan(state)), "State contains NaNs after reset"
    
    def test_reset_with_initial_state(self, svar_economy: SVAREconomy, sample_state: np.ndarray) -> None:
        """Test reset with specified initial state."""
        state = svar_economy.reset(initial_state=sample_state)
        
        np.testing.assert_array_equal(state, sample_state)
    
    def test_step_shape(self, svar_economy: SVAREconomy, sample_state: np.ndarray) -> None:
        """Test step returns correct shapes."""
        action = 3.5  # Interest rate
        
        next_state, reward, done, info = svar_economy.step(sample_state, action)
        
        assert next_state.shape == (6,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_step_state_evolution(self, svar_economy: SVAREconomy, sample_state: np.ndarray) -> None:
        """Test state properly evolves."""
        action = 3.5
        
        next_state, _, _, info = svar_economy.step(sample_state, action)
        
        # Check lag structure
        assert next_state[1] == sample_state[0], "y_{t-1} should equal y_t"
        assert next_state[3] == sample_state[2], "π_{t-1} should equal π_t"
        assert next_state[4] == action, "i_{t-1} should equal i_t"
        assert next_state[5] == sample_state[4], "i_{t-2} should equal i_{t-1}"
    
    def test_step_deterministic(self, svar_params: Dict, seed: int) -> None:
        """Test deterministic prediction without shocks."""
        economy = SVAREconomy(
            params=svar_params['coefficients'],
            shock_std={'output_gap': 0.0, 'inflation': 0.0},  # No shocks
            seed=seed
        )
        
        state = np.array([1.0, 0.5, 2.5, 2.0, 3.0, 2.5], dtype=np.float32)
        action = 3.5
        
        # Two runs should be identical
        next_state1, _, _, _ = economy.step(state, action)
        economy.reset()
        next_state2, _, _, _ = economy.step(state, action)
        
        np.testing.assert_array_almost_equal(next_state1, next_state2)
    
    def test_predict_deterministic(self, svar_economy: SVAREconomy, sample_state: np.ndarray) -> None:
        """Test deterministic prediction method."""
        action = 3.5
        
        y_next, pi_next = svar_economy.predict_deterministic(sample_state, action)
        
        assert isinstance(y_next, (float, np.floating))
        assert isinstance(pi_next, (float, np.floating))
        assert not np.isnan(y_next)
        assert not np.isnan(pi_next)


class TestANNEconomy:
    """Tests for ANN economy."""
    
    def test_initialization(self, ann_economy: ANNEconomy) -> None:
        """Test ANN economy initialization."""
        assert ann_economy.state_dim == 6
        assert ann_economy.action_dim == 1
    
    def test_reset(self, ann_economy: ANNEconomy) -> None:
        """Test environment reset."""
        state = ann_economy.reset()
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (6,)
    
    def test_step_shape(self, ann_economy: ANNEconomy, sample_state: np.ndarray) -> None:
        """Test step returns correct shapes."""
        action = 3.5
        
        next_state, reward, done, info = ann_economy.step(sample_state, action)
        
        assert next_state.shape == (6,)
        assert isinstance(reward, (float, np.floating))
        assert 'inflation' in info
        assert 'output_gap' in info
    
    def test_networks_called(self, ann_economy: ANNEconomy, sample_state: np.ndarray) -> None:
        """Test that both networks are called during step."""
        action = 3.5
        
        next_state, reward, done, info = ann_economy.step(sample_state, action)
        
        # Check that predictions exist
        assert 'predictions' in info
        assert 'y_pred' in info['predictions']
        assert 'pi_pred' in info['predictions']
    
    def test_predict_deterministic(self, ann_economy: ANNEconomy, sample_state: np.ndarray) -> None:
        """Test deterministic prediction."""
        action = 3.5
        
        y_next, pi_next = ann_economy.predict_deterministic(sample_state, action)
        
        assert isinstance(y_next, (float, np.floating))
        assert isinstance(pi_next, (float, np.floating))


class TestEconomyNetwork:
    """Tests for EconomyNetwork."""
    
    def test_initialization(self) -> None:
        """Test network initialization."""
        network = EconomyNetwork(input_dim=4, hidden_units=8)
        
        assert isinstance(network, torch.nn.Module)
    
    def test_forward_pass(self) -> None:
        """Test forward pass."""
        network = EconomyNetwork(input_dim=4, hidden_units=8)
        
        x = torch.randn(10, 4)  # Batch of 10
        output = network(x)
        
        assert output.shape == (10,)
    
    def test_weight_initialization(self) -> None:
        """Test that weights are properly initialized."""
        network = EconomyNetwork(input_dim=4, hidden_units=8)
        
        # Check that weights are not all zero
        for param in network.parameters():
            assert not torch.all(param == 0)
    
    def test_gradient_flow(self) -> None:
        """Test that gradients flow through network."""
        network = EconomyNetwork(input_dim=4, hidden_units=8)
        
        x = torch.randn(10, 4, requires_grad=True)
        output = network(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        for param in network.parameters():
            assert param.grad is not None


class TestEnvironmentConsistency:
    """Test consistency between SVAR and ANN environments."""
    
    def test_same_interface(self, svar_economy: SVAREconomy, ann_economy: ANNEconomy, sample_state: np.ndarray) -> None:
        """Test that both environments have same interface."""
        action = 3.5
        
        # Both should work with same inputs
        svar_result = svar_economy.step(sample_state, action)
        ann_result = ann_economy.step(sample_state, action)
        
        # Check output structure
        assert len(svar_result) == len(ann_result) == 4
        assert svar_result[0].shape == ann_result[0].shape  # state
    
    def test_reward_consistency(self, svar_economy: SVAREconomy, ann_economy: ANNEconomy) -> None:
        """Test that reward calculation is consistent."""
        # Both should give same reward for same variables
        reward_svar = svar_economy.compute_reward(2.5, 1.0)
        reward_ann = ann_economy.compute_reward(2.5, 1.0)
        
        assert reward_svar == reward_ann, "Reward calculation differs between environments"