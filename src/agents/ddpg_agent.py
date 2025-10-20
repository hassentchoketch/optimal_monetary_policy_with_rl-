"""
Deep Deterministic Policy Gradient (DDPG) agent.

Implements the algorithm from Table 1 of the paper, based on
Lillicrap et al. (2015).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
from copy import deepcopy

from src.agents.networks import ActorNetwork, CriticNetwork
from src.agents.replay_buffer import ReplayBuffer


class OUNoise:
    """
    Ornstein-Uhlenbeck process for exploration noise.
    
    Implements step d) from Table 1: Initialize random process N for action exploration.
    
    The OU process generates temporally correlated noise, which is more
    suitable for physical control tasks than uncorrelated Gaussian noise.
    
    dX_t = θ(μ - X_t)dt + σdW_t
    
    where θ is the mean reversion rate and σ is the volatility.
    """
    
    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1.0,
        x0: Optional[np.ndarray] = None
    ):
        """
        Initialize OU noise process.
        
        Args:
            action_dim: Dimension of action space
            mu: Mean (long-term drift)
            theta: Mean reversion rate
            sigma: Volatility
            dt: Time step
            x0: Initial state
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0 if x0 is not None else np.zeros(action_dim)
        self.reset()
    
    def reset(self):
        """Reset noise to initial state."""
        self.x = self.x0.copy()
    
    def sample(self) -> np.ndarray:
        """
        Generate noise sample.
        
        Returns:
            Noise vector of shape [action_dim]
        """
        dx = (
            self.theta * (self.mu - self.x) * self.dt +
            self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        )
        self.x = self.x + dx
        return self.x


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent.
    
    Implements the full DDPG algorithm from Table 1:
    - Actor-critic architecture with target networks
    - Experience replay
    - Soft target updates (τ = 0.001)
    - Ornstein-Uhlenbeck exploration noise
    
    The agent learns an optimal monetary policy by interacting with
    the economic environment and maximizing expected long-term reward.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        critic_hidden: int,
        actor_hidden: Optional[int],
        linear_policy: bool,
        lr_actor: float = 2.5e-5,
        lr_critic: float = 2.5e-5,
        gamma: float = 0.99,
        tau: float = 0.001,
        buffer_size: int = 10000,
        batch_size: int = 64,
        noise_std: float = 1.0,
        noise_theta: float = 0.15,
        noise_sigma: float = 0.2,
        gradient_clip: float = 1.0,
        device: str = 'cpu',
        seed: Optional[int] = None
    ):
        """
        Initialize DDPG agent.
        
        Args:
            state_dim: Dimension of state space (2 or 4)
            action_dim: Dimension of action space (1)
            critic_hidden: Number of hidden units in critic (1-10)
            actor_hidden: Number of hidden units in actor (None for linear)
            linear_policy: If True, use linear policy function
            lr_actor: Learning rate for actor (2.5e-5 in paper)
            lr_critic: Learning rate for critic (2.5e-5 in paper)
            gamma: Discount factor (0.99 in paper)
            tau: Soft update parameter (0.001 in paper)
            buffer_size: Replay buffer capacity (10000 in paper)
            batch_size: Mini-batch size (64 in paper)
            noise_std: Standard deviation of exploration noise
            noise_theta: OU noise mean reversion rate
            noise_sigma: OU noise volatility
            gradient_clip: Gradient clipping threshold
            device: Device for computation ('cpu' or 'cuda')
            seed: Random seed
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip
        
        # Step a): Initialize actor and critic networks
        self.actor = ActorNetwork(
            state_dim, actor_hidden, linear_policy
        ).to(self.device)
        
        self.critic = CriticNetwork(
            state_dim, action_dim, critic_hidden
        ).to(self.device)
        
        # Step b): Initialize target networks
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        
        # Freeze target networks (no gradient computation)
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # Optimizers (Adam as per Table A.2)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=lr_actor
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=lr_critic
        )
        
        # Step c): Initialize experience replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size, state_dim, action_dim
        )
        
        # Step d): Initialize exploration noise
        self.noise = OUNoise(
            action_dim,
            mu=0.0,
            theta=noise_theta,
            sigma=noise_sigma
        )
        self.noise_std = noise_std
        
        # Training statistics
        self.total_steps = 0
        self.episode_count = 0
    
    def select_action(
        self,
        state: np.ndarray,
        add_noise: bool = True,
        noise_scale: float = 1.0
    ) -> float:
        """
        Select action using current policy with optional exploration noise.
        
        Implements step f) from Table 1:
        i_t = P(x_t|θ^P) + N_t
        
        Args:
            state: Current state observation
            add_noise: Whether to add exploration noise
            noise_scale: Scale factor for noise (1.0 during training)
            
        Returns:
            Action (interest rate) to take
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        # Get action from actor network
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0, 0]
        self.actor.train()
        
        # Add exploration noise if requested
        if add_noise:
            noise = self.noise.sample()[0] * self.noise_std * noise_scale
            action = action + noise
        
        return action
    
    def update(self) -> Dict[str, float]:
        """
        Perform one update step using sampled mini-batch.
        
        Implements steps i) through m) from Table 1:
        i) Sample mini-batch
        j) Compute target Q-values
        k) Update critic
        l) Update actor using policy gradient
        m) Soft update target networks
        
        Returns:
            Dictionary with 'critic_loss' and 'actor_loss'
        """
        # Check if buffer has enough samples
        if not self.replay_buffer.is_ready(self.batch_size):
            return {'critic_loss': 0.0, 'actor_loss': 0.0}
        
        # Step i): Sample random mini-batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Step j): Set y_j = r_j + γ Q'(x_{j+1}, P'(x_{j+1}|θ^{P'})|θ^{Q'})
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * target_q * (1 - dones)
        
        # Step k): Update critic by minimizing loss
        # L = 1/N Σ_j (y_j - Q(x_j, i_j|θ^Q))²
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.gradient_clip
        )
        
        self.critic_optimizer.step()
        
        # Step l): Update actor using sampled policy gradient
        # ∇_θ^P J ≈ 1/N Σ_j [∇_i Q(x, i|θ^Q)|_{x=x_j, i=P(x_j)} · ∇_θ^P P(i|θ^P)|_{x_j}]
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.gradient_clip
        )
        
        self.actor_optimizer.step()
        
        # Step m): Soft update target networks
        # θ' ← τθ + (1-τ)θ'
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        self.total_steps += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def _soft_update(
        self,
        source: nn.Module,
        target: nn.Module
    ):
        """
        Soft update target network parameters.
        
        Implements: θ' ← τθ + (1-τ)θ'
        
        Args:
            source: Source network (actor or critic)
            target: Target network
        """
        for target_param, source_param in zip(
            target.parameters(), source.parameters()
        ):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def reset_noise(self):
        """Reset exploration noise."""
        self.noise.reset()
    
    def save(self, filepath: str):
        """
        Save agent state.
        
        Args:
            filepath: Path to save checkpoint
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count
        }, filepath)
    
    def load(self, filepath: str):
        """
        Load agent state.
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']
    
    def get_policy_parameters(self) -> Dict[str, float]:
        """
        Extract policy parameters for linear policies.
        
        For linear policies, extract coefficients as in Table 3:
        i_t = α_0 + β^0_π π_t + β^0_y y_t [+ β^1_π π_{t-1} + β^1_y y_{t-1}]
        
        Returns:
            Dictionary with parameter names and values
        """
        if not hasattr(self.actor.network, '__iter__'):
            # Linear policy: single layer
            params = {}
            weights = self.actor.network.weight.detach().cpu().numpy()[0]
            bias = self.actor.network.bias.detach().cpu().numpy()[0]
            
            if self.state_dim == 2:
                # No lags: [y_t, π_t]
                params['alpha_0'] = bias
                params['beta_y_0'] = weights[0]
                params['beta_pi_0'] = weights[1]
            elif self.state_dim == 4:
                # One lag: [y_t, y_{t-1}, π_t, π_{t-1}]
                params['alpha_0'] = bias
                params['beta_y_0'] = weights[0]
                params['beta_y_1'] = weights[1]
                params['beta_pi_0'] = weights[2]
                params['beta_pi_1'] = weights[3]
            
            return params
        else:
            # Nonlinear policy: return None (parameters not directly interpretable)
            return {'note': 'Nonlinear policy - parameters not directly interpretable'}