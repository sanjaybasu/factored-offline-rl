"""
Advantage-weighted regression with factored action space and embeddings.

Extends the original AWR implementation to handle:
- Factored actions: (modality, provider, goal, urgency)
- Action embeddings for parameter efficiency
- Scoring-based action selection instead of softmax over 1080 actions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add factored action space module
sys.path.append(str(Path(__file__).parent))
from factored_action_space import FactoredActionSpace


class ActionEmbeddingNetwork(nn.Module):
    """
    Network that scores state-action pairs using action embeddings.
    
    Instead of outputting 1080 logits, we:
    1. Get action embedding e_a ∈ ℝ¹²⁸ for each action
    2. Concatenate with state [s, e_a] ∈ ℝ³⁸⁴
    3. Output scalar score (bounded to prevent divergence)
    
    This is much more parameter-efficient and enables generalization.
    """
    
    def __init__(self, state_dim: int, action_space: FactoredActionSpace, hidden_dim: int = 256):
        super().__init__()
        self.action_space = action_space
        self.embedding_dim = action_space.embedding_dim * 4  # Concatenate 4 components
        
        # Scoring network: [state, action_embedding] → score
        self.fc1 = nn.Linear(state_dim + self.embedding_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # Layer normalization for stability
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)  # Layer normalization
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, states: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch_size, state_dim)
            action_embeddings: (batch_size, num_actions, embedding_dim) or (batch_size, embedding_dim)
            
        Returns:
            scores: (batch_size, num_actions) or (batch_size, 1) - bounded to [-10, 10]
        """
        # Handle both single action and multiple actions
        if action_embeddings.dim() == 2:
            # Single action per state
            x = torch.cat([states, action_embeddings], dim=-1)
            x = F.relu(self.ln1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.ln2(self.fc2(x)))
            x = self.fc3(x)
            # CRITICAL: Bound output to prevent divergence
            x = 10.0 * torch.tanh(x / 10.0)  # Soft bound to [-10, 10]
            return x
        else:
            # Multiple actions per state
            batch_size, num_actions, _ = action_embeddings.shape
            # Expand states to match actions
            states_expanded = states.unsqueeze(1).expand(batch_size, num_actions, -1)
            # Concatenate
            x = torch.cat([states_expanded, action_embeddings], dim=-1)
            # Flatten for processing
            x = x.view(-1, x.shape[-1])
            # Forward
            x = F.relu(self.ln1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.ln2(self.fc2(x)))
            x = self.fc3(x)
            # CRITICAL: Bound output to prevent divergence
            x = 10.0 * torch.tanh(x / 10.0)  # Soft bound to [-10, 10]
            # Reshape back
            return x.view(batch_size, num_actions)


class ValueNetwork(nn.Module):
    """Critic network for AWR."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(states))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FactoredActionDataset(Dataset):
    """Dataset for factored actions."""
    
    def __init__(self, states: np.ndarray, actions: List[Tuple], rewards: np.ndarray, 
                 action_space: FactoredActionSpace):
        self.states = torch.FloatTensor(states)
        self.rewards = torch.FloatTensor(rewards)
        self.action_space = action_space
        
        # Encode actions to indices and get embeddings
        self.action_indices = [action_space.encode_action_tuple(a) for a in actions]
        self.action_embeddings = torch.FloatTensor(
            [action_space.get_action_embedding(a) for a in actions]
        )
        
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int):
        return {
            'state': self.states[idx],
            'action_embedding': self.action_embeddings[idx],
            'action_idx': self.action_indices[idx],
            'reward': self.rewards[idx]
        }


def train_awr_factored(
    states: np.ndarray,
    actions: List[Tuple],
    rewards: np.ndarray,
    action_space: FactoredActionSpace,
    beta: float = 1.0,
    num_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-5,  # REDUCED from 3e-4 for stability
    device: str = 'cpu'
) -> Tuple[ActionEmbeddingNetwork, ValueNetwork]:
    """
    Train AWR with factored actions.
    
    Args:
        states: (N, state_dim) state observations
        actions: List of N (modality, provider, goal, urgency) tuples
        rewards: (N,) rewards
        action_space: FactoredActionSpace instance
        beta: Temperature for advantage weighting (lower = more conservative)
        num_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate (reduced for numerical stability)
        device: torch device
        
    Returns:
        actor: Trained action scoring network
        critic: Trained value network
    """
    state_dim = states.shape[1]
    
    # Preprocess rewards to prevent NaN and handle outliers
    print(f"Raw reward statistics: mean={rewards.mean():.4f}, std={rewards.std():.4f}, min={rewards.min():.4f}, max={rewards.max():.4f}")
    print(f"Non-zero rewards: {(rewards != 0).sum()} ({100*(rewards != 0).mean():.2f}%)")
    
    # Clip extreme outliers using robust percentile-based thresholding
    # This prevents extreme values (e.g., data errors showing 379 acute events) from destabilizing training
    if (rewards != 0).sum() > 100:  # Only clip if we have enough non-zero samples
        non_zero_rewards = rewards[rewards != 0]
        lower_clip = np.percentile(non_zero_rewards, 1)   # 1st percentile
        upper_clip = np.percentile(non_zero_rewards, 99)  # 99th percentile
        
        clipped_count = ((rewards < lower_clip) | (rewards > upper_clip)).sum()
        if clipped_count > 0:
            print(f"Clipping {clipped_count} extreme outliers to [{lower_clip:.2f}, {upper_clip:.2f}]")
            rewards = np.clip(rewards, lower_clip, upper_clip)
            print(f"After clipping: mean={rewards.mean():.4f}, std={rewards.std():.4f}, min={rewards.min():.4f}, max={rewards.max():.4f}")
    
    # Normalize rewards for stability (handle sparse rewards)
    if rewards.std() > 1e-6:
        rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        print(f"Normalized reward statistics: mean={rewards_normalized.mean():.4f}, std={rewards_normalized.std():.4f}")
    else:
        print("WARNING: Rewards have near-zero variance. Using unnormalized rewards.")
        rewards_normalized = rewards
    
    # Initialize networks
    actor = ActionEmbeddingNetwork(state_dim, action_space).to(device)
    critic = ValueNetwork(state_dim).to(device)
    
    # Optimizers with lower learning rate
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)
    
    # Dataset (use normalized rewards)
    dataset = FactoredActionDataset(states, actions, rewards_normalized, action_space)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(num_epochs):
        actor_losses = []
        critic_losses = []
        
        for batch in loader:
            states_b = batch['state'].to(device)
            action_embs_b = batch['action_embedding'].to(device)
            rewards_b = batch['reward'].to(device)
            
            # Critic update: predict values
            values = critic(states_b).squeeze()
            critic_loss = F.mse_loss(values, rewards_b)
            
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)  # More aggressive clipping
            critic_opt.step()
            
            # Actor update: advantage-weighted scoring
            with torch.no_grad():
                values = critic(states_b).squeeze()
                advantages = rewards_b - values
                # More conservative advantage weighting
                weights = torch.exp(torch.clamp(beta * advantages, -5, 5))  # Tighter clamp
                weights = weights / (weights.mean() + 1e-8)
                weights = torch.clamp(weights, 0.1, 10.0)  # Additional weight bounds
                
            # Score the taken actions
            scores = actor(states_b, action_embs_b).squeeze()
            
            # Weighted log-likelihood
            actor_loss = -(weights * scores).mean()
            
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)  # More aggressive clipping
            actor_opt.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
        
        if (epoch + 1) % 10 == 0:
            avg_actor_loss = np.mean(actor_losses)
            avg_critic_loss = np.mean(critic_losses)
            print(f"Epoch {epoch+1}/{num_epochs} - Actor Loss: {avg_actor_loss:.4f}, "
                  f"Critic Loss: {avg_critic_loss:.4f}")
            
            # Check for divergence
            if abs(avg_actor_loss) > 1e6:
                print(f"WARNING: Actor loss diverging ({avg_actor_loss:.2e}). Stopping training.")
                break
    
    return actor, critic


def select_action(
    actor: ActionEmbeddingNetwork,
    state: np.ndarray,
    action_space: FactoredActionSpace,
    top_k: int = 5,
    device: str = 'cpu'
) -> Tuple[Tuple, List[Tuple]]:
    """
    Select best action for a state by scoring all valid actions.
    
    Args:
        actor: Trained action scoring network
        state: (state_dim,) state vector
        action_space: FactoredActionSpace
        top_k: Return top-k actions
        device: torch device
        
    Returns:
        best_action: (modality, provider, goal, urgency)
        top_k_actions: List of top-k actions
    """
    actor.eval()
    
    # Sample a set of valid actions (or enumerate all for exact inference)
    # For efficiency, sample ~100-500 actions
    num_samples = min(500, action_space.num_actions)
    candidate_actions = [action_space.sample_valid_action() for _ in range(num_samples)]
    
    # Get embeddings
    action_embeddings = torch.FloatTensor(
        [action_space.get_action_embedding(a) for a in candidate_actions]
    ).to(device)
    
    # Expand state
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    action_embeddings = action_embeddings.unsqueeze(0)  # (1, num_samples, emb_dim)
    
    # Score all actions
    with torch.no_grad():
        scores = actor(state_tensor, action_embeddings).squeeze()  # (num_samples,)
    
    # Get top-k
    top_k_indices = torch.argsort(scores, descending=True)[:top_k]
    top_k_actions = [candidate_actions[i] for i in top_k_indices.cpu().numpy()]
    
    return top_k_actions[0], top_k_actions


def main():
    parser = argparse.ArgumentParser(description="AWR with factored action space")
    parser.add_argument("--traj_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="foundation/outputs/rl_factored")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    # Initialize action space
    action_space = FactoredActionSpace(embedding_dim=32)
    print(f"Action space size: {action_space.num_actions}")
    
    # Load trajectories
    # TODO: Parse real intervention data into factored actions
    # For now, using placeholder
    print(f"Loading trajectories from {args.traj_path}")
    print("Note: Integration with real data parsing pending.")
    
    # Example usage with synthetic data
    np.random.seed(42)
    num_samples = 10000
    state_dim = 256
    
    states = np.random.randn(num_samples, state_dim)
    actions = [action_space.sample_valid_action() for _ in range(num_samples)]
    rewards = np.random.randn(num_samples) * 0.1 - 0.05  # Small negative rewards
    
    print(f"Training on {num_samples} samples...")
    actor, critic = train_awr_factored(
        states, actions, rewards, action_space,
        beta=args.beta,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
    
    # Save models
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'action_space_config': {
            'embedding_dim': action_space.embedding_dim,
            'num_actions': action_space.num_actions
        }
    }, output_dir / "awr_factored.pt")
    
    print(f"Saved models to {output_dir / 'awr_factored.pt'}")
    
    # Test action selection
    test_state = np.random.randn(state_dim)
    best_action, top_k = select_action(actor, test_state, action_space, top_k=5, device=args.device)
    
    print("\nExample action selection:")
    print(f"Best action: {best_action}")
    print(f"Top 5 actions: {top_k}")


if __name__ == "__main__":
    main()
