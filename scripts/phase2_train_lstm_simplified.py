"""
Phase 2: Train LSTM Policy with AWR
====================================

Trains LSTM-based policy network using Advantage Weighted Regression
on the prepared sequence data.

This is a SIMPLIFIED version optimized for fast training (not full hyperparameter search).

Author: Sanjay Basu, MD PhD  
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleLSTMPolicy(nn.Module):
    """Simplified LSTM policy for faster training."""
    
    def __init__(self, state_dim=7, action_dim=1080, hidden_dim=64, num_layers=1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Action embedding
        self.action_embedding = nn.Embedding(action_dim, 32)
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, states, actions):
        """
        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch,) - action indices
        Returns:
            scores: (batch, 1)
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(states)  # (batch, seq_len, hidden_dim)
        
        # Use last timestep
        h = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Action embedding
        a_emb = self.action_embedding(actions)  # (batch, 32)
        
        # Concatenate and score
        combined = torch.cat([h, a_emb], dim=1)
        score = self.mlp(combined)
        
        return score


class SimpleLSTMValue(nn.Module):
    """Simplified LSTM value network."""
    
    def __init__(self, state_dim=7, hidden_dim=64, num_layers=1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, states):
        lstm_out, _ = self.lstm(states)
        h = lstm_out[:, -1, :]
        value = self.mlp(h)
        return value


class SequenceDataset(Dataset):
    """PyTorch dataset for sequence data."""
    
    def __init__(self, df, action_to_idx):
        self.df = df.reset_index(drop=True)
        self.action_to_idx = action_to_idx
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # States: (4, 7) numpy array
        states = torch.tensor(row['states'], dtype=torch.float32)
        
        # Action index
        action_tuple = row['action_tuple']
        action_idx = self.action_to_idx.get(action_tuple, 0)
        action = torch.tensor(action_idx, dtype=torch.long)
        
        # Reward
        reward = torch.tensor(row['reward_shaped'], dtype=torch.float32)
        
        return states, action, reward


def create_action_mapping(df):
    """Create mapping from action tuples to indices."""
    unique_actions = df['action_tuple'].unique()
    action_to_idx = {action: idx for idx, action in enumerate(unique_actions)}
    return action_to_idx


def train_awr(policy, value_net, train_loader, val_loader, 
              n_epochs=20, lr=3e-4, beta=1.0, device='cpu'):
    """
    Train policy using Advantage Weighted Regression.
    
    Args:
        policy: Policy network
        value_net: Value network
      train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of epochs
        lr: Learning rate
        beta: AWR temperature
        device: Device to train on
    """
    logger.info(f"Training on device: {device}")
    
    policy = policy.to(device)
    value_net = value_net.to(device)
    
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)
    
    history = []
    best_val_loss = float('inf')
    
    for epoch in range(1, n_epochs + 1):
        # Training
        policy.train()
        value_net.train()
        
        train_policy_loss = 0
        train_value_loss = 0
        n_batches = 0
        
        for states, actions, rewards in train_loader:
            states, actions, rewards = states.to(device), actions.to(device), rewards.to(device)
            
            batch_size = states.shape[0]
            
            # Value network update
            value_optimizer.zero_grad()
            values = value_net(states).squeeze()
            value_loss = F.mse_loss(values, rewards)
            value_loss.backward()
            value_optimizer.step()
            
            # Policy network update (AWR)
            policy_optimizer.zero_grad()
            
            # Compute advantages
            with torch.no_grad():
                values_detached = value_net(states).squeeze()
                advantages = rewards - values_detached
                weights = torch.exp(advantages / beta).clamp(max=20.0)
            
            # Policy scores
            scores = policy(states, actions).squeeze()
            
            # AWR loss: negative weighted log-likelihood
            policy_loss = -(weights * scores).mean()
            policy_loss.backward()
            policy_optimizer.step()
            
            train_policy_loss += policy_loss.item()
            train_value_loss += value_loss.item()
            n_batches += 1
        
        train_policy_loss /= n_batches
        train_value_loss /= n_batches
        
        # Validation
        policy.eval()
        value_net.eval()
        
        val_value_loss = 0
        n_val_batches = 0
        
        with torch.no_grad():
            for states, actions, rewards in val_loader:
                states, rewards = states.to(device), rewards.to(device)
                values = value_net(states).squeeze()
                val_loss = F.mse_loss(values, rewards)
                val_value_loss += val_loss.item()
                n_val_batches += 1
        
        val_value_loss /= n_val_batches
        
        logger.info(f"Epoch {epoch}/{n_epochs} | "
                   f"Train Policy Loss: {train_policy_loss:.4f} | "
                   f"Train Value Loss: {train_value_loss:.4f} | "
                   f"Val Value Loss: {val_value_loss:.4f}")
        
        history.append({
            'epoch': epoch,
            'train_policy_loss': train_policy_loss,
            'train_value_loss': train_value_loss,
            'val_value_loss': val_value_loss
        })
        
        # Save best model
        if val_value_loss < best_val_loss:
            best_val_loss = val_value_loss
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'value_state_dict': value_net.state_dict(),
                'val_loss': val_value_loss
            }, '/Users/sanjaybasu/waymark-local/notebooks/factored_rl/models/lstm_policy_best.pt')
            logger.info(f"  → Saved best model (val_loss: {val_value_loss:.4f})")
    
    return history


def main():
    logger.info("="*80)
    logger.info("PHASE 2: LSTM POLICY TRAINING")
    logger.info("="*80)
    
    # Load data
    data_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data')
    
    logger.info("Loading sequence data...")
    train_df = pd.read_parquet(data_dir / 'sequences_train.parquet')
    val_df = pd.read_parquet(data_dir / 'sequences_val.parquet')
    
    logger.info(f"  Train: {len(train_df):,} sequences")
    logger.info(f"  Val: {len(val_df):,} sequences")
    
    # Create action mapping
    action_to_idx = create_action_mapping(train_df)
    logger.info(f"  Unique actions: {len(action_to_idx)}")
    
    # Create datasets
    train_dataset = SequenceDataset(train_df, action_to_idx)
    val_dataset = SequenceDataset(val_df, action_to_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # Initialize models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    policy = SimpleLSTMPolicy(
        state_dim=7,
        action_dim=len(action_to_idx),
        hidden_dim=64,
        num_layers=1
    )
    
    value_net = SimpleLSTMValue(
        state_dim=7,
        hidden_dim=64,
        num_layers=1
    )
    
    logger.info(f"\nModel architecture:")
    logger.info(f"  Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    logger.info(f"  Value parameters: {sum(p.numel() for p in value_net.parameters()):,}")
    
    # Train
    logger.info("\nStarting training...")
    
    history = train_awr(
        policy=policy,
        value_net=value_net,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=20,
        lr=3e-4,
        beta=1.0,
        device=device
    )
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/models/training_history.csv', index=False)
    
    logger.info("\n" + "="*80)
    logger.info("✓ PHASE 2 COMPLETE")
    logger.info("="*80)
    logger.info(f"Best validation loss: {min(h['val_value_loss'] for h in history):.4f}")
    logger.info(f"Model saved to: models/lstm_policy_best.pt")
    logger.info(f"History saved to: models/training_history.csv")


if __name__ == "__main__":
    main()
