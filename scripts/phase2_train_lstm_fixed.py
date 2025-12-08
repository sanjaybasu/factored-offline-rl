"""
Phase 2: Train LSTM Policy (Compatible with Flattened States)
==============================================================

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
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleLSTMPolicy(nn.Module):
    """Simplified LSTM policy."""
    
    def __init__(self, state_dim=7, seq_len=4, action_dim=1080, hidden_dim=64):
        super().__init__()
        
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.action_embedding = nn.Embedding(action_dim, 32)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, states, actions):
        # states: (batch, seq_len, state_dim)
        lstm_out, _ = self.lstm(states)
        h = lstm_out[:, -1, :]  # Last timestep
        
        a_emb = self.action_embedding(actions)
        combined = torch.cat([h, a_emb], dim=1)
        score = self.mlp(combined)
        return score


class SimpleLSTMValue(nn.Module):
    """Simplified LSTM value network."""
    
    def __init__(self, state_dim=7, seq_len=4, hidden_dim=64):
        super().__init__()
        
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
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
    """PyTorch dataset for flattened sequence data."""
    
    def __init__(self, df, action_to_idx, seq_len=4):
        self.df = df.reset_index(drop=True)
        self.action_to_idx = action_to_idx
        self.seq_len = seq_len
        
        # Feature names for each timestep
        self.state_features = ['age', 'encounters', 'ed', 'ip', 'interventions', 'enrolled_days', 'total_paid']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Reconstruct state sequence from flattened columns
        states = []
        for t in range(self.seq_len):
            state_t = [row[f'{feat}_t{t}'] for feat in self.state_features]
            states.append(state_t)
        
        states = torch.tensor(states, dtype=torch.float32)  # (seq_len, state_dim)
        
        # Action
        action_tuple = row['action_tuple']
        action_idx = self.action_to_idx.get(action_tuple, 0)
        action = torch.tensor(action_idx, dtype=torch.long)
        
        # Reward
        reward = torch.tensor(row['reward_shaped'], dtype=torch.float32)
        
        return states, action, reward


def create_action_mapping(df):
    """Create mapping from action tuple strings to indices."""
    unique_actions = df['action_tuple'].unique()
    action_to_idx = {action: idx for idx, action in enumerate(unique_actions)}
    return action_to_idx


def train_epoch(policy, value_net, train_loader, policy_opt, value_opt, beta, device):
    """Train for one epoch."""
    policy.train()
    value_net.train()
    
    total_policy_loss = 0
    total_value_loss = 0
    n_batches = 0
    
    for states, actions, rewards in train_loader:
        states, actions, rewards = states.to(device), actions.to(device), rewards.to(device)
        
        # Value update
        value_opt.zero_grad()
        values = value_net(states).squeeze()
        value_loss = F.mse_loss(values, rewards)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
        value_opt.step()
        
        # Policy update (AWR)
        policy_opt.zero_grad()
        
        with torch.no_grad():
            values_detached = value_net(states).squeeze()
            advantages = rewards - values_detached
            weights = torch.exp(advantages / beta).clamp(max=20.0)
        
        scores = policy(states, actions).squeeze()
        policy_loss = -(weights * scores).mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        policy_opt.step()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        n_batches += 1
    
    return total_policy_loss / n_batches, total_value_loss / n_batches


def validate(policy, value_net, val_loader, device):
    """Validate."""
    policy.eval()
    value_net.eval()
    
    total_value_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for states, actions, rewards in val_loader:
            states, rewards = states.to(device), rewards.to(device)
            values = value_net(states).squeeze()
            val_loss = F.mse_loss(values, rewards)
            total_value_loss += val_loss.item()
            n_batches += 1
    
    return total_value_loss / n_batches


def main():
    logger.info("="*80)
    logger.info("PHASE 2: LSTM POLICY TRAINING")
    logger.info("="*80)
    
    # Paths
    data_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data')
    model_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading sequence data...")
    train_df = pd.read_parquet(data_dir / 'sequences_train.parquet')
    val_df = pd.read_parquet(data_dir / 'sequences_val.parquet')
    
    logger.info(f"  Train: {len(train_df):,} sequences")
    logger.info(f"  Val: {len(val_df):,} sequences")
    
    # Action mapping
    action_to_idx = create_action_mapping(train_df)
    logger.info(f"  Unique actions: {len(action_to_idx)}")
    
    # Datasets
    train_dataset = SequenceDataset(train_df, action_to_idx)
    val_dataset = SequenceDataset(val_df, action_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # Models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    policy = SimpleLSTMPolicy(state_dim=7, seq_len=4, action_dim=len(action_to_idx), hidden_dim=64)
    value_net = SimpleLSTMValue(state_dim=7, seq_len=4, hidden_dim=64)
    
    policy = policy.to(device)
    value_net = value_net.to(device)
    
    logger.info(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    logger.info(f"Value params: {sum(p.numel() for p in value_net.parameters()):,}")
    
    # Optimizers
    policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=3e-4)
    
    # Training loop
    logger.info("\nStarting training...")
    
    n_epochs = 20
    beta = 1.0
    best_val_loss = float('inf')
    history = []
    
    for epoch in range(1, n_epochs + 1):
        train_policy_loss, train_value_loss = train_epoch(
            policy, value_net, train_loader, policy_opt, value_opt, beta, device
        )
        
        val_value_loss = validate(policy, value_net, val_loader, device)
        
        logger.info(f"Epoch {epoch}/{n_epochs} | "
                   f"Train Policy: {train_policy_loss:.4f} | "
                   f"Train Value: {train_value_loss:.4f} | "
                   f"Val Value: {val_value_loss:.4f}")
        
        history.append({
            'epoch': epoch,
            'train_policy_loss': train_policy_loss,
            'train_value_loss': train_value_loss,
            'val_value_loss': val_value_loss
        })
        
        if val_value_loss < best_val_loss:
            best_val_loss = val_value_loss
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'value_state_dict': value_net.state_dict(),
                'action_to_idx': action_to_idx,
                'val_loss': val_value_loss
            }, model_dir / 'lstm_policy_best.pt')
            logger.info(f"  → Saved best model (val_loss: {val_value_loss:.4f})")
    
    # Save history
    pd.DataFrame(history).to_csv(model_dir / 'training_history.csv', index=False)
    
    logger.info("\n" + "="*80)
    logger.info("✓ PHASE 2 COMPLETE")
    logger.info("="*80)
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {model_dir / 'lstm_policy_best.pt'}")


if __name__ == "__main__":
    main()
