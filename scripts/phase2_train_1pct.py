"""
Phase 2: Ultra-Fast LSTM Training (1% Sample)
==============================================

Simplified training on 1% sample for rapid empirical validation.

Author: Sanjay Basu, MD PhD
Date: December 2024
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/phase2_1pct_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleLSTMPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, states):
        # states: (batch, seq_len, state_dim)
        _, (h_n, _) = self.lstm(states)
        logits = self.fc(h_n.squeeze(0))
        return logits

class SimpleLSTMValue(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM (state_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, states):
        _, (h_n, _) = self.lstm(states)
        value = self.fc(h_n.squeeze(0))
        return value.squeeze(-1)

def prepare_batch(df, state_cols, seq_len=4):
    """Prepare batch from dataframe."""
    states = []
    actions = []
    rewards = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        # Extract state sequence
        state_seq = []
        for t in range(seq_len):
            state_t = [row[f'{col}_t{t}'] for col in state_cols]
            state_seq.append(state_t)
        states.append(state_seq)
        
        actions.append(row['action_idx'])
        rewards.append(row['reward_shaped'])
    
    return (torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards))

def train_epoch(policy, value_net, optimizer_p, optimizer_v, train_loader, state_cols):
    """Train for one epoch."""
    policy.train()
    value_net.train()
    
    total_policy_loss = 0
    total_value_loss = 0
    n_batches = 0
    
    for batch_df in train_loader:
        states, actions, rewards = prepare_batch(batch_df, state_cols)
        
        # Value network update
        optimizer_v.zero_grad()
        values = value_net(states)
        value_loss = nn.MSELoss()(values, rewards)
        value_loss.backward()
        optimizer_v.step()
        
        # Policy network update (AWR)
        optimizer_p.zero_grad()
        logits = policy(states)
        log_probs = nn.LogSoftmax(dim=-1)(logits)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            advantages = rewards - value_net(states)
            weights = torch.exp(advantages / 1.0).clamp(max=20)
        
        policy_loss = -(weights * action_log_probs).mean()
        policy_loss.backward()
        optimizer_p.step()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        n_batches += 1
    
    return total_policy_loss / n_batches, total_value_loss / n_batches

def evaluate(value_net, val_loader, state_cols):
    """Evaluate on validation set."""
    value_net.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch_df in val_loader:
            states, _, rewards = prepare_batch(batch_df, state_cols)
            values = value_net(states)
            loss = nn.MSELoss()(values, rewards)
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches

def batch_generator(df, batch_size):
    """Generate batches."""
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i+batch_size]

def main():
    logger.info("="*80)
    logger.info("PHASE 2: ULTRA-FAST LSTM TRAINING (1% SAMPLE)")
    logger.info("="*80)
    
    # Paths
    data_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data_1pct')
    model_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    train_df = pd.read_parquet(data_dir / 'sequences_train.parquet')
    val_df = pd.read_parquet(data_dir / 'sequences_val.parquet')
    
    logger.info(f"  Train: {len(train_df):,} sequences")
    logger.info(f"  Val: {len(val_df):,} sequences")
    
    # Get dimensions
    state_cols = ['age', 'num_encounters_last_week', 'num_ed_visits_last_week', 
                  'num_ip_visits_last_week', 'num_interventions_last_week']
    state_dim = len(state_cols)
    num_actions = train_df['action_idx'].max() + 1
    
    logger.info(f"  State dim: {state_dim}, Actions: {num_actions}")
    
    # Create models
    hidden_dim = 32  # Smaller for fast training
    policy = SimpleLSTMPolicy(state_dim, hidden_dim, num_actions)
    value_net = SimpleLSTMValue(state_dim, hidden_dim)
    
    logger.info(f"  Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    logger.info(f"  Value params: {sum(p.numel() for p in value_net.parameters()):,}")
    
    # Optimizers
    optimizer_p = torch.optim.Adam(policy.parameters(), lr=3e-4)
    optimizer_v = torch.optim.Adam(value_net.parameters(), lr=3e-4)
    
    # Training
    batch_size = 512
    epochs = 10  # Fewer epochs for fast validation
    best_val_loss = float('inf')
    
    logger.info("\nStarting training...")
    for epoch in range(epochs):
        train_loader = batch_generator(train_df, batch_size)
        policy_loss, value_loss = train_epoch(policy, value_net, optimizer_p, optimizer_v, train_loader, state_cols)
        
        val_loader = batch_generator(val_df, batch_size)
        val_loss = evaluate(value_net, val_loader, state_cols)
        
        logger.info(f"Epoch {epoch+1}/{epochs}: "
                   f"Policy Loss={policy_loss:.4f}, "
                   f"Value Loss={value_loss:.4f}, "
                   f"Val Loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'value_state_dict': value_net.state_dict(),
                'val_loss': val_loss
            }, model_dir / 'lstm_policy_1pct_best.pt')
            logger.info(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
    
    logger.info("\n" + "="*80)
    logger.info("✓ TRAINING COMPLETE")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
