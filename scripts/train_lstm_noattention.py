"""
LSTM Without Attention - Ablation Study
========================================

Train LSTM without attention mechanism to show attention's contribution.

Author: Sanjay Basu, MD PhD
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/lstm_noattn_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LSTMPolicyNoAttention(nn.Module):
    """LSTM policy without attention mechanism."""
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        # No attention - just use last hidden state directly
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, states):
        lstm_out, _ = self.lstm(states)
        h = lstm_out[:, -1, :]  # Take last timestep (no attention)
        
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        logits = self.fc3(h)
        
        return logits, None  # No attention weights

class LSTMValueNoAttention(nn.Module):
    """LSTM value network without attention."""
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, states):
        lstm_out, _ = self.lstm(states)
        h = lstm_out[:, -1, :]
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        value = self.fc2(h)
        return value.squeeze(-1)

def prepare_batch(df, state_cols, seq_len=4, device='cpu'):
    """Prepare batch from dataframe."""
    states = []
    actions = []
    rewards = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        state_seq = []
        for t in range(seq_len):
            state_t = []
            for col in state_cols:
                col_name = f'{col}_t{t}'
                if col_name in row:
                    state_t.append(float(row[col_name]))
                else:
                    state_t.append(0.0)
            state_seq.append(state_t)
        states.append(state_seq)
        
        actions.append(int(row['action_idx']))
        rewards.append(float(row['reward_shaped']))
    
    return (torch.FloatTensor(states).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device))

def train_epoch(policy, value_net, optimizer_p, optimizer_v, train_df, state_cols, batch_size, device='cpu'):
    """Train for one epoch."""
    policy.train()
    value_net.train()
    
    total_policy_loss = 0
    total_value_loss = 0
    n_batches = 0
    
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    
    for i in range(0, len(train_df), batch_size):
        batch_df = train_df.iloc[i:i+batch_size]
        if len(batch_df) < batch_size // 2:
            continue
            
        states, actions, rewards = prepare_batch(batch_df, state_cols, device=device)
        
        # Value update
        optimizer_v.zero_grad()
        values = value_net(states)
        value_loss = F.mse_loss(values, rewards)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
        optimizer_v.step()
        
        # Policy update
        optimizer_p.zero_grad()
        logits, _ = policy(states)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            advantages = rewards - value_net(states)
            weights = torch.exp(advantages / 1.0).clamp(max=20)
        
        policy_loss = -(weights * action_log_probs).mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer_p.step()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        n_batches += 1
    
    return total_policy_loss / max(n_batches, 1), total_value_loss / max(n_batches, 1)

def evaluate(policy, value_net, val_df, state_cols, batch_size, device='cpu'):
    """Evaluate on validation set."""
    policy.eval()
    value_net.eval()
    
    total_value_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(val_df), batch_size):
            batch_df = val_df.iloc[i:i+batch_size]
            if len(batch_df) == 0:
                continue
                
            states, actions, rewards = prepare_batch(batch_df, state_cols, device=device)
            
            values = value_net(states)
            value_loss = F.mse_loss(values, rewards)
            total_value_loss += value_loss.item()
            n_batches += 1
    
    return total_value_loss / max(n_batches, 1)

def main():
    logger.info("="*80)
    logger.info("LSTM WITHOUT ATTENTION - ABLATION STUDY")
    logger.info("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Paths
    data_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data_factored')
    model_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\nLoading data...")
    train_df = pd.read_parquet(data_dir / 'sequences_train.parquet')
    val_df = pd.read_parquet(data_dir / 'sequences_val.parquet')
    
    logger.info(f"  Train: {len(train_df):,} sequences")
    logger.info(f"  Val: {len(val_df):,} sequences")
    
    # Get dimensions
    state_cols = ['age', 'num_encounters_last_week', 'num_ed_visits_last_week', 
                  'num_ip_visits_last_week', 'num_interventions_last_week']
    state_dim = len(state_cols)
    num_actions = int(train_df['action_idx'].max() + 1)
    
    logger.info(f"\n  State dim: {state_dim}")
    logger.info(f"  Num actions: {num_actions}")
    
    # Create models
    hidden_dim = 64
    policy = LSTMPolicyNoAttention(state_dim, hidden_dim, num_actions).to(device)
    value_net = LSTMValueNoAttention(state_dim, hidden_dim).to(device)
    
    logger.info(f"\n  Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    logger.info(f"  Value params: {sum(p.numel() for p in value_net.parameters()):,}")
    
    # Optimizers
    optimizer_p = torch.optim.Adam(policy.parameters(), lr=3e-4)
    optimizer_v = torch.optim.Adam(value_net.parameters(), lr=3e-4)
    
    # Training
    batch_size = 256
    epochs = 20
    patience = 3
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    history = []
    
    logger.info("\nStarting training...")
    
    for epoch in range(epochs):
        policy_loss, value_loss = train_epoch(
            policy, value_net, optimizer_p, optimizer_v, 
            train_df, state_cols, batch_size, device
        )
        
        val_loss = evaluate(policy, value_net, val_df, state_cols, batch_size, device)
        
        history.append({
            'epoch': epoch + 1,
            'train_policy_loss': policy_loss,
            'train_value_loss': value_loss,
            'val_loss': val_loss
        })
        
        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"Policy Loss={policy_loss:.4f}, "
            f"Value Loss={value_loss:.4f}, "
            f"Val Loss={val_loss:.4f}"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'value_state_dict': value_net.state_dict(),
                'val_loss': val_loss,
                'history': history,
                'config': {
                    'state_dim': state_dim,
                    'hidden_dim': hidden_dim,
                    'num_actions': num_actions,
                    'state_cols': state_cols
                }
            }, model_dir / 'lstm_noattention_best.pt')
            
            logger.info(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"\nEarly stopping after {patience} epochs")
                break
    
    logger.info("\n" + "="*80)
    logger.info("✓ LSTM NO-ATTENTION ABLATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
