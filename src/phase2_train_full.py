"""
Phase 2: Full LSTM Training (10% Sample)
=========================================

LSTM training with tracking and evaluation.

Author: Sanjay Basu, MD PhD
Date: December 2024
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from tqdm import tqdm
import time

# Resolve paths relative to repo root; override via environment variables
_REPO_ROOT = Path(__file__).resolve().parent.parent
data_dir = Path(os.environ.get('FACTORED_RL_DATA_DIR', _REPO_ROOT / 'data'))
model_dir = Path(os.environ.get('FACTORED_RL_MODEL_DIR', _REPO_ROOT / 'models'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase2_full_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LSTMPolicyNetwork(nn.Module):
    """LSTM-based policy with attention mechanism."""
    def __init__(self, state_dim, hidden_dim, num_actions, num_heads=4):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads, batch_first=True)
        self.action_embed = nn.Embedding(num_actions, 32)
        self.fc1 = nn.Linear(hidden_dim * 2 + 32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, states, prev_actions=None):
        # states: (batch, seq_len, state_dim)
        lstm_out, _ = self.lstm(states)  # (batch, seq_len, hidden*2)
        
        # Self-attention over sequence
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take last timestep
        h = attn_out[:, -1, :]  # (batch, hidden*2)
        
        # Optional action embedding
        if prev_actions is not None:
            action_emb = self.action_embed(prev_actions)
            h = torch.cat([h, action_emb], dim=-1)
        else:
            h = torch.cat([h, torch.zeros(h.size(0), 32, device=h.device)], dim=-1)
        
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        logits = self.fc3(h)
        
        return logits, attn_weights

class LSTMValueNetwork(nn.Module):
    """LSTM-based value network."""
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, 4, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, states):
        lstm_out, _ = self.lstm(states)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        h = attn_out[:, -1, :]
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
        
        # Extract state sequence
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
    """Train for one epoch with AWR."""
    policy.train()
    value_net.train()
    
    total_policy_loss = 0
    total_value_loss = 0
    n_batches = 0
    
    # Shuffle
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    
    # Create batches
    for i in range(0, len(train_df), batch_size):
        batch_df = train_df.iloc[i:i+batch_size]
        if len(batch_df) < batch_size // 2:  # Skip small final batch
            continue
            
        states, actions, rewards = prepare_batch(batch_df, state_cols, device=device)
        
        # Value network update
        optimizer_v.zero_grad()
        values = value_net(states)
        value_loss = F.mse_loss(values, rewards)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
        optimizer_v.step()
        
        # Policy network update (AWR)
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
    total_policy_value = 0
    n_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(val_df), batch_size):
            batch_df = val_df.iloc[i:i+batch_size]
            if len(batch_df) == 0:
                continue
                
            states, actions, rewards = prepare_batch(batch_df, state_cols, device=device)
            
            # Value loss
            values = value_net(states)
            value_loss = F.mse_loss(values, rewards)
            total_value_loss += value_loss.item()
            
            # Policy value (mean predicted value)
            total_policy_value += values.mean().item()
            n_batches += 1
    
    return total_value_loss / max(n_batches, 1), total_policy_value / max(n_batches, 1)

def main():
    logger.info("="*80)
    logger.info("PHASE 2: FULL LSTM TRAINING (10% SAMPLE)")
    logger.info("="*80)
    
    # Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    if device == 'cuda':
        torch.cuda.manual_seed(42)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Paths (using module-level defaults with env var overrides)
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
    
    # Check which columns actually exist
    sample_row = train_df.iloc[0]
    actual_state_cols = []
    for col in state_cols:
        if f'{col}_t0' in sample_row:
            actual_state_cols.append(col)
    state_cols = actual_state_cols
    
    state_dim = len(state_cols)
    num_actions = int(train_df['action_idx'].max() + 1)
    
    logger.info(f"\n  State dim: {state_dim}")
    logger.info(f"  State features: {state_cols}")
    logger.info(f"  Num actions: {num_actions}")
    
    # Create models
    hidden_dim = 64
    policy = LSTMPolicyNetwork(state_dim, hidden_dim, num_actions).to(device)
    value_net = LSTMValueNetwork(state_dim, hidden_dim).to(device)
    
    logger.info(f"\n  Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    logger.info(f"  Value params: {sum(p.numel() for p in value_net.parameters()):,}")
    
    # Optimizers
    optimizer_p = torch.optim.Adam(policy.parameters(), lr=3e-4)
    optimizer_v = torch.optim.Adam(value_net.parameters(), lr=3e-4)
    
    # Training configuration
    batch_size = 256
    epochs = 20
    patience = 3
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Track metrics
    history = {
        'train_policy_loss': [],
        'train_value_loss': [],
        'val_value_loss': [],
        'val_policy_value': [],
        'epoch_time': []
    }
    
    logger.info(f"\nTraining configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Early stopping patience: {patience}")
    logger.info(f"  Device: {device}")
    
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        policy_loss, value_loss = train_epoch(
            policy, value_net, optimizer_p, optimizer_v, 
            train_df, state_cols, batch_size, device
        )
        
        # Evaluate
        val_loss, val_policy_value = evaluate(
            policy, value_net, val_df, state_cols, batch_size, device
        )
        
        epoch_time = time.time() - start_time
        
        # Track
        history['train_policy_loss'].append(policy_loss)
        history['train_value_loss'].append(value_loss)
        history['val_value_loss'].append(val_loss)
        history['val_policy_value'].append(val_policy_value)
        history['epoch_time'].append(epoch_time)
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s): "
            f"Policy Loss={policy_loss:.4f}, "
            f"Value Loss={value_loss:.4f}, "
            f"Val Loss={val_loss:.4f}, "
            f"Val Policy Value={val_policy_value:.4f}"
        )
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'value_state_dict': value_net.state_dict(),
                'optimizer_p_state_dict': optimizer_p.state_dict(),
                'optimizer_v_state_dict': optimizer_v.state_dict(),
                'val_loss': val_loss,
                'val_policy_value': val_policy_value,
                'history': history,
                'config': {
                    'state_dim': state_dim,
                    'hidden_dim': hidden_dim,
                    'num_actions': num_actions,
                    'state_cols': state_cols
                }
            }, model_dir / 'lstm_policy_10pct_best.pt')
            
            logger.info(f"  Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break
    
    # Save final history
    with open(model_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Total epochs: {epoch + 1}")
    logger.info(f"Total time: {sum(history['epoch_time']):.1f}s ({sum(history['epoch_time'])/60:.1f}min)")
    logger.info(f"Model saved to: {model_dir / 'lstm_policy_10pct_best.pt'}")
    logger.info(f"History saved to: {model_dir / 'training_history.json'}")

if __name__ == "__main__":
    main()
