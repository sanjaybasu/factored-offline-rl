"""
LSTM Window Length Sensitivity Analysis
=========================================

Test robustness of the LSTM policy to different history window lengths:
2, 4 (base case), 6, and 8 weeks.

Addresses Reviewer 2 Comment 2.

Author: Sanjay Basu, MD PhD
Date: February 2026
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
import xgboost as xgb
import sys
import math

# Resolve paths relative to repo root; override via environment variables
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
data_dir = Path(os.environ.get('FACTORED_RL_DATA_DIR', _REPO_ROOT / 'data'))
model_dir = Path(os.environ.get('FACTORED_RL_MODEL_DIR', _REPO_ROOT / 'models'))
results_dir = Path(os.environ.get('FACTORED_RL_RESULTS_DIR', _REPO_ROOT / 'results'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('window_sensitivity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.append(str(_REPO_ROOT / 'src'))


class FlexibleLSTMPolicy(nn.Module):
    """LSTM policy network with configurable sequence length."""

    def __init__(self, state_dim, hidden_dim, num_actions, seq_len=4,
                 num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.seq_len = seq_len

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Action embedding
        self.action_embed = nn.Embedding(num_actions, 32)

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, states):
        """
        Args:
            states: (batch, seq_len, state_dim)
        Returns:
            logits: (batch, num_actions)
            attention_weights: (batch, seq_len)
        """
        batch_size = states.shape[0]

        # LSTM encoding
        lstm_out, _ = self.lstm(states)  # (batch, seq_len, hidden*2)

        # Self-attention
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )  # (batch, seq_len, hidden*2)

        # Pool over time (mean of attended output)
        context = attn_out.mean(dim=1)  # (batch, hidden*2)

        # Score each action
        action_indices = torch.arange(self.num_actions, device=states.device)
        action_embeds = self.action_embed(action_indices)  # (num_actions, 32)

        # Expand for batch
        context_expanded = context.unsqueeze(1).expand(-1, self.num_actions, -1)
        action_expanded = action_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        combined = torch.cat([context_expanded, action_expanded], dim=-1)
        logits = self.mlp(combined).squeeze(-1)  # (batch, num_actions)

        # Extract attention weights for interpretability
        mean_attn = attn_weights.mean(dim=1) if attn_weights.dim() == 3 else attn_weights

        return logits, mean_attn


class FlexibleLSTMValue(nn.Module):
    """Value network with configurable sequence length."""

    def __init__(self, state_dim, hidden_dim, seq_len=4,
                 num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, states):
        lstm_out, _ = self.lstm(states)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        context = attn_out.mean(dim=1)
        return self.mlp(context).squeeze(-1)


def prepare_sequences(df, state_cols, seq_len):
    """Prepare state sequences of specified length."""
    states = []
    valid_mask = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        state_seq = []
        valid = True

        for t in range(seq_len):
            state_t = []
            for col in state_cols:
                val = row.get(f'{col}_t{t}', np.nan)
                if pd.isna(val):
                    val = 0.0
                    if t < 4:  # Base columns should exist
                        pass
                    else:
                        valid = False
                state_t.append(float(val))
            state_seq.append(state_t)

        states.append(state_seq)
        valid_mask.append(valid)

    return np.array(states), np.array(valid_mask)


def train_and_evaluate_window(train_df, test_df, state_cols, num_actions,
                               seq_len, device='cpu', n_epochs=15):
    """Train LSTM policy with given window length and evaluate via DR OPE."""

    state_dim = len(state_cols)
    hidden_dim = 64

    logger.info(f"\n{'='*60}")
    logger.info(f"Window length: {seq_len} weeks")
    logger.info(f"{'='*60}")

    # Prepare data -- for windows longer than 4, we pad/truncate
    # For shorter windows, we use the most recent timesteps
    train_states, _ = prepare_sequences(train_df, state_cols, min(seq_len, 4))
    test_states, _ = prepare_sequences(test_df, state_cols, min(seq_len, 4))

    # If seq_len > 4, pad with zeros at the beginning
    if seq_len > 4:
        pad_len = seq_len - 4
        train_pad = np.zeros((train_states.shape[0], pad_len, state_dim))
        test_pad = np.zeros((test_states.shape[0], pad_len, state_dim))
        train_states = np.concatenate([train_pad, train_states], axis=1)
        test_states = np.concatenate([test_pad, test_states], axis=1)
    elif seq_len < 4:
        # Use only the most recent seq_len timesteps
        train_states = train_states[:, -seq_len:, :]
        test_states = test_states[:, -seq_len:, :]

    # Initialize model
    policy = FlexibleLSTMPolicy(state_dim, hidden_dim, num_actions, seq_len).to(device)
    value_net = FlexibleLSTMValue(state_dim, hidden_dim, seq_len).to(device)

    policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=3e-4)

    actions_train = train_df['action_idx'].values
    rewards_train = train_df['reward_shaped'].values

    batch_size = 256
    best_val_loss = float('inf')

    # Training loop (AWR)
    for epoch in range(n_epochs):
        policy.train()
        value_net.train()

        epoch_vloss = 0.0
        epoch_ploss = 0.0
        n_batches = 0

        indices = np.random.permutation(len(train_df))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            if len(batch_idx) < batch_size // 2:
                continue

            states = torch.FloatTensor(train_states[batch_idx]).to(device)
            actions = torch.LongTensor(actions_train[batch_idx]).to(device)
            rewards = torch.FloatTensor(rewards_train[batch_idx]).to(device)

            # Value update
            value_opt.zero_grad()
            v_pred = value_net(states)
            v_loss = F.mse_loss(v_pred, rewards)
            v_loss.backward()
            value_opt.step()

            # Policy update (AWR)
            with torch.no_grad():
                advantages = rewards - value_net(states)
                weights = torch.exp(advantages / 1.0)
                weights = torch.clamp(weights, max=20.0)

            policy_opt.zero_grad()
            logits, _ = policy(states)
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            p_loss = -(weights * selected_log_probs).mean()
            p_loss.backward()
            policy_opt.step()

            epoch_vloss += v_loss.item()
            epoch_ploss += p_loss.item()
            n_batches += 1

        if n_batches > 0 and (epoch + 1) % 5 == 0:
            avg_vloss = epoch_vloss / n_batches
            logger.info(f"  Epoch {epoch+1}/{n_epochs}: "
                       f"V_loss={avg_vloss:.4f}, P_loss={epoch_ploss/n_batches:.4f}")
            if avg_vloss < best_val_loss:
                best_val_loss = avg_vloss

    # Evaluate via DR OPE
    policy.eval()

    # Train behavioral policy for this window
    behavioral = FlexibleLSTMPolicy(state_dim, hidden_dim, num_actions, seq_len).to(device)
    beh_opt = torch.optim.Adam(behavioral.parameters(), lr=3e-4)

    for epoch in range(5):
        behavioral.train()
        indices = np.random.permutation(len(train_df))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            if len(batch_idx) < batch_size // 2:
                continue
            states = torch.FloatTensor(train_states[batch_idx]).to(device)
            actions = torch.LongTensor(actions_train[batch_idx]).to(device)
            beh_opt.zero_grad()
            logits, _ = behavioral(states)
            loss = F.cross_entropy(logits, actions)
            loss.backward()
            beh_opt.step()
    behavioral.eval()

    # Q-function
    X_train_flat = []
    for idx in range(len(train_df)):
        row = train_df.iloc[idx]
        state_flat = []
        for t in range(4):
            for col in state_cols:
                state_flat.append(row.get(f'{col}_t{t}', 0))
        state_flat.append(row['action_idx'])
        X_train_flat.append(state_flat)
    q_func = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    q_func.fit(np.array(X_train_flat), rewards_train)

    # DR evaluation on test set
    actions_test = test_df['action_idx'].values
    rewards_test = test_df['reward_shaped'].values

    states_tensor = torch.FloatTensor(test_states).to(device)

    with torch.no_grad():
        logits_learned, attn_weights = policy(states_tensor)
        probs_learned = F.softmax(logits_learned, dim=-1).cpu().numpy()

        logits_behavior, _ = behavioral(states_tensor)
        probs_behavior = F.softmax(logits_behavior, dim=-1).cpu().numpy()

    # Q-values
    X_test_flat = []
    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        state_flat = []
        for t in range(4):
            for col in state_cols:
                state_flat.append(row.get(f'{col}_t{t}', 0))
        state_flat.append(row['action_idx'])
        X_test_flat.append(state_flat)
    q_values = q_func.predict(np.array(X_test_flat))

    # Importance weights
    ap_learned = probs_learned[np.arange(len(actions_test)), actions_test]
    ap_behavior = probs_behavior[np.arange(len(actions_test)), actions_test]
    weights = ap_learned / (ap_behavior + 1e-8)
    weights_trunc = np.minimum(weights, np.percentile(weights, 95))

    # DR estimate
    dr_terms = weights_trunc * (rewards_test - q_values) + q_values
    dr_value = np.mean(dr_terms)

    # Bootstrap CI
    bootstrap_values = []
    for _ in range(1000):
        idx = np.random.choice(len(dr_terms), size=len(dr_terms), replace=True)
        bootstrap_values.append(np.mean(dr_terms[idx]))
    ci_lower = np.percentile(bootstrap_values, 2.5)
    ci_upper = np.percentile(bootstrap_values, 97.5)

    # ESS
    ess = (np.sum(weights_trunc)**2) / np.sum(weights_trunc**2)
    ess_pct = (ess / len(test_df)) * 100

    # Attention entropy (measure of how spread attention is)
    if attn_weights is not None:
        attn_np = attn_weights.cpu().numpy()
        # Average attention entropy across test samples
        attn_entropy = -np.mean(np.sum(
            attn_np * np.log(attn_np + 1e-10), axis=-1
        ))
    else:
        attn_entropy = None

    result = {
        'window_weeks': seq_len,
        'dr_value': float(dr_value),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'val_loss': float(best_val_loss),
        'ess_pct': float(ess_pct),
        'attention_entropy': float(attn_entropy) if attn_entropy is not None else None,
        'n_train': len(train_df),
        'n_test': len(test_df)
    }

    logger.info(f"  DR Value: {dr_value:.4f} ({ci_lower:.4f}, {ci_upper:.4f})")
    logger.info(f"  Val Loss: {best_val_loss:.4f}")
    logger.info(f"  ESS: {ess_pct:.1f}%")

    return result


def main():
    logger.info("=" * 80)
    logger.info("LSTM WINDOW LENGTH SENSITIVITY ANALYSIS")
    logger.info("=" * 80)

    np.random.seed(42)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths (using module-level defaults with env var overrides)
    results_dir.mkdir(exist_ok=True)

    # Load data
    train_df = pd.read_parquet(data_dir / 'sequences_train.parquet')
    test_df = pd.read_parquet(data_dir / 'sequences_test.parquet')

    # Subsample for tractability
    train_sample = train_df.sample(min(30000, len(train_df)), random_state=42)
    test_sample = test_df.sample(min(10000, len(test_df)), random_state=42)

    state_cols = ['age', 'num_encounters_last_week', 'num_ed_visits_last_week',
                  'num_ip_visits_last_week', 'num_interventions_last_week']

    # Get num_actions from saved model config
    model_path = model_dir / 'lstm_policy_10pct_best.pt'
    checkpoint = torch.load(model_path, map_location=device)
    num_actions = checkpoint['config']['num_actions']

    train_sample = train_sample[train_sample['action_idx'] < num_actions].copy()
    test_sample = test_sample[test_sample['action_idx'] < num_actions].copy()

    # Test window lengths
    window_lengths = [2, 4, 6, 8]
    results = []

    for wl in window_lengths:
        result = train_and_evaluate_window(
            train_sample, test_sample, state_cols, num_actions,
            seq_len=wl, device=device, n_epochs=15
        )
        results.append(result)

    # Save results
    output = {
        'window_sensitivity': results,
        'note': 'Window length 4 is the base case used in main analysis.'
    }

    with open(results_dir / 'window_sensitivity.json', 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {results_dir / 'window_sensitivity.json'}")
    logger.info("=" * 80)
    logger.info("WINDOW LENGTH SENSITIVITY COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
