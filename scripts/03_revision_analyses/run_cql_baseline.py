"""
CQL Baseline Comparison
========================

Train Conservative Q-Learning (CQL) on the same data and evaluate
using the same Doubly Robust OPE framework as the main AWR pipeline.

Addresses Reviewer 1 Comment 1 and Reviewer 2 Comment 5.

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

# Resolve paths relative to repo root; override via environment variables
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
data_dir = Path(os.environ.get('FACTORED_RL_DATA_DIR', _REPO_ROOT / 'data'))
model_dir = Path(os.environ.get('FACTORED_RL_MODEL_DIR', _REPO_ROOT / 'models'))
results_dir = Path(os.environ.get('FACTORED_RL_RESULTS_DIR', _REPO_ROOT / 'results'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cql_baseline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.append(str(_REPO_ROOT / 'src'))


class DiscreteCQLNetwork(nn.Module):
    """
    Conservative Q-Learning for discrete action spaces.

    CQL adds a conservative regularization term to the standard Q-learning
    objective, penalizing Q-values for actions not seen in the data.
    This addresses distributional shift in offline RL.

    Reference: Kumar et al., "Conservative Q-Learning for Offline
    Reinforcement Learning", NeurIPS 2020.
    """

    def __init__(self, state_dim, hidden_dim, num_actions, seq_len=4):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.seq_len = seq_len

        # Simple MLP Q-network (no LSTM -- CQL baseline uses feedforward)
        input_dim = state_dim * seq_len
        self.q_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, states):
        """
        Args:
            states: (batch, seq_len, state_dim)
        Returns:
            q_values: (batch, num_actions)
        """
        batch_size = states.shape[0]
        flat = states.reshape(batch_size, -1)
        return self.q_network(flat)


def train_cql(train_df, state_cols, num_actions, device='cpu',
              alpha=1.0, n_epochs=20, batch_size=256, lr=3e-4):
    """
    Train CQL policy.

    CQL loss = standard Bellman error + alpha * conservative penalty
    Conservative penalty = log_sum_exp(Q(s,a)) - Q(s, a_data)

    Args:
        train_df: Training data
        state_cols: State feature column names
        num_actions: Number of discrete actions
        device: torch device
        alpha: CQL conservative penalty weight
        n_epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Learning rate

    Returns:
        Trained CQL network
    """
    seq_len = 4
    state_dim = len(state_cols)
    hidden_dim = 128

    cql_net = DiscreteCQLNetwork(state_dim, hidden_dim, num_actions, seq_len).to(device)
    optimizer = torch.optim.Adam(cql_net.parameters(), lr=lr)

    logger.info(f"Training CQL with alpha={alpha}, epochs={n_epochs}, "
                f"batch_size={batch_size}, lr={lr}")
    logger.info(f"Network params: {sum(p.numel() for p in cql_net.parameters()):,}")

    for epoch in range(n_epochs):
        cql_net.train()
        epoch_loss = 0.0
        epoch_bellman = 0.0
        epoch_cql_penalty = 0.0
        n_batches = 0

        shuffled = train_df.sample(frac=1)

        for i in range(0, len(shuffled), batch_size):
            batch_df = shuffled.iloc[i:i+batch_size]
            if len(batch_df) < batch_size // 2:
                continue

            # Prepare states
            states_list = []
            for idx in range(len(batch_df)):
                row = batch_df.iloc[idx]
                state_seq = []
                for t in range(seq_len):
                    state_t = [float(row.get(f'{col}_t{t}', 0)) for col in state_cols]
                    state_seq.append(state_t)
                states_list.append(state_seq)

            states = torch.FloatTensor(np.array(states_list)).to(device)
            actions = torch.LongTensor(batch_df['action_idx'].values).to(device)
            rewards = torch.FloatTensor(batch_df['reward_shaped'].values).to(device)

            # Forward pass
            q_values = cql_net(states)  # (batch, num_actions)

            # Standard Bellman error (using reward as target since we treat
            # each observation as a single-step problem for simplicity)
            q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            bellman_loss = F.mse_loss(q_selected, rewards)

            # CQL conservative penalty:
            # Penalize high Q-values for all actions, reward Q-values for data actions
            logsumexp_q = torch.logsumexp(q_values, dim=1).mean()
            data_q = q_selected.mean()
            cql_penalty = logsumexp_q - data_q

            # Combined loss
            loss = bellman_loss + alpha * cql_penalty

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cql_net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bellman += bellman_loss.item()
            epoch_cql_penalty += cql_penalty.item()
            n_batches += 1

        if n_batches > 0:
            logger.info(f"Epoch {epoch+1}/{n_epochs}: "
                       f"Loss={epoch_loss/n_batches:.4f}, "
                       f"Bellman={epoch_bellman/n_batches:.4f}, "
                       f"CQL_penalty={epoch_cql_penalty/n_batches:.4f}")

    cql_net.eval()
    return cql_net


def evaluate_cql_dr(test_df, cql_net, behavioral_policy, q_function,
                    state_cols, num_actions, device='cpu'):
    """
    Evaluate CQL policy using the same Doubly Robust OPE as the main analysis.

    The CQL greedy policy is: pi_cql(a|s) = softmax(Q_cql(s,:) / tau)
    with temperature tau for soft policy.
    """
    seq_len = 4

    # Prepare states
    states_list = []
    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        state_seq = []
        for t in range(seq_len):
            state_t = [float(row.get(f'{col}_t{t}', 0)) for col in state_cols]
            state_seq.append(state_t)
        states_list.append(state_seq)

    states = torch.FloatTensor(np.array(states_list)).to(device)
    actions = test_df['action_idx'].values
    rewards = test_df['reward_shaped'].values

    with torch.no_grad():
        # CQL policy (softmax over Q-values with temperature)
        q_values = cql_net(states)
        cql_probs = F.softmax(q_values / 0.5, dim=-1).cpu().numpy()  # tau=0.5

        # Behavioral policy
        logits_behavior, _ = behavioral_policy(states)
        probs_behavior = F.softmax(logits_behavior, dim=-1).cpu().numpy()

    # Q-function values for DR
    X_test = []
    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        state_flat = []
        for t in range(seq_len):
            for col in state_cols:
                state_flat.append(row.get(f'{col}_t{t}', 0))
        state_flat.append(row['action_idx'])
        X_test.append(state_flat)
    q_hat = q_function.predict(np.array(X_test))

    # Importance weights
    action_probs_cql = cql_probs[np.arange(len(actions)), actions]
    action_probs_behavior = probs_behavior[np.arange(len(actions)), actions]
    weights = action_probs_cql / (action_probs_behavior + 1e-8)

    # Truncate at 95th percentile
    cutoff = np.percentile(weights, 95)
    weights_truncated = np.minimum(weights, cutoff)

    # DR estimate
    dr_terms = weights_truncated * (rewards - q_hat) + q_hat
    dr_value = np.mean(dr_terms)

    # Bootstrap CI
    n_bootstrap = 1000
    bootstrap_values = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(dr_terms), size=len(dr_terms), replace=True)
        bootstrap_values.append(np.mean(dr_terms[idx]))
    ci_lower = np.percentile(bootstrap_values, 2.5)
    ci_upper = np.percentile(bootstrap_values, 97.5)

    # ESS
    ess = (np.sum(weights_truncated)**2) / np.sum(weights_truncated**2)
    ess_pct = (ess / len(test_df)) * 100

    # Demographic parity (if race column available)
    dp_gap = None
    if 'race' in test_df.columns:
        cql_actions = np.argmax(cql_probs, axis=1)
        # Check home visit actions (high-resource)
        # Use top 10% of action indices as proxy for high-resource
        high_resource_threshold = int(num_actions * 0.9)
        high_resource = (cql_actions >= high_resource_threshold).astype(float)

        group_rates = {}
        for race in test_df['race'].unique():
            mask = (test_df['race'].values == race)
            if mask.sum() > 50:
                group_rates[str(race)] = float(high_resource[mask].mean())

        if len(group_rates) >= 2:
            rates = list(group_rates.values())
            dp_gap = max(rates) - min(rates)

    # Non-inferiority test
    behavioral_value = np.mean(rewards)  # Simple estimate
    margin = 0.01
    diff = dr_value - behavioral_value
    se_diff = np.std(bootstrap_values)
    z_noninf = (diff + margin) / (se_diff + 1e-8)
    from scipy import stats
    p_noninf = 1 - stats.norm.cdf(-z_noninf)

    return {
        'dr_value': float(dr_value),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'ess_pct': float(ess_pct),
        'dp_gap': float(dp_gap) if dp_gap is not None else None,
        'noninf_p': float(p_noninf),
        'max_weight': float(np.max(weights)),
        'mean_weight': float(np.mean(weights_truncated)),
        'n_test': len(test_df)
    }


def main():
    logger.info("=" * 80)
    logger.info("CQL BASELINE COMPARISON")
    logger.info("=" * 80)

    np.random.seed(42)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # Paths (using module-level defaults with env var overrides)
    results_dir.mkdir(exist_ok=True)

    # Load data
    logger.info("Loading data...")
    train_df = pd.read_parquet(data_dir / 'sequences_train.parquet')
    test_df = pd.read_parquet(data_dir / 'sequences_test.parquet')

    # Subsample for tractability
    train_sample = train_df.sample(min(50000, len(train_df)), random_state=42)
    test_sample = test_df.sample(min(10000, len(test_df)), random_state=42)

    state_cols = ['age', 'num_encounters_last_week', 'num_ed_visits_last_week',
                  'num_ip_visits_last_week', 'num_interventions_last_week']

    # Load the main AWR model for comparison
    from phase2_train_full import LSTMPolicyNetwork
    model_path = model_dir / 'lstm_policy_10pct_best.pt'
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    num_actions = config['num_actions']

    # Filter data to match action space
    train_sample = train_sample[train_sample['action_idx'] < num_actions].copy()
    test_sample = test_sample[test_sample['action_idx'] < num_actions].copy()
    logger.info(f"Training on {len(train_sample)} sequences, testing on {len(test_sample)}")

    # Train behavioral policy and Q-function (shared across evaluations)
    logger.info("\n--- Training shared components ---")
    behavioral_policy = LSTMPolicyNetwork(
        len(state_cols), 64, num_actions
    ).to(device)
    optimizer = torch.optim.Adam(behavioral_policy.parameters(), lr=3e-4)

    batch_size = 512
    for epoch in range(3):
        behavioral_policy.train()
        shuffled = train_sample.sample(frac=1)
        for i in range(0, len(shuffled), batch_size):
            batch_df = shuffled.iloc[i:i+batch_size]
            if len(batch_df) < batch_size // 2:
                continue
            states_list = []
            for idx in range(len(batch_df)):
                row = batch_df.iloc[idx]
                state_seq = []
                for t in range(4):
                    state_t = [float(row.get(f'{col}_t{t}', 0)) for col in state_cols]
                    state_seq.append(state_t)
                states_list.append(state_seq)
            states = torch.FloatTensor(np.array(states_list)).to(device)
            actions = torch.LongTensor(batch_df['action_idx'].values).to(device)
            optimizer.zero_grad()
            logits, _ = behavioral_policy(states)
            loss = F.cross_entropy(logits, actions)
            loss.backward()
            optimizer.step()
    behavioral_policy.eval()

    # Train Q-function
    logger.info("Training Q-function...")
    X_train = []
    y_train = []
    for idx in range(len(train_sample)):
        row = train_sample.iloc[idx]
        state_flat = []
        for t in range(4):
            for col in state_cols:
                state_flat.append(row.get(f'{col}_t{t}', 0))
        state_flat.append(row['action_idx'])
        X_train.append(state_flat)
        y_train.append(row['reward_shaped'])
    q_function = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    q_function.fit(np.array(X_train), np.array(y_train))

    # Train CQL
    logger.info("\n--- Training CQL ---")
    cql_net = train_cql(
        train_sample, state_cols, num_actions, device,
        alpha=1.0, n_epochs=20, batch_size=256, lr=3e-4
    )

    # Evaluate CQL via DR OPE
    logger.info("\n--- Evaluating CQL via Doubly Robust OPE ---")
    cql_results = evaluate_cql_dr(
        test_sample, cql_net, behavioral_policy, q_function,
        state_cols, num_actions, device
    )

    logger.info(f"\nCQL Results:")
    logger.info(f"  DR Value: {cql_results['dr_value']:.4f} "
               f"(95% CI: {cql_results['ci_lower']:.4f} to {cql_results['ci_upper']:.4f})")
    logger.info(f"  ESS: {cql_results['ess_pct']:.1f}%")
    logger.info(f"  Non-inferiority p: {cql_results['noninf_p']:.4f}")
    if cql_results['dp_gap'] is not None:
        logger.info(f"  Demographic parity gap: {cql_results['dp_gap']:.4f}")

    # Save results
    output = {
        'cql': cql_results,
        'comparison_note': 'CQL evaluated using same DR OPE framework as AWR policy. '
                          'AWR reference values from main analysis: DR value=-0.0736, ESS=44.2%.'
    }

    with open(results_dir / 'cql_baseline_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {results_dir / 'cql_baseline_results.json'}")
    logger.info("=" * 80)
    logger.info("CQL BASELINE COMPARISON COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
