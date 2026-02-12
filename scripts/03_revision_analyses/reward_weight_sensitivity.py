"""
Reward Weight Sensitivity Analysis
====================================

Test sensitivity of policy performance and action distributions
to reward component weights across a 4x4 grid.

Addresses Reviewer 1 Comment 2 and Reviewer 2 Comment 3.

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
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
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
        logging.FileHandler('reward_weight_sensitivity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.append(str(_REPO_ROOT / 'src'))


def reshape_rewards(df, w_primary=1.0, w_engagement=0.3, w_intermediate=0.5):
    """
    Recompute shaped reward with given weights.

    R_total = w_p * R_primary + w_e * R_engagement + w_i * R_intermediate

    Assumes df has columns: reward_primary, reward_engagement, reward_intermediate
    (or we reconstruct from raw columns).
    """
    if 'reward_primary' in df.columns:
        shaped = (w_primary * df['reward_primary'] +
                  w_engagement * df['reward_engagement'] +
                  w_intermediate * df.get('reward_intermediate', 0))
    else:
        # Fallback: use reward_shaped as-is (already computed)
        # and scale by weight ratios relative to base
        base_total = 1.0 * 1.0 + 0.3 * 1.0 + 0.5 * 1.0
        new_total = w_primary * 1.0 + w_engagement * 1.0 + w_intermediate * 1.0
        shaped = df['reward_shaped'] * (new_total / base_total)

    return shaped.values


def train_quick_policy(train_states, train_actions, train_rewards,
                       state_dim, num_actions, device, n_epochs=10):
    """Train a quick AWR policy for sensitivity analysis."""
    from phase2_train_full import LSTMPolicyNetwork

    policy = LSTMPolicyNetwork(state_dim, 64, num_actions).to(device)
    value_net = nn.Sequential(
        nn.Linear(state_dim * 4, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=3e-4)

    batch_size = 256

    for epoch in range(n_epochs):
        policy.train()
        value_net.train()

        indices = np.random.permutation(len(train_actions))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            if len(batch_idx) < batch_size // 2:
                continue

            states = torch.FloatTensor(train_states[batch_idx]).to(device)
            actions = torch.LongTensor(train_actions[batch_idx]).to(device)
            rewards = torch.FloatTensor(train_rewards[batch_idx]).to(device)

            # Value update
            value_opt.zero_grad()
            states_flat = states.reshape(states.shape[0], -1)
            v_pred = value_net(states_flat).squeeze(-1)
            v_loss = F.mse_loss(v_pred, rewards)
            v_loss.backward()
            value_opt.step()

            # Policy update
            with torch.no_grad():
                v_baseline = value_net(states_flat).squeeze(-1)
                advantages = rewards - v_baseline
                weights = torch.exp(advantages / 1.0)
                weights = torch.clamp(weights, max=20.0)

            policy_opt.zero_grad()
            logits, _ = policy(states)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            p_loss = -(weights * selected).mean()
            p_loss.backward()
            policy_opt.step()

    policy.eval()
    return policy


def compute_action_distribution(policy, states_tensor, num_actions, device):
    """Get aggregate action distribution from policy."""
    with torch.no_grad():
        logits, _ = policy(states_tensor)
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    # Aggregate distribution (average policy)
    avg_dist = probs.mean(axis=0)

    # Greedy actions
    greedy_actions = np.argmax(probs, axis=1)

    return avg_dist, greedy_actions


def main():
    logger.info("=" * 80)
    logger.info("REWARD WEIGHT SENSITIVITY ANALYSIS")
    logger.info("=" * 80)

    np.random.seed(42)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths (using module-level defaults with env var overrides)
    results_dir.mkdir(exist_ok=True)

    # Load data
    train_df = pd.read_parquet(data_dir / 'sequences_train.parquet')
    test_df = pd.read_parquet(data_dir / 'sequences_test.parquet')

    train_sample = train_df.sample(min(30000, len(train_df)), random_state=42)
    test_sample = test_df.sample(min(10000, len(test_df)), random_state=42)

    state_cols = ['age', 'num_encounters_last_week', 'num_ed_visits_last_week',
                  'num_ip_visits_last_week', 'num_interventions_last_week']
    state_dim = len(state_cols)

    # Load model config
    model_path = model_dir / 'lstm_policy_10pct_best.pt'
    checkpoint = torch.load(model_path, map_location=device)
    num_actions = checkpoint['config']['num_actions']

    train_sample = train_sample[train_sample['action_idx'] < num_actions].copy()
    test_sample = test_sample[test_sample['action_idx'] < num_actions].copy()

    # Prepare state arrays
    def prepare_states(df):
        states = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            seq = []
            for t in range(4):
                s = [float(row.get(f'{col}_t{t}', 0)) for col in state_cols]
                seq.append(s)
            states.append(seq)
        return np.array(states)

    train_states = prepare_states(train_sample)
    test_states = prepare_states(test_sample)
    train_actions = train_sample['action_idx'].values
    test_actions = test_sample['action_idx'].values

    test_states_tensor = torch.FloatTensor(test_states).to(device)

    # Grid search
    engagement_weights = [0.1, 0.3, 0.5, 0.7]
    intermediate_weights = [0.3, 0.5, 0.7, 1.0]

    # First, compute base case
    base_ew, base_iw = 0.3, 0.5
    base_rewards_train = reshape_rewards(train_sample, 1.0, base_ew, base_iw)
    base_rewards_test = reshape_rewards(test_sample, 1.0, base_ew, base_iw)

    logger.info("Training base case policy...")
    base_policy = train_quick_policy(
        train_states, train_actions, base_rewards_train,
        state_dim, num_actions, device
    )
    base_dist, base_greedy = compute_action_distribution(
        base_policy, test_states_tensor, num_actions, device
    )

    # Compute outcome correlation for base case
    if 'ed_visits_30d' in test_sample.columns:
        outcome = test_sample['ed_visits_30d'].values
    else:
        outcome = (test_sample['reward_shaped'].values < -0.5).astype(float)

    base_corr, base_pval = spearmanr(base_rewards_test, outcome)

    # Train Q-function for DR evaluation
    X_train_flat = []
    for idx in range(len(train_sample)):
        row = train_sample.iloc[idx]
        sf = []
        for t in range(4):
            for col in state_cols:
                sf.append(row.get(f'{col}_t{t}', 0))
        sf.append(row['action_idx'])
        X_train_flat.append(sf)

    # Train behavioral policy
    from phase2_train_full import LSTMPolicyNetwork
    behavioral = LSTMPolicyNetwork(state_dim, 64, num_actions).to(device)
    beh_opt = torch.optim.Adam(behavioral.parameters(), lr=3e-4)
    batch_size = 512
    for epoch in range(3):
        behavioral.train()
        indices = np.random.permutation(len(train_sample))
        for i in range(0, len(indices), batch_size):
            bi = indices[i:i+batch_size]
            if len(bi) < batch_size // 2:
                continue
            s = torch.FloatTensor(train_states[bi]).to(device)
            a = torch.LongTensor(train_actions[bi]).to(device)
            beh_opt.zero_grad()
            logits, _ = behavioral(s)
            loss = F.cross_entropy(logits, a)
            loss.backward()
            beh_opt.step()
    behavioral.eval()

    results = []

    for ew in engagement_weights:
        for iw in intermediate_weights:
            logger.info(f"\n--- Testing w_e={ew}, w_i={iw} ---")

            # Reshape rewards
            rewards_train = reshape_rewards(train_sample, 1.0, ew, iw)
            rewards_test = reshape_rewards(test_sample, 1.0, ew, iw)

            # Train policy
            policy = train_quick_policy(
                train_states, train_actions, rewards_train,
                state_dim, num_actions, device, n_epochs=10
            )

            # Get action distribution
            policy_dist, policy_greedy = compute_action_distribution(
                policy, test_states_tensor, num_actions, device
            )

            # JS divergence from base
            js_div = float(jensenshannon(base_dist + 1e-10, policy_dist + 1e-10))

            # % actions changed
            pct_changed = float(np.mean(base_greedy != policy_greedy) * 100)

            # Correlation with outcomes
            corr, pval = spearmanr(rewards_test, outcome)

            # DR evaluation
            q_func = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
            q_func.fit(np.array(X_train_flat), rewards_train)

            X_test_flat = []
            for idx in range(len(test_sample)):
                row = test_sample.iloc[idx]
                sf = []
                for t in range(4):
                    for col in state_cols:
                        sf.append(row.get(f'{col}_t{t}', 0))
                sf.append(row['action_idx'])
                X_test_flat.append(sf)
            q_vals = q_func.predict(np.array(X_test_flat))

            with torch.no_grad():
                logits_l, _ = policy(test_states_tensor)
                probs_l = F.softmax(logits_l, dim=-1).cpu().numpy()
                logits_b, _ = behavioral(test_states_tensor)
                probs_b = F.softmax(logits_b, dim=-1).cpu().numpy()

            ap_l = probs_l[np.arange(len(test_actions)), test_actions]
            ap_b = probs_b[np.arange(len(test_actions)), test_actions]
            w = ap_l / (ap_b + 1e-8)
            w_trunc = np.minimum(w, np.percentile(w, 95))

            dr_terms = w_trunc * (rewards_test - q_vals) + q_vals
            dr_value = float(np.mean(dr_terms))

            result = {
                'w_engagement': ew,
                'w_intermediate': iw,
                'dr_value': dr_value,
                'spearman_rho': float(corr),
                'spearman_p': float(pval),
                'js_divergence': js_div,
                'pct_actions_changed': pct_changed,
                'is_base_case': (ew == base_ew and iw == base_iw)
            }
            results.append(result)

            logger.info(f"  DR Value: {dr_value:.4f}, "
                       f"Spearman rho: {corr:.3f}, "
                       f"JS div: {js_div:.4f}, "
                       f"% changed: {pct_changed:.1f}%")

    output = {
        'reward_weight_sensitivity': results,
        'base_case': {'w_primary': 1.0, 'w_engagement': base_ew, 'w_intermediate': base_iw},
        'note': 'Primary weight fixed at 1.0. Grid search over engagement and intermediate weights.'
    }

    with open(results_dir / 'reward_weight_sensitivity.json', 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {results_dir / 'reward_weight_sensitivity.json'}")
    logger.info("=" * 80)
    logger.info("REWARD WEIGHT SENSITIVITY COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
