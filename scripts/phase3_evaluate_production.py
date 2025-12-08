"""
Phase 3: Doubly Robust Off-Policy Evaluation
============================================

Production evaluation using doubly robust OPE to estimate policy values
and compute effective sample sizes.

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
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/phase3_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_trained_models(model_dir, device='cpu'):
    """Load trained policy and value networks."""
    from phase2_train_full import LSTMPolicyNetwork, LSTMValueNetwork
    
    checkpoint = torch.load(model_dir / 'lstm_policy_10pct_best.pt', map_location=device)
    config = checkpoint['config']
    
    policy = LSTMPolicyNetwork(
        config['state_dim'],
        config['hidden_dim'],
        config['num_actions']
    ).to(device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    value_net = LSTMValueNetwork(
        config['state_dim'],
        config['hidden_dim']
    ).to(device)
    value_net.load_state_dict(checkpoint['value_state_dict'])
    value_net.eval()
    
    return policy, value_net, config

def prepare_states(df, state_cols, seq_len=4):
    """Prepare state sequences from dataframe."""
    states = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        state_seq = []
        for t in range(seq_len):
            state_t = [float(row.get(f'{col}_t{t}', 0)) for col in state_cols]
            state_seq.append(state_t)
        states.append(state_seq)
    return np.array(states)

def train_behavioral_policy(train_df, val_df, state_cols, num_actions, device='cpu'):
    """Train behavioral policy π_b(a|s) via maximum likelihood."""
    logger.info("\nTraining behavioral policy...")
    
    from phase2_train_full import LSTMPolicyNetwork
    
    behavioral_policy = LSTMPolicyNetwork(
        len(state_cols), 64, num_actions
    ).to(device)
    
    optimizer = torch.optim.Adam(behavioral_policy.parameters(), lr=3e-4)
    
    # Train for a few epochs
    batch_size = 512
    epochs = 5
    
    for epoch in range(epochs):
        behavioral_policy.train()
        total_loss = 0
        n_batches = 0
        
        shuffled = train_df.sample(frac=1)
        for i in range(0, len(shuffled), batch_size):
            batch_df = shuffled.iloc[i:i+batch_size]
            if len(batch_df) < batch_size //2:
                continue
            
            states = torch.FloatTensor(prepare_states(batch_df, state_cols)).to(device)
            actions = torch.LongTensor(batch_df['action_idx'].values).to(device)
            
            optimizer.zero_grad()
            logits, _ = behavioral_policy(states)
            loss = F.cross_entropy(logits, actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        logger.info(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/n_batches:.4f}")
    
    behavioral_policy.eval()
    return behavioral_policy

def train_q_function(train_df, val_df, state_cols):
    """Train Q-function Q(s,a) using XGBoost."""
    logger.info("\nTraining Q-function...")
    
    # Prepare features: flattened states + action
    X_train = []
    y_train = []
    
    for idx in range(len(train_df)):
        row = train_df.iloc[idx]
        # Flatten state
        state_flat = []
        for t in range(4):
            for col in state_cols:
                state_flat.append(row.get(f'{col}_t{t}', 0))
        # Add action
        state_flat.append(row['action_idx'])
        X_train.append(state_flat)
        y_train.append(row['reward_shaped'])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train, verbose=False)
    
    logger.info("  ✓ Q-function trained")
    return model

def compute_doubly_robust_value(test_df, policy, behavioral_policy, q_function, state_cols, device='cpu'):
    """Compute doubly robust policy value estimate."""
    logger.info("\nComputing doubly robust estimate...")
    
    states_np = prepare_states(test_df, state_cols)
    states = torch.FloatTensor(states_np).to(device)
    actions = test_df['action_idx'].values
    rewards = test_df['reward_shaped'].values
    
    # Get policy probabilities π(a|s)
    with torch.no_grad():
        logits_learned, _ = policy(states)
        probs_learned = F.softmax(logits_learned, dim=-1).cpu().numpy()
        
        logits_behavior, _ = behavioral_policy(states)
        probs_behavior = F.softmax(logits_behavior, dim=-1).cpu().numpy()
    
    # Get Q-values
    X_test = []
    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        state_flat = []
        for t in range(4):
            for col in state_cols:
                state_flat.append(row.get(f'{col}_t{t}', 0))
        state_flat.append(row['action_idx'])
        X_test.append(state_flat)
    X_test = np.array(X_test)
    q_values = q_function.predict(X_test)
    
    # Compute importance weights
    action_probs_learned = probs_learned[np.arange(len(actions)), actions]
    action_probs_behavior = probs_behavior[np.arange(len(actions)), actions]
    importance_weights = action_probs_learned / (action_probs_behavior + 1e-8)
    
    # Truncate weights at 95th percentile
    truncation_threshold = np.percentile(importance_weights, 95)
    importance_weights_truncated = np.minimum(importance_weights, truncation_threshold)
    
    # Doubly robust estimate
    dr_terms = importance_weights_truncated * (rewards - q_values) + q_values
    dr_value = np.mean(dr_terms)
    dr_std = np.std(dr_terms) / np.sqrt(len(dr_terms))
    
    # Effective sample size
    ess = (np.sum(importance_weights_truncated)**2) / np.sum(importance_weights_truncated**2)
    ess_pct = (ess / len(test_df)) * 100
    
    # Weighted importance sampling (for comparison)
    wis_value = np.sum(importance_weights * rewards) / np.sum(importance_weights)
    wis_ess = (np.sum(importance_weights)**2) / np.sum(importance_weights**2)
    wis_ess_pct = (wis_ess / len(test_df)) * 100
    
    # Behavioral policy value
    behavioral_value = np.mean(rewards)
    behavioral_std = np.std(rewards) / np.sqrt(len(rewards))
    
    results = {
        'doubly_robust': {
            'value': float(dr_value),
            'std': float(dr_std),
            'ci_lower': float(dr_value - 1.96 * dr_std),
            'ci_upper': float(dr_value + 1.96 * dr_std),
            'ess': float(ess),
            'ess_pct': float(ess_pct)
        },
        'wis': {
            'value': float(wis_value),
            'ess': float(wis_ess),
            'ess_pct': float(wis_ess_pct)
        },
        'behavioral': {
            'value': float(behavioral_value),
            'std': float(behavioral_std),
            'ci_lower': float(behavioral_value - 1.96 * behavioral_std),
            'ci_upper': float(behavioral_value + 1.96 * behavioral_std)
        },
        'importance_weights_stats': {
            'mean': float(np.mean(importance_weights)),
            'median': float(np.median(importance_weights)),
            'p95': float(np.percentile(importance_weights, 95)),
            'p99': float(np.percentile(importance_weights, 99)),
            'max': float(np.max(importance_weights))
        }
    }
    
    return results

def main():
    logger.info("="*80)
    logger.info("PHASE 3: DOUBLY ROBUST OFF-POLICY EVALUATION")
    logger.info("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Paths
    data_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data_10pct')
    model_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/models')
    results_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\nLoading data...")
    train_df = pd.read_parquet(data_dir / 'sequences_train.parquet')
    val_df = pd.read_parquet(data_dir / 'sequences_val.parquet')
    test_df = pd.read_parquet(data_dir / 'sequences_test.parquet')
    
    logger.info(f"  Train: {len(train_df):,}")
    logger.info(f"  Val: {len(val_df):,}")
    logger.info(f"  Test: {len(test_df):,}")
    
    # Load trained policy
    logger.info("\nLoading trained LSTM policy...")
    policy, value_net, config = load_trained_models(model_dir, device)
    state_cols = config['state_cols']
    num_actions = config['num_actions']
    logger.info(f"  State cols: {state_cols}")
    logger.info(f"  Num actions: {num_actions}")
    
    # Sample for faster evaluation (use 10% of test set)
    test_sample = test_df.sample(min(10000, len(test_df)), random_state=42)
    logger.info(f"\nUsing {len(test_sample):,} test sequences for evaluation")
    
    # Train behavioral policy
    behavioral_policy = train_behavioral_policy(
        train_df.sample(min(50000, len(train_df))),
        val_df.sample(min(10000, len(val_df))),
        state_cols, num_actions, device
    )
    
    # Train Q-function
    q_function = train_q_function(
        train_df.sample(min(50000, len(train_df))),
        val_df.sample(min(10000, len(val_df))),
        state_cols
    )
    
    # Compute doubly robust estimate
    results = compute_doubly_robust_value(
        test_sample, policy, behavioral_policy, q_function, state_cols, device
    )
    
    # Display results
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    
    logger.info(f"\nBehavioral Policy:")
    logger.info(f"  Value: {results['behavioral']['value']:.4f}")
    logger.info(f"  95% CI: [{results['behavioral']['ci_lower']:.4f}, {results['behavioral']['ci_upper']:.4f}]")
    
    logger.info(f"\nLearned Policy (Doubly Robust):")
    logger.info(f"  Value: {results['doubly_robust']['value']:.4f}")
    logger.info(f"  95% CI: [{results['doubly_robust']['ci_lower']:.4f}, {results['doubly_robust']['ci_upper']:.4f}]")
    logger.info(f"  ESS: {results['doubly_robust']['ess']:.0f} ({results['doubly_robust']['ess_pct']:.1f}%)")
    
    logger.info(f"\nLearned Policy (WIS - for comparison):")
    logger.info(f"  Value: {results['wis']['value']:.4f}")
    logger.info(f"  ESS: {results['wis']['ess']:.0f} ({results['wis']['ess_pct']:.1f}%)")
    
    logger.info(f"\nImportance Weights:")
    logger.info(f"  Mean: {results['importance_weights_stats']['mean']:.2f}")
    logger.info(f"  Median: {results['importance_weights_stats']['median']:.2f}")
    logger.info(f"  95th percentile: {results['importance_weights_stats']['p95']:.2f}")
    logger.info(f"  Max: {results['importance_weights_stats']['max']:.2f}")
    
    # Save results
    with open(results_dir / 'ope_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {results_dir / 'ope_results.json'}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ EVALUATION COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    main()
