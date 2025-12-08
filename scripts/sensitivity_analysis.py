"""
Sensitivity Analysis - Doubly Robust OPE
=========================================

Test robustness of Doubly Robust OPE to:
1. Q-function estimation error (bias)
2. Propensity model misspecification (noise)
3. Weight truncation thresholds

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
import xgboost as xgb
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/sensitivity_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.append('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/scripts')

def load_best_model(device='cpu'):
    """Load the best LSTM+Attention model."""
    model_path = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/models/lstm_policy_10pct_best.pt')
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    from phase2_train_full import LSTMPolicyNetwork
    policy = LSTMPolicyNetwork(
        config['state_dim'], config['hidden_dim'], config['num_actions']
    ).to(device)
    
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    return policy, config

def prepare_states(df, state_cols, seq_len=4):
    states = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        state_seq = []
        for t in range(seq_len):
            state_t = [float(row.get(f'{col}_t{t}', 0)) for col in state_cols]
            state_seq.append(state_t)
        states.append(state_seq)
    return np.array(states)

def train_components(train_df, state_cols, num_actions, device='cpu'):
    """Train behavioral policy and Q-function."""
    from phase2_train_full import LSTMPolicyNetwork
    
    # Behavioral Policy
    logger.info("Training behavioral policy...")
    behavioral_policy = LSTMPolicyNetwork(len(state_cols), 64, num_actions).to(device)
    optimizer = torch.optim.Adam(behavioral_policy.parameters(), lr=3e-4)
    
    batch_size = 512
    for epoch in range(3): # Quick training
        behavioral_policy.train()
        shuffled = train_df.sample(frac=1)
        for i in range(0, len(shuffled), batch_size):
            batch_df = shuffled.iloc[i:i+batch_size]
            if len(batch_df) < batch_size // 2: continue
            
            states = torch.FloatTensor(prepare_states(batch_df, state_cols)).to(device)
            actions = torch.LongTensor(batch_df['action_idx'].values).to(device)
            
            optimizer.zero_grad()
            logits, _ = behavioral_policy(states)
            loss = F.cross_entropy(logits, actions)
            loss.backward()
            optimizer.step()
    behavioral_policy.eval()
    
    # Q-function
    logger.info("Training Q-function...")
    X_train = []
    y_train = []
    for idx in range(len(train_df)):
        row = train_df.iloc[idx]
        state_flat = []
        for t in range(4):
            for col in state_cols:
                state_flat.append(row.get(f'{col}_t{t}', 0))
        state_flat.append(row['action_idx'])
        X_train.append(state_flat)
        y_train.append(row['reward_shaped'])
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(np.array(X_train), np.array(y_train))
    
    return behavioral_policy, model

def run_sensitivity_analysis(test_df, policy, behavioral_policy, q_function, state_cols, device='cpu'):
    """Run sensitivity checks."""
    states_np = prepare_states(test_df, state_cols)
    states = torch.FloatTensor(states_np).to(device)
    actions = test_df['action_idx'].values
    rewards = test_df['reward_shaped'].values
    
    # Base probabilities
    with torch.no_grad():
        logits_learned, _ = policy(states)
        probs_learned = F.softmax(logits_learned, dim=-1).cpu().numpy()
        
        logits_behavior, _ = behavioral_policy(states)
        probs_behavior = F.softmax(logits_behavior, dim=-1).cpu().numpy()
    
    # Base Q-values
    X_test = []
    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        state_flat = []
        for t in range(4):
            for col in state_cols:
                state_flat.append(row.get(f'{col}_t{t}', 0))
        state_flat.append(row['action_idx'])
        X_test.append(state_flat)
    base_q_values = q_function.predict(np.array(X_test))
    
    results = {}
    
    # 1. Q-function Bias
    logger.info("\nTesting Q-function bias sensitivity...")
    biases = [-0.2, -0.1, 0.0, 0.1, 0.2]
    q_results = []
    
    for bias in biases:
        q_biased = base_q_values * (1 + bias)
        
        # Calculate DR
        action_probs_learned = probs_learned[np.arange(len(actions)), actions]
        action_probs_behavior = probs_behavior[np.arange(len(actions)), actions]
        weights = action_probs_learned / (action_probs_behavior + 1e-8)
        weights = np.minimum(weights, np.percentile(weights, 95))
        
        dr_terms = weights * (rewards - q_biased) + q_biased
        dr_value = np.mean(dr_terms)
        
        q_results.append({
            'bias': bias,
            'dr_value': float(dr_value),
            'pct_change': float((dr_value - results.get('base_dr', dr_value)) / results.get('base_dr', dr_value) * 100) if 'base_dr' in results else 0.0
        })
        if bias == 0.0: results['base_dr'] = float(dr_value)
            
    results['q_sensitivity'] = q_results
    
    # 2. Propensity Misspecification (Noise)
    logger.info("Testing propensity misspecification...")
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    prop_results = []
    
    for noise in noise_levels:
        # Add noise to behavioral logits
        noisy_logits = logits_behavior + torch.randn_like(logits_behavior) * noise
        noisy_probs = F.softmax(noisy_logits, dim=-1).cpu().numpy()
        
        action_probs_behavior_noisy = noisy_probs[np.arange(len(actions)), actions]
        weights = action_probs_learned / (action_probs_behavior_noisy + 1e-8)
        weights = np.minimum(weights, np.percentile(weights, 95))
        
        dr_terms = weights * (rewards - base_q_values) + base_q_values
        dr_value = np.mean(dr_terms)
        
        prop_results.append({
            'noise_std': noise,
            'dr_value': float(dr_value),
            'pct_change': float((dr_value - results['base_dr']) / results['base_dr'] * 100)
        })
        
    results['propensity_sensitivity'] = prop_results
    
    # 3. Truncation Thresholds
    logger.info("Testing truncation thresholds...")
    thresholds = [90, 95, 99, 99.9]
    trunc_results = []
    
    action_probs_learned = probs_learned[np.arange(len(actions)), actions]
    action_probs_behavior = probs_behavior[np.arange(len(actions)), actions]
    raw_weights = action_probs_learned / (action_probs_behavior + 1e-8)
    
    for p in thresholds:
        cutoff = np.percentile(raw_weights, p)
        weights = np.minimum(raw_weights, cutoff)
        
        dr_terms = weights * (rewards - base_q_values) + base_q_values
        dr_value = np.mean(dr_terms)
        
        trunc_results.append({
            'percentile': p,
            'cutoff_value': float(cutoff),
            'dr_value': float(dr_value),
            'pct_change': float((dr_value - results['base_dr']) / results['base_dr'] * 100)
        })
        
    results['truncation_sensitivity'] = trunc_results
    
    return results

def main():
    logger.info("="*80)
    logger.info("SENSITIVITY ANALYSIS")
    logger.info("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    data_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data_factored')
    results_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/results')
    
    # Load data
    train_df = pd.read_parquet(data_dir / 'sequences_train.parquet').sample(min(20000, 484504))
    test_df = pd.read_parquet(data_dir / 'sequences_test.parquet').sample(min(10000, 242253))
    
    state_cols = ['age', 'num_encounters_last_week', 'num_ed_visits_last_week', 
                  'num_ip_visits_last_week', 'num_interventions_last_week']
    num_actions = int(max(train_df['action_idx'].max(), test_df['action_idx'].max()) + 1)
    
    # Load model
    policy, config = load_best_model(device)
    model_num_actions = config['num_actions']
    
    # Filter test data to match model's action space
    test_df = test_df[test_df['action_idx'] < model_num_actions].copy()
    logger.info(f"Filtered test set to {len(test_df)} sequences with actions < {model_num_actions}")
    
    # Use model's num_actions for consistency
    num_actions = model_num_actions
    
    # Train components
    behavioral_policy, q_function = train_components(train_df, state_cols, num_actions, device)
    
    # Run analysis
    results = run_sensitivity_analysis(
        test_df, policy, behavioral_policy, q_function, state_cols, device
    )
    
    # Save results
    with open(results_dir / 'sensitivity_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info("\n" + "="*80)
    logger.info("✓ SENSITIVITY ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to {results_dir / 'sensitivity_results.json'}")

if __name__ == "__main__":
    main()
