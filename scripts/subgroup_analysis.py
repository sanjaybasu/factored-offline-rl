"""
Subgroup Analysis - Fairness & Equity
======================================

Analyze policy performance across demographic subgroups:
1. Race/Ethnicity
2. Gender
3. Age Groups

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
        logging.FileHandler('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/subgroup_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.append('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/scripts')

def load_best_model(device='cpu'):
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

def compute_dr_value(df, policy, behavioral_policy, q_function, state_cols, device='cpu'):
    """Compute Doubly Robust value for a specific subgroup."""
    if len(df) == 0:
        return None
        
    states_np = prepare_states(df, state_cols)
    states = torch.FloatTensor(states_np).to(device)
    actions = df['action_idx'].values
    rewards = df['reward_shaped'].values
    
    with torch.no_grad():
        logits_learned, _ = policy(states)
        probs_learned = F.softmax(logits_learned, dim=-1).cpu().numpy()
        
        logits_behavior, _ = behavioral_policy(states)
        probs_behavior = F.softmax(logits_behavior, dim=-1).cpu().numpy()
    
    # Q-values
    X_test = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        state_flat = []
        for t in range(4):
            for col in state_cols:
                state_flat.append(row.get(f'{col}_t{t}', 0))
        state_flat.append(row['action_idx'])
        X_test.append(state_flat)
    q_values = q_function.predict(np.array(X_test))
    
    # Importance weights
    action_probs_learned = probs_learned[np.arange(len(actions)), actions]
    action_probs_behavior = probs_behavior[np.arange(len(actions)), actions]
    weights = action_probs_learned / (action_probs_behavior + 1e-8)
    
    # Truncate at 95th percentile (global)
    weights = np.minimum(weights, 20.0) # Hard cap for stability in small subgroups
    
    dr_terms = weights * (rewards - q_values) + q_values
    dr_value = np.mean(dr_terms)
    
    ess = (np.sum(weights)**2) / np.sum(weights**2)
    ess_pct = (ess / len(df)) * 100
    
    return {
        'n': len(df),
        'dr_value': float(dr_value),
        'ess_pct': float(ess_pct)
    }

def train_components(train_df, state_cols, num_actions, device='cpu'):
    from phase2_train_full import LSTMPolicyNetwork
    
    logger.info("Training behavioral policy...")
    behavioral_policy = LSTMPolicyNetwork(len(state_cols), 64, num_actions).to(device)
    optimizer = torch.optim.Adam(behavioral_policy.parameters(), lr=3e-4)
    
    batch_size = 512
    for epoch in range(3):
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

def main():
    logger.info("="*80)
    logger.info("SUBGROUP ANALYSIS - FAIRNESS")
    logger.info("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data_factored')
    results_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/results')
    
    # Load data
    train_df = pd.read_parquet(data_dir / 'sequences_train.parquet').sample(min(20000, 484504))
    test_df = pd.read_parquet(data_dir / 'sequences_test.parquet') # Use full test set for subgroups
    
    state_cols = ['age', 'num_encounters_last_week', 'num_ed_visits_last_week', 
                  'num_ip_visits_last_week', 'num_interventions_last_week']
    
    # Load model
    policy, config = load_best_model(device)
    num_actions = config['num_actions']
    
    # Filter test data
    test_df = test_df[test_df['action_idx'] < num_actions].copy()
    
    # Train components
    behavioral_policy, q_function = train_components(train_df, state_cols, num_actions, device)
    
    results = {}
    
    # 1. Race/Ethnicity
    logger.info("\nAnalyzing Race/Ethnicity subgroups...")
    race_results = {}
    for race in test_df['race'].unique():
        subgroup = test_df[test_df['race'] == race]
        res = compute_dr_value(subgroup, policy, behavioral_policy, q_function, state_cols, device)
        if res:
            race_results[str(race)] = res
            logger.info(f"  {race}: n={res['n']}, Value={res['dr_value']:.4f}, ESS={res['ess_pct']:.1f}%")
    results['race'] = race_results
    
    # 2. Gender
    logger.info("\nAnalyzing Gender subgroups...")
    gender_results = {}
    for gender in test_df['gender'].unique():
        subgroup = test_df[test_df['gender'] == gender]
        res = compute_dr_value(subgroup, policy, behavioral_policy, q_function, state_cols, device)
        if res:
            gender_results[str(gender)] = res
            logger.info(f"  {gender}: n={res['n']}, Value={res['dr_value']:.4f}, ESS={res['ess_pct']:.1f}%")
    results['gender'] = gender_results
    
    # Save results
    with open(results_dir / 'subgroup_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info("\n" + "="*80)
    logger.info("✓ SUBGROUP ANALYSIS COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    main()
