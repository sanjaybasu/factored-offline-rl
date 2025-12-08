"""
Compare All Models - Doubly Robust Evaluation
==============================================

Evaluate feedforward, LSTM-no-attn, and LSTM+attn using doubly robust OPE.

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
        logging.FileHandler('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/model_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import model architectures
import sys
sys.path.append('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/scripts')

def load_model(model_path, model_type, device='cpu'):
    """Load a trained model."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    if model_type == 'lstm_attention':
        from phase2_train_full import LSTMPolicyNetwork, LSTMValueNetwork
        policy = LSTMPolicyNetwork(
            config['state_dim'], config['hidden_dim'], config['num_actions']
        ).to(device)
        value_net = LSTMValueNetwork(
            config['state_dim'], config['hidden_dim']
        ).to(device)
    elif model_type == 'feedforward':
        from train_feedforward_baseline import FeedforwardPolicyNetwork, FeedforwardValueNetwork
        policy = FeedforwardPolicyNetwork(
            config['state_dim'], config['hidden_dim'], config['num_actions']
        ).to(device)
        value_net = FeedforwardValueNetwork(
            config['state_dim'], config['hidden_dim']
        ).to(device)
    elif model_type == 'lstm_noattention':
        from train_lstm_noattention import LSTMPolicyNoAttention, LSTMValueNoAttention
        policy = LSTMPolicyNoAttention(
            config['state_dim'], config['hidden_dim'], config['num_actions']
        ).to(device)
        value_net = LSTMValueNoAttention(
            config['state_dim'], config['hidden_dim']
        ).to(device)
    
    policy.load_state_dict(checkpoint['policy_state_dict'])
    value_net.load_state_dict(checkpoint['value_state_dict'])
    policy.eval()
    value_net.eval()
    
    return policy, value_net, config

def prepare_states(df, state_cols, seq_len=4):
    """Prepare state sequences."""
    states = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        state_seq = []
        for t in range(seq_len):
            state_t = [float(row.get(f'{col}_t{t}', 0)) for col in state_cols]
            state_seq.append(state_t)
        states.append(state_seq)
    return np.array(states)

def train_behavioral_policy(train_df, state_cols, num_actions, device='cpu'):
    """Train behavioral policy."""
    from phase2_train_full import LSTMPolicyNetwork
    
    logger.info("Training behavioral policy...")
    behavioral_policy = LSTMPolicyNetwork(
        len(state_cols), 64, num_actions
    ).to(device)
    
    optimizer = torch.optim.Adam(behavioral_policy.parameters(), lr=3e-4)
    
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

def train_q_function(train_df, state_cols):
    """Train Q-function."""
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
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
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
    """Compute doubly robust policy value."""
    states_np = prepare_states(test_df, state_cols)
    states = torch.FloatTensor(states_np).to(device)
    actions = test_df['action_idx'].values
    rewards = test_df['reward_shaped'].values
    
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
    
    # Importance weights
    action_probs_learned = probs_learned[np.arange(len(actions)), actions]
    action_probs_behavior = probs_behavior[np.arange(len(actions)), actions]
    importance_weights = action_probs_learned / (action_probs_behavior + 1e-8)
    
    # Truncate
    truncation_threshold = np.percentile(importance_weights, 95)
    importance_weights_truncated = np.minimum(importance_weights, truncation_threshold)
    
    # Doubly robust
    dr_terms = importance_weights_truncated * (rewards - q_values) + q_values
    dr_value = np.mean(dr_terms)
    dr_std = np.std(dr_terms) / np.sqrt(len(dr_terms))
    
    # ESS
    ess = (np.sum(importance_weights_truncated)**2) / np.sum(importance_weights_truncated**2)
    ess_pct = (ess / len(test_df)) * 100
    
    # WIS
    wis_value = np.sum(importance_weights * rewards) / np.sum(importance_weights)
    wis_ess = (np.sum(importance_weights)**2) / np.sum(importance_weights**2)
    wis_ess_pct = (wis_ess / len(test_df)) * 100
    
    return {
        'dr_value': float(dr_value),
        'dr_std': float(dr_std),
        'dr_ci_lower': float(dr_value - 1.96 * dr_std),
        'dr_ci_upper': float(dr_value + 1.96 * dr_std),
        'dr_ess': float(ess),
        'dr_ess_pct': float(ess_pct),
        'wis_value': float(wis_value),
        'wis_ess_pct': float(wis_ess_pct),
        'importance_weights_mean': float(np.mean(importance_weights)),
        'importance_weights_max': float(np.max(importance_weights))
    }

def main():
    logger.info("="*80)
    logger.info("MODEL COMPARISON - DOUBLY ROBUST EVALUATION")
    logger.info("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Paths
    data_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data_factored')
    model_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/models')
    results_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\nLoading data...")
    train_df = pd.read_parquet(data_dir / 'sequences_train.parquet')
    test_df = pd.read_parquet(data_dir / 'sequences_test.parquet')
    
    # Sample for faster evaluation
    test_sample = test_df.sample(min(10000, len(test_df)), random_state=42)
    logger.info(f"Using {len(test_sample):,} test sequences")
    
    state_cols = ['age', 'num_encounters_last_week', 'num_ed_visits_last_week', 
                  'num_ip_visits_last_week', 'num_interventions_last_week']
    num_actions = int(train_df['action_idx'].max() + 1)
    
    # Train shared components
    behavioral_policy = train_behavioral_policy(
        train_df.sample(min(50000, len(train_df))),
        state_cols, num_actions, device
    )
    
    q_function = train_q_function(
        train_df.sample(min(50000, len(train_df))),
        state_cols
    )
    
    # Evaluate all models
    models = {
        'feedforward': (model_dir / 'feedforward_baseline_best.pt', 'feedforward'),
        'lstm_noattention': (model_dir / 'lstm_noattention_best.pt', 'lstm_noattention'),
        'lstm_attention': (model_dir / 'lstm_policy_10pct_best.pt', 'lstm_attention')
    }
    
    results = {}
    
    for model_name, (model_path, model_type) in models.items():
        if not model_path.exists():
            logger.info(f"\n⚠️ Skipping {model_name}: model not found")
            continue
            
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*80}")
        
        policy, value_net, config = load_model(model_path, model_type, device)
        
        results[model_name] = compute_doubly_robust_value(
            test_sample, policy, behavioral_policy, q_function, state_cols, device
        )
        
        logger.info(f"\nResults for {model_name}:")
        logger.info(f"  DR Value: {results[model_name]['dr_value']:.4f}")
        logger.info(f"  DR ESS: {results[model_name]['dr_ess_pct']:.1f}%")
        logger.info(f"  WIS ESS: {results[model_name]['wis_ess_pct']:.1f}%")
    
    # Save results
    with open(results_dir / 'model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("✓ MODEL COMPARISON COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to {results_dir / 'model_comparison.json'}")

if __name__ == "__main__":
    main()
