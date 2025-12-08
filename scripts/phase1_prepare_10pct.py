"""
Phase 1: Full Data Preparation (10% Sample)
============================================

Prepare 10% of members for production LSTM training.

Author: Sanjay Basu, MD PhD
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/phase1_10pct.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info("PHASE 1: DATA PREPARATION (10% SAMPLE)")
    logger.info("="*80)
    
    # Paths
    data_path = Path('/Users/sanjaybasu/waymark-local/foundation/outputs/rl_factored_pipeline_full_real_rewards/merged_trajectories.parquet')
    output_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data_10pct')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading full dataset...")
    df = pd.read_parquet(data_path)
    logger.info(f"  Total: {len(df):,} observations from {df['member_id'].nunique():,} members")
    
    # Sample 10% of members
    np.random.seed(42)
    all_members = df['member_id'].unique()
    sample_size = int(len(all_members) * 0.10)
    sampled_members = np.random.choice(all_members, size=sample_size, replace=False)
    
    df_sample = df[df['member_id'].isin(sampled_members)].copy()
    logger.info(f"  Sampled: {len(df_sample):,} observations from {len(sampled_members):,} members (10%)")
    
    # Reward shaping
    logger.info("Computing shaped rewards...")
    df_sample['reward_primary'] = -(df_sample['future_ed_30d'] + 2 * df_sample['future_ip_30d'])
    df_sample['reward_engagement'] = df_sample['num_interventions_last_week'] * 0.1
    df_sample['reward_cost'] = -df_sample['future_total_paid_90d'] / 10000.0
    df_sample['reward_shaped'] = (
        1.0 * df_sample['reward_primary'] +
        0.3 * df_sample['reward_engagement'] +
        0.2 * df_sample['reward_cost']
    )
    
    non_zero_pct = (df_sample['reward_shaped'] != 0).mean() * 100
    logger.info(f"  Non-zero rewards: {non_zero_pct:.1f}%")
    
    # Create sequences
    logger.info("Creating 4-week sequences...")
    
    sequences = []
    seq_length = 4
    
    for member_id in sampled_members:
        member_df = df_sample[df_sample['member_id'] == member_id].sort_values('timestep_date')
        
        if len(member_df) < seq_length:
            continue
        
        # Get state features (use actual available columns)
        state_cols = ['age', 'num_encounters_last_week', 'num_ed_visits_last_week', 
                     'num_ip_visits_last_week', 'num_interventions_last_week']
        
        # Create sequences
        for i in range(len(member_df) - seq_length + 1):
            seq = member_df.iloc[i:i+seq_length]
            
            # Flatten state features across timesteps
            seq_data = {'member_id': member_id}
            
            for t in range(seq_length):
                row = seq.iloc[t]
                for col in state_cols:
                    seq_data[f'{col}_t{t}'] = row[col]
            
            # Add action and reward from last timestep
            last_row = seq.iloc[-1]
            seq_data['action'] = last_row.get('action_bin', 0)
            seq_data['reward_shaped'] = last_row['reward_shaped']
            seq_data['future_ed_30d'] = last_row['future_ed_30d']
            seq_data['future_ip_30d'] = last_row['future_ip_30d']
            
            # Add demographics for fairness
            seq_data['race'] = last_row.get('race', 'unknown')
            seq_data['gender'] = last_row.get('gender', 'unknown')
            
            sequences.append(seq_data)
    
    logger.info(f"  Created {len(sequences):,} sequences")
    
    # Convert to DataFrame
    sequences_df = pd.DataFrame(sequences)
    
    # Encode actions
    logger.info("Encoding actions...")
    action_map = {action: idx for idx, action in enumerate(sequences_df['action'].unique())}
    sequences_df['action_idx'] = sequences_df['action'].map(action_map)
    logger.info(f"  Unique actions: {len(action_map)}")
    
    # Split into train/val/test (50/25/25)
    np.random.seed(42)
    n = len(sequences_df)
    indices = np.random.permutation(n)
    
    train_size = int(0.5 * n)
    val_size = int(0.25 * n)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    
    train_df = sequences_df.iloc[train_idx].copy()
    val_df = sequences_df.iloc[val_idx].copy()
    test_df = sequences_df.iloc[test_idx].copy()
    
    logger.info(f"  Train: {len(train_df):,} sequences")
    logger.info(f"  Val: {len(val_df):,} sequences")
    logger.info(f"  Test: {len(test_df):,} sequences")
    
    # Save
    logger.info("Saving sequences...")
    train_df.to_parquet(output_dir / 'sequences_train.parquet')
    val_df.to_parquet(output_dir / 'sequences_val.parquet')
    test_df.to_parquet(output_dir / 'sequences_test.parquet')
    
    # Save action mapping
    import json
    with open(output_dir / 'action_map.json', 'w') as f:
        json.dump({str(k): v for k, v in action_map.items()}, f)
    
    logger.info(f"✓ Data saved to {output_dir}")
    logger.info("="*80)
    logger.info("✓ PHASE 1 COMPLETE (10% SAMPLE)")
    logger.info("="*80)

if __name__ == "__main__":
    main()
