"""
Phase 1: Prepare Enhanced Dataset for V3 LSTM Training
=======================================================

Adapts existing trajectory data for LSTM-based AWR training with:
- 4-week sequences for temporal modeling
- Multi-component reward shaping
- Demographics for fairness monitoring
- Train/val/test splits

Author: Sanjay Basu, MD PhD
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_prepare_data():
    """Load existing trajectory data and prepare for v3."""
    logger.info("Loading trajectories...")
    
    df = pd.read_parquet(
        '/Users/sanjaybasu/waymark-local/foundation/outputs/rl_factored_pipeline_full_real_rewards/merged_trajectories.parquet'
    )
    
    logger.info(f"Loaded {len(df):,} observations from {df['member_id'].nunique():,} members")
    
    # Convert timestep_date to datetime
    df['timestep_date'] = pd.to_datetime(df['timestep_date'])
    
    # Sort by member and date
    df = df.sort_values(['member_id', 'timestep_date'])
    
    return df


def add_reward_shaping(df):
    """
    Add multi-component reward shaping.
    
    Components:
    1. Primary: -(ED + 2*IP)
    2. Engagement: successful contacts
    3. Cost: negative total costs (scaled)
    """
    logger.info("Adding reward shaping...")
    
    # Primary outcome reward (already have future_ed_30d, future_ip_30d)
    df['reward_primary'] = -(df['future_ed_30d'] + 2 * df['future_ip_30d'])
    
    # Engagement reward (based on interventions)
    df['reward_engagement'] = df['num_interventions_last_week'] * 0.1  # Small bonus for engagement
    
    # Cost component (scale total paid to reasonable range)
    df['reward_cost'] = -df['future_total_paid_90d'] / 10000.0  # Scale to ~[-1, 0] range
    
    # Combined shaped reward
    df['reward_shaped'] = (
        1.0 * df['reward_primary'] +
        0.3 * df['reward_engagement'] +
        0.2 * df['reward_cost']
    )
    
    logger.info(f"Reward stats:")
    logger.info(f"  Primary mean: {df['reward_primary'].mean():.4f}")
    logger.info(f"  Engagement mean: {df['reward_engagement'].mean():.4f}")
    logger.info(f"  Cost mean: {df['reward_cost'].mean():.4f}")
    logger.info(f"  Shaped mean: {df['reward_shaped'].mean():.4f}")
    logger.info(f"  Non-zero shaped: {(df['reward_shaped'] != 0).mean()*100:.2f}%")
    
    return df


def create_lstm_sequences(df, sequence_length=4):
    """
    Create 4-week sequences for LSTM training.
    
    For each timestep t, create sequence [t-3, t-2, t-1, t].
    """
    logger.info(f"Creating {sequence_length}-week sequences...")
    
    sequences = []
    sequence_id = 0
    
    for member_id, member_data in df.groupby('member_id'):
        member_data = member_data.reset_index(drop=True)
        
        if len(member_data) < sequence_length:
            continue
        
        # Sliding window
        for i in range(sequence_length - 1, len(member_data)):
            window = member_data.iloc[i - sequence_length + 1:i + 1]
            
            # Check dates are roughly weekly (allow some flexibility)
            date_diffs = window['timestep_date'].diff().dt.days.dropna()
            if (date_diffs < 3).any() or (date_diffs > 14).any():
                continue  # Skip if dates are too close or too far apart
            
            sequence_data = {
                'sequence_id': sequence_id,
                'member_id': member_id,
                'end_date': member_data.iloc[i]['timestep_date'],
            }
            
            # Pack sequence states
            for t in range(sequence_length):
                obs = member_data.iloc[i - sequence_length + 1 + t]
                sequence_data[f'state_t{t}'] = {
                    'age': obs['age'],
                    'num_encounters': obs['num_encounters_last_week'],
                    'num_ed_visits': obs['num_ed_visits_last_week'],
                    'num_ip_visits': obs['num_ip_visits_last_week'],
                    'num_interventions': obs['num_interventions_last_week'],
                    'enrolled_days': obs['enrolled_days'],
                    'total_paid': obs['total_paid'],
                }
            
            # Current action and reward
            current = member_data.iloc[i]
            sequence_data['action_tuple'] = current['action_tuple']
            sequence_data['reward_shaped'] = current['reward_shaped']
            sequence_data['reward_primary'] = current['reward_primary']
            
            # Demographics for fairness
            sequence_data['gender'] = current['gender']
            sequence_data['race'] = current['race']
            
            # Outcomes for evaluation
            sequence_data['future_ed_30d'] = current['future_ed_30d']
            sequence_data['future_ip_30d'] = current['future_ip_30d']
            
            sequences.append(sequence_data)
            sequence_id += 1
            
            if sequence_id % 100000 == 0:
                logger.info(f"  Created {sequence_id:,} sequences...")
    
    sequence_df = pd.DataFrame(sequences)
    
    logger.info(f"Created {len(sequence_df):,} sequences from {sequence_df['member_id'].nunique():,} members")
    
    return sequence_df


def split_data(df, train_frac=0.5, val_frac=0.25, test_frac=0.25, seed=42):
    """Split at member level to avoid leakage."""
    logger.info("Splitting into train/val/test...")
    
    members = df['member_id'].unique()
    np.random.seed(seed)
    np.random.shuffle(members)
    
    n_train = int(len(members) * train_frac)
    n_val = int(len(members) * val_frac)
    
    train_members = members[:n_train]
    val_members = members[n_train:n_train + n_val]
    test_members = members[n_train + n_val:]
    
    train_df = df[df['member_id'].isin(train_members)]
    val_df = df[df['member_id'].isin(val_members)]
    test_df = df[df['member_id'].isin(test_members)]
    
    logger.info(f"  Train: {len(train_df):,} sequences from {len(train_members):,} members")
    logger.info(f"  Val: {len(val_df):,} sequences from {len(val_members):,} members")
    logger.info(f"  Test: {len(test_df):,} sequences from {len(test_members):,} members")
    
    return train_df, val_df, test_df


def main():
    logger.info("="*80)
    logger.info("PHASE 1: DATA PREPARATION FOR V3 LSTM TRAINING")
    logger.info("="*80)
    
    # Output directory
    output_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    df = load_and_prepare_data()
    
    # Step 2: Add reward shaping
    df = add_reward_shaping(df)
    
    # Step 3: Create LSTM sequences
    sequence_df = create_lstm_sequences(df, sequence_length=4)
    
    # Step 4: Split data
    train_df, val_df, test_df = split_data(sequence_df)
    
    # Step 5: Save
    logger.info("\nSaving datasets...")
    
    train_df.to_parquet(output_dir / 'sequences_train.parquet', compression='snappy', index=False)
    val_df.to_parquet(output_dir / 'sequences_val.parquet', compression='snappy', index=False)
    test_df.to_parquet(output_dir / 'sequences_test.parquet', compression='snappy', index=False)
    
    logger.info(f"\n✓ Datasets saved to {output_dir}")
    logger.info(f"  Train: {len(train_df):,} sequences")
    logger.info(f"  Val: {len(val_df):,} sequences")
    logger.info(f"  Test: {len(test_df):,} sequences")
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    logger.info(f"\nReward distribution (train set):")
    logger.info(f"  Mean: {train_df['reward_shaped'].mean():.4f}")
    logger.info(f"  Std: {train_df['reward_shaped'].std():.4f}")
    logger.info(f"  Min: {train_df['reward_shaped'].min():.4f}")
    logger.info(f"  Max: {train_df['reward_shaped'].max():.4f}")
    logger.info(f"  Non-zero: {(train_df['reward_shaped'] != 0).mean()*100:.2f}%")
    
    logger.info(f"\nOutcome distribution (test set):")
    logger.info(f"  ED mean: {test_df['future_ed_30d'].mean():.4f}")
    logger.info(f"  IP mean: {test_df['future_ip_30d'].mean():.4f}")  
    logger.info(f"  Any acute: {((test_df['future_ed_30d'] > 0) | (test_df['future_ip_30d'] > 0)).mean()*100:.2f}%")
    
    logger.info("\n" + "="*80)
    logger.info("✓ PHASE 1 COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
