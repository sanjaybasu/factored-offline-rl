"""
FIXED: Phase 1 Fast - Properly Serialize Arrays
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("PHASE 1 FAST: 10% DATA SAMPLE (FIXED)")
    logger.info("="*80)
    
    output_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading trajectories...")
    df = pd.read_parquet(
        '/Users/sanjaybasu/waymark-local/foundation/outputs/rl_factored_pipeline_full_real_rewards/merged_trajectories.parquet'
    )
    
    logger.info(f"Full dataset: {len(df):,} observations from {df['member_id'].nunique():,} members")
    
    # Sample 10% of members
    np.random.seed(42)
    all_members = df['member_id'].unique()
    sample_members = np.random.choice(all_members, size=int(len(all_members) * 0.1), replace=False)
    
    df = df[df['member_id'].isin(sample_members)]
    logger.info(f"Sampled 10%: {len(df):,} observations from {df['member_id'].nunique():,} members")
    
    df['timestep_date'] = pd.to_datetime(df['timestep_date'])
    df = df.sort_values(['member_id', 'timestep_date'])
    
    # Reward shaping
    logger.info("Adding reward shaping...")
    df['reward_primary'] = -(df['future_ed_30d'] + 2 * df['future_ip_30d'])
    df['reward_engagement'] = df['num_interventions_last_week'] * 0.1
    df['reward_cost'] = -df['future_total_paid_90d'] / 10000.0
    df['reward_shaped'] = (
        1.0 * df['reward_primary'] +
        0.3 * df['reward_engagement'] +
        0.2 * df['reward_cost']
    )
    
    logger.info(f"Reward shaped - mean: {df['reward_shaped'].mean():.4f}, non-zero: {(df['reward_shaped'] != 0).mean()*100:.2f}%")
    
    # Create sequences
    logger.info("Creating 4-week sequences...")
    
    sequence_length = 4
    sequences = []
    
    for member_id, member_data in df.groupby('member_id'):
        member_data = member_data.reset_index(drop=True)
        
        if len(member_data) < sequence_length:
            continue
        
        for i in range(sequence_length - 1, len(member_data)):
            window = member_data.iloc[i - sequence_length + 1:i + 1]
            
            # Check dates
            date_diffs = window['timestep_date'].diff().dt.days.dropna()
            if (date_diffs < 3).any() or (date_diffs > 14).any():
                continue
            
            # Flatten states into individual columns instead of array
            state_dict = {}
            for t in range(sequence_length):
                obs = member_data.iloc[i - sequence_length + 1 + t]
                state_dict[f'age_t{t}'] = obs['age']
                state_dict[f'encounters_t{t}'] = obs['num_encounters_last_week']
                state_dict[f'ed_t{t}'] = obs['num_ed_visits_last_week']
                state_dict[f'ip_t{t}'] = obs['num_ip_visits_last_week']
                state_dict[f'interventions_t{t}'] = obs['num_interventions_last_week']
                state_dict[f'enrolled_days_t{t}'] = obs['enrolled_days']
                state_dict[f'total_paid_t{t}'] = obs['total_paid']
            
            current = member_data.iloc[i]
            
            sequences.append({
                'member_id': member_id,
                'end_date': current['timestep_date'],
                **state_dict,  # Unpack flattened states
                'action_tuple': str(current['action_tuple']),  # Convert tuple to string
                'reward_shaped': current['reward_shaped'],
                'reward_primary': current['reward_primary'],
                'gender': current['gender'],
                'race': current['race'],
                'future_ed_30d': current['future_ed_30d'],
                'future_ip_30d': current['future_ip_30d'],
            })
        
        if len(sequences) % 10000 == 0 and len(sequences) > 0:
            logger.info(f"  Created {len(sequences):,} sequences...")
    
    sequence_df = pd.DataFrame(sequences)
    sequence_df['sequence_id'] = range(len(sequence_df))
    
    logger.info(f"Created {len(sequence_df):,} total sequences")
    
    # Split data
    logger.info("Splitting into train/val/test...")
    
    members = sequence_df['member_id'].unique()
    np.random.shuffle(members)
    
    n_train = int(len(members) * 0.5)
    n_val = int(len(members) * 0.25)
    
    train_members = members[:n_train]
    val_members = members[n_train:n_train + n_val]
    test_members = members[n_train + n_val:]
    
    train_df = sequence_df[sequence_df['member_id'].isin(train_members)]
    val_df = sequence_df[sequence_df['member_id'].isin(val_members)]
    test_df = sequence_df[sequence_df['member_id'].isin(test_members)]
    
    logger.info(f"  Train: {len(train_df):,} sequences from {len(train_members):,} members")
    logger.info(f"  Val: {len(val_df):,} sequences from {len(val_members):,} members")
    logger.info(f"  Test: {len(test_df):,} sequences from {len(test_members):,} members")
    
    # Save
    logger.info("Saving datasets...")
    
    train_df.to_parquet(output_dir / 'sequences_train.parquet', compression='snappy', index=False)
    val_df.to_parquet(output_dir / 'sequences_val.parquet', compression='snappy', index=False)
    test_df.to_parquet(output_dir / 'sequences_test.parquet', compression='snappy', index=False)
    
    logger.info(f"\n✓ Datasets saved to {output_dir}")
    logger.info(f"  Total size: {(train_df.memory_usage(deep=True).sum() + val_df.memory_usage(deep=True).sum() + test_df.memory_usage(deep=True).sum()) / 1e9:.2f} GB")
    
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total sequences: {len(sequence_df):,}")
    logger.info(f"Features per sequence: {len([c for c in sequence_df.columns if c.endswith(('_t0', '_t1', '_t2', '_t3'))])}")
    logger.info(f"Outcome rate (test): ED={test_df['future_ed_30d'].mean():.4f}, IP={test_df['future_ip_30d'].mean():.4f}")
    logger.info("\n✓ PHASE 1 FAST COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
