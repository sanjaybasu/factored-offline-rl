"""
OPTIMIZED: Phase 1 Data Preparation - Faster Sequence Creation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_member_sequences(args):
    """Process sequences for a single member (for parallel processing)."""
    member_id, member_data, sequence_length = args
    
    sequences = []
    
    if len(member_data) < sequence_length:
        return sequences
    
    member_data = member_data.reset_index(drop=True)
    
    for i in range(sequence_length - 1, len(member_data)):
        window = member_data.iloc[i - sequence_length + 1:i + 1]
        
        # Check dates are roughly weekly
        date_diffs = window['timestep_date'].diff().dt.days.dropna()
        if (date_diffs < 3).any() or (date_diffs > 14).any():
            continue
        
        sequence_data = {
            'member_id': member_id,
            'end_date': member_data.iloc[i]['timestep_date'],
        }
        
        # Pack sequence states (simplified to save memory)
        states = []
        for t in range(sequence_length):
            obs = member_data.iloc[i - sequence_length + 1 + t]
            states.append([
                obs['age'],
                obs['num_encounters_last_week'],
                obs['num_ed_visits_last_week'],
                obs['num_ip_visits_last_week'],
                obs['num_interventions_last_week'],
                obs['enrolled_days'],
                obs['total_paid'],
            ])
        
        sequence_data['states'] = np.array(states, dtype=np.float32)
        
        # Current timestep data
        current = member_data.iloc[i]
        sequence_data['action_tuple'] = current['action_tuple']
        sequence_data['reward_shaped'] = current['reward_shaped']
        sequence_data['reward_primary'] = current['reward_primary']
        sequence_data['gender'] = current['gender']
        sequence_data['race'] = current['race']
        sequence_data['future_ed_30d'] = current['future_ed_30d']
        sequence_data['future_ip_30d'] = current['future_ip_30d']
        
        sequences.append(sequence_data)
    
    return sequences


def main():
    logger.info("="*80)
    logger.info("PHASE 1: OPTIMIZED DATA PREPARATION")
    logger.info("="*80)
    
    output_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading trajectories...")
    df = pd.read_parquet(
        '/Users/sanjaybasu/waymark-local/foundation/outputs/rl_factored_pipeline_full_real_rewards/merged_trajectories.parquet'
    )
    
    logger.info(f"Loaded {len(df):,} observations from {df['member_id'].nunique():,} members")
    
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
    
    logger.info(f"Reward shaped - mean: {df['reward_shaped'].mean():.4f}, "
               f"std: {df['reward_shaped'].std():.4f}, "
               f"non-zero: {(df['reward_shaped'] != 0).mean()*100:.2f}%")
    
    # Create sequences with parallel processing
    logger.info("Creating 4-week sequences (parallel processing)...")
    
    sequence_length = 4
    member_groups = [(member_id, group, sequence_length) 
                    for member_id, group in df.groupby('member_id')]
    
    # Use parallel processing
    n_cores = min(cpu_count(), 8)  # Use up to 8 cores
    logger.info(f"Using {n_cores} cores for parallel processing...")
    
    with Pool(processes=n_cores) as pool:
        results = pool.map(process_member_sequences, member_groups)
    
    # Flatten results
    all_sequences = [seq for member_seqs in results for seq in member_seqs]
    
    logger.info(f"Created {len(all_sequences):,} sequences")
    
    # Convert to dataframe
    sequence_df = pd.DataFrame(all_sequences)
    sequence_df['sequence_id'] = range(len(sequence_df))
    
    # Split data
    logger.info("Splitting into train/val/test...")
    
    members = sequence_df['member_id'].unique()
    np.random.seed(42)
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
    logger.info(f"  Files: sequences_train.parquet, sequences_val.parquet, sequences_test.parquet")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total sequences: {len(sequence_df):,}")
    logger.info(f"Train: {len(train_df):,} ({len(train_df)/len(sequence_df)*100:.1f}%)")
    logger.info(f"Val: {len(val_df):,} ({len(val_df)/len(sequence_df)*100:.1f}%)")
    logger.info(f"Test: {len(test_df):,} ({len(test_df)/len(sequence_df)*100:.1f}%)")
    
    logger.info(f"\nOutcome rates (test set):")
    logger.info(f"  ED: {test_df['future_ed_30d'].mean():.4f} ({(test_df['future_ed_30d'] > 0).mean()*100:.2f}% non-zero)")
    logger.info(f"  IP: {test_df['future_ip_30d'].mean():.4f} ({(test_df['future_ip_30d'] > 0).mean()*100:.2f}% non-zero)")
    
    logger.info("\n" + "="*80)
    logger.info("✓ PHASE 1 COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
