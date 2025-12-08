"""
Phase 1: Proper Factored Action Space (10% Sample)
===================================================

Creates sequences with FULL factored action representation:
- Modality: PHONE_CALL, SMS_TEXT, HOME_VISIT, VIDEO_VISIT, etc.
- Provider: CARE_COORDINATOR, CHW, PHARMACIST, THERAPIST, etc.
- Goal: CARE, MEDICATION_ADHERENCE, PCP_APPOINTMENT, etc.
- Urgency: ROUTINE, URGENT

Author: Sanjay Basu, MD PhD
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/phase1_factored.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info("PHASE 1: FACTORED ACTION SPACE DATA PREP (10% SAMPLE)")
    logger.info("="*80)
    
    # Paths
    data_path = Path('/Users/sanjaybasu/waymark-local/foundation/outputs/rl_factored_pipeline_full_real_rewards/merged_trajectories.parquet')
    output_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data_factored')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading full dataset...")
    df = pd.read_parquet(data_path)
    logger.info(f"  Total: {len(df):,} observations from {df['member_id'].nunique():,} members")
    
    # Check action space
    logger.info("\nAction space structure:")
    logger.info(f"  Modalities: {df['action_modality'].nunique()} unique")
    logger.info(f"  Providers: {df['action_provider'].nunique()} unique")
    logger.info(f"  Goals: {df['action_goal'].nunique()} unique")
    logger.info(f"  Urgency: {df['action_urgency'].nunique()} unique")
    logger.info(f"  Total unique action tuples: {df['action_tuple'].nunique()}")
    
    # Sample 10% of members
    np.random.seed(42)
    all_members = df['member_id'].unique()
    sample_size = int(len(all_members) * 0.10)
    sampled_members = np.random.choice(all_members, size=sample_size, replace=False)
    
    df_sample = df[df['member_id'].isin(sampled_members)].copy()
    logger.info(f"\nSampled: {len(df_sample):,} observations from {len(sampled_members):,} members (10%)")
    
    # Reward shaping
    logger.info("\nComputing shaped rewards...")
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
    
    # Create action vocabulary from observed tuples
    logger.info("\nCreating action vocabulary...")
    observed_actions = df_sample['action_tuple'].unique()
    action_to_idx = {action: idx for idx, action in enumerate(observed_actions)}
    idx_to_action = {idx: action for action, idx in action_to_idx.items()}
    
    logger.info(f"  Unique action combinations in sample: {len(action_to_idx)}")
    
    # Show top 10 most frequent actions
    logger.info("\n  Top 10 most frequent actions:")
    action_freq = df_sample['action_tuple'].value_counts().head(10)
    for action, count in action_freq.items():
        pct = (count / len(df_sample)) * 100
        logger.info(f"    {action}: {count:,} ({pct:.2f}%)")
    
    # Create sequences
    logger.info("\nCreating 4-week sequences...")
    
    sequences = []
    seq_length = 4
    
    for member_id in sampled_members:
        member_df = df_sample[df_sample['member_id'] == member_id].sort_values('timestep_date')
        
        if len(member_df) < seq_length:
            continue
        
        # State features
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
            seq_data['action_tuple'] = last_row['action_tuple']
            seq_data['action_idx'] = action_to_idx[last_row['action_tuple']]
            
            # Store factored components for analysis
            seq_data['action_modality'] = last_row['action_modality']
            seq_data['action_provider'] = last_row['action_provider']
            seq_data['action_goal'] = last_row['action_goal']
            seq_data['action_urgency'] = last_row['action_urgency']
            
            seq_data['reward_shaped'] = last_row['reward_shaped']
            seq_data['future_ed_30d'] = last_row['future_ed_30d']
            seq_data['future_ip_30d'] = last_row['future_ip_30d']
            
            # Demographics for fairness
            seq_data['race'] = last_row.get('race', 'unknown')
            seq_data['gender'] = last_row.get('gender', 'unknown')
            
            sequences.append(seq_data)
    
    logger.info(f"  Created {len(sequences):,} sequences")
    
    # Convert to DataFrame
    sequences_df = pd.DataFrame(sequences)
    
    # Verify action distribution in sequences
    logger.info("\nAction distribution in sequences:")
    logger.info(f"  Unique actions: {sequences_df['action_idx'].nunique()}")
    logger.info(f"  Top 5 actions in sequences:")
    for action_idx, count in sequences_df['action_idx'].value_counts().head(5).items():
        action_tuple = idx_to_action[action_idx]
        pct = (count / len(sequences_df)) * 100
        logger.info(f"    {action_tuple}: {count:,} ({pct:.2f}%)")
    
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
    
    logger.info(f"\nData splits:")
    logger.info(f"  Train: {len(train_df):,} sequences")
    logger.info(f"  Val: {len(val_df):,} sequences")
    logger.info(f"  Test: {len(test_df):,} sequences")
    
    # Save
    logger.info("\nSaving sequences...")
    train_df.to_parquet(output_dir / 'sequences_train.parquet')
    val_df.to_parquet(output_dir / 'sequences_val.parquet')
    test_df.to_parquet(output_dir / 'sequences_test.parquet')
    
    # Save action mappings (both directions)
    with open(output_dir / 'action_to_idx.json', 'w') as f:
        # Convert tuple strings to regular strings for JSON
        action_map_str = {str(k): v for k, v in action_to_idx.items()}
        json.dump(action_map_str, f, indent=2)
    
    with open(output_dir / 'idx_to_action.json', 'w') as f:
        idx_map_str = {str(k): str(v) for k, v in idx_to_action.items()}
        json.dump(idx_map_str, f, indent=2)
    
    # Save metadata
    metadata = {
        'total_members_sampled': len(sampled_members),
        'total_sequences': len(sequences_df),
        'train_sequences': len(train_df),
        'val_sequences': len(val_df),
        'test_sequences': len(test_df),
        'num_unique_actions': len(action_to_idx),
        'state_features': state_cols,
        'sequence_length': seq_length
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n✓ Data saved to {output_dir}")
    logger.info("="*80)
    logger.info("✓ PHASE 1 COMPLETE (FACTORED ACTION SPACE)")
    logger.info("="*80)
    logger.info(f"\nKey Statistics:")
    logger.info(f"  Members: {len(sampled_members):,}")
    logger.info(f"  Sequences: {len(sequences_df):,}")
    logger.info(f"  Unique actions: {len(action_to_idx)}")
    logger.info(f"  State features: {len(state_cols)}")

if __name__ == "__main__":
    main()
