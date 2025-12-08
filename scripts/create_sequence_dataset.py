"""
Create Sequence Dataset for LSTM Training
=========================================

Converts daily observations into 4-week sequences for temporal modeling.
Each sequence contains states from the past 4 weeks leading up to an action.

Author: Sanjay Basu, MD PhD
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceDatasetCreator:
    """Create sequence dataset for LSTM training."""
    
    def __init__(self, sequence_length: int = 28):
        """
        Initialize sequence creator.
        
        Args:
            sequence_length: Length of sequences in days (default: 28 = 4 weeks)
        """
        self.sequence_length = sequence_length
        logger.info(f"SequenceDatasetCreator initialized with sequence_length={sequence_length}")
    
    def create_sequences(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sequences from daily observations.
        
        For each observation at time t, create a sequence containing
        states from [t-27, t-26, ..., t-1, t] (28 days total).
        
        Args:
            daily_df: Daily trajectory dataset
            
        Returns:
            DataFrame with sequences
        """
        logger.info(f"Creating {self.sequence_length}-day sequences...")
        
        # Sort by member and date
        daily_df = daily_df.sort_values(['member_id', 'observation_date'])
        
        sequences = []
        sequence_id = 0
        
        # Process each member separately
        for member_id, member_data in daily_df.groupby('member_id'):
            if len(member_data) < self.sequence_length:
                continue  # Skip members with insufficient data
            
            member_data = member_data.reset_index(drop=True)
            
            # Sliding window over member's observation
            for i in range(self.sequence_length - 1, len(member_data)):
                # Check if we have continuous days
                window_dates = member_data.iloc[i - self.sequence_length + 1:i + 1]['observation_date']
                
                if self._is_continuous_sequence(window_dates):
                    sequence_data = {
                        'sequence_id': sequence_id,
                        'member_id': member_id,
                        'sequence_end_date': member_data.iloc[i]['observation_date'],
                        'sequence_start_date': member_data.iloc[i - self.sequence_length + 1]['observation_date']
                    }
                    
                    # Extract state features for each timestep in sequence
                    for t in range(self.sequence_length):
                        obs = member_data.iloc[i - self.sequence_length + 1 + t]
                        sequence_data[f'state_t{t}'] = self._serialize_state(obs)
                    
                    # Outcome and action at current timestep
                    current_obs = member_data.iloc[i]
                    sequence_data['reward'] = current_obs['reward_shaped']  # Assume already shaped
                    sequence_data['action'] = current_obs['action_tuple']
                    sequence_data['ed_visits_7d'] = current_obs['ed_visits_7d']
                    sequence_data['hospitalizations_7d'] = current_obs['hospitalizations_7d']
                    
                    sequences.append(sequence_data)
                    sequence_id += 1
            
            if (sequence_id + 1) % 10000 == 0:
                logger.info(f"  Created {sequence_id + 1:,} sequences...")
        
        sequence_df = pd.DataFrame(sequences)
        
        logger.info(f"✓ Created {len(sequence_df):,} sequences")
        logger.info(f"  Unique members: {sequence_df['member_id'].nunique():,}")
        logger.info(f"  Avg sequences per member: {len(sequence_df) / sequence_df['member_id'].nunique():.1f}")
        
        return sequence_df
    
    def _is_continuous_sequence(self, dates: pd.Series) -> bool:
        """
        Check if dates form a continuous daily sequence.
        
        Args:
            dates: Series of dates
            
        Returns:
            True if consecutive days, False otherwise
        """
        if len(dates) < 2:
            return True
        
        diffs = dates.diff().dropna()
        return (diffs == pd.Timedelta(days=1)).all()
    
    def _serialize_state(self, obs: pd.Series) -> Dict:
        """
        Serialize observation into state dictionary.
        
        Extracts relevant features for the state representation.
        """
        # Continuous features
        continuous_features = [
            'age', 'transportation_score',
            'chronic_condition_count', 'active_medication_count',
            'hba1c_value', 'sbp_value', 'dbp_value', 'days_since_last_lab',
            'housing_stability_score', 'food_security_score', 'insurance_continuity_months',
            'days_since_last_contact', 'successful_contact_rate_30d', 'no_show_rate_90d',
            'current_episode_weeks'
        ]
        
        # Binary features
        binary_features = [
            'phone_access', 'internet_access',
            'has_diabetes', 'has_hypertension', 'has_depression', 'has_copd_asthma'
        ]
        
        # Categorical features
        categorical_features = {
            'language_preference': obs.get('language_preference', 'English'),
            'employment_status': obs.get('employment_status', 'unknown')
        }
        
        state = {
            'continuous': obs[continuous_features].values.tolist(),
            'binary': obs[binary_features].values.tolist(),
            'categorical': categorical_features
        }
        
        return state
    
    def expand_sequences_to_tensors(self, sequence_df: pd.DataFrame) -> Tuple[np.ndarray, Dict, np.ndarray, np.ndarray]:
        """
        Expand sequences into tensor format for PyTorch.
        
        Args:
            sequence_df: Sequence dataset
            
        Returns:
            Tuple of:
            - state_sequences: (N, seq_len, state_dim) numpy array
            - categorical_sequences: Dict of (N, seq_len) arrays for each categorical
            - actions: (N, action_components) numpy array
            - rewards: (N,) numpy array
        """
        logger.info("Expanding sequences to tensor format...")
        
        N = len(sequence_df)
        seq_len = self.sequence_length
        
        # Extract states
        state_list = []
        categorical_dict = {'language_preference': [], 'employment_status': []}
        
        for idx, row in sequence_df.iterrows():
            if idx % 10000 == 0:
                logger.info(f"  Processing sequence {idx:,} / {N:,}...")
            
            # Continuous states for this sequence
            sequence_states = []
            lang_seq = []
            emp_seq = []
            
            for t in range(seq_len):
                state_dict = row[f'state_t{t}']
                sequence_states.append(state_dict['continuous'] + state_dict['binary'])
                lang_seq.append(self._encode_categorical(state_dict['categorical']['language_preference'], 'language'))
                emp_seq.append(self._encode_categorical(state_dict['categorical']['employment_status'], 'employment'))
            
            state_list.append(sequence_states)
            categorical_dict['language_preference'].append(lang_seq)
            categorical_dict['employment_status'].append(emp_seq)
        
        # Convert to numpy
        state_sequences = np.array(state_list, dtype=np.float32)
        categorical_sequences = {
            k: np.array(v, dtype=np.int64)
            for k, v in categorical_dict.items()
        }
        
        # Extract actions and rewards
        actions = np.array(sequence_df['action'].tolist(), dtype=np.int64)
        rewards = sequence_df['reward'].values.astype(np.float32)
        
        logger.info(f"✓ Tensor expansion complete")
        logger.info(f"  State sequences shape: {state_sequences.shape}")
        logger.info(f"  Categorical sequences: {[(k, v.shape) for k, v in categorical_sequences.items()]}")
        logger.info(f"  Actions shape: {actions.shape}")
        logger.info(f"  Rewards shape: {rewards.shape}")
        
        return state_sequences, categorical_sequences, actions, rewards
    
    def _encode_categorical(self, value: str, feature_type: str) -> int:
        """Encode categorical feature to integer."""
        encodings = {
            'language': {'English': 0, 'Spanish': 1, 'Other': 2},
            'employment': {'employed': 0, 'unemployed': 1, 'disabled': 2, 'retired': 3, 'unknown': 4}
        }
        
        return encodings.get(feature_type, {}).get(value, 0)
    
    def split_train_val_test(self,
                            sequence_df: pd.DataFrame,
                            train_frac: float = 0.5,
                            val_frac: float = 0.25,
                            test_frac: float = 0.25,
                            random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split sequences into train, validation, and test sets.
        
        Splits at the member level to avoid leakage.
        
        Args:
            sequence_df: Sequence dataset
            train_frac: Fraction for training
            val_frac: Fraction for validation
            test_frac: Fraction for testing
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting dataset: train={train_frac}, val={val_frac}, test={test_frac}...")
        
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"
        
        # Get unique members
        members = sequence_df['member_id'].unique()
        np.random.seed(random_seed)
        np.random.shuffle(members)
        
        # Split members
        n_train = int(len(members) * train_frac)
        n_val = int(len(members) * val_frac)
        
        train_members = members[:n_train]
        val_members = members[n_train:n_train + n_val]
        test_members = members[n_train + n_val:]
        
        # Split sequences
        train_df = sequence_df[sequence_df['member_id'].isin(train_members)]
        val_df = sequence_df[sequence_df['member_id'].isin(val_members)]
        test_df = sequence_df[sequence_df['member_id'].isin(test_members)]
        
        logger.info(f"✓ Split complete:")
        logger.info(f"  Train: {len(train_df):,} sequences from {len(train_members):,} members")
        logger.info(f"  Val: {len(val_df):,} sequences from {len(val_members):,} members")
        logger.info(f"  Test: {len(test_df):,} sequences from {len(test_members):,} members")
        
        return train_df, val_df, test_df


def main():
    """Create sequence dataset from daily observations."""
    
    logger.info("="*80)
    logger.info("Sequence Dataset Creation Pipeline")
    logger.info("="*80)
    
    # Load daily dataset
    data_dir = Path("/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data")
    daily_df = pd.read_parquet(data_dir / "enhanced_trajectories_daily.parquet")
    
    logger.info(f"\nLoaded daily dataset: {daily_df.shape}")
    
    # Create sequences
    creator = SequenceDatasetCreator(sequence_length=28)
    sequence_df = creator.create_sequences(daily_df)
    
    # Split into train/val/test
    train_df, val_df, test_df = creator.split_train_val_test(
        sequence_df,
        train_frac=0.5,
        val_frac=0.25,
        test_frac=0.25
    )
    
    # Save splits
    train_df.to_parquet(data_dir / "sequence_trajectories_train.parquet", compression='snappy', index=False)
    val_df.to_parquet(data_dir / "sequence_trajectories_val.parquet", compression='snappy', index=False)
    test_df.to_parquet(data_dir / "sequence_trajectories_test.parquet", compression='snappy', index=False)
    
    logger.info(f"\n✓ Sequence datasets saved to {data_dir}")
    
    # Generate sample tensors for verification
    logger.info("\nGenerating sample tensors for verification...")
    sample_df = train_df.head(100)
    states, categoricals, actions, rewards = creator.expand_sequences_to_tensors(sample_df)
    
    logger.info(f"\nSample tensor shapes (first 100 sequences):")
    logger.info(f"  States: {states.shape}")
    logger.info(f"  Categoricals: {[(k, v.shape) for k, v in categoricals.items()]}")
    logger.info(f"  Actions: {actions.shape}")
    logger.info(f"  Rewards: {rewards.shape}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ Sequence Dataset Creation Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
