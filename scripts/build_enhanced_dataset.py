"""
Build Enhanced Daily Dataset
=============================

Merges enhanced features with trajectory data to create daily-level
observations with non-overlapping 7-day outcome windows.

Author: Sanjay Basu, MD PhD
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyDatasetBuilder:
    """Build daily-level trajectory dataset with enhanced features."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def create_daily_observations(self,
                                 start_date: str,
                                 end_date: str,
                                 members: list) -> pd.DataFrame:
        """
        Create daily observation grid for all members.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            members: List of member IDs
            
        Returns:
            DataFrame with member_id, observation_date
        """
        logger.info(f"Creating daily observation grid from {start_date} to {end_date}...")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create Cartesian product of members × dates
        obs_grid = pd.MultiIndex.from_product(
            [members, dates],
            names=['member_id', 'observation_date']
        ).to_frame(index=False)
        
        logger.info(f"Created {len(obs_grid):,} daily observations")
        
        return obs_grid
    
    def compute_7day_outcomes(self,
                             obs_df: pd.DataFrame,
                             utilization_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute non-overlapping 7-day outcome windows.
        
        For each observation date, count ED visits and hospitalizations
        in the following 7 days (not including the observation date itself).
        
        Args:
            obs_df: Observation grid (member_id, observation_date)
            utilization_df: Utilization events (member_id, event_date, event_type)
            
        Returns:
            obs_df with ed_visits_7d and hospitalizations_7d columns
        """
        logger.info("Computing 7-day outcome windows...")
        
        obs_df = obs_df.copy()
        obs_df['ed_visits_7d'] = 0
        obs_df['hospitalizations_7d'] = 0
        
        # Group utilization by member
        util_by_member = {
            member_id: group
            for member_id, group in utilization_df.groupby('member_id')
        }
        
        # Process in batches for efficiency
        batch_size = 10000
        for i in range(0, len(obs_df), batch_size):
            if i % 100000 == 0:
                logger.info(f"Processing observations {i:,} to {i+batch_size:,}...")
            
            batch = obs_df.iloc[i:i+batch_size]
            
            for idx, row in batch.iterrows():
                member_id = row['member_id']
                obs_date = row['observation_date']
                
                if member_id not in util_by_member:
                    continue
                
                member_util = util_by_member[member_id]
                
                # 7-day window: (obs_date, obs_date + 7 days)
                window_start = obs_date + timedelta(days=1)
                window_end = obs_date + timedelta(days=7)
                
                window_util = member_util[
                    (member_util['event_date'] >= window_start) &
                    (member_util['event_date'] <= window_end)
                ]
                
                obs_df.loc[idx, 'ed_visits_7d'] = (window_util['event_type'] == 'ED').sum()
                obs_df.loc[idx, 'hospitalizations_7d'] = (window_util['event_type'] == 'Hospitalization').sum()
        
        logger.info("✓ 7-day outcomes computed")
        logger.info(f"  Mean ED visits: {obs_df['ed_visits_7d'].mean():.4f}")
        logger.info(f"  Mean hospitalizations: {obs_df['hospitalizations_7d'].mean():.4f}")
        logger.info(f"  Non-zero outcomes: {(obs_df[['ed_visits_7d', 'hospitalizations_7d']].sum(axis=1) > 0).mean()*100:.2f}%")
        
        return obs_df
    
    def extract_engagement_outcomes(self,
                                   obs_df: pd.DataFrame,
                                   intervention_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract engagement outcomes for reward shaping.
        
        Features:
        - successful_contact: Did we successfully reach patient on this day?
        - attempted_contact: Did we try to contact?
        - no_show: Did patient no-show for appointment?
        - appointment_attended: Did patient attend appointment?
        - pcp_appointment_attended: Attended PCP appointment?
        - medication_filled: Filled prescription?
        - lab_completed: Completed lab?
        - specialist_appointment_attended: Attended specialist?
        - missed_appointment: Missed any appointment?
        """
        logger.info("Extracting engagement outcomes...")
        
        obs_df = obs_df.copy()
        
        # Initialize all to False
        engagement_cols = [
            'successful_contact', 'attempted_contact', 'no_show',
            'appointment_attended', 'pcp_appointment_attended',
            'medication_filled', 'lab_completed',
            'specialist_appointment_attended', 'missed_appointment'
        ]
        
        for col in engagement_cols:
            obs_df[col] = False
        
        # Join with interventions on same day
        intervention_df['observation_date'] = pd.to_datetime(intervention_df['contact_date'])
        
        merged = obs_df.merge(
            intervention_df,
            on=['member_id', 'observation_date'],
            how='left'
        )
        
        # Fill engagement indicators
        obs_df['successful_contact'] = merged['contact_successful'].fillna(False)
        obs_df['attempted_contact'] = merged['contact_attempted'].fillna(False)
        obs_df['no_show'] = merged['no_show'].fillna(False)
        obs_df['appointment_attended'] = merged['appointment_attended'].fillna(False)
        obs_df['pcp_appointment_attended'] = merged['pcp_appointment'].fillna(False)
        obs_df['medication_filled'] = merged['med_filled'].fillna(False)
        obs_df['lab_completed'] = merged['lab_done'].fillna(False)
        obs_df['specialist_appointment_attended'] = merged['specialist_appt'].fillna(False)
        obs_df['missed_appointment'] = merged['missed_appt'].fillna(False)
        
        logger.info("✓ Engagement outcomes extracted")
        for col in engagement_cols:
            logger.info(f"  {col}: {obs_df[col].mean()*100:.2f}%")
        
        return obs_df
    
    def merge_with_enhanced_features(self,
                                    obs_df: pd.DataFrame,
                                    features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge observations with enhanced features.
        
        Args:
            obs_df: Observation grid with outcomes
            features_df: Enhanced features (from extract_enhanced_features.py)
            
        Returns:
            Complete daily dataset
        """
        logger.info("Merging with enhanced features...")
        
        # Merge on member_id and observation_date
        merged = obs_df.merge(
            features_df,
            on=['member_id', 'observation_date'],
            how='left'
        )
        
        logger.info(f"✓ Merged dataset: {merged.shape}")
        logger.info(f"  Columns: {len(merged.columns)}")
        logger.info(f"  Observations: {len(merged):,}")
        
        # Check for missing values
        missing_pct = merged.isnull().mean() * 100
        high_missing = missing_pct[missing_pct > 10]
        
        if len(high_missing) > 0:
            logger.warning("⚠️ High missing values detected:")
            for col, pct in high_missing.items():
                logger.warning(f"  {col}: {pct:.1f}%")
        
        return merged
    
    def filter_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter and clean the dataset.
        
        Removes:
        - Observations before member enrollment
        - Observations after member disenrollment
        - Members with < 28 days of data (need for 4-week sequences)
        """
        logger.info("Filtering and cleaning dataset...")
        
        initial_size = len(df)
        
        # Remove observations outside enrollment period
        df = df[
            (df['observation_date'] >= df['enrollment_start']) &
            (df['observation_date'] <= df['enrollment_end'])
        ]
        logger.info(f"  Removed {initial_size - len(df):,} obs outside enrollment")
        
        # Remove members with insufficient data
        member_days = df.groupby('member_id')['observation_date'].nunique()
        valid_members = member_days[member_days >= 28].index
        df = df[df['member_id'].isin(valid_members)]
        logger.info(f"  Kept {len(valid_members):,} members with ≥28 days")
        
        # Remove extreme outliers (safety check)
        df = df[df['ed_visits_7d'] <= 10]
        df = df[df['hospitalizations_7d'] <= 5]
        
        logger.info(f"✓ Final dataset: {len(df):,} observations")
        
        return df
    
    def build_complete_dataset(self,
                              start_date: str = "2023-01-01",
                              end_date: str = "2025-11-30") -> pd.DataFrame:
        """
        Build complete enhanced daily dataset.
        
        Pipeline:
        1. Create daily observation grid
        2. Compute 7-day outcomes
        3. Extract engagement outcomes
        4. Merge with enhanced features
        5. Filter and clean
        
        Returns:
            Complete daily trajectory dataset
        """
        logger.info("="*80)
        logger.info("Building Enhanced Daily Dataset")
        logger.info("="*80)
        
        # Load source data
        logger.info("\n1. Loading source data...")
        members_df = pd.read_parquet(self.data_dir / "members.parquet")
        utilization_df = pd.read_parquet(self.data_dir / "utilization.parquet")
        intervention_df = pd.read_parquet(self.data_dir / "interventions.parquet")
        features_df = pd.read_parquet(self.data_dir / "enhanced_features_daily.parquet")
        
        members = members_df['member_id'].unique().tolist()
        logger.info(f"  {len(members):,} unique members")
        
        # Create daily observations
        logger.info("\n2. Creating daily observation grid...")
        obs_df = self.create_daily_observations(start_date, end_date, members)
        
        # Compute outcomes
        logger.info("\n3. Computing 7-day outcomes...")
        obs_df = self.compute_7day_outcomes(obs_df, utilization_df)
        
        # Extract engagement
        logger.info("\n4. Extracting engagement outcomes...")
        obs_df = self.extract_engagement_outcomes(obs_df, intervention_df)
        
        # Merge features
        logger.info("\n5. Merging with enhanced features...")
        complete_df = self.merge_with_enhanced_features(obs_df, features_df)
        
        # Add enrollment dates from members_df
        complete_df = complete_df.merge(
            members_df[['member_id', 'enrollment_start', 'enrollment_end']],
            on='member_id',
            how='left'
        )
        
        # Filter and clean
        logger.info("\n6. Filtering and cleaning...")
        complete_df = self.filter_and_clean(complete_df)
        
        logger.info("\n" + "="*80)
        logger.info("✓ Enhanced Daily Dataset Complete!")
        logger.info("="*80)
        logger.info(f"Final shape: {complete_df.shape}")
        logger.info(f"Date range: {complete_df['observation_date'].min()} to {complete_df['observation_date'].max()}")
        logger.info(f"Members: {complete_df['member_id'].nunique():,}")
        logger.info(f"Avg observations per member: {len(complete_df) / complete_df['member_id'].nunique():.1f}")
        
        return complete_df


def main():
    """Build enhanced daily dataset."""
    
    # Configuration
    data_dir = Path("/Users/sanjaybasu/waymark-local/data")
    output_dir = Path("/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build dataset
    builder = DailyDatasetBuilder(data_dir=str(data_dir))
    daily_df = builder.build_complete_dataset(
        start_date="2023-01-01",
        end_date="2025-11-30"
    )
    
    # Save
    output_path = output_dir / "enhanced_trajectories_daily.parquet"
    daily_df.to_parquet(output_path, compression='snappy', index=False)
    logger.info(f"\n✓ Dataset saved to: {output_path}")
    logger.info(f"  Size: {output_path.stat().st_size / 1e9:.2f} GB")
    
    # Generate summary statistics
    logger.info("\n" + "="*80)
    logger.info("Dataset Summary Statistics")
    logger.info("="*80)
    
    print("\nContinuous Features:")
    continuous_cols = ['age', 'active_medication_count', 'hba1c_value', 'sbp_value']
    print(daily_df[continuous_cols].describe())
    
    print("\nBinary Features:")
    binary_cols = ['has_diabetes', 'has_hypertension', 'phone_access', 'internet_access']
    print(daily_df[binary_cols].mean())
    
    print("\nOutcome Statistics:")
    print(f"ED visits (7-day): {daily_df['ed_visits_7d'].mean():.4f} ± {daily_df['ed_visits_7d'].std():.4f}")
    print(f"Hospitalizations (7-day): {daily_df['hospitalizations_7d'].mean():.4f} ± {daily_df['hospitalizations_7d'].std():.4f}")
    print(f"Non-zero outcomes: {(daily_df[['ed_visits_7d', 'hospitalizations_7d']].sum(axis=1) > 0).mean()*100:.2f}%")
    
    logger.info("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()
