"""
Enhanced Feature Extraction for RL v3.0
=========================================

Extracts 25+ features from EHR and operational databases to create
a rich state representation for the LSTM-based policy.

Author: Sanjay Basu, MD PhD
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedFeatureExtractor:
    """Extract clinical, demographic, and SDOH features for enhanced RL state."""
    
    def __init__(self, data_dir: str = "/Users/sanjaybasu/waymark-local/data"):
        self.data_dir = Path(data_dir)
        self.features_extracted = []
        
    def extract_demographics_and_access(self, member_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract demographic and access features.
        
        Features:
        - age (continuous)
        - language_preference (categorical: English, Spanish, Other)
        - phone_access (binary: has phone number on file)
        - internet_access (binary: has email or portal activity)
        - transportation_score (0-10: derived from SDOH assessments)
        """
        logger.info("Extracting demographics and access features...")
        
        features = pd.DataFrame()
        features['member_id'] = member_df['member_id']
        
        # Age (already in member_df typically)
        features['age'] = member_df['age']
        
        # Language preference
        features['language_preference'] = member_df.get('preferred_language', 'English')
        features['language_preference'] = features['language_preference'].fillna('English')
        
        # Phone access (has valid phone number)
        features['phone_access'] = (~member_df['phone'].isna()).astype(int)
        
        # Internet access (has email OR portal login in last 90 days)
        features['internet_access'] = (
            (~member_df['email'].isna()) | 
            (member_df['last_portal_login_days_ago'] < 90)
        ).astype(int)
        
        # Transportation score (from SDOH file if available, else impute median)
        if 'transportation_score' in member_df.columns:
            features['transportation_score'] = member_df['transportation_score'].fillna(5.0)
        else:
            features['transportation_score'] = 5.0  # Neutral default
            
        self.features_extracted.extend([
            'age', 'language_preference', 'phone_access', 
            'internet_access', 'transportation_score'
        ])
        
        return features
    
    def extract_clinical_conditions(self, claims_df: pd.DataFrame, 
                                    members: List[str]) -> pd.DataFrame:
        """
        Extract chronic condition indicators from claims data.
        
        Features:
        - has_diabetes (binary)
        - has_hypertension (binary)
        - has_depression (binary)
        - has_copd_asthma (binary)
        - chronic_condition_count (0-10+)
        """
        logger.info("Extracting clinical condition features...")
        
        # ICD-10 code mappings
        icd10_maps = {
            'diabetes': ['E10', 'E11', 'E13'],  # Diabetes
            'hypertension': ['I10', 'I11', 'I12', 'I13'],  # HTN
            'depression': ['F32', 'F33', 'F34.1'],  # Depression
            'copd_asthma': ['J44', 'J45']  # COPD/Asthma
        }
        
        features = pd.DataFrame({'member_id': members})
        
        for condition, icd_prefixes in icd10_maps.items():
            # Check if any claim has these ICD codes
            has_condition = claims_df.groupby('member_id').apply(
                lambda x: x['icd10_code'].str.startswith(tuple(icd_prefixes)).any()
            ).astype(int)
            
            features[f'has_{condition}'] = features['member_id'].map(has_condition).fillna(0)
        
        # Chronic condition count (up to 10)
        features['chronic_condition_count'] = features[[
            'has_diabetes', 'has_hypertension', 'has_depression', 'has_copd_asthma'
        ]].sum(axis=1)
        
        # Add additional chronic conditions from claims (e.g., CKD, CHF)
        # This is a simplified version - extend as needed
        additional_chronic = claims_df.groupby('member_id')['chronic_flag'].sum().clip(0, 10)
        features['chronic_condition_count'] += features['member_id'].map(additional_chronic).fillna(0)
        features['chronic_condition_count'] = features['chronic_condition_count'].clip(0, 10)
        
        self.features_extracted.extend([
            'has_diabetes', 'has_hypertension', 'has_depression', 
            'has_copd_asthma', 'chronic_condition_count'
        ])
        
        return features
    
    def extract_medications_and_labs(self, rx_df: pd.DataFrame, 
                                     lab_df: pd.DataFrame,
                                     members: List[str],
                                     observation_date: pd.Timestamp) -> pd.DataFrame:
        """
        Extract medication and lab features.
        
        Features:
        - active_medication_count (0-20+)
        - hba1c_value (if diabetic, else NA → imputed to mean)
        - sbp_value (systolic blood pressure)
        - dbp_value (diastolic blood pressure)
        - days_since_last_lab (0-365+)
        """
        logger.info(f"Extracting medications and labs for {observation_date}...")
        
        features = pd.DataFrame({'member_id': members})
        
        # Active medication count (prescriptions filled in last 90 days)
        recent_rx = rx_df[
            (rx_df['fill_date'] >= observation_date - timedelta(days=90)) &
            (rx_df['fill_date'] <= observation_date)
        ]
        med_count = recent_rx.groupby('member_id')['ndc_code'].nunique().clip(0, 20)
        features['active_medication_count'] = features['member_id'].map(med_count).fillna(0)
        
        # Labs (most recent before observation_date)
        recent_labs = lab_df[lab_df['lab_date'] <= observation_date]
        
        # HbA1c (for diabetics)
        hba1c = recent_labs[recent_labs['lab_type'] == 'HbA1c'].groupby('member_id')['value'].last()
        features['hba1c_value'] = features['member_id'].map(hba1c)
        # Impute missing HbA1c with median (non-diabetics get median too)
        features['hba1c_value'] = features['hba1c_value'].fillna(features['hba1c_value'].median())
        
        # Blood pressure
        sbp = recent_labs[recent_labs['lab_type'] == 'SBP'].groupby('member_id')['value'].last()
        dbp = recent_labs[recent_labs['lab_type'] == 'DBP'].groupby('member_id')['value'].last()
        features['sbp_value'] = features['member_id'].map(sbp).fillna(120)  # Normal default
        features['dbp_value'] = features['member_id'].map(dbp).fillna(80)
        
        # Days since last lab
        last_lab_date = recent_labs.groupby('member_id')['lab_date'].max()
        features['days_since_last_lab'] = (
            observation_date - features['member_id'].map(last_lab_date)
        ).dt.days.fillna(365).clip(0, 365)
        
        self.features_extracted.extend([
            'active_medication_count', 'hba1c_value', 'sbp_value', 
            'dbp_value', 'days_since_last_lab'
        ])
        
        return features
    
    def extract_sdoh_features(self, sdoh_df: pd.DataFrame, members: List[str]) -> pd.DataFrame:
        """
        Extract social determinants of health features.
        
        Features:
        - housing_stability_score (0-10)
        - food_security_score (0-10)
        - employment_status (categorical: employed, unemployed, disabled, retired)
        - insurance_continuity_months (continuous)
        """
        logger.info("Extracting SDOH features...")
        
        features = pd.DataFrame({'member_id': members})
        
        # Housing stability (from SDOH assessments)
        housing = sdoh_df.groupby('member_id')['housing_score'].last()
        features['housing_stability_score'] = features['member_id'].map(housing).fillna(7.0)
        
        # Food security
        food = sdoh_df.groupby('member_id')['food_security_score'].last()
        features['food_security_score'] = features['member_id'].map(food).fillna(7.0)
        
        # Employment status (from most recent assessment)
        employment = sdoh_df.groupby('member_id')['employment_status'].last()
        features['employment_status'] = features['member_id'].map(employment).fillna('unknown')
        
        # Insurance continuity (months enrolled without gap)
        continuity = sdoh_df.groupby('member_id')['enrollment_months'].max()
        features['insurance_continuity_months'] = features['member_id'].map(continuity).fillna(1)
        
        self.features_extracted.extend([
            'housing_stability_score', 'food_security_score', 
            'employment_status', 'insurance_continuity_months'
        ])
        
        return features
    
    def extract_engagement_features(self, intervention_df: pd.DataFrame,
                                    members: List[str],
                                    observation_date: pd.Timestamp) -> pd.DataFrame:
        """
        Extract care engagement features.
        
        Features:
        - days_since_last_contact (0-365+)
        - successful_contact_rate_30d (0-1)
        - no_show_rate_90d (0-1)
        - episode_id (integer, resets after 90-day gap)
        - current_episode_weeks (continuous)
        """
        logger.info(f"Extracting engagement features for {observation_date}...")
        
        features = pd.DataFrame({'member_id': members})
        
        # Days since last contact
        recent_contacts = intervention_df[intervention_df['contact_date'] <= observation_date]
        last_contact = recent_contacts.groupby('member_id')['contact_date'].max()
        features['days_since_last_contact'] = (
            observation_date - features['member_id'].map(last_contact)
        ).dt.days.fillna(365).clip(0, 365)
        
        # Successful contact rate (last 30 days)
        window_30d = intervention_df[
            (intervention_df['contact_date'] >= observation_date - timedelta(days=30)) &
            (intervention_df['contact_date'] <= observation_date)
        ]
        total_attempts = window_30d.groupby('member_id').size()
        successful = window_30d[window_30d['contact_successful'] == True].groupby('member_id').size()
        contact_rate = (successful / total_attempts).fillna(0)
        features['successful_contact_rate_30d'] = features['member_id'].map(contact_rate).fillna(0)
        
        # No-show rate (last 90 days)
        window_90d = intervention_df[
            (intervention_df['contact_date'] >= observation_date - timedelta(days=90)) &
            (intervention_df['contact_date'] <= observation_date)
        ]
        appointments = window_90d[window_90d['appointment_scheduled'] == True]
        total_appts = appointments.groupby('member_id').size()
        no_shows = appointments[appointments['no_show'] == True].groupby('member_id').size()
        no_show_rate = (no_shows / total_appts).fillna(0)
        features['no_show_rate_90d'] = features['member_id'].map(no_show_rate).fillna(0)
        
        # Episode tracking (simplified - assign episode ID based on 90-day gaps)
        # This is a placeholder - actual implementation would be more sophisticated
        features['episode_id'] = features['member_id'].apply(hash).abs() % 10000
        features['current_episode_weeks'] = features['days_since_last_contact'].apply(
            lambda x: max(0, (90 - x) / 7) if x <= 90 else 0
        )
        
        self.features_extracted.extend([
            'days_since_last_contact', 'successful_contact_rate_30d', 
            'no_show_rate_90d', 'episode_id', 'current_episode_weeks'
        ])
        
        return features
    
    def build_complete_feature_set(self, 
                                   member_df: pd.DataFrame,
                                   claims_df: pd.DataFrame,
                                   rx_df: pd.DataFrame,
                                   lab_df: pd.DataFrame,
                                   sdoh_df: pd.DataFrame,
                                   intervention_df: pd.DataFrame,
                                   observation_dates: List[pd.Timestamp]) -> pd.DataFrame:
        """
        Build complete enhanced feature set for all members at all observation dates.
        
        Returns:
            DataFrame with columns: member_id, observation_date, feature1, feature2, ...
        """
        logger.info(f"Building complete feature set for {len(observation_dates)} observation dates...")
        
        all_features = []
        
        for i, obs_date in enumerate(observation_dates):
            if i % 100 == 0:
                logger.info(f"Processing observation date {i+1}/{len(observation_dates)}: {obs_date}")
            
            members = member_df['member_id'].tolist()
            
            # Extract each feature category
            demo_features = self.extract_demographics_and_access(member_df)
            clinical_features = self.extract_clinical_conditions(claims_df, members)
            med_lab_features = self.extract_medications_and_labs(rx_df, lab_df, members, obs_date)
            sdoh_features = self.extract_sdoh_features(sdoh_df, members)
            engagement_features = self.extract_engagement_features(intervention_df, members, obs_date)
            
            # Merge all features
            obs_features = demo_features.merge(clinical_features, on='member_id') \
                                       .merge(med_lab_features, on='member_id') \
                                       .merge(sdoh_features, on='member_id') \
                                       .merge(engagement_features, on='member_id')
            
            obs_features['observation_date'] = obs_date
            all_features.append(obs_features)
        
        # Concatenate all observation dates
        complete_features = pd.concat(all_features, ignore_index=True)
        
        logger.info(f"Complete feature set built: {complete_features.shape}")
        logger.info(f"Features extracted: {len(self.features_extracted)}")
        logger.info(f"Feature list: {self.features_extracted}")
        
        return complete_features


def main():
    """
    Main execution: Load data and extract enhanced features.
    """
    logger.info("=" * 80)
    logger.info("Enhanced Feature Extraction Pipeline Starting")
    logger.info("=" * 80)
    
    # Initialize extractor
    extractor = EnhancedFeatureExtractor()
    
    # Load source data (placeholder - adjust paths to your actual data)
    data_dir = Path("/Users/sanjaybasu/waymark-local/data")
    
    logger.info("Loading source datasets...")
    
    # These would be loaded from your actual database/files
    # For now, creating placeholder structure
    member_df = pd.read_parquet(data_dir / "members.parquet")
    claims_df = pd.read_parquet(data_dir / "claims.parquet")
    rx_df = pd.read_parquet(data_dir / "prescriptions.parquet")
    lab_df = pd.read_parquet(data_dir / "labs.parquet")
    sdoh_df = pd.read_parquet(data_dir / "sdoh_assessments.parquet")
    intervention_df = pd.read_parquet(data_dir / "interventions.parquet")
    
    # Define observation dates (daily from start to end date)
    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2025-11-30")
    observation_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    logger.info(f"Observation period: {start_date} to {end_date}")
    logger.info(f"Total observation dates: {len(observation_dates)}")
    
    # Extract complete feature set
    enhanced_features = extractor.build_complete_feature_set(
        member_df=member_df,
        claims_df=claims_df,
        rx_df=rx_df,
        lab_df=lab_df,
        sdoh_df=sdoh_df,
        intervention_df=intervention_df,
        observation_dates=observation_dates.tolist()
    )
    
    # Save to parquet
    output_path = Path("/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data")
    output_path.mkdir(parents=True, exist_ok=True)
    
    enhanced_features.to_parquet(
        output_path / "enhanced_features_daily.parquet",
        compression='snappy',
        index=False
    )
    
    logger.info(f"Enhanced features saved to: {output_path / 'enhanced_features_daily.parquet'}")
    logger.info(f"Total records: {len(enhanced_features):,}")
    logger.info(f"Total features: {len(extractor.features_extracted)}")
    
    # Generate summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("Feature Summary Statistics")
    logger.info("=" * 80)
    print(enhanced_features.describe())
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Enhanced Feature Extraction Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
