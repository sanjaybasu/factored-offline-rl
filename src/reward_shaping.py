"""
Multi-Component Reward Shaping for RL v3.0
===========================================

Implements shaped reward function with:
1. Primary outcome (ED visits + hospitalizations)
2. Engagement signals (contact success, no-shows)
3. Intermediate milestones (PCP appointments, medication fills, labs)

Author: Sanjay Basu, MD PhD
Date: December 2024
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging

# Resolve paths relative to repo root; override via environment variables
_REPO_ROOT = Path(__file__).resolve().parent.parent
data_dir = Path(os.environ.get('FACTORED_RL_DATA_DIR', _REPO_ROOT / 'data'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardShaper:
    """Compute shaped rewards from multiple signal sources."""
    
    def __init__(self, 
                 primary_weight: float = 1.0,
                 engagement_weight: float = 0.3,
                 intermediate_weight: float = 0.5):
        """
        Initialize reward shaper with component weights.
        
        Args:
            primary_weight: Weight for primary outcome (ED/hospitalization)
            engagement_weight: Weight for engagement signals
            intermediate_weight: Weight for intermediate clinical milestones
        """
        self.w_primary = primary_weight
        self.w_engagement = engagement_weight
        self.w_intermediate = intermediate_weight
        
        logger.info(f"RewardShaper initialized with weights: "
                   f"primary={primary_weight}, "
                   f"engagement={engagement_weight}, "
                   f"intermediate={intermediate_weight}")
    
    def compute_primary_reward(self, ed_visits: int, hospitalizations: int) -> float:
        """
        Compute primary outcome reward.
        
        Formula: R_primary = -(ED_visits + 2 * hospitalizations)
        
        Hospitalizations weighted 2x (higher cost and severity).
        
        Args:
            ed_visits: Number of ED visits in outcome window
            hospitalizations: Number of hospitalizations in outcome window
            
        Returns:
            Primary reward (always ≤ 0)
        """
        return -(ed_visits + 2 * hospitalizations)
    
    def compute_engagement_reward(self, 
                                  successful_contact: bool,
                                  attempted_contact: bool,
                                  no_show: bool,
                                  appointment_attended: bool) -> float:
        """
        Compute engagement-based reward.
        
        Rewards:
        +0.2: Successful contact made
        +0.1: Contact attempted (even if unsuccessful)
        -0.1: Patient no-showed for scheduled appointment
        +0.15: Patient attended appointment
        
        These are additive within a timestep.
        
        Args:
            successful_contact: Did we successfully reach the patient?
            attempted_contact: Did we try to contact (regardless of success)?
            no_show: Did patient no-show for scheduled appointment?
            appointment_attended: Did patient attend a scheduled appointment?
            
        Returns:
            Engagement reward (range: -0.1 to +0.45)
        """
        reward = 0.0
        
        if successful_contact:
            reward += 0.2
        elif attempted_contact:  # Attempted but unsuccessful
            reward += 0.1
            
        if no_show:
            reward -= 0.1
            
        if appointment_attended:
            reward += 0.15
            
        return reward
    
    def compute_intermediate_reward(self,
                                    pcp_appointment_attended: bool,
                                    medication_filled: bool,
                                    lab_completed: bool,
                                    specialist_appointment_attended: bool,
                                    missed_appointment: bool) -> float:
        """
        Compute intermediate clinical milestone reward.
        
        Rewards for positive milestones:
        +0.3: Attended PCP appointment (high value)
        +0.2: Filled medication as prescribed
        +0.1: Completed lab work
        +0.2: Attended specialist appointment
        
        Penalties:
        -0.2: Missed any scheduled appointment
        
        Args:
            pcp_appointment_attended: Attended primary care appointment
            medication_filled: Filled prescription within 7 days of due date
            lab_completed: Completed ordered lab work
            specialist_appointment_attended: Attended specialist appointment
            missed_appointment: Missed any scheduled appointment
            
        Returns:
            Intermediate reward (range: -0.2 to +0.8)
        """
        reward = 0.0
        
        if pcp_appointment_attended:
            reward += 0.3
            
        if medication_filled:
            reward += 0.2
            
        if lab_completed:
            reward += 0.1
            
        if specialist_appointment_attended:
            reward += 0.2
            
        if missed_appointment:
            reward -= 0.2
            
        return reward
    
    def compute_total_reward(self,
                           ed_visits: int,
                           hospitalizations: int,
                           successful_contact: bool,
                           attempted_contact: bool,
                           no_show: bool,
                           appointment_attended: bool,
                           pcp_appointment_attended: bool,
                           medication_filled: bool,
                           lab_completed: bool,
                           specialist_appointment_attended: bool,
                           missed_appointment: bool) -> Dict[str, float]:
        """
        Compute total shaped reward from all components.
        
        Formula:
        R_total = w1 * R_primary + w2 * R_engagement + w3 * R_intermediate
        
        Returns:
            Dictionary with:
            - 'primary': Primary outcome reward
            - 'engagement': Engagement reward
            - 'intermediate': Intermediate milestone reward
            - 'total': Weighted sum of all components
        """
        r_primary = self.compute_primary_reward(ed_visits, hospitalizations)
        r_engagement = self.compute_engagement_reward(
            successful_contact, attempted_contact, no_show, appointment_attended
        )
        r_intermediate = self.compute_intermediate_reward(
            pcp_appointment_attended, medication_filled, lab_completed,
            specialist_appointment_attended, missed_appointment
        )
        
        r_total = (
            self.w_primary * r_primary +
            self.w_engagement * r_engagement +
            self.w_intermediate * r_intermediate
        )
        
        return {
            'primary': r_primary,
            'engagement': r_engagement,
            'intermediate': r_intermediate,
            'total': r_total
        }
    
    def shape_trajectory_rewards(self, trajectory_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply reward shaping to entire trajectory dataset.
        
        Expects trajectory_df to have columns:
        - ed_visits_7d
        - hospitalizations_7d
        - successful_contact
        - attempted_contact
        - no_show
        - appointment_attended
        - pcp_appointment_attended
        - medication_filled
        - lab_completed
        - specialist_appointment_attended
        - missed_appointment
        
        Adds columns:
        - reward_primary
        - reward_engagement
        - reward_intermediate
        - reward_shaped (total)
        
        Args:
            trajectory_df: Trajectory dataframe with required columns
            
        Returns:
            Trajectory dataframe with reward columns added
        """
        logger.info(f"Shaping rewards for {len(trajectory_df):,} observations...")
        
        # Compute each reward component
        rewards = trajectory_df.apply(lambda row: self.compute_total_reward(
            ed_visits=row['ed_visits_7d'],
            hospitalizations=row['hospitalizations_7d'],
            successful_contact=row.get('successful_contact', False),
            attempted_contact=row.get('attempted_contact', False),
            no_show=row.get('no_show', False),
            appointment_attended=row.get('appointment_attended', False),
            pcp_appointment_attended=row.get('pcp_appointment_attended', False),
            medication_filled=row.get('medication_filled', False),
            lab_completed=row.get('lab_completed', False),
            specialist_appointment_attended=row.get('specialist_appointment_attended', False),
            missed_appointment=row.get('missed_appointment', False)
        ), axis=1, result_type='expand')
        
        trajectory_df['reward_primary'] = rewards['primary']
        trajectory_df['reward_engagement'] = rewards['engagement']
        trajectory_df['reward_intermediate'] = rewards['intermediate']
        trajectory_df['reward_shaped'] = rewards['total']
        
        # Log statistics
        logger.info("Reward shaping complete. Statistics:")
        logger.info(f"  Mean primary reward: {trajectory_df['reward_primary'].mean():.4f}")
        logger.info(f"  Mean engagement reward: {trajectory_df['reward_engagement'].mean():.4f}")
        logger.info(f"  Mean intermediate reward: {trajectory_df['reward_intermediate'].mean():.4f}")
        logger.info(f"  Mean shaped reward: {trajectory_df['reward_shaped'].mean():.4f}")
        logger.info(f"  Std shaped reward: {trajectory_df['reward_shaped'].std():.4f}")
        logger.info(f"  Non-zero shaped rewards: {(trajectory_df['reward_shaped'] != 0).sum():,} "
                   f"({(trajectory_df['reward_shaped'] != 0).mean()*100:.1f}%)")
        
        return trajectory_df


def tune_reward_weights(trajectory_df: pd.DataFrame,
                       validation_policy_performance: callable) -> Tuple[float, float, float]:
    """
    Tune reward weights to maximize validation policy performance.
    
    Grid search over:
    - engagement_weight: [0.1, 0.3, 0.5, 0.7]
    - intermediate_weight: [0.3, 0.5, 0.7, 1.0]
    - primary_weight: fixed at 1.0
    
    Args:
        trajectory_df: Training trajectory data
        validation_policy_performance: Function that trains policy and returns validation score
        
    Returns:
        Tuple of (primary_weight, engagement_weight, intermediate_weight)
    """
    logger.info("Tuning reward weights via grid search...")
    
    best_score = -np.inf
    best_weights = (1.0, 0.3, 0.5)  # Default
    
    engagement_values = [0.1, 0.3, 0.5, 0.7]
    intermediate_values = [0.3, 0.5, 0.7, 1.0]
    
    for eng_w in engagement_values:
        for int_w in intermediate_values:
            logger.info(f"Testing weights: engagement={eng_w}, intermediate={int_w}")
            
            # Shape rewards with these weights
            shaper = RewardShaper(
                primary_weight=1.0,
                engagement_weight=eng_w,
                intermediate_weight=int_w
            )
            shaped_df = shaper.shape_trajectory_rewards(trajectory_df.copy())
            
            # Train policy and evaluate
            score = validation_policy_performance(shaped_df)
            
            logger.info(f"  Validation score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_weights = (1.0, eng_w, int_w)
                logger.info(f"  New best weights")
    
    logger.info(f"\nBest weights: primary={best_weights[0]}, "
               f"engagement={best_weights[1]}, intermediate={best_weights[2]}")
    logger.info(f"Best validation score: {best_score:.4f}")
    
    return best_weights


def main():
    """
    Example usage: Shape rewards for a trajectory dataset.
    """
    logger.info("=" * 80)
    logger.info("Reward Shaping Pipeline")
    logger.info("=" * 80)
    
    # Load trajectory data
    trajectory_df = pd.read_parquet(data_dir / "enhanced_trajectories_daily.parquet")
    
    # Initialize shaper with default weights
    shaper = RewardShaper(
        primary_weight=1.0,
        engagement_weight=0.3,
        intermediate_weight=0.5
    )
    
    # Shape rewards
    shaped_df = shaper.shape_trajectory_rewards(trajectory_df)
    
    # Save shaped rewards
    output_path = data_dir / "trajectories_with_shaped_rewards.parquet"
    shaped_df.to_parquet(output_path, compression='snappy', index=False)
    
    logger.info(f"\nShaped rewards saved to: {output_path}")
    logger.info(f"Total observations: {len(shaped_df):,}")
    
    # Plot reward distributions
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    shaped_df['reward_primary'].hist(bins=50, ax=axes[0, 0])
    axes[0, 0].set_title('Primary Reward Distribution')
    axes[0, 0].set_xlabel('Reward Value')
    
    shaped_df['reward_engagement'].hist(bins=50, ax=axes[0, 1])
    axes[0, 1].set_title('Engagement Reward Distribution')
    axes[0, 1].set_xlabel('Reward Value')
    
    shaped_df['reward_intermediate'].hist(bins=50, ax=axes[1, 0])
    axes[1, 0].set_title('Intermediate Reward Distribution')
    axes[1, 0].set_xlabel('Reward Value')
    
    shaped_df['reward_shaped'].hist(bins=50, ax=axes[1, 1])
    axes[1, 1].set_title('Total Shaped Reward Distribution')
    axes[1, 1].set_xlabel('Reward Value')
    
    plt.tight_layout()
    plt.savefig(_REPO_ROOT / 'figures' / 'reward_distributions.png', dpi=300)
    logger.info(f"Reward distribution plots saved")
    
    logger.info("\n" + "=" * 80)
    logger.info("Reward shaping complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
