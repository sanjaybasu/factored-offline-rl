"""
Generate synthetic care management data for demonstration purposes.

This synthetic data mimics the structure of real Medicaid care management data
but contains NO real patient information. It is for code demonstration only.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


def generate_synthetic_data(
    n_patients: int = 1000,
    n_weeks_per_patient: int = 52,
    seed: int = 42
) -> Tuple[np.ndarray, List[Tuple], np.ndarray]:
    """
    Generate synthetic care management trajectories.
    
    Args:
        n_patients: Number of synthetic patients
        n_weeks_per_patient: Number of weeks of data per patient
        seed: Random seed for reproducibility
        
    Returns:
        states: (N, 7) array of state features
        actions: List of (modality, provider, goal, urgency) tuples
        rewards: (N,) array of rewards (negative utilization)
    """
    np.random.seed(seed)
    
    n_timesteps = n_patients * n_weeks_per_patient
    
    # Action space definitions
    modalities = ['NONE', 'PHONE_CALL', 'SMS_TEXT', 'VIDEO_VISIT', 'HOME_VISIT', 'EHR_COMMUNICATION']
    providers = ['NONE', 'CHW', 'CARE_COORDINATOR', 'PHARMACIST', 'THERAPIST']
    goals = [
        'NONE', 'MEDICATION_ADHERENCE', 'HYPERTENSION', 'DIABETES', 'ASTHMA_COPD',
        'DEPRESSION', 'MENTAL_HEALTH', 'OTHER_MENTAL_BEHAVIORAL', 'FOOD_INSECURITY',
        'HOUSING_INSECURITY', 'TRANSPORTATION', 'FINANCIAL', 'EMPLOYMENT',
        'PCP_APPOINTMENT', 'INSURANCE_COVERAGE', 'DENTAL', 'CARE', 
        'OTHER', 'MEDICATION_OPTIMIZATION'
    ]
    urgencies = ['ROUTINE', 'URGENT']
    
    # Generate synthetic states
    # State features: [encounters, ed_visits, ip_visits, calls, texts, enrolled_days, costs]
    states = np.zeros((n_timesteps, 7))
    
    # Encounters (0-5 per week)
    states[:, 0] = np.random.poisson(0.5, n_timesteps)
    
    # ED visits (sparse, mostly 0)
    states[:, 1] = np.random.binomial(1, 0.02, n_timesteps)
    
    # Hospitalizations (very sparse)
    states[:, 2] = np.random.binomial(1, 0.01, n_timesteps)
    
    # Phone calls (0-3 per week)
    states[:, 3] = np.random.poisson(0.3, n_timesteps)
    
    # Text messages (0-2 per week)
    states[:, 4] = np.random.poisson(0.2, n_timesteps)
    
    # Enrolled days (increasing over time within patient)
    for i in range(n_patients):
        start_idx = i * n_weeks_per_patient
        end_idx = start_idx + n_weeks_per_patient
        states[start_idx:end_idx, 5] = np.arange(7, 7 + n_weeks_per_patient * 7, 7)
    
    # Healthcare costs (90-day window, $0-$50k)
    states[:, 6] = np.abs(np.random.gamma(2, 2000, n_timesteps))
    
    # Generate synthetic actions
    actions = []
    for _ in range(n_timesteps):
        # 30% chance of no action
        if np.random.random() < 0.3:
            action = ('NONE', 'NONE', 'NONE', 'ROUTINE')
        else:
            modality = np.random.choice(modalities[1:])  # Exclude NONE
            provider = np.random.choice(providers[1:])  # Exclude NONE
            goal = np.random.choice(goals[1:])  # Exclude NONE
            urgency = np.random.choice(urgencies, p=[0.85, 0.15])  # 85% routine
            action = (modality, provider, goal, urgency)
        actions.append(action)
    
    # Generate synthetic rewards
    # Reward = -(ED visits + hospitalizations in next 30 days)
    # Sparse: most are 0, some are -1 to -3
    base_rate = 0.01  # 1% base adverse event rate
    
    # Higher engagement (more encounters) slightly reduces risk
    engagement_factor = np.clip(1 - states[:, 0] * 0.05, 0.5, 1.0)
    
    # Recent utilization increases risk
    recent_util_factor = 1 + states[:, 1] * 2 + states[:, 2] * 3
    
    event_prob = base_rate * engagement_factor * recent_util_factor
    event_prob = np.clip(event_prob, 0, 0.1)  # Cap at 10%
    
    # Sample events
    has_event = np.random.binomial(1, event_prob)
    event_count = has_event * (1 + np.random.poisson(0.2, n_timesteps))
    
    rewards = -event_count
    
    print(f"Generated {n_timesteps:,} synthetic timesteps")
    print(f"  Unique patients: {n_patients}")
    print(f"  Weeks per patient: {n_weeks_per_patient}")
    print(f"  State dimension: {states.shape[1]}")
    print(f"  % with adverse events: {100 * (rewards < 0).mean():.2f}%")
    print(f"  Mean reward: {rewards.mean():.4f}")
    
    return states, actions, rewards


def save_synthetic_data(
    states: np.ndarray,
    actions: List[Tuple],
    rewards: np.ndarray,
    output_path: str = 'synthetic_trajectories.parquet'
):
    """Save synthetic data to parquet file."""
    
    df = pd.DataFrame({
        'num_encounters_last_week': states[:, 0],
        'num_ed_visits_last_week': states[:, 1],
        'num_ip_visits_last_week': states[:, 2],
        'num_calls_last_week': states[:, 3],
        'num_texts_last_week': states[:, 4],
        'enrolled_days': states[:, 5],
        'total_paid': states[:, 6],
        'action_modality': [a[0] for a in actions],
        'action_provider': [a[1] for a in actions],
        'action_goal': [a[2] for a in actions],
        'action_urgency': [a[3] for a in actions],
        'reward': rewards
    })
    
    df.to_parquet(output_path, index=False)
    print(f"Saved synthetic data to {output_path}")


if __name__ == '__main__':
    # Generate synthetic data
    states, actions, rewards = generate_synthetic_data(
        n_patients=1000,
        n_weeks_per_patient=52,
        seed=42
    )
    
    # Save to file
    save_synthetic_data(states, actions, rewards)
    
    # Demonstrate training
    print("\nTo train a policy on this synthetic data:")
    print("  from src.factored_action_space import FactoredActionSpace")
    print("  from src.offline_rl_policy_awr import train_awr_factored")
    print("  ")
    print("  action_space = FactoredActionSpace(embedding_dim=32)")
    print("  actor, critic = train_awr_factored(states, actions, rewards, action_space)")
