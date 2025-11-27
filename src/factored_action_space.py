"""
Encode expanded action space from real intervention data.
Maps (modality, provider, goal, urgency) tuples to action indices and embeddings.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class ActionComponent:
    """A single dimension of the factored action space."""
    name: str
    values: List[str]
    
    def __len__(self) -> int:
        return len(self.values)
    
    def encode(self, value: str) -> int:
        """Map value to index."""
        if value not in self.values:
            raise ValueError(f"Unknown {self.name}: {value}")
        return self.values.index(value)


# Define action space components based on real data analysis
MODALITIES = ActionComponent(
    name="modality",
    values=["NONE", "PHONE_CALL", "SMS_TEXT", "VIDEO_VISIT", "HOME_VISIT", "EHR_COMMUNICATION"]
)

PROVIDERS = ActionComponent(
    name="provider",
    values=["NONE", "CHW", "CARE_COORDINATOR", "PHARMACIST", "THERAPIST"]
)

GOALS = ActionComponent(
    name="goal",
    values=[
        "NONE",
        # Medical
        "MEDICATION_ADHERENCE",
        "HYPERTENSION",
        "DIABETES",
        "ASTHMA_COPD",
        "DEPRESSION",
        # Behavioral Health
        "MENTAL_HEALTH",
        "OTHER_MENTAL_BEHAVIORAL",
        # Social Determinants
        "FOOD_INSECURITY",
        "HOUSING_INSECURITY",
        "TRANSPORTATION",
        "FINANCIAL",
        "EMPLOYMENT",
        # Care Coordination
        "PCP_APPOINTMENT",
        "INSURANCE_COVERAGE",
        "DENTAL",
        "CARE",  # General care coordination
        "OTHER",
        "MEDICATION_OPTIMIZATION"
    ]
)

URGENCY = ActionComponent(
    name="urgency",
    values=["ROUTINE", "URGENT"]
)


class FactoredActionSpace:
    """
    Factored action space with ~1,000+ distinct actions.
    Actions are tuples: (modality, provider, goal, urgency)
    """
    
    def __init__(self, embedding_dim: int = 32):
        self.modalities = MODALITIES
        self.providers = PROVIDERS
        self.goals = GOALS
        self.urgency = URGENCY
        self.embedding_dim = embedding_dim
        
        # Compute total action space size
        self.num_actions = (
            len(self.modalities) * 
            len(self.providers) * 
            len(self.goals) * 
            len(self.urgency)
        )
        
        # Initialize learnable embeddings for each component
        np.random.seed(42)
        self.modality_embeddings = np.random.randn(len(self.modalities), embedding_dim) * 0.1
        self.provider_embeddings = np.random.randn(len(self.providers), embedding_dim) * 0.1
        self.goal_embeddings = np.random.randn(len(self.goals), embedding_dim) * 0.1
        self.urgency_embeddings = np.random.randn(len(self.urgency), embedding_dim) * 0.1
        
    def encode_action_tuple(self, action: Tuple[str, str, str, str]) -> int:
        """
        Encode action tuple to a single integer index.
        Returns index in [0, num_actions).
        """
        modality, provider, goal, urgency = action
        
        m_idx = self.modalities.encode(modality)
        p_idx = self.providers.encode(provider)
        g_idx = self.goals.encode(goal)
        u_idx = self.urgency.encode(urgency)
        
        # Flatten multi-dimensional index
        action_idx = (
            m_idx * (len(self.providers) * len(self.goals) * len(self.urgency)) +
            p_idx * (len(self.goals) * len(self.urgency)) +
            g_idx * len(self.urgency) +
            u_idx
        )
        return action_idx
    
    def decode_action_index(self, action_idx: int) -> Tuple[str, str, str, str]:
        """
        Decode integer index back to action tuple.
        """
        num_u = len(self.urgency)
        num_g = len(self.goals)
        num_p = len(self.providers)
        
        u_idx = action_idx % num_u
        action_idx //= num_u
        g_idx = action_idx % num_g
        action_idx //= num_g
        p_idx = action_idx % num_p
        m_idx = action_idx // num_p
        
        return (
            self.modalities.values[m_idx],
            self.providers.values[p_idx],
            self.goals.values[g_idx],
            self.urgency.values[u_idx]
        )
    
    def get_action_embedding(self, action: Tuple[str, str, str, str]) -> np.ndarray:
        """
        Get dense embedding for action by concatenating component embeddings.
        Returns: embedding of shape (4 * embedding_dim,)
        """
        modality, provider, goal, urgency = action
        
        m_emb = self.modality_embeddings[self.modalities.encode(modality)]
        p_emb = self.provider_embeddings[self.providers.encode(provider)]
        g_emb = self.goal_embeddings[self.goals.encode(goal)]
        u_emb = self.urgency_embeddings[self.urgency.encode(urgency)]
        
        return np.concatenate([m_emb, p_emb, g_emb, u_emb])
    
    def sample_valid_action(self) -> Tuple[str, str, str, str]:
        """
        Sample a random valid action.
        Enforces constraints (e.g., NONE modality => NONE provider/goal).
        """
        modality = np.random.choice(self.modalities.values)
        
        if modality == "NONE":
            return ("NONE", "NONE", "NONE", "ROUTINE")
        
        # Sample provider appropriate for modality
        if modality in ["PHONE_CALL", "SMS_TEXT", "HOME_VISIT"]:
            # CHWs do most of these
            provider = np.random.choice(self.providers.values[1:], p=[0.6, 0.2, 0.1, 0.1])
        else:
            provider = np.random.choice(self.providers.values[1:])
        
        # Sample goal based on provider specialty
        if provider == "PHARMACIST":
            goal = np.random.choice(["MEDICATION_ADHERENCE", "MEDICATION_OPTIMIZATION", "OTHER"])
        elif provider == "THERAPIST":
            goal = np.random.choice(["MENTAL_HEALTH", "DEPRESSION", "OTHER_MENTAL_BEHAVIORAL"])
        elif provider == "CHW":
            # CHWs handle SDOH and general coordination
            goal = np.random.choice([
                "FOOD_INSECURITY", "HOUSING_INSECURITY", "TRANSPORTATION",
                "FINANCIAL", "EMPLOYMENT", "CARE", "PCP_APPOINTMENT"
            ])
        else:  # CARE_COORDINATOR
            goal = np.random.choice([
                "PCP_APPOINTMENT", "INSURANCE_COVERAGE", "DENTAL", "CARE", "MEDICATION_OPTIMIZATION"
            ])
        
        urgency = np.random.choice(self.urgency.values, p=[0.85, 0.15])
        
        return (modality, provider, goal, urgency)


def parse_real_interventions(interventions_df: pd.DataFrame, goals_df: pd.DataFrame) -> List[Tuple]:
    """
    Parse real intervention data into factored actions.
    
    Args:
        interventions_df: DataFrame with columns [person_key, intervention_date, intervention]
        goals_df: DataFrame with columns [patient_id, category, status]
    
    Returns:
        List of (modality, provider, goal, urgency) tuples
    """
    actions = []
    
    # Map intervention types to modalities
    modality_map = {
        "SMS_TEXT": "SMS_TEXT",
        "SMS_TEXT_CMT": "SMS_TEXT",
        "PHONE_CALL": "PHONE_CALL",
        "PHONE_CALL_CMT": "PHONE_CALL",
        "VIDEO_VISIT": "VIDEO_VISIT",
        "HOME_VISIT": "HOME_VISIT",
        "EHR_COMMUNICATION": "EHR_COMMUNICATION",
    }
    
    # For each intervention, try to match with active goals
    for _, row in interventions_df.iterrows():
        person_key = row['person_key']
        intervention_type = row['intervention']
        
        if pd.isna(intervention_type) or intervention_type not in modality_map:
            continue
            
        modality = modality_map[intervention_type]
        
        # Find active goals for this member around intervention time
        matching_goals = goals_df[goals_df['patient_id'] == person_key]
        
        if len(matching_goals) > 0:
            # Use most recent active goal
            goal_category = matching_goals.iloc[0]['category']
            goal = goal_category if goal_category in GOALS.values else "OTHER"
        else:
            goal = "CARE"  # Default general care
        
        # Infer provider from modality and goal
        if goal in ["MEDICATION_ADHERENCE", "MEDICATION_OPTIMIZATION"]:
            provider = "PHARMACIST"
        elif goal in ["MENTAL_HEALTH", "DEPRESSION", "OTHER_MENTAL_BEHAVIORAL"]:
            provider = "THERAPIST"
        elif modality in ["HOME_VISIT", "SMS_TEXT"]:
            provider = "CHW"
        else:
            provider = "CARE_COORDINATOR"
        
        # Default to routine urgency
        urgency = "ROUTINE"
        
        actions.append((modality, provider, goal, urgency))
    
    return actions


if __name__ == "__main__":
    # Example usage
    action_space = FactoredActionSpace(embedding_dim=32)
    
    print(f"Total action space size: {action_space.num_actions}")
    print(f"Embedding dimension per action: {action_space.embedding_dim * 4}")
    
    # Test encoding/decoding
    test_action = ("PHONE_CALL", "CHW", "FOOD_INSECURITY", "ROUTINE")
    idx = action_space.encode_action_tuple(test_action)
    decoded = action_space.decode_action_index(idx)
    print(f"\nTest action: {test_action}")
    print(f"Encoded index: {idx}")
    print(f"Decoded: {decoded}")
    print(f"Match: {test_action == decoded}")
    
    # Test embedding
    emb = action_space.get_action_embedding(test_action)
    print(f"\nAction embedding shape: {emb.shape}")
    
    # Sample some valid actions
    print("\nSample valid actions:")
    for _ in range(5):
        action = action_space.sample_valid_action()
        print(f"  {action}")
