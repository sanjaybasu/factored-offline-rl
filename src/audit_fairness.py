import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append('/Users/sanjaybasu/waymark-local/foundation')
from scripts.factored_action_space import FactoredActionSpace
from scripts.offline_rl_policy_awr_factored import ActionEmbeddingNetwork

def audit_fairness(
    data_path: str,
    policy_path: str,
    output_dir: str = 'outputs/fairness_audit',
    n_samples: int = 10000
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Filter for valid demographics
    df = df[df['age'].notna() & df['gender'].notna() & df['race'].notna()]
    
    # Sample data
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
    
    print(f"Auditing on {len(df)} samples...")
    
    # Load policy
    checkpoint = torch.load(policy_path, map_location='cpu')
    action_space = FactoredActionSpace()
    # fc1 input is state_dim + action_embedding_dim (128)
    state_dim = checkpoint['actor']['fc1.weight'].shape[1] - 128
    
    actor = ActionEmbeddingNetwork(state_dim, action_space)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()
    
    # Prepare states
    state_cols = [
        'num_encounters_last_week', 'num_ed_visits_last_week', 'num_ip_visits_last_week',
        'num_calls_last_week', 'num_texts_last_week', 'enrolled_days', 'total_paid'
    ]
    
    # Use raw states (fillna 0) as in training pipeline
    states = df[state_cols].fillna(0).values
    
    print("Generating policy recommendations...")
    
    from scripts.offline_rl_policy_awr_factored import select_action
    
    recommendations = []
    with torch.no_grad():
        for i in range(len(states)):
            state = states[i]
            # Use the exact same selection logic as evaluation
            best_action, _ = select_action(actor, state, action_space, top_k=1, device='cpu')
            recommendations.append(best_action)
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i+1}/{len(states)}", end='\r')
    print()
            
    # Add recommendations to dataframe
    df['rec_modality'] = [r[0] for r in recommendations]
    df['rec_provider'] = [r[1] for r in recommendations]
    df['rec_goal'] = [r[2] for r in recommendations]
    df['rec_urgency'] = [r[3] for r in recommendations]
    
    # --- Fairness Analysis ---
    
    results = {}
    
    # 1. Race/Ethnicity Analysis
    print("\nAnalyzing Race/Ethnicity...")
    race_dist = df.groupby('race')['rec_modality'].value_counts(normalize=True).unstack().fillna(0)
    results['race_modality'] = race_dist.to_dict()
    
    # Check for disparity in "High Touch" (Home Visit) vs "Low Touch" (SMS/None)
    df['is_home_visit'] = df['rec_modality'] == 'HOME_VISIT'
    home_visit_rates = df.groupby('race')['is_home_visit'].mean()
    results['home_visit_rates_by_race'] = home_visit_rates.to_dict()
    
    # 2. Age Analysis
    print("\nAnalyzing Age...")
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '65+'])
    age_dist = df.groupby('age_group')['rec_modality'].value_counts(normalize=True).unstack().fillna(0)
    results['age_modality'] = age_dist.to_dict()
    
    # Check digital modality (Video/Text) usage by age
    df['is_digital'] = df['rec_modality'].isin(['VIDEO_VISIT', 'SMS_TEXT'])
    digital_rates = df.groupby('age_group')['is_digital'].mean()
    results['digital_rates_by_age'] = digital_rates.to_dict()
    
    # 3. Gender Analysis
    print("\nAnalyzing Gender...")
    gender_dist = df.groupby('gender')['rec_modality'].value_counts(normalize=True).unstack().fillna(0)
    results['gender_modality'] = gender_dist.to_dict()
    
    # Save results
    with open(output_path / 'fairness_audit_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Fairness audit complete. Results saved to {output_path}")
    
    # Print summary
    print("\n--- Fairness Audit Summary ---")
    print("\nHome Visit Rates by Race:")
    print(home_visit_rates)
    print("\nDigital Modality Rates by Age:")
    print(digital_rates)

if __name__ == "__main__":
    audit_fairness(
        data_path='/Users/sanjaybasu/waymark-local/foundation/outputs/rl_factored_pipeline_real_50pct/merged_trajectories.parquet',
        policy_path='/Users/sanjaybasu/waymark-local/foundation/outputs/rl_factored_pipeline_real_50pct/trained_policy.pt'
    )
