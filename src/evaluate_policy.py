import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
import sys
import json

# Add script directory to path
sys.path.append(str(Path(__file__).parent))
from factored_action_space import FactoredActionSpace
from offline_rl_policy_awr_factored import ActionEmbeddingNetwork, select_action

def evaluate_policy(
    policy_path: str,
    data_path: str,
    output_dir: str,
    num_samples: int = 5000,
    device: str = 'cpu'
):
    """
    Evaluate trained policy and generate visualizations.
    """
    print(f"Loading policy from {policy_path}...")
    checkpoint = torch.load(policy_path, map_location=device)
    
    # Initialize action space
    action_space = FactoredActionSpace(embedding_dim=32)
    
    # Initialize actor
    # We need to know state_dim from data or config. 
    # Assuming standard state_dim=7 based on previous logs (or we can infer from data)
    # Let's load a bit of data first to be sure.
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Extract state columns
    if 'state_cols' in checkpoint:
        state_cols = checkpoint['state_cols']
        print(f"Loaded state columns from checkpoint: {state_cols}")
    else:
        # Fallback to known columns from run_factored_rl_pipeline_optimized.py
        print("Warning: state_cols not found in checkpoint. Using default list.")
        state_cols = [
            'num_encounters_last_week', 'num_ed_visits_last_week', 'num_ip_visits_last_week',
            'num_calls_last_week', 'num_texts_last_week', 'enrolled_days', 'total_paid'
        ]
            
    # Ensure these exist in data
    available_cols = [c for c in state_cols if c in df.columns]
    if len(available_cols) < len(state_cols):
        print(f"Warning: Missing state columns. Found: {available_cols}")
        print(f"Missing: {set(state_cols) - set(available_cols)}")
        # Fill missing with 0
        for col in set(state_cols) - set(available_cols):
            df[col] = 0
            
    print(f"Using state columns: {state_cols}")
    state_dim = len(state_cols)
    
    actor = ActionEmbeddingNetwork(state_dim, action_space).to(device)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()
    
    # Sample states for evaluation
    sample_df = df.sample(n=min(num_samples, len(df)), random_state=42)
    states = sample_df[state_cols].fillna(0).values
    
    print(f"Running inference on {len(states)} states...")
    
    recommended_actions = []
    
    with torch.no_grad():
        for i in range(len(states)):
            state = states[i]
            # Select best action
            best_action, _ = select_action(actor, state, action_space, top_k=1, device=device)
            recommended_actions.append(best_action)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(states)}", end='\r')
                
    print("\nInference complete.")
    
    # Convert to DataFrame for analysis
    rec_df = pd.DataFrame(recommended_actions, columns=['modality', 'provider', 'goal', 'urgency'])
    
    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Action Distribution Analysis
    print("Generating action distribution plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sns.countplot(data=rec_df, y='modality', ax=axes[0,0], order=rec_df['modality'].value_counts().index)
    axes[0,0].set_title('Recommended Modalities')
    
    sns.countplot(data=rec_df, y='provider', ax=axes[0,1], order=rec_df['provider'].value_counts().index)
    axes[0,1].set_title('Recommended Providers')
    
    sns.countplot(data=rec_df, y='goal', ax=axes[1,0], order=rec_df['goal'].value_counts().index)
    axes[1,0].set_title('Recommended Goals')
    
    sns.countplot(data=rec_df, y='urgency', ax=axes[1,1], order=rec_df['urgency'].value_counts().index)
    axes[1,1].set_title('Recommended Urgency')
    
    plt.tight_layout()
    plt.savefig(out_path / 'action_distributions.png')
    plt.close()
    
    # 2. Action Embedding t-SNE
    print("Generating embedding t-SNE...")
    
    # Get all valid actions and their embeddings
    # Sampling if too many
    num_viz_actions = min(2000, action_space.num_actions)
    viz_actions = [action_space.sample_valid_action() for _ in range(num_viz_actions)]
    
    embeddings = np.array([action_space.get_action_embedding(a) for a in viz_actions])
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    emb_2d = tsne.fit_transform(embeddings)
    
    # Plot colored by Modality
    plt.figure(figsize=(12, 8))
    modalities = [a[0] for a in viz_actions]
    sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=modalities, palette='viridis', s=50)
    plt.title('Action Embeddings t-SNE (by Modality)')
    plt.savefig(out_path / 'embeddings_tsne_modality.png')
    plt.close()
    
    # Plot colored by Goal
    plt.figure(figsize=(12, 8))
    goals = [a[2] for a in viz_actions]
    # Simplify goals for legend if too many
    sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=goals, palette='tab20', s=50)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Action Embeddings t-SNE (by Goal)')
    plt.tight_layout()
    plt.savefig(out_path / 'embeddings_tsne_goal.png')
    plt.close()
    
    # 3. Save summary stats
    summary = {
        'total_samples': len(states),
        'modality_counts': rec_df['modality'].value_counts().to_dict(),
        'provider_counts': rec_df['provider'].value_counts().to_dict(),
        'goal_counts': rec_df['goal'].value_counts().to_dict(),
        'urgency_counts': rec_df['urgency'].value_counts().to_dict()
    }
    
    with open(out_path / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    evaluate_policy(args.policy, args.data, args.output)
