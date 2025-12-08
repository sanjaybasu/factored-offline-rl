"""
Attention Weight Visualization
===============================

Extract and visualize attention patterns from trained LSTM policy.

Author: Sanjay Basu, MD PhD
Date: December 2024
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 8
plt.rcParams['figure.dpi'] = 300

def load_model(model_dir, device='cpu'):
    """Load trained policy."""
    from phase2_train_full import LSTMPolicyNetwork
    
    checkpoint = torch.load(model_dir / 'lstm_policy_10pct_best.pt', map_location=device)
    config = checkpoint['config']
    
    policy = LSTMPolicyNetwork(
        config['state_dim'],
        config['hidden_dim'],
        config['num_actions']
    ).to(device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    return policy, config

def extract_attention_weights(policy, test_df, state_cols, n_samples=1000, device='cpu'):
    """Extract attention weights from model."""
    logger.info(f"Extracting attention weights from {n_samples} samples...")
    
    sample_df = test_df.sample(min(n_samples, len(test_df)), random_state=42)
    
    all_attention_weights = []
    
    with torch.no_grad():
        for idx in range(len(sample_df)):
            row = sample_df.iloc[idx]
            
            # Prepare state sequence
            state_seq = []
            for t in range(4):
                state_t = [float(row.get(f'{col}_t{t}', 0)) for col in state_cols]
                state_seq.append(state_t)
            
            states = torch.FloatTensor([state_seq]).to(device)
            
            # Forward pass to get attention weights
            _, attn_weights = policy(states)
            
            # attn_weights shape varies - handle different cases
            if attn_weights is not None:
                # Get attention weights for the current sample
                if len(attn_weights.shape) == 4:  # (batch, heads, seq, seq)
                    attn = attn_weights[0].mean(dim=0)  # Average over heads
                    attn_last = attn[-1, :].cpu().numpy()  # Last position
                elif len(attn_weights.shape) == 3:  # (batch, seq, seq) - already averaged
                    attn_last = attn_weights[0, -1, :].cpu().numpy()
                elif len(attn_weights.shape) == 2:  # (seq, seq) - single sample
                    attn_last = attn_weights[-1, :].cpu().numpy()
                else:  # (seq,) - just the weights we need
                    attn_last = attn_weights.cpu().numpy()
                
                # Ensure we have 4 values for 4-week sequence
                if len(attn_last) == 4:
                    all_attention_weights.append(attn_last)
    
    return np.array(all_attention_weights)

def plot_attention_analysis(attention_weights, output_dir):
    """Generate attention visualization figures."""
    logger.info("Generating attention visualizations...")
    
    # Create figure
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    
    # Panel A: Mean attention profile
    ax1 = fig.add_subplot(gs[0, 0])
    
    mean_weights = attention_weights.mean(axis=0)
    std_weights = attention_weights.std(axis=0)
    positions = ['t-3', 't-2', 't-1', 't (Current)']
    
    ax1.plot(positions, mean_weights, 'o-', linewidth=2, markersize=8, color='#0066CC')
    ax1.fill_between(range(4), mean_weights - std_weights, mean_weights + std_weights, 
                     alpha=0.2, color='#0066CC')
    
    ax1.set_ylabel('Mean Attention Weight')
    ax1.set_title('A. Temporal Attention Profile', fontweight='bold')
    ax1.set_ylim(0, max(mean_weights) * 1.2)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (pos, val) in enumerate(zip(positions, mean_weights)):
        ax1.text(i, val + 0.02, f'{val:.3f}', ha='center', fontsize=7)
    
    # Panel B: Attention distribution by position
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create violin plot
    data_for_violin = [attention_weights[:, i] for i in range(4)]
    parts = ax2.violinplot(data_for_violin, positions=range(4), showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#4682B4')
        pc.set_alpha(0.6)
    
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(positions)
    ax2.set_ylabel('Attention Weight')
    ax2.set_title('B. Attention Weight Distribution by Position', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'Attention_Analysis_Real.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved {output_dir / 'Attention_Analysis_Real.png'}")
    
    # Print statistics
    logger.info("\nAttention Weight Statistics:")
    for i, pos in enumerate(positions):
        logger.info(f"  {pos}: mean={mean_weights[i]:.3f}, std={std_weights[i]:.3f}")

def main():
    logger.info("="*80)
    logger.info("ATTENTION WEIGHT VISUALIZATION")
    logger.info("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    data_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/data_10pct')
    model_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/models')
    figures_dir = Path('/Users/sanjaybasu/waymark-local/notebooks/factored_rl/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("\nLoading trained model...")
    policy, config = load_model(model_dir, device)
    state_cols = config['state_cols']
    
    # Load test data
    logger.info("Loading test data...")
    test_df = pd.read_parquet(data_dir / 'sequences_test.parquet')
    logger.info(f"  Test: {len(test_df):,} sequences")
    
    # Extract attention weights
    attention_weights = extract_attention_weights(
        policy, test_df, state_cols, n_samples=1000, device=device
    )
    
    # Generate visualizations
    plot_attention_analysis(attention_weights, figures_dir)
    
    logger.info("\n" + "="*80)
    logger.info("✓ VISUALIZATION COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    main()
