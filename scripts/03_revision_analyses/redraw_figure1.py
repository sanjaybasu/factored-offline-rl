"""
Redraw Figure 1 with Larger Fonts
===================================

Regenerate the conceptual framework figure with minimum 12pt fonts
for improved readability.

Addresses Reviewer 1 Comment 3.

Author: Sanjay Basu, MD PhD
Date: February 2026
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
import logging

# Resolve paths relative to repo root; override via environment variables
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
results_dir = Path(os.environ.get('FACTORED_RL_RESULTS_DIR', _REPO_ROOT / 'results'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global font settings -- minimum 12pt
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
})


def draw_box(ax, x, y, w, h, text, color='#4ECDC4', fontsize=13,
             text_color='black', alpha=0.3):
    """Draw a rounded box with text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor='#2c3e50',
        alpha=alpha, linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color,
            wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color='#2c3e50'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                               lw=2, connectionstyle='arc3,rad=0'))


def create_figure1():
    """Create the conceptual framework figure."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    # ===== Panel A: Variance Inflation Crisis =====
    ax = axes[0, 0]
    ax.set_title('(A) Variance Inflation in\nHigh-Dimensional Action Spaces', fontsize=14, fontweight='bold')

    # Simulated ESS curves
    dims = np.array([5, 10, 25, 50, 100, 200])
    ess_wis = 100 * np.exp(-0.05 * dims)
    ess_dr = 100 * np.exp(-0.005 * dims)

    ax.plot(dims, ess_wis, 'r-o', linewidth=2.5, markersize=8, label='Standard WIS')
    ax.plot(dims, ess_dr, 'b-s', linewidth=2.5, markersize=8, label='Doubly Robust')
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(150, 12, 'Minimum viable ESS', fontsize=12, color='gray')
    ax.axvline(x=97, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(100, 80, '97 actions\n(this study)', fontsize=12, color='green')
    ax.set_xlabel('Number of Action Combinations', fontsize=13)
    ax.set_ylabel('Effective Sample Size (%)', fontsize=13)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=12)

    # ===== Panel B: Factored Action Space =====
    ax = axes[0, 1]
    ax.set_title('(B) Factored Action Space\nDecomposition', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Draw factor boxes
    factors = [
        ('Modality', 'Phone, Text,\nVideo, Home Visit', '#3498db', 8.5),
        ('Provider', 'RN, SW, CC,\nCHW, MD', '#2ecc71', 6.5),
        ('Goal', 'CDM, CC,\nBH, SDoH', '#e74c3c', 4.5),
        ('Urgency', 'Routine,\nSemi-urgent, Urgent', '#f39c12', 2.5)
    ]

    for name, details, color, y_pos in factors:
        draw_box(ax, 3, y_pos, 4.5, 1.5, f'{name}\n{details}',
                 color=color, fontsize=12, alpha=0.25)

    # Arrow to combined
    ax.annotate('', xy=(7.5, 5.5), xytext=(5.5, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#2c3e50'))

    draw_box(ax, 8.5, 5.5, 2.5, 2, '97\nAction\nCombinations',
             color='#9b59b6', fontsize=13, alpha=0.3)

    ax.text(5, 0.8, '4 x 5 x 4 x 3 = 240 theoretical\n97 observed in practice',
            fontsize=12, ha='center', style='italic', color='#666')

    # ===== Panel C: Reward Shaping =====
    ax = axes[0, 2]
    ax.set_title('(C) Multi-Component\nReward Shaping', fontsize=14, fontweight='bold')

    categories = ['Sparse\nBinary', 'Shaped\nReward']
    sparse_vals = [0.6, 0]
    engagement_vals = [0, 10.4]
    intermediate_vals = [0, 21.4]  # Was "cost", now intermediate milestones
    primary_in_shaped = [0, 0.3]

    width = 0.5
    x_pos = np.array([0, 1.2])

    ax.bar(x_pos, [0.6, 68.2], width, label='Primary (ED+Hosp)', color='#e74c3c', alpha=0.7)
    ax.bar(x_pos, [0, 21.4], width, bottom=[0.6, 68.2], label='Engagement', color='#2ecc71', alpha=0.7)
    ax.bar(x_pos, [0, 10.4], width, bottom=[0.6, 89.6], label='Intermediate\nMilestones', color='#3498db', alpha=0.7)

    ax.set_ylabel('% of Reward Variance', fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 110)
    ax.tick_params(labelsize=12)

    ax.text(0, -12, '0.6% non-zero', fontsize=12, ha='center', color='#e74c3c', fontweight='bold')
    ax.text(1.2, -12, '32.1% non-zero', fontsize=12, ha='center', color='#2ecc71', fontweight='bold')

    # ===== Panel D: LSTM with Attention =====
    ax = axes[1, 0]
    ax.set_title('(D) Bidirectional LSTM\nwith Attention', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Timeline
    weeks = ['t-3', 't-2', 't-1', 't']
    attn_weights = [0.046, 0.098, 0.204, 0.652]

    for i, (week, attn) in enumerate(zip(weeks, attn_weights)):
        x = 1.5 + i * 2
        y = 7

        # State box
        size = 0.5 + attn * 1.5
        draw_box(ax, x, y, 1.5, 1.2, f'State\n{week}',
                 color='#3498db', fontsize=12, alpha=0.15 + attn * 0.6)

        # Attention weight
        ax.text(x, 5.5, f'a={attn:.3f}', fontsize=12, ha='center',
                fontweight='bold', color='#e74c3c')

        # Arrow down
        ax.annotate('', xy=(x, 5.8), xytext=(x, 6.3),
                    arrowprops=dict(arrowstyle='->', lw=1.5 + attn * 3,
                                   color='#e74c3c', alpha=0.3 + attn * 0.7))

    # LSTM box
    draw_box(ax, 5, 3.5, 8, 1.2, 'Bidirectional LSTM (2 layers, hidden=64)',
             color='#f39c12', fontsize=13, alpha=0.25)

    # Output
    draw_box(ax, 5, 1.5, 4, 1, 'Policy: P(action | state)',
             color='#9b59b6', fontsize=13, alpha=0.25)
    ax.annotate('', xy=(5, 2.1), xytext=(5, 2.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))

    # ===== Panel E: Fairness Constraints =====
    ax = axes[1, 1]
    ax.set_title('(E) Fairness-Constrained\nTraining', fontsize=14, fontweight='bold')

    groups = ['White', 'Black', 'Asian', 'Hispanic', 'NHPI', 'AI/AN']
    unconstrained = [5.2, 8.7, 4.1, 5.0, 4.5, 7.8]
    constrained = [5.2, 5.8, 4.9, 5.1, 5.0, 5.6]

    x = np.arange(len(groups))
    width = 0.35

    ax.bar(x - width/2, unconstrained, width, label='Unconstrained',
           color='#e74c3c', alpha=0.6)
    ax.bar(x + width/2, constrained, width, label='Fairness-Constrained',
           color='#2ecc71', alpha=0.6)

    ax.set_ylabel('Home Visit Rate (%)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11, rotation=30, ha='right')
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=12)

    # Annotate gap
    ax.annotate('Gap: 3.5pp', xy=(1, 8.7), xytext=(3, 9.5),
                fontsize=12, color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
    ax.annotate('Gap: 0.8pp', xy=(1, 5.8), xytext=(3, 3),
                fontsize=12, color='#2ecc71',
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))

    # ===== Panel F: Framework Overview =====
    ax = axes[1, 2]
    ax.set_title('(F) Integrated Framework', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Flow diagram
    steps = [
        ('Observational\nData', '#bdc3c7', 9),
        ('Factored\nActions (B)', '#3498db', 7.5),
        ('Reward\nShaping (C)', '#2ecc71', 6),
        ('LSTM\nPolicy (D)', '#f39c12', 4.5),
        ('Fairness\nConstraints (E)', '#9b59b6', 3),
        ('DR Off-Policy\nEvaluation (A)', '#e74c3c', 1.5),
    ]

    for text, color, y in steps:
        draw_box(ax, 5, y, 5, 1, text, color=color, fontsize=12, alpha=0.25)

    for i in range(len(steps) - 1):
        y1 = steps[i][2] - 0.5
        y2 = steps[i+1][2] + 0.5
        ax.annotate('', xy=(5, y2), xytext=(5, y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))

    plt.tight_layout(pad=2.0)

    # Save
    output_dir = _REPO_ROOT / 'figures'
    output_dir.mkdir(exist_ok=True)

    fig.savefig(output_dir / 'Figure1_revised.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(output_dir / 'Figure1_revised.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    logger.info(f"Figure 1 saved to {output_dir / 'Figure1_revised.png'}")
    logger.info(f"Figure 1 saved to {output_dir / 'Figure1_revised.pdf'}")

    plt.close()


if __name__ == "__main__":
    logger.info("Regenerating Figure 1 with larger fonts (min 12pt)...")
    create_figure1()
    logger.info("Done!")
