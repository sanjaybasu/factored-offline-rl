"""
Proportional Parity Fairness Analysis
========================================

Compute proportional parity metric alongside demographic parity.
Proportional parity allocates interventions proportional to each
group's clinical need (baseline acute care utilization rate).

Addresses Reviewer 2 Comment 4.

Author: Sanjay Basu, MD PhD
Date: February 2026
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import sys

# Resolve paths relative to repo root; override via environment variables
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
data_dir = Path(os.environ.get('FACTORED_RL_DATA_DIR', _REPO_ROOT / 'data'))
model_dir = Path(os.environ.get('FACTORED_RL_MODEL_DIR', _REPO_ROOT / 'models'))
results_dir = Path(os.environ.get('FACTORED_RL_RESULTS_DIR', _REPO_ROOT / 'results'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('proportional_parity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.append(str(_REPO_ROOT / 'src'))


def compute_proportional_parity(test_df, policy_actions, num_actions,
                                 race_col='race',
                                 high_resource_actions=None):
    """
    Compute proportional parity metric.

    Proportional parity: each group's share of high-resource interventions
    should be proportional to the group's share of clinical need.

    Need is defined as the group's acute care utilization rate
    (ED visits + hospitalizations).

    Metric: For each group g,
        PP_ratio(g) = [P(high_resource | group=g)] / [P(need | group=g) / P(need)]

    A ratio of 1.0 = perfectly proportional allocation.

    Args:
        test_df: Test dataframe with outcomes and demographics
        policy_actions: Array of policy-recommended actions (greedy)
        num_actions: Total number of actions
        race_col: Column name for race/ethnicity
        high_resource_actions: Set of action indices considered high-resource
                              (if None, uses top 10% of action space)

    Returns:
        Dict with per-group results and aggregate metrics
    """
    if high_resource_actions is None:
        # Use top 10% of action indices as proxy
        high_resource_actions = set(range(int(num_actions * 0.9), num_actions))

    is_high_resource = np.array([a in high_resource_actions for a in policy_actions])

    # Need proxy: acute care utilization rate per group
    # Try multiple column name patterns
    need_cols = [
        ('ed_visits_30d', 'hospitalizations_30d'),
        ('num_ed_visits_last_week_t3', 'num_ip_visits_last_week_t3'),
        ('ed_visits_7d_t3', 'hospitalizations_7d_t3'),
    ]

    need = None
    for ed_col, ip_col in need_cols:
        if ed_col in test_df.columns and ip_col in test_df.columns:
            need = (test_df[ed_col].values > 0).astype(float) + \
                   (test_df[ip_col].values > 0).astype(float)
            need = (need > 0).astype(float)  # Binary: any acute care event
            logger.info(f"Using need columns: {ed_col}, {ip_col}")
            break

    if need is None:
        # Fallback: use reward_shaped as proxy (more negative = more need)
        logger.warning("No direct outcome columns found; using reward_shaped as need proxy")
        need = (test_df['reward_shaped'].values < test_df['reward_shaped'].median()).astype(float)

    overall_need_rate = need.mean()
    overall_intervention_rate = is_high_resource.mean()

    results = []
    groups = test_df[race_col].unique()

    for group in sorted(groups, key=str):
        mask = (test_df[race_col].values == group)
        n = mask.sum()

        if n < 50:
            logger.info(f"  Skipping group '{group}' (n={n} < 50)")
            continue

        group_need_rate = need[mask].mean()
        group_intervention_rate = is_high_resource[mask].mean()

        # Proportional parity ratio
        # Expected intervention rate if proportional to need:
        #   expected = overall_intervention_rate * (group_need / overall_need)
        if overall_need_rate > 0 and group_need_rate > 0:
            expected_rate = overall_intervention_rate * (group_need_rate / overall_need_rate)
            pp_ratio = group_intervention_rate / expected_rate if expected_rate > 0 else float('inf')
        else:
            pp_ratio = 1.0  # If no need, ratio is undefined; default to 1

        # Demographic parity (absolute intervention rate)
        dp_rate = group_intervention_rate

        result = {
            'group': str(group),
            'n': int(n),
            'population_pct': float(n / len(test_df) * 100),
            'need_rate': float(group_need_rate * 100),
            'intervention_rate': float(group_intervention_rate * 100),
            'pp_ratio': float(pp_ratio),
            'dp_rate': float(dp_rate * 100)
        }
        results.append(result)

        logger.info(f"  {group}: n={n}, need={group_need_rate*100:.2f}%, "
                    f"intervention={group_intervention_rate*100:.2f}%, "
                    f"PP ratio={pp_ratio:.3f}")

    # Aggregate metrics
    if len(results) >= 2:
        pp_ratios = [r['pp_ratio'] for r in results if r['pp_ratio'] != float('inf')]
        dp_rates = [r['dp_rate'] for r in results]

        pp_gap = max(pp_ratios) - min(pp_ratios) if pp_ratios else None
        pp_max_deviation = max(abs(r - 1.0) for r in pp_ratios) if pp_ratios else None
        dp_gap = max(dp_rates) - min(dp_rates)
    else:
        pp_gap = None
        pp_max_deviation = None
        dp_gap = None

    return {
        'per_group': results,
        'aggregate': {
            'pp_ratio_range': [float(min(pp_ratios)), float(max(pp_ratios))] if pp_ratios else None,
            'pp_max_deviation_from_1': float(pp_max_deviation) if pp_max_deviation is not None else None,
            'dp_gap_pp': float(dp_gap) if dp_gap is not None else None,
            'overall_need_rate': float(overall_need_rate * 100),
            'overall_intervention_rate': float(overall_intervention_rate * 100)
        }
    }


def main():
    logger.info("=" * 80)
    logger.info("PROPORTIONAL PARITY FAIRNESS ANALYSIS")
    logger.info("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths (using module-level defaults with env var overrides)
    results_dir.mkdir(exist_ok=True)

    # Load data
    test_df = pd.read_parquet(data_dir / 'sequences_test.parquet')

    state_cols = ['age', 'num_encounters_last_week', 'num_ed_visits_last_week',
                  'num_ip_visits_last_week', 'num_interventions_last_week']

    # Load trained policy
    from phase2_train_full import LSTMPolicyNetwork
    model_path = model_dir / 'lstm_policy_10pct_best.pt'
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    num_actions = config['num_actions']

    policy = LSTMPolicyNetwork(
        config['state_dim'], config['hidden_dim'], num_actions
    ).to(device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()

    # Filter test data
    test_df = test_df[test_df['action_idx'] < num_actions].copy()

    # Subsample if very large
    if len(test_df) > 50000:
        test_sample = test_df.sample(50000, random_state=42)
    else:
        test_sample = test_df

    logger.info(f"Test set: {len(test_sample)} sequences")

    # Get policy actions
    states_list = []
    for idx in range(len(test_sample)):
        row = test_sample.iloc[idx]
        seq = []
        for t in range(4):
            s = [float(row.get(f'{col}_t{t}', 0)) for col in state_cols]
            seq.append(s)
        states_list.append(seq)

    states_tensor = torch.FloatTensor(np.array(states_list)).to(device)

    with torch.no_grad():
        logits, _ = policy(states_tensor)
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    greedy_actions = np.argmax(probs, axis=1)

    # Determine high-resource action indices
    # Check if action metadata is available
    high_resource = None
    action_meta_path = data_dir / 'action_metadata.json'
    if action_meta_path.exists():
        with open(action_meta_path) as f:
            action_meta = json.load(f)
        # Look for home visit, physician actions
        high_resource = set()
        for idx, meta in enumerate(action_meta):
            if any(kw in str(meta).lower() for kw in ['home visit', 'physician', 'in-person']):
                high_resource.add(idx)

    if not high_resource:
        # Use top 10% as proxy
        high_resource = set(range(int(num_actions * 0.9), num_actions))
        logger.info(f"Using top 10% of actions as high-resource proxy: {len(high_resource)} actions")

    # Check available race column
    race_col = None
    for col in ['race', 'race_ethnicity', 'race_t3', 'race_t0']:
        if col in test_sample.columns:
            race_col = col
            break

    if race_col is None:
        logger.error("No race/ethnicity column found in test data!")
        logger.info(f"Available columns: {list(test_sample.columns[:30])}")
        return

    logger.info(f"Using race column: {race_col}")
    logger.info(f"Race groups: {test_sample[race_col].value_counts().to_dict()}")

    # Compute proportional parity
    pp_results = compute_proportional_parity(
        test_sample, greedy_actions, num_actions,
        race_col=race_col,
        high_resource_actions=high_resource
    )

    # Save
    with open(results_dir / 'proportional_parity_results.json', 'w') as f:
        json.dump(pp_results, f, indent=2)

    logger.info(f"\nResults saved to {results_dir / 'proportional_parity_results.json'}")
    logger.info(f"\nAggregate metrics:")
    logger.info(f"  PP ratio range: {pp_results['aggregate']['pp_ratio_range']}")
    logger.info(f"  PP max deviation from 1.0: {pp_results['aggregate']['pp_max_deviation_from_1']}")
    logger.info(f"  DP gap: {pp_results['aggregate']['dp_gap_pp']}")

    logger.info("=" * 80)
    logger.info("PROPORTIONAL PARITY ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
