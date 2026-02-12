"""
Race-Outcome Mediation Analysis
=================================

Regression-based decomposition to estimate the proportion of the
race-outcome association explained by observed SDoH mediators.

Model 1 (total): outcome ~ race + age + sex
Model 2 (direct): outcome ~ race + age + sex + SDoH mediators

Proportion mediated = 1 - beta_race(M2) / beta_race(M1)

Addresses Reviewer 2 Comment 1.

Author: Sanjay Basu, MD PhD
Date: February 2026
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Resolve paths relative to repo root; override via environment variables
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
data_dir = Path(os.environ.get('FACTORED_RL_DATA_DIR', _REPO_ROOT / 'data'))
results_dir = Path(os.environ.get('FACTORED_RL_RESULTS_DIR', _REPO_ROOT / 'results'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mediation_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_mediation_analysis(df, race_col, outcome_col,
                           basic_covariates, sdoh_mediators,
                           reference_group='White',
                           n_bootstrap=500):
    """
    Run regression-based mediation decomposition.

    Args:
        df: DataFrame with all variables
        race_col: Column name for race/ethnicity
        outcome_col: Column for binary outcome
        basic_covariates: List of basic adjustment variables (age, sex)
        sdoh_mediators: List of SDoH mediator columns
        reference_group: Reference race group
        n_bootstrap: Number of bootstrap iterations for CIs

    Returns:
        Dict with per-group mediation results
    """
    # Prepare data
    analysis_df = df.dropna(subset=[race_col, outcome_col]).copy()

    # Create race dummies
    race_groups = analysis_df[race_col].unique()
    race_groups = [g for g in race_groups if str(g) != reference_group and str(g) != 'Unknown']

    logger.info(f"Reference group: {reference_group}")
    logger.info(f"Comparison groups: {race_groups}")
    logger.info(f"N = {len(analysis_df)}")

    # Encode race as dummies
    race_dummies = pd.get_dummies(analysis_df[race_col], prefix='race', drop_first=False)

    # Ensure reference group column exists
    ref_col = f'race_{reference_group}'
    if ref_col not in race_dummies.columns:
        # Try matching
        for col in race_dummies.columns:
            if reference_group.lower() in col.lower():
                ref_col = col
                break

    # Prepare covariates
    scaler = StandardScaler()

    # Basic covariates
    X_basic = []
    basic_cols_available = []
    for col in basic_covariates:
        matches = [c for c in analysis_df.columns if col in c.lower()]
        if matches:
            basic_cols_available.append(matches[0])

    if basic_cols_available:
        X_basic_raw = analysis_df[basic_cols_available].fillna(0).values
        X_basic_scaled = scaler.fit_transform(X_basic_raw)
    else:
        X_basic_scaled = np.zeros((len(analysis_df), 0))

    # SDoH mediators
    sdoh_cols_available = []
    for col in sdoh_mediators:
        matches = [c for c in analysis_df.columns if col in c.lower()]
        if matches:
            sdoh_cols_available.append(matches[0])

    if sdoh_cols_available:
        X_sdoh_raw = analysis_df[sdoh_cols_available].fillna(0).values
        X_sdoh_scaled = StandardScaler().fit_transform(X_sdoh_raw)
    else:
        logger.warning("No SDoH mediator columns found!")
        X_sdoh_scaled = np.zeros((len(analysis_df), 0))

    logger.info(f"Basic covariates found: {basic_cols_available}")
    logger.info(f"SDoH mediators found: {sdoh_cols_available}")

    # Outcome
    y = analysis_df[outcome_col].values
    if not np.all(np.isin(y, [0, 1])):
        y = (y < np.median(y)).astype(float)  # Binarize if needed

    results = []

    for group in race_groups:
        group_str = str(group)
        logger.info(f"\n--- Analyzing: {group_str} vs {reference_group} ---")

        # Create binary indicator for this group vs reference
        group_col = f'race_{group_str}'
        if group_col not in race_dummies.columns:
            # Try fuzzy match
            matches = [c for c in race_dummies.columns if group_str.lower() in c.lower()]
            if matches:
                group_col = matches[0]
            else:
                logger.warning(f"  Cannot find column for group {group_str}")
                continue

        # Subset to reference + this group only
        mask = (analysis_df[race_col].values == group) | \
               (analysis_df[race_col].values == reference_group)

        if mask.sum() < 100:
            logger.info(f"  Too few observations ({mask.sum()}), skipping")
            continue

        y_sub = y[mask]
        race_indicator = (analysis_df[race_col].values[mask] == group).astype(float)
        X_basic_sub = X_basic_scaled[mask]
        X_sdoh_sub = X_sdoh_scaled[mask]

        # Model 1: outcome ~ race + basic
        X1 = np.column_stack([race_indicator, X_basic_sub]) if X_basic_sub.shape[1] > 0 \
             else race_indicator.reshape(-1, 1)

        # Model 2: outcome ~ race + basic + SDoH
        if X_sdoh_sub.shape[1] > 0:
            X2 = np.column_stack([race_indicator, X_basic_sub, X_sdoh_sub]) \
                 if X_basic_sub.shape[1] > 0 \
                 else np.column_stack([race_indicator, X_sdoh_sub])
        else:
            X2 = X1  # No mediators available

        try:
            # Fit Model 1
            m1 = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
            m1.fit(X1, y_sub)
            beta_race_m1 = m1.coef_[0][0]  # Coefficient on race indicator
            or_total = np.exp(beta_race_m1)

            # Fit Model 2
            m2 = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
            m2.fit(X2, y_sub)
            beta_race_m2 = m2.coef_[0][0]
            or_adjusted = np.exp(beta_race_m2)

            # Proportion mediated
            if beta_race_m1 != 0:
                prop_mediated = 1 - (beta_race_m2 / beta_race_m1)
            else:
                prop_mediated = 0

            # Bootstrap CIs
            boot_prop_mediated = []
            boot_or_total = []
            boot_or_adjusted = []

            for b in range(n_bootstrap):
                idx = np.random.choice(len(y_sub), size=len(y_sub), replace=True)
                y_b = y_sub[idx]
                X1_b = X1[idx]
                X2_b = X2[idx]

                if len(np.unique(y_b)) < 2:
                    continue

                try:
                    m1_b = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
                    m1_b.fit(X1_b, y_b)
                    b1 = m1_b.coef_[0][0]

                    m2_b = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
                    m2_b.fit(X2_b, y_b)
                    b2 = m2_b.coef_[0][0]

                    boot_or_total.append(np.exp(b1))
                    boot_or_adjusted.append(np.exp(b2))
                    if b1 != 0:
                        boot_prop_mediated.append(1 - b2/b1)
                except Exception:
                    continue

            result = {
                'group': group_str,
                'reference': reference_group,
                'n_group': int(np.sum(race_indicator)),
                'n_reference': int(np.sum(1 - race_indicator)),
                'or_total': float(or_total),
                'or_total_ci': [float(np.percentile(boot_or_total, 2.5)),
                                float(np.percentile(boot_or_total, 97.5))] if boot_or_total else None,
                'or_adjusted': float(or_adjusted),
                'or_adjusted_ci': [float(np.percentile(boot_or_adjusted, 2.5)),
                                   float(np.percentile(boot_or_adjusted, 97.5))] if boot_or_adjusted else None,
                'proportion_mediated': float(prop_mediated),
                'proportion_mediated_ci': [float(np.percentile(boot_prop_mediated, 2.5)),
                                           float(np.percentile(boot_prop_mediated, 97.5))] if boot_prop_mediated else None,
                'key_mediators': sdoh_cols_available
            }

            results.append(result)

            logger.info(f"  OR (total): {or_total:.3f}")
            logger.info(f"  OR (adjusted): {or_adjusted:.3f}")
            logger.info(f"  Proportion mediated: {prop_mediated*100:.1f}%")

        except Exception as e:
            logger.warning(f"  Error fitting models for {group_str}: {e}")
            continue

    return results


def main():
    logger.info("=" * 80)
    logger.info("RACE-OUTCOME MEDIATION ANALYSIS")
    logger.info("=" * 80)

    # Paths (using module-level defaults with env var overrides)
    results_dir.mkdir(exist_ok=True)

    # Load data
    test_df = pd.read_parquet(data_dir / 'sequences_test.parquet')

    # Subsample for tractability
    if len(test_df) > 100000:
        test_df = test_df.sample(100000, random_state=42)

    logger.info(f"Analysis dataset: {len(test_df)} observations")
    logger.info(f"Columns available: {list(test_df.columns[:30])}")

    # Find race column
    race_col = None
    for col in ['race', 'race_ethnicity', 'race_t3', 'race_t0']:
        if col in test_df.columns:
            race_col = col
            break

    if race_col is None:
        logger.error("No race column found!")
        return

    logger.info(f"Race column: {race_col}")
    logger.info(f"Distribution: {test_df[race_col].value_counts().to_dict()}")

    # Find outcome column
    outcome_col = None
    for col in ['ed_visits_30d', 'any_acute_care', 'reward_shaped',
                'num_ed_visits_last_week_t3', 'ed_visits_7d_t3']:
        if col in test_df.columns:
            outcome_col = col
            break

    if outcome_col is None:
        # Use reward_shaped as proxy (negative = adverse outcome)
        outcome_col = 'reward_shaped'

    logger.info(f"Outcome column: {outcome_col}")

    # If outcome is continuous, binarize
    if outcome_col == 'reward_shaped':
        test_df['outcome_binary'] = (test_df['reward_shaped'] < test_df['reward_shaped'].quantile(0.10)).astype(int)
        outcome_col = 'outcome_binary'
    elif test_df[outcome_col].nunique() > 2:
        test_df['outcome_binary'] = (test_df[outcome_col] > 0).astype(int)
        outcome_col = 'outcome_binary'

    # Basic covariates
    basic_covariates = ['age', 'gender', 'sex', 'female']

    # SDoH mediators
    sdoh_mediators = [
        'housing', 'food', 'transportation', 'insurance',
        'deprivation', 'employment', 'sdoh',
        'social_determinant', 'adi', 'zip'
    ]

    # Find most common race group as reference
    race_counts = test_df[race_col].value_counts()
    reference_group = str(race_counts.index[0])
    logger.info(f"Reference group (largest): {reference_group}")

    # Run mediation analysis
    results = run_mediation_analysis(
        test_df, race_col, outcome_col,
        basic_covariates, sdoh_mediators,
        reference_group=reference_group,
        n_bootstrap=500
    )

    output = {
        'mediation_results': results,
        'reference_group': reference_group,
        'outcome_variable': outcome_col,
        'basic_covariates_searched': basic_covariates,
        'sdoh_mediators_searched': sdoh_mediators,
        'note': 'Proportion mediated = 1 - beta_race(adjusted) / beta_race(unadjusted). '
                'Values >0 indicate SDoH variables explain part of race-outcome association.'
    }

    with open(results_dir / 'mediation_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {results_dir / 'mediation_analysis.json'}")
    logger.info("=" * 80)
    logger.info("MEDIATION ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
