# Factored Offline Reinforcement Learning for Medicaid Care Management

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official code repository for **"Factored Offline Reinforcement Learning for Personalized Medicaid Care Management: A Methodological Framework with Doubly Robust Evaluation"** 

---

## Paper Summary

This work presents a methodological framework for applying offline reinforcement learning to healthcare resource allocation, addressing three key challenges:

1. **Sparse rewards** (< 1% of observations have non-zero outcomes)
2. **High-dimensional action spaces** (97 observed action combinations from 240 theoretical)
3. **Confounding by indication** (sicker patients receive more intensive care)

---

## Repository Structure

```
factored_rl_github/
├── README.md                                    # This file
├── LICENSE                                      # Apache 2.0
├── pyproject.toml                               # Package metadata and dependencies
├── requirements.txt                             # Python dependencies (pip)
│
├── src/                                         # Core library code
│   ├── __init__.py                              # Package init
│   ├── factored_action_space.py                 # Action space definition
│   ├── offline_rl_policy_awr.py                 # AWR algorithm
│   ├── phase2_train_full.py                     # LSTM policy training pipeline
│   ├── phase3_evaluate.py                       # DR off-policy evaluation
│   ├── reward_shaping.py                        # Multi-component reward shaping
│   ├── sensitivity_analysis.py                  # OPE sensitivity analyses
│   ├── subgroup_analysis.py                     # Demographic subgroup analysis
│   ├── audit_fairness.py                        # Fairness audit
│   └── evaluate_policy.py                       # Policy evaluation utilities
│
├── scripts/                                     # Analysis pipeline
│
├── examples/                                    # Usage examples
│   └── synthetic_data_example.py                # Synthetic data example
│
└── docs/                                        # Documentation
    └── architecture_diagram.md                  # System architecture

```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/sanjaybasu/factored-offline-rl.git
cd factored-offline-rl

# Create environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Or install as a package:
# pip install -e .
```

### Synthetic Data Example

Since real data cannot be shared due to HIPAA, we provide a synthetic example:

```python
from examples.synthetic_data_example import generate_synthetic_data

# Generate synthetic data matching paper structure
states, action_tuples, rewards = generate_synthetic_data(
    n_patients=1000,
    n_weeks_per_patient=52,
    seed=42
)

print(f"States shape: {states.shape}")
print(f"Number of trajectories: {len(action_tuples)}")
print(f"Reward rate: {(rewards != 0).mean():.3f}")
```

---

## Analysis Pipeline

**Note:** Exact reproduction requires the original Medicaid dataset which cannot be shared due to HIPAA. The code in `src/` implements the full training and evaluation pipeline. Scripts in `scripts/03_revision_analyses/` implement additional analyses requested during peer review.

### Core Pipeline (`src/`)

1. **`phase2_train_full.py`** — Train LSTM policy with bidirectional LSTM + attention via AWR
2. **`phase3_evaluate.py`** — Doubly robust off-policy evaluation with bootstrap CIs
3. **`reward_shaping.py`** — Multi-component reward shaping (primary, engagement, intermediate milestones; weights 1.0, 0.3, 0.5)
4. **`sensitivity_analysis.py`** — Q-model bias, propensity noise, truncation threshold sensitivity
5. **`subgroup_analysis.py`** — Demographic subgroup DR evaluation
6. **`audit_fairness.py`** — Demographic parity fairness audit

### Revision Analyses (`scripts/03_revision_analyses/`)

| Script | Reviewer Comment | Analysis |
|--------|-----------------|----------|
| `run_cql_baseline.py` | R1.1, R2.5 | CQL comparison baseline |
| `window_length_sensitivity.py` | R2.2 | LSTM window = 2, 4, 6, 8 weeks |
| `reward_weight_sensitivity.py` | R1.2, R2.3 | 4x4 reward weight grid search |
| `proportional_parity_analysis.py` | R2.4 | Need-proportional fairness metric |
| `mediation_analysis.py` | R2.1 | Race-outcome SDoH mediation |
| `redraw_figure1.py` | R1.3 | Figure 1 with >= 12pt fonts |

---

## Key Results

### Off-Policy Evaluation

The doubly robust estimator achieves stable policy value estimates with 44.2% effective sample size, compared to 0.0% for naive importance sampling in this 97-action space. The learned policy achieves performance parity with observed clinician behavior.

### Fairness

Fairness-constrained training reduces the demographic parity gap in home visit allocation from 3.5 to 0.8 percentage points across racial/ethnic groups, with proportional parity ratios close to 1.0 for all groups.

### Sensitivity Analyses

- **CQL comparison**: AWR and CQL produce comparable policy values (Table S12)
- **Window length**: DR values stable across 2–8 week windows (Table S13)
- **Reward weights**: Policy robust across 16 weight configurations (Table S14)
- **Mediation**: SDoH mediators explain substantial proportion of race-outcome associations (Table S16)


---

## System Requirements

### Software Dependencies
- **Python**: 3.12+
- **PyTorch**: 2.0+
- **NumPy**: < 2.0 (for compatibility)
- **Pandas**: 2.0+
- **Scikit-learn**: 1.3+
- **Matplotlib**: 3.7+
- **Seaborn**: 0.12+

### Hardware
- **CPU**: 16+ cores recommended
- **RAM**: 32 GB minimum
- **GPU**: Not required (CPU-only training)
- **Storage**: 50 GB for full pipeline

### Tested Environments
- macOS Sonoma (Apple Silicon)
- Ubuntu 22.04 LTS
- AWS EC2 c5.4xlarge

---

## Documentation

### Methods Overview

**Factored Action Space:**
Actions decompose into (modality, provider, goal, urgency) where:
- Modality: {Phone, Text/SMS, Video, Home Visit}
- Provider: {RN, Social Worker, Care Coordinator, CHW, MD}
- Goal: {Chronic Disease Management, Care Coordination, Behavioral Health, SDoH}
- Urgency: {Routine, Semi-urgent, Urgent}

This yields 4 x 5 x 4 x 3 = 240 theoretical combinations; 97 are observed in practice.

**Multi-Component Reward Shaping:**

R_total = 1.0 * R_primary + 0.3 * R_engagement + 0.5 * R_intermediate

Components: primary outcome (ED visits + hospitalizations), engagement (contact success, appointment adherence), intermediate milestones (PCP visits, medication fills, lab completion).

**Doubly Robust Off-Policy Evaluation:**

V_DR = (1/n) * sum[ rho_i * (Y_i - Q(X_i, A_i)) + Q(X_i, A_i) ]

where rho_i = pi(A_i|X_i) / pi_b(A_i|X_i) and Q is the outcome model (XGBoost).

**Fairness Constraints:**
Demographic parity penalty during training ensures equitable intervention allocation across racial/ethnic groups. Proportional parity metric validates allocation is proportional to clinical need.

---

## Limitations

1. **Observational Data**: Policy evaluated via off-policy estimation, not randomized trials
2. **Single Setting**: Results from one Medicaid program may not generalize
3. **Minimal State Space**: 5 state features exclude detailed clinical information
4. **Race as Proxy**: Race/ethnicity variables may incompletely capture structural determinants (see mediation analysis, Table S16)
5. **Reward Shaping Assumptions**: Engagement and intermediate milestone weights selected via grid search, not derived from causal models

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{basu2026factored,
  title={Factored Offline Reinforcement Learning for Personalized Medicaid Care Management: A Methodological Framework with Doubly Robust Evaluation},
  author={Basu, Sanjay},
  year={2026}}
```

---

## Contact

**Sanjay Basu, MD, PhD**  
Email: sanjay.basu@waymarkcare.com  

---

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

Copyright 2026 Waymark 
