# Factored Offline Reinforcement Learning for Medicaid Care Management

This repository contains the code for the paper **"Factored Offline Reinforcement Learning for Personalized Medicaid Care Management"**.

## Overview

This work demonstrates how offline reinforcement learning with factored action spaces can learn diverse, clinically interpretable care management policies from observational healthcare data with sparse rewards.

### Key Innovations
1. **Factored Action Space**: Decomposes actions into interpretable components (modality, provider, goal, urgency)
2. **Learned Action Embeddings**: 32-dimensional representations for parameter-efficient learning
3. **Advantage-Weighted Regression (AWR)**: Conservative offline RL algorithm that prevents action collapse

## Repository Structure

```
factored_rl_github/
├── README.md                          # This file
├── LICENSE                            # Apache 2.0 License
├── requirements.txt                   # Python dependencies
├── src/
│   ├── factored_action_space.py      # Action space definition and embeddings
│   ├── offline_rl_policy_awr.py      # AWR algorithm implementation
│   ├── evaluate_policy.py            # Policy evaluation and visualization
│   └── audit_fairness.py             # Fairness audit across demographics
├── examples/
│   └── synthetic_data_example.py     # Example with synthetic data
└── docs/
    └── architecture_diagram.md        # Model architecture description
```

## Installation

### Requirements
- Python 3.12+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn

### Setup

```bash
# Clone the repository
git clone https://github.com/[username]/factored-rl-medicaid.git
cd factored-rl-medicaid

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a Policy

The core training pipeline requires longitudinal patient data with the following structure:
- **States**: Recent utilization patterns, care team engagement, program tenure, costs
- **Actions**: Factored into (modality, provider, goal, urgency)
- **Rewards**: Negative of acute utilization (ED visits + hospitalizations)

```python
from src.factored_action_space import FactoredActionSpace
from src.offline_rl_policy_awr import train_awr_factored

# Initialize action space
action_space = FactoredActionSpace(embedding_dim=32)

# Train policy
actor, critic = train_awr_factored(
    states=states,           # (N, 7) array
    actions=actions,         # List of (modality, provider, goal, urgency) tuples
    rewards=rewards,         # (N,) array of negative utilization
    action_space=action_space,
    beta=1.0,
    num_epochs=50
)
```

### Evaluating a Policy

```python
from src.evaluate_policy import evaluate_policy

evaluate_policy(
    policy_path='trained_policy.pt',
    data_path='evaluation_data.parquet',
    output_dir='outputs/evaluation/'
)
```

### Fairness Audit

```python
from src.audit_fairness import audit_fairness

audit_fairness(
    data_path='evaluation_data.parquet',
    policy_path='trained_policy.pt',
    output_dir='outputs/fairness_audit/'
)
```

## Data Requirements

Due to HIPAA regulations and patient privacy, we **cannot release the training data**. However, the code is designed to work with any longitudinal care management dataset with the following schema:

### Required Columns
- **State features** (7 features):
  - `num_encounters_last_week`: Count of care team contacts
  - `num_ed_visits_last_week`: Emergency department visits
  - `num_ip_visits_last_week`: Hospitalizations
  - `num_calls_last_week`: Phone interventions
  - `num_texts_last_week`: SMS interventions
  - `enrolled_days`: Days in program
  - `total_paid`: Healthcare spending (90-day window)

- **Action components**:
  - `action_modality`: {NONE, PHONE_CALL, SMS_TEXT, VIDEO_VISIT, HOME_VISIT, EHR_COMMUNICATION}
  - `action_provider`: {NONE, CHW, CARE_COORDINATOR, PHARMACIST, THERAPIST}
  - `action_goal`: {18 categories including medication, mental health, social determinants}
  - `action_urgency`: {ROUTINE, URGENT}

- **Outcomes**:
  - `future_ed_30d`: ED visits in next 30 days
  - `future_ip_30d`: Hospitalizations in next 30 days

### Synthetic Data Example

We provide a synthetic data generator for demonstration purposes:

```python
from examples.synthetic_data_example import generate_synthetic_data

# Generate synthetic trajectories
states, actions, rewards = generate_synthetic_data(
    n_patients=1000,
    n_weeks_per_patient=52
)
```

**Note**: The synthetic data is for demonstration only and will not reproduce the exact results in the paper, which used real Medicaid claims and care management data.

## Reproducibility Notes

### Results from the Paper
- **Training set size**: 4,999,070 timesteps (50% of 9.9M observations)
- **Evaluation set**: 5,000 held-out states
- **Training time**: 3.5 hours on 16-core CPU
- **Key hyperparameters**:
  - Learning rate: 3×10⁻⁵
  - Batch size: 256
  - Epochs: 50
  - Gradient clipping: 0.5
  - Embedding dimension: 32

### Hardware
- CPU-only training (no GPU required)
- Tested on AWS EC2 c5.4xlarge (16 vCPUs, 32 GB RAM)

### Random Seeds
All experiments used fixed random seeds for reproducibility:
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- Action space embeddings: seed 42 (in `FactoredActionSpace.__init__`)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{basu2025factored,
  title={Factored Offline Reinforcement Learning for Personalized Medicaid Care Management},
  author={Basu, Sanjay and Patel, Sadiq Y. and Sheth, Parth and Muralidharan, Bhairavi and Elamaran, Namrata and Kinra, Aakriti and Batniji, Rajaie},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the code or collaboration inquiries, please contact:
- **Sanjay Basu, MD, PhD**: sanjay.basu@waymarkcare.com

## Acknowledgments

We thank the Waymark care teams for their dedication to Medicaid beneficiaries and meticulous documentation that enabled this analysis.
