# Offline Reinforcement Learning for Population Health Care Management

## Overview

We present an offline reinforcement learning framework designed for high-dimensional clinical action spaces, sparse outcomes, and fairness constraints
## Repository Structure

- `scripts/`: Python scripts for data preparation, training, and evaluation.
- `models/`: Model architecture definitions (LSTM policy, fairness constraints).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
The scripts in `scripts/` assume access to a formatted dataset. Due to privacy regulations (HIPAA), the raw data is not provided. However, the data processing logic is available in:
- `scripts/extract_enhanced_features.py`
- `scripts/build_enhanced_dataset.py`

### 2. Training
To train the policy network:
```bash
python scripts/train_lstm_policy_final.py
```

### 3. Evaluation
To evaluate the policy using doubly robust estimation:
```bash
python scripts/phase3_evaluate.py
```

## Citation

If you use this code in your research, please cite:

> Basu S, et al. Offline Reinforcement Learning for Population Health Care Management: Addressing Sparse Rewards, Temporal Dependencies, and Fairness in Medicaid Populations. 2025.

## License

This project is licensed under the MIT License.
