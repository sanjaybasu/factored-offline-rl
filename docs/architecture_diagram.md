# Model Architecture

This document describes the architecture of the factored offline reinforcement learning model.

## Overview

The model consists of three main components:
1. **Factored Action Space**: Decomposes actions into interpretable factors
2. **Action Embedding Network**: Learns semantic representations of actions
3. **Actor-Critic Networks**: Policy and value function approximation

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT STATE                              │
│  [encounters, ed_visits, ip_visits, calls, texts, days, costs]  │
│                           (7 features)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │      STATE ENCODER           │
          │   FC(7 → 256) + LayerNorm    │
          │   + ReLU + Dropout(0.1)      │
          └──────────────┬───────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │   HIDDEN LAYER               │
          │   FC(256 → 256) + LayerNorm  │
          │   + ReLU + Dropout(0.1)      │
          └──────────────┬───────────────┘
                         │
                ┌────────┴────────┐
                │                 │
                ▼                 ▼
         ┌────────────┐    ┌─────────────┐
         │   ACTOR    │    │   CRITIC    │
         └─────┬──────┘    └──────┬──────┘
               │                  │
               ▼                  ▼
    ┌──────────────────┐   ┌──────────┐
    │ ACTION SCORING   │   │  VALUE   │
    │                  │   │ V(s) ∈ ℝ │
    │ For each action: │   └──────────┘
    │ 1. Get embedding │
    │ 2. Concat [s,e]  │
    │ 3. Score ∈ ℝ     │
    └──────────────────┘
```

## Component Details

### 1. Factored Action Space

Actions are decomposed into 4 interpretable components:

| Component | Categories | Dimension |
|-----------|-----------|-----------|
| Modality | 6 (None, Phone, SMS, Video, Home Visit, EHR) | Embedding: 32 |
| Provider | 5 (None, CHW, Care Coordinator, Pharmacist, Therapist) | Embedding: 32 |
| Goal | 18 (medication, mental health, social determinants, etc.) | Embedding: 32 |
| Urgency | 2 (Routine, Urgent) | Embedding: 32 |

**Total action embedding dimension**: 4 × 32 = 128

### 2. Action Embeddings

Each component is embedded into a 32-dimensional learned representation:

```python
# Initialization (random, seed=42)
modality_embedding = nn.Embedding(6, 32)
provider_embedding = nn.Embedding(5, 32)
goal_embedding = nn.Embedding(18, 32)
urgency_embedding = nn.Embedding(2, 32)

# Concatenation for full action
action_emb = concat([
    modality_embedding[m],
    provider_embedding[p],
    goal_embedding[g],
    urgency_embedding[u]
])  # Shape: (128,)
```

### 3. Actor Network

The actor scores state-action pairs:

```
Input: state (7) + action_embedding (128) = 135 dimensions
  ↓
FC(135 → 256) + LayerNorm + ReLU + Dropout(0.1)
  ↓
FC(256 → 128) + LayerNorm + ReLU
  ↓
FC(128 → 1)
  ↓
Tanh-bounded score ∈ [-10, 10]
```

**Key design choice**: The actor does NOT output 1,080 logits (one per action combination). Instead, it scores each candidate action individually. This enables:
- Parameter sharing across related actions
- Generalization to under-observed action combinations
- Efficient exploration of the combinatorial action space

### 4. Critic Network

The critic estimates state value:

```
Input: state (7 dimensions)
  ↓
FC(7 → 256) + ReLU + Dropout(0.1)
  ↓
FC(256 → 128) + ReLU
  ↓
FC(128 → 1)
  ↓
Value V(s) ∈ ℝ
```

### 5. Advantage-Weighted Regression (AWR)

AWR learns by weighting updates according to advantage:

```python
# For each training sample (s, a, r):
advantage = r - V(s)  # Critic provides baseline

# Weight samples by advantage
weight = exp(β * advantage)
weight = clip(weight, 0.1, 10.0)  # Prevent extremes

# Actor loss
actor_loss = -weight * log_prob(a | s)

# Critic loss
critic_loss = (V(s) - r)²
```

**Hyperparameter β = 1.0**: Controls the strength of advantage weighting. Higher values amplify high-advantage samples.

## Training Details

### Optimization
- Optimizer: Adam
- Learning rate: 3×10⁻⁵
- Batch size: 256
- Gradient clipping: 0.5 (max norm)

### Regularization
- Dropout: 0.1 (between layers)
- Layer Normalization: Applied after each linear layer
- Advantage weight clipping: [0.1, 10.0]

### Computational Cost
- **Parameters**: ~330k (actor: 198k, critic: 131k)
- **Training time**: 3.5 hours on 16-core CPU for 50 epochs
- **Inference**: ~100ms per state (sampling 500 candidate actions)

## Action Selection at Inference

At test time, we select the highest-scoring action:

```python
def select_action(actor, state, action_space, top_k=1):
    # Sample candidate actions
    candidates = [action_space.sample_valid_action() 
                  for _ in range(500)]
    
    # Get embeddings
    embeddings = [action_space.get_action_embedding(a) 
                  for a in candidates]
    
    # Score each candidate
    scores = actor(state, embeddings)
    
    # Return top-k
    best_idx = scores.argmax()
    return candidates[best_idx], scores[best_idx]
```

This approximate inference balances computational efficiency with action space coverage.
