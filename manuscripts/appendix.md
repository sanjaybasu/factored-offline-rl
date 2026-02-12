# Supplementary Materials

## Offline Reinforcement Learning for Care Management: Addressing Sparse Rewards, Temporal Dependencies, and Fairness in Medicaid Populations

---

## Supplementary Methods

### S1. Detailed State Feature Definitions

**Demographic Features (5)**
- Age: continuous, years (range: 18-89, capped for privacy)
- Gender: binary (female, male)  
- Race/ethnicity: categorical (8 levels per US Census)
- Preferred language: categorical (English, Spanish, Other)
- Employment status: categorical (employed, unemployed, disabled, retired, unknown)

**Clinical Features (10)**
- Active diagnosis count: continuous (0-25+)
- Active medication count: continuous (0-20+)
- HbA1c (last 90 days): continuous (4.0-14.0%), missing indicator
- Systolic BP (last 90 days): continuous (80-200 mmHg), missing indicator
- Diastolic BP (last 90 days): continuous (40-120 mmHg), missing indicator
- Days since last clinical encounter: continuous (0-365+)
- Has diabetes: binary
- Has hypertension: binary
- Has depression/anxiety: binary
- Has COPD/asthma: binary

**Utilization Features (6)**
- ED visits (prior 7 days): count (0-5+)
- Hospitalizations (prior 7 days): count (0-3+)
- ED visits (prior 30 days): count (0-10+)
- Hospitalizations (prior 30 days): count (0-5+)
- ED visits (prior 90 days): count (0-20+)
- Hospitalizations (prior 90 days): count (0-10+)

**Engagement Features (5)**
- Intervention count (prior 30 days): count (0-15+)
- Successful contact rate (prior 30 days): proportion (0-1)
- Appointment no-show rate (prior 90 days): proportion (0-1)
- Days since last contact: continuous (0-365+)
- Current episode duration: weeks (0-96)

**Social Determinants (5)**
- Housing stability score: ordinal (1-5)
- Food security score: ordinal (1-5)
- Transportation access: binary
- Insurance continuity: months (0-24)
- Zip code-level deprivation index: continuous (0-100)

---

### S2. Long Short-Term Memory Network Architecture Details

**Policy Network Architecture:**

```
Input: State sequence {s_{t-3}, s_{t-2}, s_{t-1}, s_t} ∈ ℝ^{4×31}
       Previous actions {a_{t-3}, a_{t-2}, a_{t-1}} ∈ {1...97}^3

1. State Embedding Layer:
   - Continuous features (26): Linear(26 → 64) + BatchNorm + ReLU
   - Categorical embeddings:
     * Race/ethnicity: Embedding(8 → 16)
     * Language: Embedding(3 → 8)  
     * Employment: Embedding(5 → 8)
   - Concatenate → 96-dim state representation

2. Bidirectional LSTM:
   - Input: (batch, 4, 96)
   - Hidden dim: 64 (128 total for bidirectional)
   - Layers: 2
   - Dropout: 0.2
   - Output: (batch, 4, 128)

3. Multi-Head Attention:
   - Heads: 4
   - Key/Query/Value dim: 16 each (64 total)
   - Self-attention over 4-week sequence
   - Output: (batch, 128)

4. Action Embedding:
   - Action index: Embedding(97 → 32)
   
5. MLP Head:
   - Input: Concat(attention_output, action_emb) → 160-dim
   - Layer 1: Linear(160 → 128) + ReLU + Dropout(0.1)
   - Layer 2: Linear(128 → 64) + ReLU + Dropout(0.1)
   - Output: Linear(64 → 1) → action score

Total Parameters: 34,337
```

---

### S3. Advantage-Weighted Regression Algorithm

**Algorithm 1: Offline AWR Training**

```
Input: Dataset D = {(s_i, a_i, r_i)}
Hyperparameters: β (temperature), λ (fairness weight)

Initialize: Policy π_θ, Value V_φ

For epoch = 1 to N:
    
    # Value Network Update
    For minibatch B ⊂ D:
        V̂ = V_φ(s)
        L_V = MSE(V̂, r)
        φ ← φ - α∇_φ L_V
    
    # Policy Network Update  
    For minibatch B ⊂ D:
        # Compute advantages
        V̂ = V_φ(s) [stop gradient]
        A = r - V̂
        
        # Compute importance weights
        w = exp(A / β)
        w = clip(w, max=20)
        
        # Policy scores
        score = π_θ(a | s)
        
        # AWR loss
        L_π = -mean(w · score)
        
        # Fairness penalty
        L_fair = λ · DemographicParityLoss(π_θ, demographics)
        
        # Combined update
        θ ← θ - α∇_θ(L_π + L_fair)

Return: π_θ, V_φ
```

---

### S4. Doubly Robust Estimator Derivation

The doubly robust estimator for policy value combines regression and importance sampling:

$$\hat{V}^{DR}(\pi) = \frac{1}{n}\sum_{i=1}^n \left[\rho_i(R_i - \hat{Q}(s_i,a_i)) + \hat{Q}(s_i,a_i)\right]$$

**Bias Analysis:**

If $$\hat{Q}$$ is correctly specified: $$E[\hat{Q}(s,a)] = E[R|s,a]$$
$$E[\hat{V}^{DR}] = E[\hat{Q}(s,a)] = E[R|s,a] = V(\pi)$$

If $$\hat{\pi}_b$$ is correctly specified but $$\hat{Q}$$ is biased:
$$E[\hat{V}^{DR}] = E[\rho R] = V(\pi)$$

Thus the estimator is **doubly robust**: unbiased if **either** model is correct.

**Variance:**

$$\text{Var}(\hat{V}^{DR}) \approx \frac{1}{n}E\left[(\rho(R - \hat{Q}))^2\right]$$

This is substantially lower than pure IS variance $$E[\rho^2 R^2]/n$$ when $$\hat{Q}$$ is approximately correct.

---

## Supplementary Tables

**Table S1. Factored Action Space Structure and Observed Distributions**

| Action Factor | Levels | Top 3 Categories (% of Interventions) |
|--------------|--------|--------------------------------------|
| Modality | 4 | Telephone (58.2%), Text message (24.7%), In-person home visit (12.4%) |
| Provider | 5 | Care coordinator (42.3%), Nurse (31.8%), Social worker (15.6%) |
| Goal | 4 | Chronic disease management (45.1%), Care coordination (28.9%), SDOH support (18.3%) |
| Urgency | 3 | Routine (76.4%), Semi-urgent (18.9%), Urgent (4.7%) |
| **Combinations** | **97** | **(modality, provider, goal, urgency)** |

SDOH: social determinants of health. Factorization reduces theoretical action space from 4×5×4×3=240 potential combinations to 97 combinations observed in clinical practice across 9,998,139 observations. Most frequent specific combinations: (Telephone, Care coordinator, Chronic disease management, Routine) at 31.2% of interventions; (Text, Nurse, Care coordination, Routine) at 18.7%; (In-person, Social worker, SDOH support, Routine) at 9.4%. Action embedding dimension: 32, learned jointly with policy network to capture semantic relationships between intervention types.

---

**Table S2. Sensitivity Analysis: Varying Key Assumptions**

| Assumption Varied | Base Case | Variation Range | Impact on Point Estimate | Robustness |
|-------------------|-----------|-----------------|--------------------------|------------|
| Q-function bias | 0% | ±20% | < 0.06% | High |
| Propensity error | 0% | 0.3 SD Noise | < 0.07% | High |
| Weight truncation | 95th percentile | 90th-99th | < 0.2% | High |
| Weight truncation | 95th percentile | 99.9th | Unstable | Low |
| Reward weights | (1.0, 0.3, 0.2) | ±50% each | < 2.5% | Moderate |

*Note: Base case represents main analysis configuration. Variations applied independently while holding other parameters constant. Impact quantified as percentage change from base case estimate. Truncation at 99.9th percentile introduces instability due to extreme weights (max > 16 million), validating the choice of 95th percentile truncation.*

---

**Table S3. Multi-Component Reward Shaping Impact on Learning Signal Availability**

| Reward Component | Mean | SD | Non-Zero Rate (%) | Contribution to Total Variance (%) |
|-----------------|------|-----|-------------------|-----------------------------------|
| Primary (ED + 2×IP) | -0.041 | 0.312 | 0.60 | 68.2 |
| Engagement | 0.002 | 0.029 | 1.18 | 21.4 |
| Intermediate Milestones | -0.018 | 0.094 | 31.2 | 10.4 |
| **Shaped (Combined)** | **-0.083** | **0.877** | **32.1** | **100.0** |

ED: emergency department; IP: inpatient hospitalization; SD: standard deviation. Primary reward component weights emergency department visits (coefficient: −1) and hospitalizations (coefficient: −2) to reflect clinical severity. Engagement component provides graded reinforcement: +0.2 for successful contact, +0.1 for attempted contact, −0.1 for no-show, +0.15 for appointment attendance. Intermediate milestone component rewards care process indicators: +0.3 for primary care visits, +0.2 for medication fills, +0.1 for lab completions, +0.2 for specialist visits, −0.2 for missed appointments. Component contributions to total variance computed through variance decomposition of linear combination with weights: primary=1.0, engagement=0.3, intermediate=0.5. Shaped reward increases non-zero signal rate 53-fold compared to sparse binary outcomes alone (32.1% vs 0.60%).

---

**Table S4. Ablation Study Results**

| Model Configuration | Val Loss | DR Value | ESS (%) | Training Time |
|---------------------|----------|----------|---------|---------------|
| Full Model (LSTM+Attn) | 0.4574 | -0.0736 | 44.2 | 30.8 min |
| w/o Attention | 0.4680 | -0.0784 | 44.2 | 28.5 min |
| w/o LSTM (FeedForward) | 0.4849 | -0.0759 | 44.2 | 5.2 min |
| Standard WIS Evaluation | - | -2.6949 | 0.0 | - |

*Note: Val Loss = Validation set loss (lower is better). DR Value = Doubly Robust policy value estimate. ESS = Effective Sample Size. FeedForward model uses only current state features without temporal history. WIS fails completely (0.0% ESS).*

---

**Table S5. Learned Attention Weights**

| Temporal Position | Mean Weight | Std Dev | 95% CI |
|-------------------|-------------|---------|--------|
| Week $t$ (Current) | 0.652 | 0.145 | (0.58, 0.72) |
| Week $t-1$ | 0.204 | 0.089 | (0.16, 0.25) |
| Week $t-2$ | 0.098 | 0.054 | (0.07, 0.13) |
| Week $t-3$ | 0.046 | 0.032 | (0.03, 0.06) |

*Note: Weights sum to 1.0. While the model prioritizes the most recent observation, it retains significant attention on historical weeks (35% total weight on $t-1$ to $t-3$), validating the importance of temporal modeling.*

---

**Table S6. Fairness Analysis Across Racial and Ethnic Groups**

| Group | Population (%) | ED Rate (%) | IP Rate (%) | Home Visit Rate (%) |
|-------|---------------|-------------|-------------|---------------------|
| White | 46.1 | 0.59 | 0.09 | 5.2 |
| Black or African American | 26.4 | 0.80 | 0.11 | 5.8 |
| Asian | 7.4 | 0.34 | 0.05 | 4.9 |
| Hispanic | 6.7 | 0.60 | 0.08 | 5.1 |
| Native Hawaiian/Pacific Islander | 2.0 | 0.42 | 0.06 | 5.0 |
| American Indian/Alaska Native | 1.3 | 0.77 | 0.13 | 5.6 |
| Other | 1.7 | 0.54 | 0.07 | 5.3 |
| Unknown | 8.4 | 0.35 | 0.05 | 4.8 |
| **Demographic Parity Gap** | - | **0.46pp** | **0.08pp** | **0.8pp** |

ED: emergency department; IP: inpatient hospitalization; pp: percentage points. Demographic parity gap computed as maximum pairwise difference across groups. Home visit rates projected under fairness-constrained learned policy, demonstrating equitable access to high-resource interventions (gap: 0.8pp) compared to unconstrained historical baseline (gap: 3.5pp). Fairness constraint implemented as soft penalty on demographic parity loss with λ=0.01 during training.

---

**Table S7. Comparison to Prior Reinforcement Learning Approaches in Healthcare**

| Study | Algorithm | Action Space | Temporal Model | OPE Method | Fairness | Sample Size |
|-------|-----------|--------------|----------------|------------|----------|-------------|
| Komorowski 2018<sup>31</sup> | DQN | 25 discrete | MDP | None | No | 17,082 |
| Raghu 2017<sup>30</sup> | DQN | 5 discrete | Continuous-state | Simulation | No | 11,060 |
| Nemati 2016<sup>32</sup> | Q-learning | Continuous | None | None | No | 352 |
| Laber 2014<sup>36</sup> | Q-learning | Binary | None | WIS | No | 150 |
| Basu 2025<sup>37</sup> | SARSA | Factored (4×5×4×3) | None | Counterfactual | Post-hoc | 3,175 |
| **This study** | **AWR** | **Factored (97)** | **LSTM+Attention** | **Doubly Robust** | **Constrained** | **160,264** |

AWR: Advantage-Weighted Regression; DQN: Deep Q-Network; LSTM: Long Short-Term Memory; MDP: Markov Decision Process; OPE: Off-Policy Evaluation; SARSA: State-Action-Reward-State-Action; WIS: Weighted Importance Sampling.

---

**Table S8. Model Architecture and Hyperparameter Configuration**

| Component | Configuration |
|-----------|--------------|
| **Policy Network** | |
| LSTM layers | 2 bidirectional |
| LSTM hidden dimension | 64 |
| LSTM dropout | 0.2 |
| Attention heads | 4 |
| Attention dimension | 16 (key, query, value) |
| Action embedding dimension | 32 |
| MLP hidden layers | [128, 64] |
| MLP dropout | 0.1 |
| **Value Network** | |
| LSTM layers | 2 bidirectional |
| LSTM hidden dimension | 64 |
| MLP hidden layers | [64] |
| **Training** | |
| Algorithm | Advantage-Weighted Regression |
| Learning rate | 3×10⁻⁴ |
| Batch size | 256 |
| AWR temperature (β) | 1.0 |
| Fairness penalty (λ) | 0.01 |
| Sequence length | 4 weeks |
| Training epochs | 20 |
| **Optimization** | |
| Method | Bayesian optimization (Optuna TPE) |
| Search trials | 100 |
| Validation metric | Doubly robust policy value |
| **Total Parameters** | |
| Policy network | 34,337 |
| Value network | 22,913 |

AWR: Advantage-Weighted Regression; LSTM: long short-term memory; MLP: multilayer perceptron; TPE: Tree-structured Parzen Estimator. All hyperparameters selected through systematic Bayesian optimization on validation set using Tree-structured Parzen Estimator algorithm with median pruner for early stopping of unpromising configurations. Architecture configurations represent best-performing combination from 100 trials. Policy and value networks share LSTM encoder architecture but differ in output heads (policy: action scoring via MLP + softmax; value: scalar return prediction).

---

**Table S9. Hyperparameter Search Results (Top 10 Configurations)**

| Rank | LSTM Hidden | LSTM Layers | Attention Heads | LR | Batch Size | Val Value | ESS (%) |
|------|-------------|-------------|-----------------|-------|------------|-----------|---------|
| 1 | 64 | 2 | 4 | 3e-4 | 256 | -0.081 | 44.2 |
| 2 | 128 | 2 | 4 | 2e-4 | 256 | -0.082 | 43.8 |
| 3 | 64 | 2 | 2 | 3e-4 | 256 | -0.084 | 44.5 |
| 4 | 64 | 3 | 4 | 2e-4 | 128 | -0.085 | 42.1 |
| 5 | 128 | 1 | 4 | 3e-4 | 256 | -0.086 | 43.9 |
| 6 | 64 | 2 | 8 | 3e-4 | 512 | -0.087 | 41.5 |
| 7 | 32 | 2 | 4 | 4e-4 | 256 | -0.088 | 43.2 |
| 8 | 64 | 1 | 4 | 3e-4 | 128 | -0.089 | 42.7 |
| 9 | 128 | 2 | 2 | 3e-4 | 512 | -0.090 | 43.5 |
| 10 | 64 | 2 | 4 | 1e-4 | 256 | -0.091 | 44.0 |

ESS: Effective Sample Size; LR: Learning Rate; Val Value: Validation set doubly robust policy value. Selected configuration (Rank 1) shown in bold. Search conducted via Bayesian optimization (100 trials total) using Tree-structured Parzen Estimator with median pruner.

---

**Table S10. TRIPOD Checklist for Prediction Model Development**

| Item | Location | Completed |
|------|----------|-----------|
| Title: Identify as prediction model | Title, Abstract | Yes |
| Abstract: Structured summary | Abstract | Yes |
| Background: Rationale | Introduction | Yes |
| Objectives: Specify aims | Introduction | Yes |
| Source of data | Methods | Yes |
| Participants: Eligibility | Methods | Yes |
| Outcome: Definition | Methods | Yes |
| Predictors: Definition | Methods, Appendix S1 | Yes |
| Sample size: Justification | Methods | Yes |
| Missing data: Handling | Methods | Yes |
| Statistical analysis methods | Methods | Yes |
| Model development | Methods, Appendix S2-S3 | Yes |
| Model evaluation | Methods, Results | Yes |
| Participants: Flow diagram | Results | Yes |
| Model performance | Results, Table 3 | Yes |
| Model specification | Table 5, Appendix S2 | Yes |
| Limitations | Discussion | Yes |
| Implications | Discussion | Yes |

---

**Table S11. DECIDE-AI Checklist for AI Clinical Evaluation**

| Item | Location | Completed |
|------|----------|-----------|
| Healthcare context | Introduction, Methods | Yes |
| System description | Methods | Yes |
| Integration requirements | Discussion | Yes |
| Data sources | Methods | Yes |
| Model architecture | Methods, Table 5, Appendix | Yes |
| Validation approach | Methods | Yes |
| Performance metrics | Results, Tables 3-5 | Yes |
| Fairness assessment | Results, Table 4 | Yes |
| Clinical workflow | Methods, Discussion | Yes |
| Intended use | Discussion | Yes |
| Limitations | Discussion | Yes |

---

**Table S12. Comparison of AWR and CQL Offline RL Algorithms**

| Policy | DR Value (95% CI) | ESS (%) | Dem. Parity Gap (pp) | Non-Inferiority p |
|--------|-------------------|---------|---------------------|-------------------|
| AWR (LSTM+Attn) | −0.0736 (−0.0827, −0.0646) | 44.2 | 0.8 | < 0.001 |
| CQL (Feedforward) | −0.0811 (−0.0903, −0.0719) | 41.8 | 1.1 | < 0.001 |
| Behavioral Policy | −0.0783 (−0.0934, −0.0632) | — | 3.5 | Reference |

AWR: Advantage-Weighted Regression; CQL: Conservative Q-Learning; DR: Doubly Robust; ESS: Effective Sample Size; pp: percentage points. CQL baseline trained with conservative regularization coefficient α=1.0 using feedforward Q-network with same hidden dimensions. Both learned policies evaluated using identical doubly robust estimator, behavioral policy model, and Q-function to ensure consistent comparison. CQL achieves non-inferiority to behavioral policy but modestly lower DR value than AWR, consistent with temporal modeling advantages of the LSTM architecture.

---

**Table S13. LSTM Window Length Sensitivity Analysis**

| Window Length (weeks) | Sequence Steps | DR Value (95% CI) | Validation Loss | ESS (%) | Attention Entropy |
|----------------------|----------------|-------------------|-----------------|---------|-------------------|
| 2 | 2 | −0.082 (−0.091, −0.073) | 0.478 | 43.1 | 0.51 |
| 4 (base case) | 4 | −0.074 (−0.083, −0.065) | 0.457 | 44.2 | 0.89 |
| 6 | 6 | −0.077 (−0.086, −0.068) | 0.462 | 42.7 | 1.24 |
| 8 | 8 | −0.079 (−0.089, −0.069) | 0.468 | 41.3 | 1.48 |

DR: Doubly Robust; ESS: Effective Sample Size. Window length specifies the number of consecutive weeks used as input to the LSTM. The 4-week window achieves the best validation loss and DR policy value. Shorter windows (2 weeks) show modestly worse performance consistent with insufficient temporal context. Longer windows (6, 8 weeks) show marginal degradation reflecting the bias–variance tradeoff: longer windows capture more history but increase sequence modeling difficulty and reduce valid training sequences. Attention entropy increases with window length as the model distributes attention across more timesteps. ESS is comparable across all window lengths (41–45%), confirming evaluation methodology robustness.

---

**Table S14. Reward Weight Sensitivity Analysis**

| w_engagement | w_intermediate | DR Value | Spearman ρ with Outcomes | JS Divergence from Base | % Actions Changed |
|-------------|----------------|----------|--------------------------|------------------------|-------------------|
| 0.1 | 0.3 | −0.071 | 0.71 | 0.018 | 6.2 |
| 0.1 | 0.5 | −0.072 | 0.72 | 0.012 | 4.8 |
| 0.1 | 0.7 | −0.073 | 0.72 | 0.015 | 5.5 |
| 0.1 | 1.0 | −0.074 | 0.71 | 0.019 | 6.8 |
| 0.3 | 0.3 | −0.072 | 0.73 | 0.010 | 3.9 |
| **0.3** | **0.5 (base)** | **−0.074** | **0.74** | **0.000** | **0.0** |
| 0.3 | 0.7 | −0.074 | 0.73 | 0.008 | 3.2 |
| 0.3 | 1.0 | −0.075 | 0.73 | 0.014 | 5.1 |
| 0.5 | 0.3 | −0.073 | 0.72 | 0.013 | 5.0 |
| 0.5 | 0.5 | −0.074 | 0.73 | 0.007 | 2.8 |
| 0.5 | 0.7 | −0.075 | 0.73 | 0.011 | 4.3 |
| 0.5 | 1.0 | −0.076 | 0.72 | 0.017 | 6.1 |
| 0.7 | 0.3 | −0.074 | 0.71 | 0.016 | 5.8 |
| 0.7 | 0.5 | −0.075 | 0.72 | 0.010 | 3.7 |
| 0.7 | 0.7 | −0.076 | 0.72 | 0.015 | 5.4 |
| 0.7 | 1.0 | −0.076 | 0.71 | 0.021 | 7.6 |

w_engagement: engagement component weight; w_intermediate: intermediate milestone component weight; primary weight fixed at 1.0. DR Value: Doubly Robust policy value estimate. Spearman ρ: rank correlation between shaped reward and 30-day acute care outcomes. JS Divergence: Jensen-Shannon divergence of policy action distribution relative to base case (w_e=0.3, w_i=0.5). % Actions Changed: fraction of test observations where the greedy action differs from base case. Results demonstrate that policy recommendations are robust across the full weight grid, with < 8% of actions changing and DR values varying by < 3%. The primary outcome component dominates policy behavior regardless of engagement and intermediate weight choices.

---

**Table S15. Proportional Parity Fairness Analysis**

| Group | Population (%) | Need Rate (ED+IP %) | Intervention Rate (%) | Proportional Parity Ratio | DP Gap (pp) |
|-------|---------------|---------------------|----------------------|--------------------------|-------------|
| White | 46.1 | 0.68 | 5.2 | 1.00 (reference) | — |
| Black or African American | 26.4 | 0.91 | 6.8 | 0.98 | 0.6 |
| Asian | 7.4 | 0.39 | 3.1 | 1.04 | 0.3 |
| Hispanic | 6.7 | 0.68 | 5.1 | 0.98 | 0.1 |
| Native Hawaiian/Pacific Islander | 2.0 | 0.48 | 3.9 | 1.06 | 0.2 |
| American Indian/Alaska Native | 1.3 | 0.90 | 6.6 | 0.96 | 0.4 |
| Other | 1.7 | 0.61 | 4.8 | 1.03 | 0.2 |
| Unknown | 8.4 | 0.40 | 3.2 | 1.05 | 0.3 |
| **Aggregate** | | | | **Range: 0.96–1.06** | **Max: 0.8** |

ED: emergency department; IP: inpatient hospitalization; DP: demographic parity; pp: percentage points. Need Rate defined as baseline acute care utilization rate (ED visits + hospitalizations per week). Proportional Parity Ratio = (group intervention rate / overall intervention rate) / (group need rate / overall need rate). A ratio of 1.0 indicates perfectly need-proportional allocation. Ratios range from 0.96 to 1.06, indicating near-proportional allocation across all racial and ethnic groups. Demographic parity gap computed as absolute difference in intervention rate from overall mean.

---

**Table S16. Race-Outcome Mediation Analysis by Social Determinants of Health**

| Race/Ethnicity (vs. White) | Total OR (95% CI) | Adjusted OR (95% CI) | % Mediated by SDoH | Key Mediators |
|---------------------------|-------------------|---------------------|-------------------|---------------|
| Black or African American | 1.35 (1.28, 1.43) | 1.18 (1.11, 1.26) | 49% (38%, 60%) | Housing instability, transportation |
| Hispanic | 1.02 (0.94, 1.11) | 0.97 (0.89, 1.06) | 62% (41%, 83%) | Insurance continuity, food security |
| Asian | 0.58 (0.51, 0.66) | 0.62 (0.54, 0.71) | — (protective) | — |
| Native Hawaiian/Pacific Islander | 0.71 (0.60, 0.84) | 0.74 (0.62, 0.88) | — (protective) | — |
| American Indian/Alaska Native | 1.31 (1.15, 1.49) | 1.15 (1.01, 1.32) | 52% (31%, 73%) | Housing instability, area deprivation |

OR: Odds Ratio; SDoH: Social Determinants of Health. Total OR from logistic regression Model 1: outcome ~ race + age + sex. Adjusted OR from Model 2: outcome ~ race + age + sex + housing stability + food security + transportation access + insurance continuity + area deprivation index. Proportion mediated = 1 − log(OR_adjusted) / log(OR_total). Bootstrap 95% CIs from 500 replicates. Mediation percentages indicate the proportion of the race-outcome association explained by included SDoH variables. Values of 47–62% suggest that observed mediators capture a substantial but incomplete portion of structural disadvantage associated with race. Groups with protective associations (OR < 1) do not have interpretable mediation percentages.

---

## Supplementary References

S1. Schulte PJ, Tsiatis AA, Laber EB, Davidian M. Q- and A-learning methods for estimating optimal dynamic treatment regimes. *Stat Sci*. 2014;29(4):640-661.

S2. Chakraborty B, Moodie EEM. *Statistical Methods for Dynamic Treatment Regimes*. Springer; 2013.

S3. Tian L, Alizadeh AA, Gentles AJ, Tibshirani R. A simple method for estimating interactions between a treatment and a large number of covariates. *J Am Stat Assoc*. 2014;109(508):1517-1532.

S4. Athey S, Tibshirani J, Wager S. Generalized random forests. *Ann Stat*. 2019;47(2):1148-1178.

S5. Chernozhukov V, Chetverikov D, Demirer M, et al. Double/debiased machine learning for treatment and structural parameters. *Econom J*. 2018;21(1):C1-C68.

---

*End of Supplementary Materials*
