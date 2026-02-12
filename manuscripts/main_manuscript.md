# Offline Reinforcement Learning for Care Management: Addressing Sparse Rewards, Temporal Dependencies, and Fairness in Medicaid Populations

Sanjay Basu, MD, PhD<sup>1,2,*</sup>, Sadiq Y. Patel, MSW, PhD<sup>1,3</sup>, Parth Sheth, MSE<sup>1,3</sup>, Bhairavi Muralidharan, MSE<sup>1</sup>, Namrata Elamaran, MSE<sup>1</sup>, Aakriti Kinra, MS<sup>1</sup>, Rajaie Batniji, MD, PhD<sup>1</sup>

<sup>1</sup>Waymark, San Francisco, CA, USA  
<sup>2</sup>University of California San Francisco, San Francisco, CA, USA  
<sup>3</sup>University of Pennsylvania, Philadelphia, PA, USA

**Correspondence:**  
Sanjay Basu, MD, PhD  
2120 Fillmore St, San Francisco, CA 94115  
Email: sanjay.basu@waymarkcare.com

---

## Abstract
Reinforcement learning offers a principled framework for optimizing sequential clinical decisions, yet deployment in healthcare remains limited by a persistent evaluation challenge. Off-policy evaluation methods must estimate policy performance from observational data without prospective trials, but standard importance sampling approaches suffer from extreme variance inflation in high-dimensional action spaces, rendering policy value estimates statistically unreliable. This variance challenge interacts with additional challenges inherent to healthcare applications: sparse reward signals from infrequent clinical outcomes, complex temporal dependencies in patient trajectories, confounding by indication in observational data, and fairness requirements across demographic subgroups. Here we present an offline reinforcement learning framework centered on doubly robust off-policy evaluation to address the variance inflation challenge. We demonstrate that doubly robust methods can achieve stable policy assessment in high-dimensional clinical action spaces where standard importance sampling produces unreliable estimates. To enable doubly robust evaluation in realistic clinical settings, we integrated four supporting methodological components: factored action space decomposition to reduce dimensionality while preserving clinical interpretability, multi-component reward shaping to address sparse binary outcomes, bidirectional long short-term memory networks with attention mechanisms for temporal modeling, and fairness-constrained training to ensure demographic equity, including intersectional and proportional parity analyses. We applied this framework to 9,998,139 weekly observations from 160,264 Medicaid beneficiaries enrolled in population health management programs across multiple states from 2023 through 2025. In this cohort, the factored action space comprised 97 unique intervention combinations. Standard weighted importance sampling failed to evaluate the policy reliably, achieving an effective sample size of 0.0% due to extreme variance inflation. In contrast, doubly robust evaluation, utilizing truncated importance weights, achieved an effective sample size of 44.2% (147-fold improvement) from a pre-truncation ESS of <0.1%, enabling stable policy assessment. The learned LSTM policy achieved a value statistically non-inferior to the behavioral policy. Ablation studies confirmed that temporal modeling with attention significantly outperformed feedforward baselines. Doubly robust off-policy evaluation addresses the variance inflation challenge inherent in high-dimensional clinical action spaces, enabling reliable, safe evaluation of reinforcement learning policies for population health management. This framework provides the methodological infrastructure necessary for prospective clinical trials of reinforcement learning-based decision support systems.



**Keywords:** reinforcement learning; machine learning; clinical decision support; population health management; health equity; causal inference; Medicaid

---

## Introduction

Reinforcement learning provides a mathematically principled framework for optimizing sequential clinical decisions by learning policies that maximize long-term patient outcomes<sup>1,2,3</sup>. Despite two decades of theoretical development, clinical deployment remains limited by persistent evaluation challenges. Healthcare reinforcement learning systems must learn from observational data without prospective randomized trials, requiring off-policy evaluation methods that estimate policy performance from historical records of clinician behavior<sup>4,5</sup>. Standard importance sampling approaches reweight observed outcomes by the probability ratio between the learned policy and the behavioral policy, but this reweighting suffers from extreme variance inflation when action spaces are high-dimensional<sup>6</sup>. In population health management settings, where clinicians select from dozens of intervention modalities, provider types, clinical goals, and urgency levels, the resulting combinatorial action spaces create importance weight distributions so heavy-tailed that policy value estimates become statistically unreliable.

This variance inflation challenge interacts with four additional challenges that distinguish healthcare applications from domains where reinforcement learning has succeeded. First, clinical outcomes occur infrequently, creating sparse reward signals that provide minimal gradient information for policy optimization<sup>7</sup>. In population health management populations, adverse events such as emergency department visits and hospitalizations occur in less than one percent of observation periods, yielding reward sparsity that fundamentally undermines value function estimation. Second, patient trajectories exhibit complex temporal dependencies across weeks to months<sup>8</sup>, requiring models that capture sequential patterns rather than assuming Markovian state transitions where current state alone determines optimal actions. Third, observational healthcare data exhibits confounding by indication, where sicker patients systematically receive more intensive interventions<sup>9</sup>, violating the randomness assumptions underlying standard reinforcement learning algorithms. Fourth, policies must maintain fairness across demographic subgroups to avoid perpetuating or amplifying healthcare disparities<sup>10</sup>, requiring explicit constraints during policy optimization rather than post-hoc auditing.

Existing work has addressed individual challenges in isolation but has not adequately addressed the central evaluation challenge. Doubly robust estimation combines outcome regression with importance weighting to reduce variance in single-decision causal inference problems<sup>11,12</sup>, but its application to sequential reinforcement learning evaluation in healthcare remains unexplored. Reward shaping methods augment sparse outcomes with intermediate signals<sup>13</sup>, but lack systematic frameworks for incorporating clinical domain knowledge about meaningful milestones. Factored action representations reduce dimensionality in robotics and multi-agent systems<sup>14,15</sup>, but have not been adapted to the specific structure of clinical interventions. Long short-term memory networks model temporal sequences effectively<sup>16</sup>, but are rarely integrated into policy architectures for healthcare applications. Fairness constraints have been developed for supervised learning<sup>17,18</sup>, but their incorporation into policy gradient methods for sequential decision-making remains limited.

This study presents an offline reinforcement learning framework centered on doubly robust off-policy evaluation to address the variance inflation challenge (Figure 1). We demonstrate that doubly robust methods can achieve stable policy assessment in high-dimensional clinical action spaces where standard importance sampling produces unreliable estimates. To enable doubly robust evaluation in realistic healthcare settings, we integrate four supporting methodological components that address sparse rewards, temporal dependencies, confounding, and fairness. Application to a large-scale Medicaid population health management program establishes that rigorous policy learning from observational healthcare data becomes feasible when the evaluation challenge is properly addressed.

---



---

## Results

### Dataset Characteristics

The study population comprised 160,264 Medicaid beneficiaries contributing 9,998,139 weekly observations over the 23-month study period. Demographic characteristics reflected the diversity of Medicaid populations (Table 1): 54.2% female, mean age 42.3 years (SD: 15.8), racial and ethnic distribution of 46.1% White, 26.4% Black or African American, 7.4% Asian, 6.7% Hispanic, 2.0% Native Hawaiian or Pacific Islander, 1.3% American Indian or Alaska Native, 1.7% other race, and 8.4% unknown or declined to state. The population exhibited high rates of chronic conditions (diabetes: 28.4%, hypertension: 41.2%, depression or anxiety: 35.7%, chronic obstructive pulmonary disease or asthma: 18.9%) and social complexity (housing instability: 12.3%, food insecurity: 18.7%, transportation barriers: 22.1%).

Primary outcome events occurred in 0.60% of weekly observations for emergency department visits and 0.09% for hospitalizations, yielding a combined acute care utilization rate of 0.63% per week. This sparse outcome distribution created fundamental challenges for standard reinforcement learning approaches relying solely on binary reward signals. Observed interventions occurred in 1.2% of weekly observations, with 97 distinct action combinations documented across the factored space of modality, provider, goal, and urgency dimensions (Supplementary Table S1).

### Variance Inflation and Doubly Robust Evaluation

Standard weighted importance sampling (WIS) failed to provide reliable policy evaluation in the high-dimensional action space (97 unique intervention combinations). The effective sample size (ESS) for WIS was 0.0% (3 out of 10,000 test sequences), with importance weights exhibiting extreme variance (maximum weight 605.74). This resulted in an unstable and invalid policy value estimate of −2.69 (Figure 2C).

In contrast, the truncated importance weights used in the doubly robust (DR) estimator achieved an effective sample size of 44.2% (4,423 effective sequences), representing a 147-fold improvement over WIS (Figure 2A). Pre-truncation weights yielded an ESS of <0.1%, confirming that both weight truncation and the control variate (outcome model) were necessary for stability. The DR estimator produced a stable policy value of −0.0736 (95% CI: −0.0827 to −0.0646) (Figure 2C). This result empirically demonstrates that doubly robust methods can address the variance inflation challenge inherent in factored clinical action spaces (Figure 2D, Supplementary Table S2).

The CQL baseline, evaluated using the same doubly robust estimator, achieved a DR policy value of −0.0811 (95% CI: −0.0903 to −0.0719) with an effective sample size of 41.8% (Supplementary Table S12). The CQL policy was also non-inferior to the behavioral policy, confirming that the doubly robust evaluation framework produces stable estimates regardless of the upstream RL algorithm. The AWR-based LSTM policy modestly outperformed CQL, consistent with the temporal modeling advantages of the LSTM architecture over the feedforward Q-network used in CQL.

### Methodological Innovations: Reward Shaping and Temporal Modeling

The multi-component reward function dramatically increased learning signal availability while maintaining alignment with clinical objectives (Figure 3A). Sparse binary rewards provided non-zero signals in only 0.60% of observations. Incorporating engagement indicators and intermediate clinical milestones through the shaped reward function increased non-zero reward occurrences to 32.1% of observations, a 53-fold improvement in signal density (Supplementary Table S3). The shaped reward exhibited strong correlation with long-term acute care outcomes (Spearman ρ = 0.74, p < 0.001). Component contributions to the overall reward distribution demonstrated appropriate weighting: primary acute care outcomes contributed 68.2% of reward variance, engagement signals 21.4%, and intermediate milestones 10.4% (Figure 3B).

Temporal modeling with attention mechanisms proved essential for performance. The long short-term memory (LSTM) policy achieved validation doubly robust value of -0.081 (95% CI: -0.084 to -0.078) compared to -0.095 (95% CI: -0.098 to -0.092) for an equivalent-capacity feedforward network (p < 0.001) (Figure 3D, Supplementary Table S4). Attention weight analysis revealed interpretable temporal focus patterns (Figure 3C, Supplementary Table S5), with the model allocating highest attention to the most recent week (mean weight: 0.65) while retaining significant capacity to attend to historical context (week t-1: 0.20; week t-2: 0.10). Differences between global mean attention (Figure 3C) and high-risk sequence attention (Table S5) reflect the model's dynamic adaptation to context; Table S5 reports weights conditional on decision-relevant high-risk sequences where history is most predictive. Sensitivity analysis over window lengths of 2, 4, 6, and 8 weeks confirmed that the 4-week window achieved the best validation loss and DR policy value, while all window lengths yielded comparable effective sample sizes (41–45%), demonstrating robustness to this design choice (Supplementary Table S13).

### Policy Performance and Fairness

The learned LSTM policy achieved a value of −0.0736, which was statistically non-inferior to the behavioral policy value of −0.0783 (95% CI: −0.0934 to −0.0632) (Figure 4B). Non-inferiority was established using a one-sided test with a pre-specified margin of 0.01 (representing ~10% of the outcome rate), confirming that the automated policy did not degrade performance relative to expert clinicians (p < 0.001 for non-inferiority).

Fairness analysis demonstrated that the learned policy maintained equitable intervention rates across demographic subgroups. High-resource intervention rates (in-person home visits, physician consultations) showed demographic parity within pre-specified bounds (Figure 4C, Supplementary Table S6). The maximum pairwise difference in home visit rates across racial and ethnic groups was 0.8 percentage points (95% CI: 0.6 to 1.1), substantially lower than the unconstrained baseline difference of 3.5 percentage points. Proportional parity analysis, which evaluates whether intervention rates are allocated proportionally to each group's clinical need (baseline acute care utilization rate), showed need-proportional ratios ranging from 0.92 to 1.08 across racial and ethnic groups, indicating near-proportional allocation (Supplementary Table S15). Intersectional analyses (e.g., Race × Gender) did not reveal significant disparities beyond those observed in the marginal groups (data not shown). However, policy evaluation reliability varied by subgroup, with lower effective sample sizes for Black or African American members (32.8%) compared to Asian members (76.8%) (Figure 4D), suggesting that fairness estimates for smaller or less-represented subgroups should be interpreted with caution due to higher evaluation variance.

---

## Discussion

This study substantially addresses the evaluation challenge that has limited clinical deployment of reinforcement learning for healthcare decision support. Standard importance sampling methods suffer from extreme variance inflation in high-dimensional action spaces, rendering policy value estimates statistically unreliable for clinical decision-making. We demonstrate that doubly robust off-policy evaluation achieves stable policy assessment where standard approaches produce unreliable estimates, increasing effective sample size from 0.0% to 44.2% (147-fold improvement). This mitigation of the variance inflation challenge, combined with integrated methodological components addressing sparse rewards, temporal dependencies, confounding, and fairness, enables rigorous policy learning from observational healthcare data. Application to a large-scale Medicaid population health management program established that learned policies can achieve non-inferiority to expert clinician behavior under rigorous evaluation, validating the framework's readiness for prospective clinical trials.

### Principal Findings

The resolution of the evaluation variance challenge represents the central methodological advance. Standard weighted importance sampling achieved only 0.0% effective sample size (3 out of 10,000 sequences), with importance weights exhibiting such extreme variance (maximum weight 605.74) that policy value estimates became statistically unreliable. Doubly robust estimation achieved 44.2% effective sample size, transforming off-policy evaluation from a theoretical construct into practical infrastructure for reliable performance assessment. This finding establishes that doubly robust methods provide essential variance reduction for reliable performance assessment in any healthcare reinforcement learning system operating on observational data. The magnitude of improvement—147-fold increase in effective sample size—quantifies the critical importance of variance reduction methods in making off-policy evaluation practically viable for clinical deployment.

The demonstration of non-inferiority to observed clinician behavior validates the complete methodological pipeline. Prior reinforcement learning systems in healthcare have either claimed superiority based on simulation without real deployment<sup>30</sup>, or failed to provide rigorous off-policy evaluation of learned policies<sup>31</sup>. Our framework achieves statistical non-inferiority under doubly robust evaluation methodology that properly accounts for confounding and variance, establishing that offline reinforcement learning can learn clinically coherent policies without performance degradation. This validation milestone creates the foundation for prospective clinical trials where actual policy deployment becomes ethically defensible.

The multi-component reward shaping approach increased learning signal density 53-fold while maintaining strong correlation with ultimate clinical outcomes. This addresses the fundamental challenge that meaningful outcome events (hospitalizations, emergency department visits) occur too infrequently to provide sufficient gradient signal for policy optimization. Prior reward shaping methods in healthcare have relied on hand-crafted potential functions<sup>32</sup> or assume access to detailed trajectory labels unavailable in real settings<sup>33</sup>. Our framework demonstrates that clinically meaningful intermediate signals - care engagement, successful contacts, disease management milestones - can augment sparse ultimate outcomes to enable learning while preserving alignment with what truly matters for patient health.

Temporal modeling through long short-term memory networks with attention mechanisms yielded 14.7% improvement in policy value compared to feedforward architectures. This demonstrates that care trajectories exhibit memory beyond the Markovian assumption underlying standard reinforcement learning, where current state alone is assumed sufficient for optimal decision-making. The learned attention patterns revealed interpretable focus on recent events while maintaining awareness of longer-term patterns, particularly during care disruptions. This finding validates that recurrent architectures are essential for healthcare reinforcement learning, not merely incremental enhancements.

Fairness-constrained training reduced disparities in high-resource intervention access from 3.5 to 0.8 percentage points across racial and ethnic groups while maintaining policy performance. This demonstrates that equity can be actively enforced during policy optimization rather than assessed post-hoc and addressed through separate interventions. The demographic parity constraint enabled equitable access to beneficial interventions while avoiding the use of race or ethnicity as direct decision inputs, addressing ethical concerns about algorithmic discrimination in healthcare artificial intelligence<sup>34</sup>.

The engagement component of the shaped reward function may implicitly capture socioeconomic status, since patients with greater healthcare access, stable housing, and reliable transportation may be more reachable and more likely to attend appointments. If engagement acts as a socioeconomic proxy, reward shaping could inadvertently assign higher rewards to already-advantaged patients. However, two features of our framework mitigate this concern. First, the demographic parity constraint during training explicitly penalizes differential intervention rates across racial and ethnic groups, regardless of engagement-driven reward differences. Second, the reward weight sensitivity analysis (Supplementary Table S14) demonstrates that even when the engagement weight is reduced to 0.1 (one-third of the base case), the policy's action distribution changes minimally (Jensen-Shannon divergence < 0.02), indicating that the primary outcome signal dominates policy behavior. The proportional parity analysis (Supplementary Table S15) further confirms that high-resource interventions are allocated proportionally to clinical need across demographic subgroups.

Given the extremely rare primary outcomes (0.60% weekly emergency department visits, 0.09% hospitalizations), the use of zero-inflated or count models warrants consideration. Zero-inflated models (e.g., zero-inflated Poisson, hurdle models) are designed for modeling zero-heavy count distributions in prediction or regression settings. However, in the reinforcement learning framework, the agent does not directly model the outcome distribution; rather, it learns a policy to maximize cumulative reward. The sparsity challenge in RL is not that the outcome distribution is zero-inflated per se, but that sparse binary rewards provide insufficient gradient signal for policy optimization. Our multi-component reward shaping addresses this by converting near-binary outcomes into continuous, information-rich learning signals through engagement and intermediate milestone components, increasing non-zero reward occurrences from 0.60% to 32.1%. This approach preserves policy optimality properties under potential-based reward shaping theory<sup>12</sup>. The Q-function model used in doubly robust evaluation does implicitly accommodate the zero-heavy distribution through its gradient-boosted regression structure (XGBoost), which naturally handles skewed and zero-heavy outcome distributions without requiring explicit zero-inflated parameterization.

### Comparison to Prior Work

Our framework advances beyond prior reinforcement learning applications in healthcare across multiple dimensions (Supplementary Table S7). The RL-CDS system for sepsis treatment<sup>35</sup> used on-policy learning requiring environment interaction, making it unsuitable for learning from observational data without prospective trials. The treatment policy learning framework for HIV<sup>36</sup> relied on standard importance sampling for off-policy evaluation, suffering from variance inflation we demonstrated renders such estimates unreliable in high-dimensional action spaces. The care management reinforcement learning system in our prior work<sup>37</sup> used simple SARSA algorithms with limited state representations and no systematic fairness constraints, temporal modeling, or rigorous hyperparameter optimization.

The doubly robust evaluation methodology we employ extends prior work on offline policy evaluation<sup>38,39</sup> to complex healthcare decision sequences with temporal dependencies. While doubly robust estimators have been applied to single-decision problems in epidemiology<sup>40</sup>, their integration into sequential reinforcement learning evaluation for healthcare represents a novel contribution. Our demonstration that effective sample size increases from 0.0% to 44.2% quantifies the critical importance of variance reduction methods in making off-policy evaluation practically viable for clinical deployment. Additionally, the CQL baseline comparison (Supplementary Table S12) demonstrates that the doubly robust evaluation framework produces stable estimates regardless of the upstream RL algorithm, suggesting the evaluation methodology generalizes beyond the specific AWR policy used in the primary analysis. A broader comparison across additional offline RL methods (e.g., Implicit Q-Learning, Batch-Constrained Q-Learning) would further strengthen these conclusions and represents a direction for future work.

The factored action space approach adapts techniques from robotics and multi-agent learning<sup>41,42</sup> to the specific structure of clinical interventions. Prior factorization methods assumed independent factors or required extensive domain-specific engineering. Our framework leverages the natural clinical taxonomy of population health management interventions (what, who, why, when) to achieve dimensionality reduction without sacrificing interpretability or requiring manual feature engineering beyond standard care categorization.

### Limitations

Several limitations warrant consideration. First, the observational study design precludes definitive causal conclusions about policy superiority. While doubly robust evaluation provides unbiased estimates under correct model specification, unmeasured confounding could bias results if critically important state variables are omitted or behavioral policy estimates are severely miscalibrated. The non-inferiority finding mitigates concern that the learned policy is dangerously different from current practice, but prospective randomized evaluation remains necessary to establish definitive comparative effectiveness.

Second, the temporal resolution of weekly observations may miss important within-week dynamics. Daily or sub-daily resolution would capture more granular decision-making but would exponentially increase computational requirements and exacerbate sparsity challenges. The four-week sequence length balances capturing relevant history against maintaining tractable sequence modeling, but  longer dependencies may exist in chronic disease trajectories extending months to years.

Third, the 30-day outcome window for acute care events represents a compromise between attribution confidence and outcome frequency. Longer windows would increase outcome rates but weaken causal links between interventions and subsequent events. Alternative outcome definitions, such as preventable hospitalizations using standard algorithms<sup>43</sup>, might better isolate events potentially modifiable through population health management.

Fourth, the framework's exclusion of race and ethnicity from policy inputs implicitly assumes that observed mediators (socioeconomic indicators, utilization history, social determinants of health) sufficiently capture structural disadvantage associated with race and ethnicity. A regression-based mediation analysis (Supplementary Table S16) estimated that 47–62% of the race-outcome association is explained by included social determinants of health variables, depending on the racial and ethnic group. This suggests that observed mediators capture a substantial but incomplete portion of structural disadvantage. Residual race-related pathways may persist through unmeasured mediators such as provider implicit bias, neighborhood-level environmental exposures, and historical disenrollment patterns. Future work should integrate richer social determinants of health data and formal causal mediation frameworks to more fully account for structural pathways.

Fifth, the fairness constraints focused on demographic parity for high-resource interventions. Alternative fairness definitions such as equalized odds (equal true and false positive rates across groups) or calibration (equal positive predictive value across groups) might be more appropriate depending on the decision context<sup>44</sup>. The optimal fairness criterion for healthcare reinforcement learning remains an open ethical and technical question requiring stakeholder engagement.

Sixth, generalization beyond Medicaid populations and population health management contexts remains uncertain. The patient-centered medical home model, chronic care model, and population health management frameworks share similar sequential decision structures<sup>45,46</sup>, suggesting potential transferability. However, adaptation to settings with different intervention modalities (pharmacotherapy decisions, surgical planning), time scales (emergency medicine, critical care), or data availability (limited electronic health record access) would require domain-specific modifications.

Seventh, model interpretability remains challenging despite architectural choices favoring transparency. While attention weights and action embeddings provide some insight into learned decision rules, the complete policy mapping from patient states to intervention probabilities resists simple explanation. Developing interpretability methods specific to healthcare reinforcement learning policies represents an important direction for future work to support clinician trust and regulatory approval.

Eighth, our framework assumes a stationary environment where patient populations, clinician behaviors, and outcome distributions remain constant. However, Medicaid populations exhibit substantial temporal instability due to enrollment cycling (30-40% annual churn), policy changes (redetermination periods, benefit modifications), and evolving social contexts (housing crises, economic shifts)<sup>51</sup>. Recent advances in forecasting-augmented offline reinforcement learning<sup>52</sup> could enable proactive adaptation to anticipated environmental shifts rather than reactive drift correction. Additionally, label-free drift detection methods<sup>53</sup> would allow earlier intervention before outcome degradation manifests, critical given the 30-day lag in our primary outcome window. Multi-calibration approaches that monitor and correct for subgroup-specific drift<sup>54</sup> would provide more robust fairness guarantees, particularly for intersectional subgroups and when race/ethnicity data is incomplete or substituted with ZIP-level deprivation indices. Future deployments should incorporate continuous monitoring of state and action distributions and automated retraining triggers when statistical thresholds are exceeded. Finally, while we demonstrate non-inferiority, prospective deployment would require strict safety constraints, including "do-no-harm" guardrails for high-risk states and human-in-the-loop override mechanisms for edge cases where model uncertainty is high.

### Future Directions

The validation of non-inferiority establishes the foundation for prospective randomized trials where learned policies guide actual clinical decisions. A pragmatic cluster-randomized trial comparing machine-learned policy recommendations to usual care, with randomization at the care team or facility level, would provide definitive evidence on comparative effectiveness while avoiding individual-level randomization challenges. Such trials could incorporate adaptive elements where policies continue learning from accruing trial data, combining experiment and optimization objectives.

Extensions to multi-agent reinforcement learning could model coordination between care managers, primary care physicians, specialists, and community health workers. Current formulations treat the care manager as sole decision-maker, but real care delivery involves distributed decision-making across multiple actors. Multi-agent approaches<sup>47</sup> could optimize joint policies accounting for information asymmetries, communication constraints, and resource conflicts inherent in team-based care.

Integration with large language models could enhance state representations through automated extraction from clinical notes and enable natural language policy explanations. Current feature engineering relies on structured data elements, missing rich contextual information in free-text notes. Recent advances in clinical language models<sup>48</sup> could augment state vectors while maintaining privacy and providing human-readable rationales for policy recommendations.

Causal discovery methods could identify modifiable risk factors and intermediate outcomes suitable for reward shaping from observational data. Our reward function incorporated engagement and intermediate milestones based on clinical domain knowledge, but data-driven approaches using causal inference<sup>49</sup> could systematically discover which intermediate events most strongly predict ultimate outcomes and where interventions have causal leverage.

Transfer learning across populations and settings could enable deployment in data-scarce environments. Meta-reinforcement learning approaches<sup>50</sup> could leverage experience from multiple population health management programs to quickly adapt policies to new populations with limited local data. This would address the cold-start problem limiting deployment in smaller health systems.

Continuous learning systems that update policies as new data accumulates could maintain performance as patient populations and best practices evolve. Current formulations assume stationary environments, but healthcare continuously changes through new treatments, care models, and population characteristics. Online learning methods that balance exploration and exploitation while maintaining safety guarantees represent critical infrastructure for deployed systems.

Prospective deployment requires explicit safety frameworks beyond non-inferiority validation. Recent work on safe reinforcement learning with human-preference bottlenecks<sup>55</sup> provides principled methods for combining learned policies with clinician oversight, formalizing the "override rules" common in clinical decision support while preserving adaptive advantages. Such systems could restrict policy actions to a state-dependent safe set, log overrides for continuous improvement, and escalate edge cases to human review. Integration with test-time adaptation methods<sup>56</sup> would further enable deployment in new Medicaid regions with minimal local data requirements, addressing limitations in transfer learning across diverse state policies and population characteristics.

### Conclusions

This framework addresses the evaluation challenge that has limited clinical deployment of reinforcement learning for healthcare decision support. Doubly robust off-policy evaluation achieves stable policy assessment in high-dimensional clinical action spaces where standard importance sampling produces unreliable estimates, increasing effective sample size from 0.0% to 44.2% (147-fold improvement). Integration with methodological components addressing sparse rewards, temporal dependencies, confounding, and fairness enables rigorous policy learning from observational healthcare data. The demonstration that learned policies achieve non-inferiority to expert clinician behavior under rigorous doubly robust evaluation establishes that offline reinforcement learning has matured sufficiently for prospective clinical validation. Future work should focus on prospective randomized trials, multi-agent coordination, natural language integration, causal discovery for reward engineering, transfer learning across settings, and continuous updating to maintain performance as clinical practice evolves.

---

## Methods

### Study Design and Data Source

We conducted a retrospective observational study using data from Medicaid beneficiaries enrolled in population health management programs across multiple states from January 1, 2023, through December 1, 2025. The population health management programs provided comprehensive support for individuals with complex medical and social needs, including chronic disease management, care coordination, and social determinants of health interventions. All beneficiaries provided informed consent for program participation and data use for quality improvement and research purposes. The study protocol received approval from WCG IRB (protocol #20253751).

The dataset comprised 9,998,139 weekly observations from 160,264 unique members, with mean observation period of 62.4 weeks per member (range: 4-96 weeks). Weekly observation periods were selected to align with standard population health management workflow cycles and provide sufficient time for intervention effects to manifest while maintaining temporal resolution for sequential decision modeling. Data sources included electronic health records (diagnoses, procedures, medications, laboratory values), administrative claims (utilization, costs), population health management system records (interventions, encounter notes), and social determinants of health assessments (housing stability, food security, transportation access).

### Outcome Measures

The primary outcome was occurrence of acute care utilization within 30 days, defined as emergency department visits or hospitalizations ascertained through claims data. Emergency department visits were identified using revenue codes 0450-0459 or 0981. Hospitalizations were identified through inpatient facility claims with admission dates. The 30-day prediction window was selected based on prior validation studies demonstrating that this interval captures preventable acute care while maintaining sufficient outcome frequency for model training<sup>17,18</sup>.

Secondary outcomes included care engagement (successful patient contact, appointment attendance), cost of care (total paid amount for all services), and intermediate clinical milestones (medication adherence, laboratory test completion, specialist visit attendance). Outcomes were extracted from multiple data sources and validated through manual chart review of a random sample of 500 episodes (Cohen's kappa=0.94 for emergency department visits, 0.97 for hospitalizations).

### Factored Action Space

The intervention space was decomposed into four clinically meaningful factors: modality (telephone, text message, video visit, in-person home visit), provider type (registered nurse, social worker, care coordinator, community health worker, physician), clinical goal (chronic disease management, care coordination, behavioral health support, social determinants of health), and urgency level (routine, semi-urgent, urgent). This factored representation reduced the theoretical action space from $$4 \times 5 \times 4 \times 3 = 240$$ possible combinations to 97 combinations observed in practice, enabling tractable policy optimization while preserving the clinical structure and interpretability required for implementation.

Each action was represented as a tuple $$a = (m, p, g, u)$$ where $$m \in \mathcal{M}$$ denotes modality, $$p \in \mathcal{P}$$ denotes provider, $$g \in \mathcal{G}$$ denotes goal, and $$u \in \mathcal{U}$$ denotes urgency. The policy $$\pi(a|s)$$ produced a probability distribution over the discrete action space conditioned on patient state $$s$$. Action embeddings were learned jointly with the policy network to capture semantic relationships between intervention types (embedding dimension: 32).

### State Representation

Patient states were constructed from 31 features spanning five domains: demographic characteristics (age, gender, race/ethnicity, preferred language, employment status), clinical status (active diagnoses, medication count, recent laboratory values for hemoglobin A1c and blood pressure, days since last clinical encounter), utilization history (emergency department visits and hospitalizations in prior 7, 30, and 90 days), engagement patterns (intervention count, successful contact rate, appointment no-show rate in prior 30 days), and social determinants (housing stability score, food security score, transportation access, insurance continuity). Features were selected based on clinical domain expertise and prior literature on risk stratification for complex populations<sup>19,20</sup>.

Continuous features were standardized to zero mean and unit variance. Categorical features were one-hot encoded for demographic attributes and embedded for high-cardinality variables. Missing laboratory values (23% of observations) were imputed using last observation carried forward with missingness indicator features. State vectors were constructed at weekly intervals aligned with population health management workflow cycles.

### Multi-Component Reward Shaping

To address the fundamental challenge of sparse binary outcomes (0.60% emergency department visits, 0.09% hospitalizations across all observations), we developed a multi-component reward function combining primary outcomes with engagement signals and intermediate clinical milestones:

$$R_{\text{shaped}}(s, a, s') = w_p R_p(s') + w_e R_e(s, a) + w_i R_i(s')$$

The primary component $$R_p(s') = -(ED_{30d} + 2 \times IP_{30d})$$ quantified acute care utilization, weighting hospitalizations twice as heavily as emergency department visits to reflect relative clinical severity and cost. The engagement component $$R_e(s, a)$$ provided graded reinforcement for productive patient interactions: +0.2 for successful patient contact, +0.1 for attempted but unsuccessful contact, −0.1 for appointment no-shows, and +0.15 for appointment attendance, based on validated associations between engagement and outcome improvement<sup>21</sup>. The intermediate milestone component $$R_i(s')$$ rewarded clinically meaningful care process indicators: +0.3 for primary care physician appointment attendance, +0.2 for medication fills, +0.1 for laboratory test completions, +0.2 for specialist visit attendance, and −0.2 for missed appointments.

Component weights ($$w_p=1.0, w_e=0.3, w_i=0.5$$) were determined through systematic grid search on held-out validation data, searching over $$w_e \in \{0.1, 0.3, 0.5, 0.7\}$$ and $$w_i \in \{0.3, 0.5, 0.7, 1.0\}$$ with $$w_p$$ fixed at 1.0, optimizing for correlation with long-term acute care outcomes (Supplementary Table S14). The selected weights achieved a Spearman correlation of ρ = 0.74 (p < 0.001) between shaped rewards and 30-day acute care outcomes. Sensitivity analysis across the full 4 × 4 grid of weight configurations demonstrated that policy recommendations were robust, with fewer than 8% of actions changing across all tested weight combinations and doubly robust policy values varying by less than 3% (Supplementary Table S14). This reward shaping increased the proportion of non-zero reward signals from 0.60% (sparse binary outcomes alone) to 32.1% of observations, providing substantially richer learning signal while preserving alignment with ultimate clinical objectives.

### Temporal Modeling with Long Short-Term Memory Networks

Patient trajectories were modeled as sequences of four consecutive weeks (28 days), selected to capture typical care episode duration while maintaining computational tractability. For each decision point at week $$t$$, the observed sequence comprised states $$\{s_{t-3}, s_{t-2}, s_{t-1}, s_t\}$$, actions $$\{a_{t-3}, a_{t-2}, a_{t-1}\}$$, and rewards $$\{r_{t-3}, r_{t-2}, r_{t-1}\}$$.

The policy network employed a bidirectional long short-term memory architecture with multi-head attention to model temporal dependencies (Supplementary Table S8). The bidirectional encoder processed the fixed-length historical window $$\{s_{t-3}, \dots, s_t\}$$ to extract retrospective features; crucially, no future information ($$s_{t+1}$$) was used in the decision process, ensuring causal validity for deployment.

$$h_t^{\text{LSTM}} = \text{LSTM}(\{s_{\tau}\}_{\tau=t-3}^t, \{a_{\tau}\}_{\tau=t-3}^{t-1})$$

$$\alpha_t = \text{Attention}(h_t^{\text{LSTM}})$$

$$\pi(a_t | s_{t-3:t}) = \text{softmax}(\text{MLP}(\alpha_t \odot h_t^{\text{LSTM}}, \text{embed}(a_t)))$$

The long short-term memory component (hidden dimension: 64, layers: 2, dropout: 0.2) captured sequential dependencies across the four-week window. The attention mechanism (heads: 4, key/query/value dimension: 16) enabled the model to selectively focus on relevant historical timesteps when evaluating current action values. Action embeddings (dimension: 32) were concatenated with the attended long short-term memory output and passed through a multilayer perceptron (hidden layers: 128, 64; dropout: 0.1) to produce action scores.

The value network employed an identical long short-term memory architecture without action conditioning:

$$V(s_{t-3:t}) = \text{MLP}(\text{Attention}(\text{LSTM}(\{s_{\tau}\}_{\tau=t-3}^t)))$$

Both networks were trained using the Advantage-Weighted Regression algorithm<sup>22</sup>, an offline reinforcement learning method suitable for learning from observational data without environment interaction. The advantage function $$A(s,a) = R(s,a) - V(s)$$ was estimated using the trained value network, and policy updates were weighted by exponentiated normalized advantages $$\exp(A(s,a)/\beta)$$ with temperature parameter $$\beta=1.0$$ to control the aggressiveness of policy improvement.

### Fairness-Constrained Training

To ensure equitable policy recommendations across demographic subgroups while avoiding the use of race or ethnicity as explicit decision features, we implemented demographic parity constraints during training. The fairness objective penalized differences in high-resource intervention rates (in-person home visits, physician consultations) across racial and ethnic groups:

$$\mathcal{L}_{\text{fairness}} = \lambda \sum_{d_1, d_2} \sum_{a \in \mathcal{A}_{\text{high}}} \left| P(A=a | D=d_1) - P(A=a | D=d_2) \right| $$

where $$\mathcal{A}_{\text{high}}$$ denotes the subset of resource-intensive interventions, $$D$$ represents demographic group membership, and $$\lambda=0.01$$ controls the strength of the fairness penalty. This constraint was incorporated as a differentiable soft penalty in the policy gradient, enabling end-to-end optimization for both performance and fairness objectives without requiring post-hoc adjustments.

Demographic group membership was used solely for fairness monitoring and constraint enforcement, not as input to the policy decision function. This approach maintains equity by ensuring comparable access to beneficial interventions while avoiding potential discrimination from direct use of protected attributes in decision-making<sup>23</sup>. In addition to demographic parity, we computed proportional parity, defined as the ratio of each group's high-resource intervention rate to its baseline clinical need (measured as the group's acute care utilization rate). A proportional parity ratio of 1.0 indicates perfectly need-proportional allocation, accounting for differential baseline risk across groups (Supplementary Table S15).

### Doubly Robust Off-Policy Evaluation

Evaluating policy performance from observational data without prospective deployment requires off-policy evaluation methods that estimate expected outcomes under a learned policy using data collected under a different behavioral policy<sup>24</sup>. Standard weighted importance sampling estimators suffer from prohibitive variance when action spaces are high-dimensional, as importance weights $$\rho(a|s) = \pi(a|s) / \pi_b(a|s)$$ can vary by orders of magnitude.

We employed doubly robust estimation, which combines outcome regression modeling with importance weighting to achieve lower variance than either approach alone<sup>25</sup>:

$$\hat{V}^{DR}(\pi) = \frac{1}{n} \sum_{i=1}^n \left[ \rho_i (R_i - \hat{Q}(s_i, a_i)) + \hat{Q}(s_i, a_i) \right]$$

where $$\hat{Q}(s,a)$$ is a separately trained regression model predicting expected return for state-action pairs, $$\rho_i = \pi(a_i|s_i) / \hat{\pi}_b(a_i|s_i)$$ is the importance weight, and $$R_i$$ is the observed return. The doubly robust estimator is unbiased if either the outcome model or the propensity model is correctly specified, and exhibits dramatically reduced variance compared to pure importance sampling when both models are approximately correct.

The outcome regression model $$\hat{Q}(s,a)$$ was implemented as a gradient boosted tree ensemble (XGBoost; max depth: 6, learning rate: 0.1, 500 trees) trained on the same observational data to predict shaped rewards conditional on states and actions. Model calibration was assessed via reliability diagrams, and root mean squared error (RMSE) was monitored on the validation set (RMSE: 0.312). The behavioral policy $$\hat{\pi}_b(a|s)$$ was estimated through maximum likelihood estimation on observed state-action pairs using a separate neural network (architecture: multilayer perceptron with hidden layers 256, 128, 64; dropout: 0.2). Propensity scores were calibrated using Platt scaling to ensure accurate probability estimates. Importance weights were truncated at the 95th percentile, a threshold selected to minimize mean squared error in validation experiments (Figure 2D) by balancing bias from truncation against variance from extreme weights.

We quantified evaluation reliability using effective sample size, defined as $$\text{ESS} = (\sum \rho_i)^2 / \sum \rho_i^2$$, which measures the equivalent number of independent observations contributing to the estimate. Values approaching the actual sample size indicate low variance and reliable evaluation, while values approaching zero indicate variance inflation that renders estimates unreliable<sup>26</sup>.

### Hyperparameter Optimization

All hyperparameters were systematically optimized through Bayesian optimization using the Tree-structured Parzen Estimator algorithm implemented in Optuna<sup>27</sup> (Supplementary Table S9). The search space encompassed long short-term memory architecture (hidden dimensions: 32-256, layers: 1-3, dropout: 0.1-0.3), attention mechanisms (heads: 1-8), multilayer perceptron structure (hidden dimensions: 64-512, layers: 1-3, dropout: 0.1-0.3), embedding dimensions (8-128), learning rates (1×10<sup>-5</sup> to 3×10<sup>-4</sup>), batch sizes (128-512), and Advantage-Weighted Regression temperature (0.5-2.0). The optimization objective was validation set policy value estimated via doubly robust off-policy evaluation.

The search ran for 100 trials with median pruning to terminate unpromising configurations after 5 training epochs. Final hyperparameters were selected based on best validation performance with at least 3 replications to ensure stability. This systematic optimization replaced manual tuning and provided documented justification for all architectural choices.

### Comparison to Alternative Offline RL Algorithms

To assess whether the observed policy performance arises specifically from the Advantage-Weighted Regression algorithm or generalizes across offline RL methods, we additionally trained a Conservative Q-Learning (CQL) baseline using the same state representation, reward function, and doubly robust evaluation framework. CQL penalizes Q-values for out-of-distribution actions through a conservative regularization term, providing an alternative approach to handling distributional shift in offline settings<sup>57</sup>. The CQL policy was implemented as a feedforward Q-network with the same hidden dimensions and was evaluated using the identical doubly robust estimator and behavioral policy model employed for the primary analysis (Supplementary Table S12).

### Statistical Analysis

The dataset was partitioned at the member level into training (50%), validation (25%), and test (25%) sets using stratified random sampling to maintain outcome distributions. This split was selected for three reasons: first, the central methodological contribution is the doubly robust evaluation framework, which requires a large held-out test set (25%) for unbiased policy assessment; second, the 25% validation set supports 100-trial Bayesian hyperparameter optimization with stable early-stopping decisions; and third, with 9,998,139 total observations, the 50% training set comprises approximately 5 million observations, yielding a parameter-to-observation ratio exceeding 145:1 for the LSTM architecture (34,337 parameters), which substantially exceeds standard recommendations for neural network training. This split is also consistent with standard practice in offline RL evaluation studies where the evaluation component is methodologically central<sup>24</sup>. All model development and hyperparameter selection used only training and validation data. Final evaluation was performed once on the held-out test set to provide unbiased performance estimates.

Policy value was estimated using doubly robust off-policy evaluation with bootstrap confidence intervals (1,000 replicates, bias-corrected and accelerated method). Non-inferiority of the learned policy to observed behavioral policy was assessed using a two-sided equivalence test with equivalence margin of ±0.01 on the shaped reward scale, corresponding to approximately 1% change in acute care events. P-values less than 0.05 for tests falling within the equivalence bounds indicated statistically significant non-inferiority.

Fairness metrics were computed as maximal pairwise differences in intervention rates across racial and ethnic groups for high-resource actions. Uncertainty was quantified through cluster bootstrapping at the member level to account for within-person correlation. All analyses were conducted using Python 3.9.7 with PyTorch 2.0.1, XGBoost 1.7.4, and Optuna 3.1.0. Code for full reproduction is available at https://github.com/sanjaybasu/factored-offline-rl.

This study adheres to the TRIPOD reporting guideline for prediction model development<sup>28</sup> and the DECIDE-AI guideline for clinical artificial intelligence<sup>29</sup>. The completed TRIPOD and DECIDE-AI checklists are provided in Supplementary Tables S10 and S11.

---



---

## Author Contributions

S.B. conceived the study, designed the methodology, conducted analyses, and wrote the manuscript. S.Y.P., P.S., B.M., N.E., and A.K. assisted with data preparation, model implementation, and interpretation of results. R.B. provided strategic guidance and critical revision of the manuscript. All authors approved the final version.

---

## Competing Interests

All authors are employed by the public benefit organization Waymark, which provides free social and healthcare services to Medicaid beneficiaries.

---

## Data Availability

Code and data available at https://github.com/sanjaybasu/factored-offline-rl

---

## References

1. Bodenheimer T, Chen E, Bennett HD. Confronting the growing burden of chronic disease: can the U.S. health care workforce do the job? *Health Aff (Millwood)*. 2009;28(1):64-74.

2. Peikes D, Chen A, Schore J, Brown R. Effects of care coordination on hospitalization, quality of care, and health care expenditures among Medicare beneficiaries: 15 randomized trials. *JAMA*. 2009;301(6):603-618.

3. Sutton RS, Barto AG. *Reinforcement Learning: An Introduction*. 2nd ed. MIT Press; 2018.

4. Yu C, Liu J, Nemati S, Yin G. Reinforcement learning in healthcare: a survey. *ACM Comput Surv*. 2023;55(1):Article 5.

5. Gottesman O, Johansson F, Komorowski M, Faisal A, Sontag D, Doshi-Velez F, Celi LA. Guidelines for reinforcement learning in healthcare. *Nat Med*. 2019;25(1):16-18.

6. Shortreed SM, Laber E, Lizotte DJ, Stroup TS, Pineau J, Murphy SA. Informing sequential clinical decision-making through reinforcement learning: an empirical study. *Mach Learn*. 2011;84(1-2):109-136.

7. Murphy SA. Optimal dynamic treatment regimes. *J R Stat Soc Series B Stat Methodol*. 2003;65(2):331-355.

8. Liu N, Koh ZX, Goh EC, et al. Prediction of adverse cardiac events in emergency department patients with chest pain using machine learning for variable selection. *BMC Med Inform Decis Mak*. 2014;14:75.

9. Hernán MA, Robins JM. *Causal Inference: What If*. Chapman & Hall/CRC; 2020.

10. Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. *Science*. 2019;366(6464):447-453.

11. Precup D, Sutton RS, Singh S. Eligibility traces for off-policy evaluation. In: *Proceedings of the 17th International Conference on Machine Learning*. Morgan Kaufmann; 2000:759-766.

12. Ng AY, Harada D, Russell S. Policy invariance under reward transformations: theory and application to reward shaping. In: *Proceedings of the 16th International Conference on Machine Learning*. Morgan Kaufmann; 1999:278-287.

13. Guestrin C, Koller D, Parr R, Venkataraman S. Efficient solution algorithms for factored MDPs. *J Artif Intell Res*. 2003;19:399-468.

14. Hochreiter S, Schmidhuber J. Long short-term memory. *Neural Comput*. 1997;9(8):1735-1780.

15. Bang H, Robins JM. Doubly robust estimation in missing data and causal inference models. *Biometrics*. 2005;61(4):962-973.

16. Hardt M, Price E, Srebro N. Equality of opportunity in supervised learning. In: *Advances in Neural Information Processing Systems* 29. 2016:3315-3323.

17. Kansagara D, Englander H, Salanitro A, et al. Risk prediction models for hospital readmission: a systematic review. *JAMA*. 2011;306(15):1688-1698.

18. Futoma J, Morris J, Lucas J. A comparison of models for predicting early hospital readmissions. *J Biomed Inform*. 2015;56:229-238.

19. Billings J, Dixon J, Mijanovich T, Wennberg D. Case finding for patients at risk of readmission to hospital: development of algorithm to identify high risk patients. *BMJ*. 2006;333(7563):327.

20. Haas LR, Takahashi PY, Shah ND, et al. Risk-stratification methods for identifying patients for care coordination. *Am J Manag Care*. 2013;19(9):725-732.

21. Hibbard JH, Greene J, Overton V. Patients with lower activation associated with higher costs; delivery systems should know their patients' 'scores'. *Health Aff (Millwood)*. 2013;32(2):216-222.

22. Peng XB, Kumar A, Zhang G, Levine S. Advantage-weighted regression: simple and scalable off-policy reinforcement learning. *arXiv preprint* arXiv:1910.00177. 2019.

23. Chouldechova A, Roth A. A snapshot of the frontiers of fairness in machine learning. *Commun ACM*. 2020;63(5):82-89.

24. Thomas P, Brunskill E. Data-efficient off-policy policy evaluation for reinforcement learning. In: *Proceedings of the 33rd International Conference on Machine Learning*. PMLR; 2016:2139-2148.

25. Dudík M, Langford J, Li L. Doubly robust policy evaluation and learning. In: *Proceedings of the 28th International Conference on Machine Learning*. 2011:1097-1104.

26. Owen AB. *Monte Carlo Theory, Methods and Examples*. 2013. http://statweb.stanford.edu/~owen/mc/

27. Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: a next-generation hyperparameter optimization framework. In: *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. 2019:2623-2631.

28. Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. *BMJ*. 2015;350:g7594.

29. Vasey B, Nagendran M, Campbell B, et al. Reporting guideline for the early stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI. *BMJ*. 2022;377:e070904.

30. Raghu A, Komorowski M, Celi LA, Szolovits P, Ghassemi M. Continuous state-space models for optimal sepsis treatment: a deep reinforcement learning approach. In: *Proceedings of Machine Learning for Healthcare*. PMLR; 2017:147-163.

31. Komorowski M, Celi LA, Badawi O, Gordon AC, Faisal AA. The artificial intelligence clinician learns optimal treatment strategies for sepsis in intensive care. *Nat Med*. 2018;24(11):1716-1720.

32. Nemati S, Ghassemi MM, Clifford GD. Optimal medication dosing from suboptimal clinical examples: a deep reinforcement learning approach. In: *Proceedings of the 38th Annual International Conference of the IEEE Engineering in Medicine and Biology Society*. IEEE; 2016:2978-2981.

33. Liu Y, Logan B, Liu N, Xu Z, Tang J, Wang Y. Deep reinforcement learning for dynamic treatment regimes on medical registry data. In: *Proceedings of the 2017 IEEE International Conference on Healthcare Informatics*. IEEE; 2017:380-385.

34. Char DS, Shah NH, Magnus D. Implementing machine learning in health care - addressing ethical challenges. *N Engl J Med*. 2018;378(11):981-983.

35. Killian TW, Daulton S, Konidaris G, Doshi-Velez F. Robust and efficient transfer learning with hidden parameter Markov decision processes. In: *Advances in Neural Information Processing Systems* 30. 2017:6405-6416.

36. Laber EB, Lizotte DJ, Qian M, Pelham WE, Murphy SA. Dynamic treatment regimes: technical challenges and applications. *Electron J Stat*. 2014;8(1):1225-1272.

37. Basu S, Muralidharan B, Sheth P, Wanek D, Morgan J, Patel SY. Reinforcement learning to prevent acute care events among Medicaid populations: mixed methods study. *JMIR AI*. 2025;4(1):e74264.

38. Jiang N, Li L. Doubly robust off-policy value evaluation for reinforcement learning. In: *Proceedings of the 33rd International Conference on Machine Learning*. PMLR; 2016:652-661.

39. Thomas P, Brunskill E. Policy evaluation using the Ω-return. In: *Advances in Neural Information Processing Systems* 28. 2015:334-342.

40. Funk MJ, Westreich D, Wiesen C, Stürmer T, Brookhart MA, Davidian M. Doubly robust estimation of causal effects. *Am J Epidemiol*. 2011;173(7):761-767.

41. Guestrin C, Venkataraman S, Koller D. Context-specific multiagent coordination and planning with factored MDPs. In: *Proceedings of the 18th National Conference on Artificial Intelligence*. AAAI Press; 2002:253-259.

42. Dibangoye JS, Amato C, Buffet O, Charpillet F. Optimally solving Dec-POMDPs as continuous-state MDPs. In: *Proceedings of the 23rd International Joint Conference on Artificial Intelligence*. AAAI Press; 2013:90-96.

43. Agency for Healthcare Research and Quality. *Prevention Quality Indicators Technical Specifications*. Version 2024. AHRQ; 2024.

44. Mitchell S, Potash E, Barocas S, D'Amour A, Lum K. Algorithmic fairness: choices, assumptions, and definitions. *Annu Rev Stat Appl*. 2021;8:141-163.

45. Wagner EH, Austin BT, Davis C, Hindmarsh M, Schaefer J, Bonomi A. Improving chronic illness care: translating evidence into action. *Health Aff (Millwood)*. 2001;20(6):64-78.

46. Starfield B, Shi L, Macinko J. Contribution of primary care to health systems and health. *Milbank Q*. 2005;83(3):457-502.

47. Bu L, Babu R, De Schutter B. A comprehensive survey of multiagent reinforcement learning. *IEEE Trans Syst Man Cybern C Appl Rev*. 2008;38(2):156-172.

48. Lee J, Yoon W, Kim S, Kim D, Kim S, So CH, Kang J. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*. 2020;36(4):1234-1240.

49. Pearl J. *Causality: Models, Reasoning, and Inference*. 2nd ed. Cambridge University Press; 2009.

50. Finn C, Abbeel P, Levine S. Model-agnostic meta-learning for fast adaptation of deep networks. In: *Proceedings of the 34th International Conference on Machine Learning*. PMLR; 2017:1126-1135.

51. Sommers BD, Gruber J. Federal funding insulated state budgets from increased spending related to Medicaid expansion. *Health Aff (Millwood)*. 2017;36(5):938-944.

52. Ada SE, Martius G, Ugur E, Oztop E. Forecasting in offline reinforcement learning for non-stationary environments. In: *Advances in Neural Information Processing Systems* 39. 2025. arXiv:2512.01987.

53. Finlayson SG, Subbaswamy A, Singh K, et al. The clinician and dataset shift in artificial intelligence. *N Engl J Med*. 2021;385(3):283-286.

54. Hébert-Johnson U, Kim MP, Reingold O, Rothblum GN. Multicalibration: calibration for the (computationally-identifiable) masses. In: *Proceedings of the 35th International Conference on Machine Learning*. PMLR; 2018:1939-1948.

55. Christiano PF, Leike J, Brown T, Martic M, Legg S, Amodei D. Deep reinforcement learning from human preferences. In: *Advances in Neural Information Processing Systems* 30. 2017:4299-4307.

56. Sun Y, Wang X, Liu Z, et al. Test-time training with self-supervision for generalization under distribution shifts. In: *Proceedings of the 37th International Conference on Machine Learning*. PMLR; 2020:9229-9248.

57. Kumar A, Zhou A, Tucker G, Levine S. Conservative Q-learning for offline reinforcement learning. In: *Advances in Neural Information Processing Systems* 33. 2020:1179-1191.

---

## Tables

**Table 1. Demographic and Clinical Characteristics of the Study Population**

| Characteristic | N = 160,264 | % or Mean (SD) |
|----------------|-------------|----------------|
| **Demographics** | | |
| Age, years | | 42.3 (15.8) |
| Female sex | 86,863 | 54.2% |
| **Race and Ethnicity** | | |
| White | 73,882 | 46.1% |
| Black or African American | 42,310 | 26.4% |
| Asian | 11,859 | 7.4% |
| Hispanic | 10,738 | 6.7% |
| Native Hawaiian or Pacific Islander | 3,205 | 2.0% |
| American Indian or Alaska Native | 2,083 | 1.3% |
| Other | 2,725 | 1.7% |
| Unknown / Declined | 13,462 | 8.4% |
| **Clinical Conditions** | | |
| Hypertension | 66,029 | 41.2% |
| Depression or Anxiety | 57,214 | 35.7% |
| Diabetes Mellitus | 45,515 | 28.4% |
| COPD or Asthma | 30,290 | 18.9% |
| **Social Complexity** | | |
| Transportation Barriers | 35,418 | 22.1% |
| Food Insecurity | 29,969 | 18.7% |
| Housing Instability | 19,712 | 12.3% |
| **Utilization (Weekly Rate)** | | |
| Emergency Department Visits | - | 0.60% |
| Inpatient Hospitalizations | - | 0.09% |

*Note: SD = Standard Deviation; COPD = Chronic Obstructive Pulmonary Disease. Clinical conditions identified via ICD-10 diagnosis codes in the 12 months prior to study entry. Social complexity factors assessed via standardized screening instruments.*

---

## Figure Legends

**Figure 1: Conceptual Framework.** (A) The variance inflation challenge in healthcare RL: standard importance sampling produces unreliable estimates in high-dimensional action spaces (red), while doubly robust evaluation (blue) remains stable. (B) Factored action space decomposition reduces dimensionality while preserving clinical semantics. (C) Multi-component reward shaping addresses sparse outcomes by incorporating engagement and intermediate milestone signals. (D) Bidirectional LSTM with attention captures complex temporal dependencies in patient trajectories. (E) Fairness constraints ensure equitable policy performance across demographic subgroups.

**Figure 2: Variance Inflation Challenge and Resolution.** (A) Effective sample size (ESS) comparison between standard weighted importance sampling (WIS, 0.0%) and doubly robust evaluation (DR, 44.2%). (B) Distribution of importance weights, showing extreme tails for WIS (max > 600) vs. stabilized weights for DR. (C) Policy value estimates with 95% confidence intervals, demonstrating the invalidity of WIS estimates (-2.69) compared to the stable DR estimate (-0.0736). (D) Sensitivity analysis of policy value estimates to key assumptions (Q-function bias, propensity error, weight truncation), demonstrating robustness of the DR estimator.

**Figure 3: Methodological Innovations.** (A) Reward signal density comparison: sparse binary outcomes (0.6%) vs. shaped multi-component rewards (32.1%). (B) Variance decomposition of the shaped reward function, showing relative contributions of acute care outcomes (68.2%), engagement (21.4%), and intermediate milestones (10.4%). (C) Attention weight profile showing the model's focus on recent events (t) and relevant historical context (t-3). (D) Validation loss comparison across model architectures, demonstrating the superiority of the LSTM+Attention model (0.457) over feedforward baselines (0.485).

**Figure 4: Clinical Validation and Fairness.** (A) Learning curves showing convergence of the policy value on training and validation sets. (B) Non-inferiority assessment: the learned policy value (-0.0736) is statistically non-inferior to the behavioral policy (-0.0783), with the 95% CI of the difference excluding clinically significant degradation. (C) Demographic parity assessment for high-resource interventions (home visits), showing reduced disparity under the fairness-constrained policy (0.8 pp). (D) Subgroup effective sample sizes, highlighting variance in evaluation reliability across racial/ethnic groups (e.g., lower ESS for Black/African American members).
