"""
Fairness Constraints for Policy Training
========================================

Implements fairness metrics and constraint losses to ensure
demographic equity in policy recommendations.

Focuses on:
- Demographic parity for high-resource interventions
- Equalized odds across demographic groups
- Calibration of predicted outcomes

Author: Sanjay Basu, MD PhD
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FairnessMetrics:
    """Compute fairness metrics for policy evaluation."""
    
    def __init__(self, protected_attributes: List[str] = ['race', 'age_group', 'sex']):
        """
        Initialize fairness metrics calculator.
        
        Args:
            protected_attributes: List of demographic attributes to monitor
        """
        self.protected_attributes = protected_attributes
        
    def demographic_parity(self,
                          actions: torch.Tensor,
                          demographics: Dict[str, torch.Tensor],
                          sensitive_action_indices: List[int] = None) -> Dict[str, float]:
        """
        Compute demographic parity metric.
        
        Measures: |P(A=a|D=d1) - P(A=a|D=d2)|
        
        Where A is a sensitive action (e.g., home visit), D is demographic group.
        
        Args:
            actions: Action indices (batch_size,)
            demographics: Dict mapping attribute names to indices (batch_size,)
            sensitive_action_indices: List of action indices to check (e.g., [42] for home visit)
            
        Returns:
            Dict of {attribute: parity_gap} for each protected attribute
        """
        if sensitive_action_indices is None:
            # Default: check all actions
            sensitive_action_indices = range(actions.max().item() + 1)
        
        parity_gaps = {}
        
        for attr in self.protected_attributes:
            if attr not in demographics:
                continue
            
            demo_values = demographics[attr]
            unique_values = demo_values.unique()
            
            max_gap = 0.0
            
            for action_idx in sensitive_action_indices:
                action_probs = []
                
                for demo_val in unique_values:
                    mask = (demo_values == demo_val)
                    if mask.sum() == 0:
                        continue
                    
                    # P(A=action_idx | D=demo_val)
                    prob = (actions[mask] == action_idx).float().mean().item()
                    action_probs.append(prob)
                
                if len(action_probs) >= 2:
                    # Compute max pairwise gap
                    gap = max(action_probs) - min(action_probs)
                    max_gap = max(max_gap, gap)
            
            parity_gaps[attr] = max_gap
        
        return parity_gaps
    
    def equalized_odds(self,
                      actions: torch.Tensor,
                      demographics: Dict[str, torch.Tensor],
                      outcomes: torch.Tensor,
                      sensitive_action_indices: List[int] = None) -> Dict[str, float]:
        """
        Compute equalized odds metric.
        
        Measures: |P(A=a|D=d1,Y=y) - P(A=a|D=d2,Y=y)|
        
        Requires conditioning on outcome Y.
        
        Args:
            actions: Action indices (batch_size,)
            demographics: Dict of demographics (batch_size,)
            outcomes: Binary outcomes (batch_size,) - 1 if adverse event, 0 otherwise
            sensitive_action_indices: Actions to check
            
        Returns:
            Dict of {attribute: equalized_odds_gap}
        """
        if sensitive_action_indices is None:
            sensitive_action_indices = range(actions.max().item() + 1)
        
        eq_odds_gaps = {}
        
        for attr in self.protected_attributes:
            if attr not in demographics:
                continue
            
            demo_values = demographics[attr]
            unique_values = demo_values.unique()
            
            max_gap = 0.0
            
            # Stratify by outcome
            for outcome_val in [0, 1]:
                outcome_mask = (outcomes == outcome_val)
                
                for action_idx in sensitive_action_indices:
                    action_probs = []
                    
                    for demo_val in unique_values:
                        mask = outcome_mask & (demo_values == demo_val)
                        if mask.sum() == 0:
                            continue
                        
                        # P(A=action_idx | D=demo_val, Y=outcome_val)
                        prob = (actions[mask] == action_idx).float().mean().item()
                        action_probs.append(prob)
                    
                    if len(action_probs) >= 2:
                        gap = max(action_probs) - min(action_probs)
                        max_gap = max(max_gap, gap)
            
            eq_odds_gaps[attr] = max_gap
        
        return eq_odds_gaps
    
    def calibration_error(self,
                         predicted_outcomes: torch.Tensor,
                         actual_outcomes: torch.Tensor,
                         demographics: Dict[str, torch.Tensor],
                         n_bins: int = 10) -> Dict[str, float]:
        """
        Compute calibration error across demographic groups.
        
        Measures: E[|P(Y=1|score,D=d1) - P(Y=1|score,D=d2)|]
        
        Args:
            predicted_outcomes: Predicted outcome probabilities or values (batch_size,)
            actual_outcomes: Actual binary outcomes (batch_size,)
            demographics: Dict of demographics
            n_bins: Number of bins for calibration curves
            
        Returns:
            Dict of {attribute: calibration_error}
        """
        calibration_errors = {}
        
        for attr in self.protected_attributes:
            if attr not in demographics:
                continue
            
            demo_values = demographics[attr]
            unique_values = demo_values.unique()
            
            if len(unique_values) < 2:
                continue
            
            # Create bins based on predicted scores
            bins = torch.linspace(
                predicted_outcomes.min().item(),
                predicted_outcomes.max().item(),
                n_bins + 1
            )
            
            max_error = 0.0
            
            for i in range(n_bins):
                bin_mask = (predicted_outcomes >= bins[i]) & (predicted_outcomes < bins[i+1])
                
                if bin_mask.sum() < 10:  # Skip bins with too few samples
                    continue
                
                # Actual positive rate in each demographic group for this bin
                pos_rates = []
                
                for demo_val in unique_values:
                    mask = bin_mask & (demo_values == demo_val)
                    if mask.sum() == 0:
                        continue
                    
                    pos_rate = actual_outcomes[mask].float().mean().item()
                    pos_rates.append(pos_rate)
                
                if len(pos_rates) >= 2:
                    error = max(pos_rates) - min(pos_rates)
                    max_error = max(max_error, error)
            
            calibration_errors[attr] = max_error
        
        return calibration_errors


class FairnessConstraintLoss(nn.Module):
    """Fairness constraint loss for policy training."""
    
    def __init__(self,
                 constraint_type: str = 'demographic_parity',
                 sensitive_actions: List[int] = None,
                 lambda_fairness: float = 0.01):
        """
        Initialize fairness constraint loss.
        
        Args:
            constraint_type: Type of fairness constraint ('demographic_parity', 'equalized_odds')
            sensitive_actions: List of sensitive action indices (e.g., high-resource interventions)
            lambda_fairness: Weight for fairness penalty
        """
        super().__init__()
        
        self.constraint_type = constraint_type
        self.sensitive_actions = sensitive_actions or []
        self.lambda_fairness = lambda_fairness
        
        logger.info(f"FairnessConstraintLoss initialized:")
        logger.info(f"  Constraint type: {constraint_type}")
        logger.info(f"  Sensitive actions: {sensitive_actions}")
        logger.info(f"  Lambda: {lambda_fairness}")
    
    def forward(self,
                action_probs: torch.Tensor,
                demographics: Dict[str, torch.Tensor],
                outcomes: torch.Tensor = None) -> torch.Tensor:
        """
        Compute fairness constraint loss.
        
        Args:
            action_probs: Action probabilities (batch_size, num_actions)
            demographics: Dict mapping attribute to indices (batch_size,)
            outcomes: Binary outcomes if using equalized odds (batch_size,)
            
        Returns:
            Fairness loss (scalar)
        """
        if self.lambda_fairness == 0.0:
            return torch.tensor(0.0, device=action_probs.device)
        
        if self.constraint_type == 'demographic_parity':
            return self._demographic_parity_loss(action_probs, demographics)
        elif self.constraint_type == 'equalized_odds':
            assert outcomes is not None, "Outcomes required for equalized odds"
            return self._equalized_odds_loss(action_probs, demographics, outcomes)
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")
    
    def _demographic_parity_loss(self,
                                action_probs: torch.Tensor,
                                demographics: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute demographic parity loss.
        
        Loss = Σ_actions Σ_demographics |P(A=a|D=d1) - P(A=a|D=d2)|
        
        This is differentiable and can be backpropagated.
        """
        total_loss = 0.0
        num_comparisons = 0
        
        for attr in demographics.keys():
            demo_values = demographics[attr]
            unique_values = demo_values.unique()
            
            # For each sensitive action
            for action_idx in self.sensitive_actions:
                # Compute P(A=action_idx | D=d) for each demographic group
                group_probs = []
                
                for demo_val in unique_values:
                    mask = (demo_values == demo_val).float()
                    if mask.sum() == 0:
                        continue
                    
                    # Expected probability of this action for this demographic group
                    group_prob = (action_probs[:, action_idx] * mask).sum() / mask.sum()
                    group_probs.append(group_prob)
                
                # Compute pairwise differences                
                for i in range(len(group_probs)):
                    for j in range(i + 1, len(group_probs)):
                        diff = torch.abs(group_probs[i] - group_probs[j])
                        total_loss += diff
                        num_comparisons += 1
        
        if num_comparisons == 0:
            return torch.tensor(0.0, device=action_probs.device)
        
        return self.lambda_fairness * (total_loss / num_comparisons)
    
    def _equalized_odds_loss(self,
                            action_probs: torch.Tensor,
                            demographics: Dict[str, torch.Tensor],
                            outcomes: torch.Tensor) -> torch.Tensor:
        """
        Compute equalized odds loss.
        
        Loss = Σ_actions Σ_demographics Σ_outcomes |P(A=a|D=d1,Y=y) - P(A=a|D=d2,Y=y)|
        """
        total_loss = 0.0
        num_comparisons = 0
        
        # Binarize outcomes
        binary_outcomes = (outcomes > 0).float()
        
        for attr in demographics.keys():
            demo_values = demographics[attr]
            unique_values = demo_values.unique()
            
            # Stratify by outcome
            for outcome_val in [0.0, 1.0]:
                outcome_mask = (binary_outcomes == outcome_val).float()
                
                if outcome_mask.sum() < 10:  # Skip if too few samples
                    continue
                
                for action_idx in self.sensitive_actions:
                    group_probs = []
                    
                    for demo_val in unique_values:
                        demo_mask = (demo_values == demo_val).float()
                        combined_mask = outcome_mask * demo_mask
                        
                        if combined_mask.sum() == 0:
                            continue
                        
                        # P(A=action_idx | D=demo_val, Y=outcome_val)
                        group_prob = (action_probs[:, action_idx] * combined_mask).sum() / combined_mask.sum()
                        group_probs.append(group_prob)
                    
                    # Pairwise differences
                    for i in range(len(group_probs)):
                        for j in range(i + 1, len(group_probs)):
                            diff = torch.abs(group_probs[i] - group_probs[j])
                            total_loss += diff
                            num_comparisons += 1
        
        if num_comparisons == 0:
            return torch.tensor(0.0, device=action_probs.device)
        
        return self.lambda_fairness * (total_loss / num_comparisons)


def test_fairness_constraints():
    """Test fairness constraint implementation."""
    print("Testing Fairness Constraints...")
    
    batch_size = 1000
    num_actions = 100
    
    # Create dummy data
    action_probs = F.softmax(torch.randn(batch_size, num_actions), dim=-1)
    demographics = {
        'race': torch.randint(0, 5, (batch_size,)),  # 5 race categories
        'age_group': torch.randint(0, 4, (batch_size,))  # 4 age groups
    }
    outcomes = torch.randint(0, 2, (batch_size,)).float()
    
    # Test demographic parity loss
    dp_loss = FairnessConstraintLoss(
        constraint_type='demographic_parity',
        sensitive_actions=[10, 20, 30],  # Home visit, specialist, etc.
        lambda_fairness=0.1
    )
    
    loss_dp = dp_loss(action_probs, demographics)
    print(f"✓ Demographic parity loss: {loss_dp.item():.6f}")
    
    # Test equalized odds loss
    eo_loss = FairnessConstraintLoss(
        constraint_type='equalized_odds',
        sensitive_actions=[10, 20, 30],
        lambda_fairness=0.1
    )
    
    loss_eo = eo_loss(action_probs, demographics, outcomes)
    print(f"✓ Equalized odds loss: {loss_eo.item():.6f}")
    
    # Test metrics
    metrics = FairnessMetrics()
    
    actions = action_probs.argmax(dim=-1)
    dp_metrics = metrics.demographic_parity(actions, demographics, sensitive_action_indices=[10, 20, 30])
    print(f"✓ Demographic parity gaps: {dp_metrics}")
    
    eo_metrics = metrics.equalized_odds(actions, demographics, outcomes, sensitive_action_indices=[10, 20, 30])
    print(f"✓ Equalized odds gaps: {eo_metrics}")
    
    print("\n✓ Fairness constraints test passed!")


if __name__ == "__main__":
    test_fairness_constraints()
