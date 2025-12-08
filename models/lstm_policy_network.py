"""
LSTM-Based Policy Network for Temporal Care Management
======================================================

Replaces feedforward MLP with bidirectional LSTM + attention mechanism
to model temporal dependencies in patient trajectories.

Architecture:
- Input: Sequence of patient states (last 4 weeks)
- Bidirectional LSTM with 2 layers
- Multi-head attention over sequence
- Action embedding concatenation
- MLP head for action scoring

Author: Sanjay Basu, MD PhD
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for sequence modeling."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional mask of shape (batch, seq_len)
            
        Returns:
            output: Attended output (batch, seq_len, hidden_dim)
            attention_weights: Attention weights (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            # Expand mask for heads: (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Final linear projection
        output = self.out(attended)
        
        return output, attention_weights


class LSTMPolicyNetwork(nn.Module):
    """
    LSTM-based policy network for temporal patient state modeling.
    
    Architecture:
    1. Embedding layer for categorical features
    2. Bidirectional LSTM (2 layers)
    3. Multi-head attention over sequence
    4. Concatenate with action embedding
    5. MLP head for action scoring
    """
    
    def __init__(self,
                 state_dim: int,
                 action_embedding_dim: int,
                 categorical_dims: dict,
                 lstm_hidden_dim: int = 128,
                 lstm_layers: int = 2,
                 lstm_dropout: float = 0.2,
                 attention_heads: int = 4,
                 mlp_hidden_dim: int = 256,
                 mlp_dropout: float = 0.1,
                 sequence_length: int = 4):
        """
        Initialize LSTM policy network.
        
        Args:
            state_dim: Dimension of continuous state features
            action_embedding_dim: Dimension of action embeddings
            categorical_dims: Dict mapping categorical feature names to num categories
            lstm_hidden_dim: Hidden dimension for LSTM
            lstm_layers: Number of LSTM layers
            lstm_dropout: Dropout for LSTM
            attention_heads: Number of attention heads
            mlp_hidden_dim: Hidden dimension for MLP head
            mlp_dropout: Dropout for MLP
            sequence_length: Length of input sequence (number of timesteps)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_embedding_dim = action_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.sequence_length = sequence_length
        
        # Embedding layers for categorical features
        self.categorical_embeddings = nn.ModuleDict()
        self.categorical_embedding_dim = 16  # Fixed embedding size for categoricals
        total_categorical_dim = 0
        
        for feat_name, num_categories in categorical_dims.items():
            self.categorical_embeddings[feat_name] = nn.Embedding(
                num_categories, 
                self.categorical_embedding_dim
            )
            total_categorical_dim += self.categorical_embedding_dim
        
        # Total input dimension to LSTM
        self.lstm_input_dim = state_dim + total_categorical_dim
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            hidden_dim=2 * lstm_hidden_dim,  # *2 because bidirectional
            num_heads=attention_heads,
            dropout=lstm_dropout
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(2 * lstm_hidden_dim)
        
        # MLP head for action scoring
        # Input: attended LSTM output + action embedding
        self.mlp = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dim + action_embedding_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.LayerNorm(mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim // 2, 1)
        )
        
        # Action score bounding
        self.score_bound = 10.0
        
    def forward(self, 
                state_sequence: torch.Tensor,
                categorical_sequence: dict,
                action_embedding: torch.Tensor,
                sequence_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM policy network.
        
        Args:
            state_sequence: Continuous state features (batch, seq_len, state_dim)
            categorical_sequence: Dict of categorical features (batch, seq_len) for each category
            action_embedding: Action embeddings (batch, action_embedding_dim)
            sequence_mask: Binary mask for valid timesteps (batch, seq_len)
            
        Returns:
            action_score: Scalar score for state-action pair (batch, 1)
            attention_weights: Attention weights (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = state_sequence.shape
        
        # Embed categorical features
        embedded_categorical = []
        for feat_name, indices in categorical_sequence.items():
            if feat_name in self.categorical_embeddings:
                # indices shape: (batch, seq_len)
                embedded = self.categorical_embeddings[feat_name](indices)
                embedded_categorical.append(embedded)
        
        # Concatenate continuous and categorical features
        if embedded_categorical:
            categorical_features = torch.cat(embedded_categorical, dim=-1)
            lstm_input = torch.cat([state_sequence, categorical_features], dim=-1)
        else:
            lstm_input = state_sequence
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        # lstm_out shape: (batch, seq_len, 2*hidden_dim) because bidirectional
        
        # Apply multi-head attention
        attended, attention_weights = self.attention(lstm_out, mask=sequence_mask)
        
        # Residual connection + layer norm
        attended = self.norm1(lstm_out + attended)
        
        # Pool over sequence (mean pooling over non-masked positions)
        if sequence_mask is not None:
            # Expand mask to match attended shape
            mask_expanded = sequence_mask.unsqueeze(-1).expand_as(attended)
            attended_masked = attended * mask_expanded
            pooled = attended_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = attended.mean(dim=1)
        
        # Concatenate pooled state representation with action embedding
        combined = torch.cat([pooled, action_embedding], dim=-1)
        
        # Pass through MLP to get action score
        score = self.mlp(combined)
        
        # Bound score using tanh
        score = self.score_bound * torch.tanh(score / self.score_bound)
        
        return score, attention_weights
    
    def get_action_scores(self,
                         state_sequence: torch.Tensor,
                         categorical_sequence: dict,
                         action_embeddings: torch.Tensor,
                         sequence_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute scores for multiple actions given a single state sequence.
        
        Useful for inference when selecting best action from candidate set.
        
        Args:
            state_sequence: State sequence (batch, seq_len, state_dim)
            categorical_sequence: Categorical features dict
            action_embeddings: Multiple action embeddings (batch, num_actions, action_embedding_dim)
            sequence_mask: Sequence mask (batch, seq_len)
            
        Returns:
            scores: Action scores (batch, num_actions)
        """
        batch_size, num_actions, _ = action_embeddings.shape
        
        # Expand state sequence to match number of actions
        state_expanded = state_sequence.unsqueeze(1).expand(-1, num_actions, -1, -1)
        state_expanded = state_expanded.reshape(batch_size * num_actions, self.sequence_length, self.state_dim)
        
        # Expand categorical sequences
        categorical_expanded = {}
        for feat_name, indices in categorical_sequence.items():
            expanded = indices.unsqueeze(1).expand(-1, num_actions, -1)
            categorical_expanded[feat_name] = expanded.reshape(batch_size * num_actions, self.sequence_length)
        
        # Expand mask
        if sequence_mask is not None:
            mask_expanded = sequence_mask.unsqueeze(1).expand(-1, num_actions, -1)
            mask_expanded = mask_expanded.reshape(batch_size * num_actions, self.sequence_length)
        else:
            mask_expanded = None
        
        # Reshape action embeddings
        actions_flat = action_embeddings.reshape(batch_size * num_actions, self.action_embedding_dim)
        
        # Forward pass
        scores_flat, _ = self.forward(state_expanded, categorical_expanded, actions_flat, mask_expanded)
        
        # Reshape back to (batch, num_actions)
        scores = scores_flat.view(batch_size, num_actions)
        
        return scores


class LSTMValueNetwork(nn.Module):
    """LSTM-based value network for state value estimation."""
    
    def __init__(self,
                 state_dim: int,
                 categorical_dims: dict,
                 lstm_hidden_dim: int = 128,
                 lstm_layers: int = 2,
                 lstm_dropout: float = 0.2,
                 mlp_hidden_dim: int = 256,
                 mlp_dropout: float = 0.1,
                 sequence_length: int = 4):
        """Initialize LSTM value network."""
        super().__init__()
        
        self.state_dim = state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.sequence_length = sequence_length
        
        # Embedding layers for categorical features
        self.categorical_embeddings = nn.ModuleDict()
        self.categorical_embedding_dim = 16
        total_categorical_dim = 0
        
        for feat_name, num_categories in categorical_dims.items():
            self.categorical_embeddings[feat_name] = nn.Embedding(
                num_categories,
                self.categorical_embedding_dim
            )
            total_categorical_dim += self.categorical_embedding_dim
        
        self.lstm_input_dim = state_dim + total_categorical_dim
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # MLP head for value estimation
        self.mlp = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim // 2, 1)
        )
        
    def forward(self,
                state_sequence: torch.Tensor,
                categorical_sequence: dict,
                sequence_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to estimate state value.
        
        Args:
            state_sequence: Continuous state features (batch, seq_len, state_dim)
            categorical_sequence: Dict of categorical features
            sequence_mask: Binary mask for valid timesteps
            
        Returns:
            value: Estimated state value (batch, 1)
        """
        # Embed categorical features
        embedded_categorical = []
        for feat_name, indices in categorical_sequence.items():
            if feat_name in self.categorical_embeddings:
                embedded = self.categorical_embeddings[feat_name](indices)
                embedded_categorical.append(embedded)
        
        # Concatenate features
        if embedded_categorical:
            categorical_features = torch.cat(embedded_categorical, dim=-1)
            lstm_input = torch.cat([state_sequence, categorical_features], dim=-1)
        else:
            lstm_input = state_sequence
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # Pool over sequence
        if sequence_mask is not None:
            mask_expanded = sequence_mask.unsqueeze(-1).expand_as(lstm_out)
            lstm_masked = lstm_out * mask_expanded
            pooled = lstm_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = lstm_out.mean(dim=1)
        
        # MLP to get value estimate
        value = self.mlp(pooled)
        
        return value


def test_lstm_policy():
    """Test LSTM policy network with dummy data."""
    print("Testing LSTM Policy Network...")
    
    # Hyperparameters
    batch_size = 32
    seq_length = 4
    state_dim = 20  # 20 continuous features
    action_embedding_dim = 128
    num_actions = 1080
    
    categorical_dims = {
        'language_preference': 3,
        'employment_status': 4
    }
    
    # Create model
    policy = LSTMPolicyNetwork(
        state_dim=state_dim,
        action_embedding_dim=action_embedding_dim,
        categorical_dims=categorical_dims,
        lstm_hidden_dim=128,
        lstm_layers=2,
        sequence_length=seq_length
    )
    
    # Dummy input
   state_sequence = torch.randn(batch_size, seq_length, state_dim)
    categorical_sequence = {
        'language_preference': torch.randint(0, 3, (batch_size, seq_length)),
        'employment_status': torch.randint(0, 4, (batch_size, seq_length))
    }
    action_embedding = torch.randn(batch_size, action_embedding_dim)
    
    # Forward pass
    score, attention = policy(state_sequence, categorical_sequence, action_embedding)
    
    print(f"✓ Policy output shape: {score.shape}")
    print(f"✓ Attention weights shape: {attention.shape}")
    print(f"✓ Score range: [{score.min().item():.2f}, {score.max().item():.2f}]")
    
    # Test multi-action scoring
    action_embeddings = torch.randn(batch_size, 10, action_embedding_dim)
    scores = policy.get_action_scores(state_sequence, categorical_sequence, action_embeddings)
    print(f"✓ Multi-action scores shape: {scores.shape}")
    
    print("\n✓ LSTM Policy Network test passed!")


if __name__ == "__main__":
    test_lstm_policy()
