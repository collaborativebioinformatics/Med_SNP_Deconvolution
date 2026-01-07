"""
Interpretable Deep Learning Models for SNP/Genotype Data Analysis

This module implements deep learning architectures for GWAS/SNP analysis
with a focus on interpretability for identifying medically relevant SNPs.

Architectures:
- SNP_CNN: 1D CNN for local LD pattern extraction
- SNP_CNN_Transformer: Hybrid CNN + Transformer for causal SNP identification (recommended)

References:
- DPCformer (2025): CNN + Self-Attention for genotype-phenotype modeling
- DeepGWAS: Deep learning with functional annotations
- Hybrid CNN-Transformer models for causal SNP identification

Author: Generated for Haploblock Analysis Pipeline
Date: 2026-01-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional, Tuple, Dict, List
import numpy as np


# ============================================================================
# 1. GENOTYPE ENCODING AND PREPROCESSING
# ============================================================================

class GenotypeEncoder:
    """
    Encodes genotype data for deep learning models.

    Standard encoding schemes:
    - Additive: 0 (Ref/Ref), 1 (Ref/Alt), 2 (Alt/Alt)
    - One-hot: 3-dimensional binary vectors
    - 8-dimensional: Advanced encoding capturing haplotype information
    """

    @staticmethod
    def additive_encoding(genotypes: np.ndarray) -> torch.Tensor:
        """
        Standard additive encoding: 0/1/2 for Ref/Ref, Ref/Alt, Alt/Alt

        Args:
            genotypes: (n_samples, n_snps) array
        Returns:
            torch.Tensor of shape (n_samples, n_snps, 1)
        """
        encoded = torch.from_numpy(genotypes).float().unsqueeze(-1)
        return encoded

    @staticmethod
    def onehot_encoding(genotypes: np.ndarray) -> torch.Tensor:
        """
        One-hot encoding for 3 genotype states

        Args:
            genotypes: (n_samples, n_snps) array with values in {0, 1, 2}
        Returns:
            torch.Tensor of shape (n_samples, n_snps, 3)
        """
        n_samples, n_snps = genotypes.shape
        encoded = torch.zeros(n_samples, n_snps, 3)
        for i in range(3):
            encoded[:, :, i] = (genotypes == i).astype(float)
        return encoded

    @staticmethod
    def haplotype_encoding(genotypes: np.ndarray,
                          phased: bool = False) -> torch.Tensor:
        """
        8-dimensional encoding capturing haplotype information (DPCformer style)

        For unphased data:
        - Position 0-2: One-hot genotype (0/0, 0/1, 1/1)
        - Position 3-4: Allele counts
        - Position 5: Heterozygosity indicator
        - Position 6-7: Minor/Major allele presence

        Args:
            genotypes: (n_samples, n_snps) array
            phased: Whether data is phased
        Returns:
            torch.Tensor of shape (n_samples, n_snps, 8)
        """
        n_samples, n_snps = genotypes.shape
        encoded = torch.zeros(n_samples, n_snps, 8)

        # One-hot genotype (positions 0-2)
        for i in range(3):
            encoded[:, :, i] = torch.from_numpy((genotypes == i).astype(float))

        # Allele counts (positions 3-4)
        encoded[:, :, 3] = torch.from_numpy((genotypes >= 1).astype(float))  # Has alt allele
        encoded[:, :, 4] = torch.from_numpy((genotypes == 2).astype(float))  # Homozygous alt

        # Heterozygosity (position 5)
        encoded[:, :, 5] = torch.from_numpy((genotypes == 1).astype(float))

        # Minor/Major allele indicators (positions 6-7)
        maf = genotypes.mean(axis=0) / 2.0  # Minor allele frequency
        encoded[:, :, 6] = torch.from_numpy((genotypes > 0).astype(float))  # Has any alt
        encoded[:, :, 7] = torch.from_numpy((maf > 0.5).astype(float))  # MAF > 0.5

        return encoded

    @staticmethod
    def normalize_genotypes(genotypes: torch.Tensor,
                           method: str = 'standardize') -> torch.Tensor:
        """
        Normalize genotype data

        Args:
            genotypes: Encoded genotype tensor
            method: 'standardize', 'minmax', or 'none'
        Returns:
            Normalized tensor
        """
        if method == 'standardize':
            mean = genotypes.mean(dim=(0, 1), keepdim=True)
            std = genotypes.std(dim=(0, 1), keepdim=True) + 1e-8
            return (genotypes - mean) / std
        elif method == 'minmax':
            min_val = genotypes.min()
            max_val = genotypes.max()
            return (genotypes - min_val) / (max_val - min_val + 1e-8)
        else:
            return genotypes


# ============================================================================
# 2. ATTENTION MECHANISMS FOR INTERPRETABILITY
# ============================================================================

class SNPAttention(nn.Module):
    """
    Multi-head attention mechanism for SNP importance scoring.
    Attention weights can be extracted to identify causal SNPs.
    """

    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = input_dim // num_heads

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # Store for interpretability

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, n_snps, input_dim)
            return_attention: Whether to return attention weights
        Returns:
            output: (batch, n_snps, input_dim)
            attention_weights (optional): (batch, num_heads, n_snps, n_snps)
        """
        batch_size, n_snps, _ = x.shape

        # Linear transformations
        Q = self.query(x)  # (batch, n_snps, input_dim)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, n_snps, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_snps, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_snps, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        self.attention_weights = attention.detach()  # Store for interpretability

        # Apply attention to values
        attended = torch.matmul(self.dropout(attention), V)

        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, n_snps, self.input_dim)

        # Final linear transformation
        output = self.fc_out(attended)

        if return_attention:
            return output, attention
        return output

    def get_snp_importance(self) -> torch.Tensor:
        """
        Extract SNP importance scores from attention weights.
        Returns: (batch, n_snps) - averaged attention scores
        """
        if self.attention_weights is None:
            raise ValueError("No attention weights available. Run forward pass first.")

        # Average across heads and query positions
        importance = self.attention_weights.mean(dim=(1, 2))  # (batch, n_snps)
        return importance


# ============================================================================
# 3. CNN-BASED ARCHITECTURE (Best for regulatory effect estimation)
# ============================================================================

class SNP_CNN(nn.Module):
    """
    1D CNN for SNP data analysis.

    According to 2025 research, CNN architectures are most reliable for
    estimating enhancer regulatory effects of SNPs.

    Architecture:
    - 1D convolutions to capture local LD patterns
    - Batch normalization for stability
    - Max pooling for feature extraction
    - Attention layer for interpretability
    """

    def __init__(self,
                 n_snps: int,
                 encoding_dim: int = 8,
                 num_classes: int = 2,
                 conv_channels: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [3, 5, 7],
                 use_attention: bool = True,
                 dropout: float = 0.3):
        super().__init__()

        self.n_snps = n_snps
        self.encoding_dim = encoding_dim
        self.use_attention = use_attention

        # Convolutional layers with different kernel sizes (multi-scale)
        self.conv_layers = nn.ModuleList()
        in_channels = encoding_dim

        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                         kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ELU(),  # ELU activation (recommended in literature)
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels

        # Calculate output size after convolutions and pooling
        pooled_length = n_snps // (2 ** len(conv_channels))

        # Optional attention layer for interpretability
        if use_attention:
            self.attention = SNPAttention(conv_channels[-1], num_heads=8)
            self.attn_pooling = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        fc_input_dim = conv_channels[-1] * pooled_length
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, n_snps, encoding_dim) genotype tensor
        Returns:
            logits: (batch, num_classes)
        """
        # Transpose for 1D convolution: (batch, encoding_dim, n_snps)
        x = x.transpose(1, 2)

        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Transpose back: (batch, n_snps', channels)
        x = x.transpose(1, 2)

        # Optional attention
        attention_weights = None
        if self.use_attention:
            if return_attention:
                x, attention_weights = self.attention(x, return_attention=True)
            else:
                x = self.attention(x)
            x = x.transpose(1, 2)  # (batch, channels, n_snps')
        else:
            x = x.transpose(1, 2)

        # Fully connected layers
        logits = self.fc_layers(x)

        if return_attention and attention_weights is not None:
            return logits, attention_weights
        return logits


# ============================================================================
# 4. HYBRID CNN-TRANSFORMER (Best for causal SNP identification)
# ============================================================================

class SNP_CNN_Transformer(nn.Module):
    """
    Hybrid CNN-Transformer architecture.

    According to 2025 research, hybrid CNN-Transformer models are superior
    for causal SNP identification within LD blocks.

    Architecture:
    - CNN layers extract local LD patterns
    - Transformer layers model long-range dependencies
    - Multi-head attention provides interpretability
    """

    def __init__(self,
                 n_snps: int,
                 encoding_dim: int = 8,
                 num_classes: int = 2,
                 cnn_channels: List[int] = [32, 64],
                 kernel_size: int = 5,
                 transformer_dim: int = 128,
                 num_transformer_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.2):
        super().__init__()

        self.n_snps = n_snps

        # CNN feature extractor (local patterns)
        self.cnn = nn.Sequential(
            nn.Conv1d(encoding_dim, cnn_channels[0],
                     kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels[0], cnn_channels[1],
                     kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # Project to transformer dimension
        self.cnn_to_transformer = nn.Linear(cnn_channels[-1], transformer_dim)

        # Positional encoding (chromosome position aware)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, n_snps, transformer_dim) * 0.02
        )

        # Transformer encoder (long-range dependencies)
        encoder_layers = TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(
            encoder_layers,
            num_layers=num_transformer_layers
        )

        # Attention pooling for interpretability
        self.attention_pool = nn.Sequential(
            nn.Linear(transformer_dim, 1),
            nn.Softmax(dim=1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self.attention_weights = None

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, n_snps, encoding_dim)
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.shape[0]

        # CNN feature extraction
        x_cnn = x.transpose(1, 2)  # (batch, encoding_dim, n_snps)
        x_cnn = self.cnn(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)  # (batch, n_snps, cnn_channels)

        # Project to transformer dimension
        x_trans = self.cnn_to_transformer(x_cnn)

        # Add positional encoding
        x_trans = x_trans + self.positional_encoding

        # Transformer encoding
        x_trans = self.transformer(x_trans)  # (batch, n_snps, transformer_dim)

        # Attention pooling (interpretable aggregation)
        attention_scores = self.attention_pool(x_trans)  # (batch, n_snps, 1)
        self.attention_weights = attention_scores.squeeze(-1).detach()

        # Weighted sum
        x_pooled = (x_trans * attention_scores).sum(dim=1)  # (batch, transformer_dim)

        # Classification
        logits = self.classifier(x_pooled)

        if return_attention:
            return logits, self.attention_weights
        return logits

    def get_snp_importance(self) -> torch.Tensor:
        """Extract SNP importance from attention pooling weights"""
        if self.attention_weights is None:
            raise ValueError("No attention weights. Run forward pass first.")
        return self.attention_weights


# ============================================================================
# 5. INTERPRETABILITY MODULE
# ============================================================================

class IntegratedGradients:
    """
    Integrated Gradients for feature attribution.
    Identifies which SNPs contribute most to predictions.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def compute_attributions(self,
                            inputs: torch.Tensor,
                            target_class: int,
                            baseline: Optional[torch.Tensor] = None,
                            n_steps: int = 50) -> torch.Tensor:
        """
        Compute integrated gradients attribution.

        Args:
            inputs: (batch, n_snps, encoding_dim)
            target_class: Class index to compute attribution for
            baseline: Baseline input (default: zeros)
            n_steps: Number of integration steps
        Returns:
            attributions: (batch, n_snps, encoding_dim)
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, n_steps).to(inputs.device)
        interpolated_inputs = []

        for alpha in alphas:
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated_inputs.append(interpolated)

        interpolated_inputs = torch.cat(interpolated_inputs, dim=0)
        interpolated_inputs.requires_grad_(True)

        # Forward pass
        outputs = self.model(interpolated_inputs)

        # Backward pass for target class
        target_outputs = outputs[:, target_class]
        self.model.zero_grad()
        target_outputs.sum().backward()

        # Compute gradients
        gradients = interpolated_inputs.grad

        # Reshape and average
        batch_size = inputs.shape[0]
        gradients = gradients.view(n_steps, batch_size, *inputs.shape[1:])
        avg_gradients = gradients.mean(dim=0)

        # Integrated gradients = (input - baseline) * avg_gradients
        attributions = (inputs - baseline) * avg_gradients

        return attributions

    def get_snp_importance(self, attributions: torch.Tensor) -> torch.Tensor:
        """
        Aggregate attribution across encoding dimensions.

        Args:
            attributions: (batch, n_snps, encoding_dim)
        Returns:
            importance: (batch, n_snps)
        """
        return attributions.abs().sum(dim=-1)


# ============================================================================
# 6. COMPLETE INTERPRETABLE SNP MODEL
# ============================================================================

class InterpretableSNPModel(nn.Module):
    """
    Complete model with multiple interpretability methods.

    Features:
    - Choice of architecture (CNN, CNN-Transformer)
    - Built-in attention mechanisms
    - Integrated Gradients support
    - SNP importance extraction
    """

    def __init__(self,
                 n_snps: int,
                 encoding_dim: int = 8,
                 num_classes: int = 2,
                 architecture: str = 'cnn_transformer',
                 **kwargs):
        super().__init__()

        self.n_snps = n_snps
        self.encoding_dim = encoding_dim
        self.architecture = architecture

        # Select architecture
        if architecture == 'cnn':
            self.model = SNP_CNN(n_snps, encoding_dim, num_classes, **kwargs)
        elif architecture == 'cnn_transformer':
            self.model = SNP_CNN_Transformer(n_snps, encoding_dim, num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Use 'cnn' or 'cnn_transformer'")

        self.ig_explainer = IntegratedGradients(self.model)

    def forward(self, x: torch.Tensor, **kwargs):
        return self.model(x, **kwargs)

    def predict_with_interpretation(self,
                                    x: torch.Tensor,
                                    methods: List[str] = ['attention', 'integrated_gradients']
                                    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions and provide interpretability.

        Args:
            x: (batch, n_snps, encoding_dim)
            methods: List of interpretability methods
        Returns:
            dict with 'predictions' and importance scores
        """
        results = {}

        # Forward pass
        if 'attention' in methods:
            logits, attention = self.model(x, return_attention=True)
            results['attention_weights'] = attention
            results['snp_importance_attention'] = attention.mean(dim=1)  # Average over heads
        else:
            logits = self.model(x)

        results['logits'] = logits
        results['predictions'] = torch.argmax(logits, dim=-1)
        results['probabilities'] = F.softmax(logits, dim=-1)

        # Integrated Gradients
        if 'integrated_gradients' in methods:
            target_class = results['predictions'][0].item()
            attributions = self.ig_explainer.compute_attributions(x, target_class)
            results['attributions'] = attributions
            results['snp_importance_ig'] = self.ig_explainer.get_snp_importance(attributions)

        return results

    def identify_causal_snps(self,
                            x: torch.Tensor,
                            top_k: int = 10,
                            method: str = 'ensemble') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identify top causal SNPs.

        Args:
            x: Input genotypes
            top_k: Number of top SNPs to return
            method: 'attention', 'integrated_gradients', or 'ensemble'
        Returns:
            (indices, scores) of top SNPs
        """
        results = self.predict_with_interpretation(x, methods=['attention', 'integrated_gradients'])

        if method == 'attention':
            importance = results.get('snp_importance_attention', None)
        elif method == 'integrated_gradients':
            importance = results.get('snp_importance_ig', None)
        elif method == 'ensemble':
            # Average multiple methods
            attn = results.get('snp_importance_attention', torch.zeros(1, self.n_snps))
            ig = results.get('snp_importance_ig', torch.zeros(1, self.n_snps))
            # Normalize and average
            attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
            ig_norm = (ig - ig.min()) / (ig.max() - ig.min() + 1e-8)
            importance = (attn_norm + ig_norm) / 2
        else:
            raise ValueError(f"Unknown method: {method}")

        if importance is None:
            raise ValueError(f"Importance scores not available for method: {method}")

        # Get top-k SNPs
        top_scores, top_indices = torch.topk(importance[0], k=min(top_k, self.n_snps))

        return top_indices, top_scores


# ============================================================================
# 8. TRAINING UTILITIES
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Used in GPformer and other genomic prediction models.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Example usage of the interpretable SNP models"""

    # Simulate data
    batch_size = 16
    n_snps = 1000
    encoding_dim = 8
    num_classes = 2

    # Create sample genotype data
    genotypes = np.random.randint(0, 3, size=(batch_size, n_snps))

    # Encode genotypes
    encoder = GenotypeEncoder()
    x = encoder.haplotype_encoding(genotypes)  # (16, 1000, 8)
    x = encoder.normalize_genotypes(x, method='standardize')

    print(f"Encoded genotypes shape: {x.shape}")

    # Create model
    model = InterpretableSNPModel(
        n_snps=n_snps,
        encoding_dim=encoding_dim,
        num_classes=num_classes,
        architecture='cnn_transformer',
        dropout=0.2
    )

    print(f"\nModel architecture: {model.architecture}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Make predictions with interpretation
    results = model.predict_with_interpretation(x, methods=['attention', 'integrated_gradients'])

    print(f"\nPredictions shape: {results['predictions'].shape}")
    print(f"Predictions: {results['predictions']}")
    print(f"Probabilities shape: {results['probabilities'].shape}")

    # Identify causal SNPs
    top_indices, top_scores = model.identify_causal_snps(x, top_k=10, method='ensemble')

    print(f"\nTop 10 causal SNPs (indices): {top_indices.numpy()}")
    print(f"Importance scores: {top_scores.numpy()}")

    # Demonstrate different architectures
    print("\n" + "="*80)
    print("COMPARING ARCHITECTURES")
    print("="*80)

    for arch in ['cnn', 'cnn_transformer']:
        model = InterpretableSNPModel(
            n_snps=n_snps,
            encoding_dim=encoding_dim,
            num_classes=num_classes,
            architecture=arch
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{arch.upper():20s}: {n_params:>10,} parameters")
