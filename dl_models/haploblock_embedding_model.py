"""
Haploblock Embedding Model for SNP Deconvolution

Pipeline output structure (step5_variant_hashes.py):
    individual_hash = strand_hash(4) + chr_hash + haploblock_hash(20) + cluster_hash(20)

Key insight: The CLUSTER ID is the meaningful categorical feature!
- Each individual belongs to a cluster within each haploblock
- Cluster membership is the key discriminative feature for population classification

Architecture:
    Cluster IDs → Embedding → CNN/Transformer → Classification

Author: Generated for Haploblock Analysis Pipeline
Date: 2026-01-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional, Tuple, Dict, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HaploblockEmbedding(nn.Module):
    """
    Embedding layer for haploblock hashes.

    Each haploblock has its own embedding table since hash spaces are independent.
    This allows the model to learn meaningful representations for each hash value.

    Args:
        n_haploblocks: Number of haploblocks
        vocab_sizes: List of vocabulary sizes for each haploblock (max hash ID + 1)
        embedding_dim: Embedding dimension
        padding_idx: Index for padding/unknown hashes (default: 0)

    Input: (batch, n_haploblocks) - hash IDs per haploblock
    Output: (batch, n_haploblocks, embedding_dim)
    """

    def __init__(
        self,
        n_haploblocks: int,
        vocab_sizes: List[int],
        embedding_dim: int = 32,
        padding_idx: int = 0
    ):
        super().__init__()
        self.n_haploblocks = n_haploblocks
        self.embedding_dim = embedding_dim

        # One embedding table per haploblock
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx
            )
            for vocab_size in vocab_sizes
        ])

        # Initialize embeddings
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0, std=0.02)
            if padding_idx is not None:
                emb.weight.data[padding_idx].zero_()

        logger.info(
            f"HaploblockEmbedding: {n_haploblocks} haploblocks, "
            f"dim={embedding_dim}, vocab_sizes={vocab_sizes[:3]}..."
        )

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hash_ids: (batch, n_haploblocks) tensor of hash IDs

        Returns:
            (batch, n_haploblocks, embedding_dim) embedded representations
        """
        batch_size = hash_ids.shape[0]
        embedded = torch.zeros(
            batch_size, self.n_haploblocks, self.embedding_dim,
            device=hash_ids.device, dtype=torch.float32
        )

        for i, emb in enumerate(self.embeddings):
            embedded[:, i, :] = emb(hash_ids[:, i])

        return embedded


class CompositeHaploblockEmbedding(nn.Module):
    """
    Composite embedding that decomposes hash components and adds them in shared space.

    Hash structure: strand(4) + chr(10) + haploblock(20) + cluster(20)

    Final embedding = strand_emb + position_emb + cluster_emb
    (Similar to BERT: token_emb + position_emb + segment_emb)

    This allows the model to learn:
    - Strand-specific patterns (maternal vs paternal)
    - Position-dependent patterns (where on chromosome)
    - Cluster-specific patterns (which sequence cluster)

    Args:
        n_haploblocks: Number of haploblocks
        vocab_sizes: List of cluster vocabulary sizes per haploblock
        embedding_dim: Shared embedding dimension
        n_strands: Number of strands (default: 2 for diploid)
        n_chromosomes: Number of chromosomes (default: 23)
        use_strand_embedding: Whether to include strand embedding
        use_position_embedding: Whether to include position embedding
        use_chromosome_embedding: Whether to include chromosome embedding
        dropout: Embedding dropout rate

    Input:
        cluster_ids: (batch, n_haploblocks) cluster assignments
        strand_ids: (batch,) or (batch, n_haploblocks) strand IDs (0 or 1)
        chr_ids: (batch,) chromosome ID (optional)

    Output: (batch, n_haploblocks, embedding_dim)
    """

    def __init__(
        self,
        n_haploblocks: int,
        vocab_sizes: List[int],
        embedding_dim: int = 64,
        n_strands: int = 2,
        n_chromosomes: int = 23,
        use_strand_embedding: bool = True,
        use_position_embedding: bool = True,
        use_chromosome_embedding: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_haploblocks = n_haploblocks
        self.embedding_dim = embedding_dim
        self.use_strand_embedding = use_strand_embedding
        self.use_position_embedding = use_position_embedding
        self.use_chromosome_embedding = use_chromosome_embedding

        # Cluster embeddings (one per haploblock, main semantic feature)
        self.cluster_embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=0
            )
            for vocab_size in vocab_sizes
        ])

        # Strand embedding (maternal=0, paternal=1)
        if use_strand_embedding:
            self.strand_embedding = nn.Embedding(
                num_embeddings=n_strands,
                embedding_dim=embedding_dim
            )

        # Positional embedding for haploblock position
        if use_position_embedding:
            self.position_embedding = nn.Embedding(
                num_embeddings=n_haploblocks,
                embedding_dim=embedding_dim
            )
            # Register position indices as buffer
            self.register_buffer(
                'position_ids',
                torch.arange(n_haploblocks).unsqueeze(0)
            )

        # Chromosome embedding (optional, for multi-chromosome models)
        if use_chromosome_embedding:
            self.chromosome_embedding = nn.Embedding(
                num_embeddings=n_chromosomes,
                embedding_dim=embedding_dim
            )

        # LayerNorm and dropout for regularization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize
        self._init_weights()

        logger.info(
            f"CompositeHaploblockEmbedding: {n_haploblocks} haploblocks, "
            f"dim={embedding_dim}, strand={use_strand_embedding}, "
            f"position={use_position_embedding}, chr={use_chromosome_embedding}"
        )

    def _init_weights(self):
        """Initialize embedding weights."""
        for emb in self.cluster_embeddings:
            nn.init.normal_(emb.weight, mean=0, std=0.02)
            emb.weight.data[0].zero_()  # padding

        if self.use_strand_embedding:
            nn.init.normal_(self.strand_embedding.weight, mean=0, std=0.02)

        if self.use_position_embedding:
            nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)

        if self.use_chromosome_embedding:
            nn.init.normal_(self.chromosome_embedding.weight, mean=0, std=0.02)

    def forward(
        self,
        cluster_ids: torch.Tensor,
        strand_ids: Optional[torch.Tensor] = None,
        chr_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute composite embedding by summing components.

        Args:
            cluster_ids: (batch, n_haploblocks) cluster assignments
            strand_ids: (batch,) or (batch, n_haploblocks) strand IDs
            chr_ids: (batch,) chromosome IDs

        Returns:
            (batch, n_haploblocks, embedding_dim) composite embeddings
        """
        batch_size = cluster_ids.shape[0]
        device = cluster_ids.device

        # 1. Cluster embedding (main semantic feature)
        cluster_emb = torch.zeros(
            batch_size, self.n_haploblocks, self.embedding_dim,
            device=device, dtype=torch.float32
        )
        for i, emb in enumerate(self.cluster_embeddings):
            cluster_emb[:, i, :] = emb(cluster_ids[:, i])

        # 2. Position embedding
        if self.use_position_embedding:
            position_ids = self.position_ids.expand(batch_size, -1)
            position_emb = self.position_embedding(position_ids)
            cluster_emb = cluster_emb + position_emb

        # 3. Strand embedding
        if self.use_strand_embedding and strand_ids is not None:
            if strand_ids.dim() == 1:
                # (batch,) -> (batch, n_haploblocks, dim)
                strand_emb = self.strand_embedding(strand_ids).unsqueeze(1)
                strand_emb = strand_emb.expand(-1, self.n_haploblocks, -1)
            else:
                # (batch, n_haploblocks) -> (batch, n_haploblocks, dim)
                strand_emb = self.strand_embedding(strand_ids)
            cluster_emb = cluster_emb + strand_emb

        # 4. Chromosome embedding (optional)
        if self.use_chromosome_embedding and chr_ids is not None:
            chr_emb = self.chromosome_embedding(chr_ids).unsqueeze(1)
            chr_emb = chr_emb.expand(-1, self.n_haploblocks, -1)
            cluster_emb = cluster_emb + chr_emb

        # LayerNorm + Dropout
        output = self.layer_norm(cluster_emb)
        output = self.dropout(output)

        return output


class SharedHaploblockEmbedding(nn.Module):
    """
    Shared embedding for all haploblocks (when hash spaces are unified).

    Use this when all haploblocks share the same hash vocabulary.
    More parameter-efficient than per-haploblock embeddings.

    Args:
        vocab_size: Total vocabulary size (max hash ID + 1)
        embedding_dim: Embedding dimension
        n_haploblocks: Number of haploblocks (for positional encoding)
        use_positional: Add positional encoding for haploblock position

    Input: (batch, n_haploblocks) - hash IDs
    Output: (batch, n_haploblocks, embedding_dim)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        n_haploblocks: int = 100,
        use_positional: bool = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_positional = use_positional

        # Shared hash embedding
        self.hash_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # Positional encoding for haploblock position
        if use_positional:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, n_haploblocks, embedding_dim) * 0.02
            )

        nn.init.normal_(self.hash_embedding.weight, mean=0, std=0.02)
        self.hash_embedding.weight.data[0].zero_()

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hash_ids: (batch, n_haploblocks)

        Returns:
            (batch, n_haploblocks, embedding_dim)
        """
        embedded = self.hash_embedding(hash_ids)

        if self.use_positional:
            embedded = embedded + self.positional_encoding

        return embedded


class HaploblockTransformer(nn.Module):
    """
    Transformer model for haploblock-based population classification.

    Architecture:
        Hash IDs → Embedding → Transformer → Attention Pooling → Classification

    Features:
    - Learnable hash embeddings (no manual feature engineering)
    - Transformer captures inter-haploblock relationships
    - Attention pooling provides interpretability

    Embedding modes:
    - 'simple': Per-haploblock embedding (default)
    - 'shared': Shared embedding across haploblocks
    - 'composite': Additive embedding (cluster + position + strand)

    Args:
        n_haploblocks: Number of haploblocks
        vocab_sizes: Vocabulary sizes per haploblock (or single int for shared)
        embedding_dim: Hash embedding dimension
        transformer_dim: Transformer hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_classes: Number of output classes
        dropout: Dropout rate
        embedding_mode: 'simple', 'shared', or 'composite'
        use_strand_embedding: For composite mode, include strand embedding
        use_chromosome_embedding: For composite mode, include chromosome embedding
    """

    def __init__(
        self,
        n_haploblocks: int,
        vocab_sizes: List[int],
        embedding_dim: int = 32,
        transformer_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        num_classes: int = 3,
        dropout: float = 0.2,
        embedding_mode: str = 'simple',
        use_strand_embedding: bool = True,
        use_chromosome_embedding: bool = False,
        shared_embedding: bool = False  # Deprecated, use embedding_mode='shared'
    ):
        super().__init__()
        self.n_haploblocks = n_haploblocks
        self.embedding_dim = embedding_dim
        self.embedding_mode = embedding_mode

        # Handle deprecated parameter
        if shared_embedding:
            embedding_mode = 'shared'
            self.embedding_mode = embedding_mode

        # Select embedding type
        if embedding_mode == 'composite':
            self.embedding = CompositeHaploblockEmbedding(
                n_haploblocks=n_haploblocks,
                vocab_sizes=vocab_sizes,
                embedding_dim=embedding_dim,
                use_strand_embedding=use_strand_embedding,
                use_position_embedding=True,
                use_chromosome_embedding=use_chromosome_embedding,
                dropout=dropout
            )
        elif embedding_mode == 'shared':
            max_vocab = max(vocab_sizes) if isinstance(vocab_sizes, list) else vocab_sizes
            self.embedding = SharedHaploblockEmbedding(
                vocab_size=max_vocab,
                embedding_dim=embedding_dim,
                n_haploblocks=n_haploblocks
            )
        else:  # 'simple'
            self.embedding = HaploblockEmbedding(
                n_haploblocks=n_haploblocks,
                vocab_sizes=vocab_sizes,
                embedding_dim=embedding_dim
            )

        # Project embedding to transformer dimension
        self.input_proj = nn.Linear(embedding_dim, transformer_dim)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling for interpretability
        self.attention_pool = nn.Sequential(
            nn.Linear(transformer_dim, 1),
            nn.Softmax(dim=1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self.attention_weights = None

    def forward(
        self,
        hash_ids: torch.Tensor,
        strand_ids: Optional[torch.Tensor] = None,
        chr_ids: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            hash_ids: (batch, n_haploblocks) cluster IDs
            strand_ids: (batch,) or (batch, n_haploblocks) strand IDs (for composite)
            chr_ids: (batch,) chromosome IDs (for composite)
            return_attention: Return attention weights for interpretability

        Returns:
            logits: (batch, num_classes)
            attention_weights (optional): (batch, n_haploblocks)
        """
        # Embed hashes (composite mode uses strand_ids and chr_ids)
        if self.embedding_mode == 'composite':
            x = self.embedding(hash_ids, strand_ids=strand_ids, chr_ids=chr_ids)
        else:
            x = self.embedding(hash_ids)  # (batch, n_haploblocks, embedding_dim)

        # Project to transformer dimension
        x = self.input_proj(x)  # (batch, n_haploblocks, transformer_dim)

        # Transformer encoding
        x = self.transformer(x)  # (batch, n_haploblocks, transformer_dim)

        # Attention pooling
        attn_scores = self.attention_pool(x)  # (batch, n_haploblocks, 1)
        self.attention_weights = attn_scores.squeeze(-1).detach()

        # Weighted sum
        x_pooled = (x * attn_scores).sum(dim=1)  # (batch, transformer_dim)

        # Classification
        logits = self.classifier(x_pooled)

        if return_attention:
            return logits, self.attention_weights
        return logits

    def get_haploblock_importance(self) -> torch.Tensor:
        """Get importance scores for each haploblock."""
        if self.attention_weights is None:
            raise ValueError("Run forward pass first")
        return self.attention_weights


class HaploblockCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer for haploblock classification.

    Combines:
    - Hash embeddings for learning representations
    - CNN for local patterns across nearby haploblocks
    - Transformer for global relationships

    Embedding modes:
    - 'simple': Per-haploblock embedding (default)
    - 'composite': Additive embedding (cluster + position + strand)

    Args:
        n_haploblocks: Number of haploblocks
        vocab_sizes: Vocabulary sizes per haploblock
        embedding_dim: Hash embedding dimension
        cnn_channels: CNN channel dimensions
        transformer_dim: Transformer hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_classes: Number of output classes
        dropout: Dropout rate
        embedding_mode: 'simple' or 'composite'
        use_strand_embedding: For composite mode, include strand embedding
        use_chromosome_embedding: For composite mode, include chromosome embedding
    """

    def __init__(
        self,
        n_haploblocks: int,
        vocab_sizes: List[int],
        embedding_dim: int = 32,
        cnn_channels: List[int] = [64, 128],
        transformer_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        num_classes: int = 3,
        dropout: float = 0.2,
        embedding_mode: str = 'simple',
        use_strand_embedding: bool = True,
        use_chromosome_embedding: bool = False
    ):
        super().__init__()
        self.embedding_mode = embedding_mode

        # Select embedding type
        if embedding_mode == 'composite':
            self.embedding = CompositeHaploblockEmbedding(
                n_haploblocks=n_haploblocks,
                vocab_sizes=vocab_sizes,
                embedding_dim=embedding_dim,
                use_strand_embedding=use_strand_embedding,
                use_position_embedding=True,
                use_chromosome_embedding=use_chromosome_embedding,
                dropout=dropout
            )
        else:  # 'simple'
            self.embedding = HaploblockEmbedding(
                n_haploblocks=n_haploblocks,
                vocab_sizes=vocab_sizes,
                embedding_dim=embedding_dim
            )

        # CNN for local patterns
        self.cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, cnn_channels[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Project to transformer dimension
        self.cnn_to_transformer = nn.Linear(cnn_channels[-1], transformer_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, n_haploblocks, transformer_dim) * 0.02
        )

        # Transformer
        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(transformer_dim, 1),
            nn.Softmax(dim=1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self.attention_weights = None

    def forward(
        self,
        hash_ids: torch.Tensor,
        strand_ids: Optional[torch.Tensor] = None,
        chr_ids: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            hash_ids: (batch, n_haploblocks) cluster IDs
            strand_ids: (batch,) or (batch, n_haploblocks) strand IDs (for composite)
            chr_ids: (batch,) chromosome IDs (for composite)
            return_attention: Return attention weights for interpretability

        Returns:
            logits: (batch, num_classes)
        """
        # Embed (composite mode uses strand_ids and chr_ids)
        if self.embedding_mode == 'composite':
            x = self.embedding(hash_ids, strand_ids=strand_ids, chr_ids=chr_ids)
        else:
            x = self.embedding(hash_ids)  # (batch, n_haploblocks, embedding_dim)

        # CNN (needs channel-first)
        x = x.transpose(1, 2)  # (batch, embedding_dim, n_haploblocks)
        x = self.cnn(x)  # (batch, cnn_channels[-1], n_haploblocks)
        x = x.transpose(1, 2)  # (batch, n_haploblocks, cnn_channels[-1])

        # Project to transformer dim
        x = self.cnn_to_transformer(x)
        x = x + self.positional_encoding

        # Transformer
        x = self.transformer(x)

        # Attention pooling
        attn_scores = self.attention_pool(x)
        self.attention_weights = attn_scores.squeeze(-1).detach()

        x_pooled = (x * attn_scores).sum(dim=1)

        # Classify
        logits = self.classifier(x_pooled)

        if return_attention:
            return logits, self.attention_weights
        return logits


def create_vocab_from_hashes(hash_df) -> Tuple[List[int], Dict[str, Dict]]:
    """
    Create vocabulary mapping from hash DataFrame.

    Args:
        hash_df: DataFrame with hash values (samples x haploblocks)

    Returns:
        vocab_sizes: List of vocabulary sizes per haploblock
        hash_to_id: Dict mapping {haploblock: {hash_value: id}}
    """
    vocab_sizes = []
    hash_to_id = {}

    for col in hash_df.columns:
        unique_hashes = hash_df[col].unique()
        col_vocab = {h: idx + 1 for idx, h in enumerate(unique_hashes)}  # 0 reserved for padding
        col_vocab[None] = 0  # Unknown/missing
        hash_to_id[col] = col_vocab
        vocab_sizes.append(len(col_vocab))

    return vocab_sizes, hash_to_id


def encode_hashes_to_ids(hash_df, hash_to_id: Dict) -> np.ndarray:
    """
    Convert hash values to integer IDs for embedding lookup.

    Args:
        hash_df: DataFrame with hash values
        hash_to_id: Vocabulary mapping from create_vocab_from_hashes

    Returns:
        (n_samples, n_haploblocks) array of integer IDs
    """
    n_samples = len(hash_df)
    n_haploblocks = len(hash_df.columns)
    ids = np.zeros((n_samples, n_haploblocks), dtype=np.int64)

    for col_idx, col in enumerate(hash_df.columns):
        vocab = hash_to_id[col]
        ids[:, col_idx] = hash_df[col].apply(
            lambda x: vocab.get(x, 0)  # 0 for unknown
        ).values

    return ids


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simulate haploblock hash data
    batch_size = 32
    n_haploblocks = 50
    num_classes = 3

    # Simulate different vocab sizes per haploblock
    vocab_sizes = [100 + i * 10 for i in range(n_haploblocks)]

    # Random hash IDs
    hash_ids = torch.stack([
        torch.randint(0, vs, (batch_size,))
        for vs in vocab_sizes
    ], dim=1)  # (batch, n_haploblocks)

    print(f"Input shape: {hash_ids.shape}")
    print(f"Vocab sizes (first 5): {vocab_sizes[:5]}")

    # Test HaploblockTransformer
    print("\n=== HaploblockTransformer ===")
    model = HaploblockTransformer(
        n_haploblocks=n_haploblocks,
        vocab_sizes=vocab_sizes,
        embedding_dim=32,
        transformer_dim=128,
        num_layers=4,
        num_heads=8,
        num_classes=num_classes
    )

    logits, attention = model(hash_ids, return_attention=True)
    print(f"Output shape: {logits.shape}")
    print(f"Attention shape: {attention.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test HaploblockCNNTransformer
    print("\n=== HaploblockCNNTransformer ===")
    model2 = HaploblockCNNTransformer(
        n_haploblocks=n_haploblocks,
        vocab_sizes=vocab_sizes,
        embedding_dim=32,
        num_classes=num_classes
    )

    logits2 = model2(hash_ids)
    print(f"Output shape: {logits2.shape}")
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")

    # Test Composite Embedding (cluster + position + strand)
    print("\n=== CompositeHaploblockEmbedding ===")
    model3 = HaploblockTransformer(
        n_haploblocks=n_haploblocks,
        vocab_sizes=vocab_sizes,
        embedding_dim=64,
        transformer_dim=128,
        num_layers=4,
        num_heads=8,
        num_classes=num_classes,
        embedding_mode='composite',  # Use composite embedding
        use_strand_embedding=True,
        use_chromosome_embedding=True
    )

    # Create strand and chromosome IDs
    strand_ids = torch.randint(0, 2, (batch_size,))  # 0 or 1
    chr_ids = torch.zeros(batch_size, dtype=torch.long)  # chromosome 6

    logits3, attention3 = model3(
        hash_ids,
        strand_ids=strand_ids,
        chr_ids=chr_ids,
        return_attention=True
    )
    print(f"Output shape: {logits3.shape}")
    print(f"Attention shape: {attention3.shape}")
    print(f"Parameters: {sum(p.numel() for p in model3.parameters()):,}")

    # Show embedding breakdown
    print("\n=== Embedding Breakdown ===")
    print("Composite embedding = cluster_emb + position_emb + strand_emb [+ chr_emb]")
    print("All components share the same dimension and are added in the same space.")
