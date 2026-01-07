# Deep Learning Model Architecture

## Overview

```mermaid
graph TD
    subgraph Input ["Input Layer"]
        RAW["Pipeline Output<br/>(clusters/*.tsv)"]
        RAW --> CID["Cluster IDs<br/>(n_samples × n_haploblocks)"]
        RAW --> STR["Strand IDs<br/>(0 or 1)"]
        RAW --> CHR["Chromosome ID"]
    end

    subgraph Embedding ["Embedding Layer"]
        CID --> EMB_SELECT{{"Embedding Mode"}}

        EMB_SELECT --> |simple| SIMPLE["Per-Haploblock<br/>Embedding"]
        EMB_SELECT --> |shared| SHARED["Shared<br/>Embedding"]
        EMB_SELECT --> |composite| COMP["Composite<br/>Embedding"]

        STR --> COMP
        CHR -.-> COMP
    end

    subgraph Model ["Model Architecture"]
        SIMPLE --> ARCH{{"Architecture"}}
        SHARED --> ARCH
        COMP --> ARCH

        ARCH --> |transformer| TRANS["HaploblockTransformer"]
        ARCH --> |cnn_transformer| CNN_TRANS["HaploblockCNNTransformer"]
    end

    subgraph Output ["Output Layer"]
        TRANS --> ATTN["Attention Pooling"]
        CNN_TRANS --> ATTN
        ATTN --> CLASS["Classification<br/>(num_classes)"]
    end

    style COMP fill:#ff9800,stroke:#ef6c00,stroke-width:2px
    style EMB_SELECT fill:#e3f2fd,stroke:#1565c0
    style ARCH fill:#e3f2fd,stroke:#1565c0
```

## Embedding Modes

### 1. Simple Embedding (Per-Haploblock)

```mermaid
graph LR
    subgraph Input
        CID["Cluster ID<br/>(batch, n_hb)"]
    end

    subgraph Embedding ["HaploblockEmbedding"]
        CID --> HB1["Emb[0]<br/>vocab=v₀"]
        CID --> HB2["Emb[1]<br/>vocab=v₁"]
        CID --> HBN["Emb[n]<br/>vocab=vₙ"]
    end

    subgraph Output
        HB1 --> OUT["(batch, n_hb, dim)"]
        HB2 --> OUT
        HBN --> OUT
    end

    style Embedding fill:#e8f5e9,stroke:#2e7d32
```

- Each haploblock has independent embedding table
- Vocab size varies per haploblock (cluster count)
- Parameters: `Σ(vocab_sizes[i] × embedding_dim)`

### 2. Shared Embedding

```mermaid
graph LR
    subgraph Input
        CID["Cluster ID<br/>(batch, n_hb)"]
    end

    subgraph Embedding ["SharedHaploblockEmbedding"]
        CID --> SHARED["Shared Embedding<br/>vocab=max(v)"]
        SHARED --> POS["+ Positional<br/>Encoding"]
    end

    subgraph Output
        POS --> OUT["(batch, n_hb, dim)"]
    end

    style Embedding fill:#fff3e0,stroke:#ef6c00
```

- Single embedding table for all haploblocks
- Positional encoding distinguishes positions
- More parameter-efficient

### 3. Composite Embedding (Recommended)

```mermaid
graph TD
    subgraph Input ["Input Components"]
        CID["Cluster IDs<br/>(batch, n_hb)"]
        STR["Strand IDs<br/>(batch,)"]
        POS["Position<br/>(0..n_hb-1)"]
        CHR["Chromosome<br/>(batch,)"]
    end

    subgraph Embeddings ["Separate Embedding Tables"]
        CID --> CE["Cluster Embedding<br/>(per-haploblock)"]
        POS --> PE["Position Embedding<br/>(n_hb × dim)"]
        STR --> SE["Strand Embedding<br/>(2 × dim)"]
        CHR --> CHE["Chromosome Embedding<br/>(23 × dim)"]
    end

    subgraph Combine ["Additive Combination"]
        CE --> SUM(("+"))
        PE --> SUM
        SE --> SUM
        CHE -.->|optional| SUM
        SUM --> LN["LayerNorm"]
        LN --> DROP["Dropout"]
    end

    subgraph Output
        DROP --> OUT["(batch, n_hb, dim)"]
    end

    style SUM fill:#ff9800,stroke:#ef6c00,stroke-width:3px
    style Combine fill:#e3f2fd,stroke:#1565c0
```

**Key Insight**: Similar to BERT's `token_emb + position_emb + segment_emb`

```
final_embedding = cluster_emb + position_emb + strand_emb [+ chr_emb]
```

## Model Architectures

### HaploblockTransformer

```mermaid
graph TD
    subgraph Input
        X["Cluster IDs<br/>(batch, n_hb)"]
    end

    subgraph Embedding
        X --> EMB["Embedding Layer<br/>(simple/shared/composite)"]
        EMB --> PROJ["Linear Projection<br/>emb_dim → transformer_dim"]
    end

    subgraph Transformer ["Transformer Encoder"]
        PROJ --> TE1["TransformerEncoderLayer 1"]
        TE1 --> TE2["TransformerEncoderLayer 2"]
        TE2 --> TEN["TransformerEncoderLayer N"]
    end

    subgraph Pooling ["Attention Pooling"]
        TEN --> ATT_W["Linear → Softmax<br/>(n_hb weights)"]
        TEN --> FEAT["Features"]
        ATT_W --> |"weighted sum"| POOL["Pooled Vector<br/>(batch, transformer_dim)"]
        FEAT --> POOL
    end

    subgraph Classification
        POOL --> FC1["Linear(128) + GELU"]
        FC1 --> DROP["Dropout"]
        DROP --> FC2["Linear(num_classes)"]
        FC2 --> OUT["Logits<br/>(batch, num_classes)"]
    end

    style Transformer fill:#e3f2fd,stroke:#1565c0
    style Pooling fill:#fff3e0,stroke:#ef6c00
```

### HaploblockCNNTransformer

```mermaid
graph TD
    subgraph Input
        X["Cluster IDs<br/>(batch, n_hb)"]
    end

    subgraph Embedding
        X --> EMB["Embedding Layer"]
    end

    subgraph CNN ["CNN for Local Patterns"]
        EMB --> |"transpose"| CONV1["Conv1d(64, k=5)<br/>+ BatchNorm + GELU"]
        CONV1 --> CONV2["Conv1d(128, k=5)<br/>+ BatchNorm + GELU"]
        CONV2 --> |"transpose"| CNN_OUT["CNN Features"]
    end

    subgraph Transformer ["Transformer for Global"]
        CNN_OUT --> PROJ["Linear Projection"]
        PROJ --> POS["+ Positional Encoding"]
        POS --> TE["Transformer Encoder<br/>(N layers)"]
    end

    subgraph Pooling
        TE --> ATT["Attention Pooling"]
    end

    subgraph Classification
        ATT --> CLASS["Classifier"]
        CLASS --> OUT["Logits"]
    end

    style CNN fill:#e8f5e9,stroke:#2e7d32
    style Transformer fill:#e3f2fd,stroke:#1565c0
```

## Parameter Comparison

| Model | Embedding | Parameters | Notes |
|-------|-----------|------------|-------|
| HaploblockTransformer | simple | ~1.37M | Per-haploblock embedding |
| HaploblockTransformer | composite | ~1.93M | + strand/position embedding |
| HaploblockCNNTransformer | simple | ~1.85M | + CNN layers |
| HaploblockCNNTransformer | composite | ~2.00M | Full model |

## Usage Examples

### Simple Mode

```python
model = HaploblockTransformer(
    n_haploblocks=2288,
    vocab_sizes=vocab_sizes,
    embedding_dim=32,
    transformer_dim=128,
    num_classes=3,
    embedding_mode='simple'
)

logits = model(cluster_ids)
```

### Composite Mode (Recommended)

```python
model = HaploblockTransformer(
    n_haploblocks=2288,
    vocab_sizes=vocab_sizes,
    embedding_dim=64,
    transformer_dim=128,
    num_classes=3,
    embedding_mode='composite',
    use_strand_embedding=True,
    use_chromosome_embedding=False
)

logits, attention = model(
    cluster_ids,
    strand_ids=strand_ids,
    return_attention=True
)

# Interpretability: attention weights show haploblock importance
important_haploblocks = attention.mean(dim=0).argsort(descending=True)[:10]
```

## Data Flow Summary

```mermaid
graph LR
    subgraph Pipeline ["Haploblock Pipeline"]
        VCF["VCF"] --> HB["Haploblocks"]
        HB --> SEQ["Sequences"]
        SEQ --> CLU["MMSeqs2"]
        CLU --> HASH["Hashes"]
    end

    subgraph Extract ["Feature Extraction"]
        HASH --> |"ClusterFeatureLoader"| CID["Cluster IDs"]
        HASH --> |"parse"| STR["Strand IDs"]
    end

    subgraph DL ["Deep Learning"]
        CID --> EMB["Embedding"]
        STR --> EMB
        EMB --> MODEL["CNN/Transformer"]
        MODEL --> POP["Population<br/>Classification"]
    end

    style Extract fill:#fff3e0,stroke:#ef6c00
    style DL fill:#e3f2fd,stroke:#1565c0
```
