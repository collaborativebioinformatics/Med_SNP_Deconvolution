# Data Integration Module

Memory-efficient data loading and feature engineering for SNP deconvolution machine learning pipelines.

## Overview

This module provides tools for:
- Loading haploblock pipeline outputs (variant counts, hashes, boundaries)
- Loading population labels from 1000 Genomes Project files
- Converting VCF genotypes to sparse matrices for memory efficiency
- Feature engineering from haploblock data
- Integration with ML frameworks (XGBoost, PyTorch)

## Components

### 1. HaploblockMLDataLoader

Load and integrate data from haploblock pipeline outputs.

```python
from snp_deconvolution.data_integration import HaploblockMLDataLoader

# Initialize with pipeline output directory
loader = HaploblockMLDataLoader("/path/to/pipeline/output")

# Load haploblock features
features_df = loader.load_haploblock_features()
# Returns: DataFrame (samples x features) with variant counts and hashes

# Load population labels
population_files = [
    "/path/to/data/igsr-chb.tsv.tsv",  # Chinese Han Beijing (CHB)
    "/path/to/data/igsr-gbr.tsv.tsv",  # British (GBR)
    "/path/to/data/igsr-pur.tsv.tsv",  # Puerto Rican (PUR)
]
labels = loader.load_population_labels(population_files)
# Returns: Dict[sample_name -> population_id]

# Load cluster assignments
clusters = loader.load_cluster_assignments("/path/to/clusters/")
# Returns: Dict[file_name -> Dict[sample_name -> cluster_id]]

# Create complete ML dataset
dataset = loader.create_ml_dataset(
    population_files=population_files,
    clusters_dir="/path/to/clusters/"
)
# Returns: Dict with 'features', 'labels', 'sample_ids', 'metadata'
```

### 2. SparseGenotypeMatrix

Memory-efficient sparse matrix representation for genotypes.

```python
from snp_deconvolution.data_integration import SparseGenotypeMatrix

# Load VCF as sparse matrix
sparse_matrix, sample_ids, snp_info = SparseGenotypeMatrix.from_vcf(
    vcf_path="/path/to/genotypes.vcf.gz",
    region="chr6:1000000-5000000",
    maf_threshold=0.01,
    max_missing=0.1
)
# Returns: (csr_matrix, sample_list, variant_metadata)

# Convert numpy array to sparse
import numpy as np
genotypes = np.random.randint(0, 3, size=(100, 10000))
sparse_matrix = SparseGenotypeMatrix.from_numpy(genotypes)

# Convert to XGBoost DMatrix
dmatrix = SparseGenotypeMatrix.to_xgboost_dmatrix(
    sparse_matrix,
    labels=labels
)

# Convert to PyTorch tensor
tensor = SparseGenotypeMatrix.to_pytorch_tensor(
    sparse_matrix,
    device="cuda",
    keep_sparse=False  # Use dense tensor on GPU
)

# Save/load sparse matrix
SparseGenotypeMatrix.save_npz(sparse_matrix, "genotypes.npz")
loaded_matrix = SparseGenotypeMatrix.load_npz("genotypes.npz")
```

### 3. Feature Engineering

Utilities for creating and transforming haploblock features.

```python
from snp_deconvolution.data_integration import (
    encode_hashes_as_features,
    create_haploblock_features,
)

# Encode haploblock hashes as binary features
hash_features = encode_hashes_as_features(
    "haploblock_hashes.tsv",
    binary_encoding=True  # Convert to binary vector
)
# Returns: ndarray (samples x n_haploblocks*hash_bits)

# Create haploblock-level features
haploblock_features = create_haploblock_features(
    boundaries_file="haploblock_boundaries.tsv",
    variant_counts_file="variant_counts.tsv",
    include_length=True,
    include_density=True,
    include_statistics=True
)
# Returns: DataFrame with aggregate haploblock features
```

## Example Workflow

### Complete ML Pipeline

```python
from pathlib import Path
from snp_deconvolution.data_integration import (
    HaploblockMLDataLoader,
    SparseGenotypeMatrix,
    create_haploblock_features,
)

# 1. Load haploblock features
loader = HaploblockMLDataLoader("/data/pipeline_output")

population_files = [
    Path("/data/igsr-chb.tsv.tsv"),
    Path("/data/igsr-gbr.tsv.tsv"),
    Path("/data/igsr-pur.tsv.tsv"),
]

dataset = loader.create_ml_dataset(
    population_files=population_files
)

# 2. Load genotypes as sparse matrix
sparse_matrix, sample_ids, snp_info = SparseGenotypeMatrix.from_vcf(
    vcf_path="/data/chr6.vcf.gz",
    region="chr6:25000000-35000000",
    maf_threshold=0.05,
)

# 3. Align features and genotypes
common_samples = set(dataset['sample_ids']) & set(sample_ids)
print(f"Common samples: {len(common_samples)}")

# 4. Convert to ML framework
import xgboost as xgb

# Extract labels for common samples
sample_to_idx = {s: i for i, s in enumerate(sample_ids)}
common_indices = [sample_to_idx[s] for s in common_samples if s in sample_to_idx]

X = sparse_matrix[common_indices, :]
y = [dataset['label_map'][s] for s in common_samples]

# Create train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train XGBoost model
dtrain = SparseGenotypeMatrix.to_xgboost_dmatrix(X_train, y_train)
dtest = SparseGenotypeMatrix.to_xgboost_dmatrix(X_test, y_test)

params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 6,
    'eta': 0.3,
}

model = xgb.train(params, dtrain, num_boost_round=100)

# Evaluate
predictions = model.predict(dtest)
accuracy = (predictions == y_test).mean()
print(f"Accuracy: {accuracy:.3f}")
```

## Data Formats

### Haploblock Pipeline Outputs

**variant_counts.tsv**
```
sample_id    haploblock_0    haploblock_1    haploblock_2    ...
NA18531      42              38              51              ...
NA18536      39              40              49              ...
```

**haploblock_hashes.tsv**
```
sample_id    haploblock_0    haploblock_1    haploblock_2    ...
NA18531      0xA1B2C3D4      0xE5F6A7B8      0xC9D0E1F2      ...
NA18536      0xA1B2C3D4      0xE5F6A7B8      0xC9D0E1F2      ...
```

**haploblock_boundaries.tsv**
```
START    END
1        711055
711055   736514
736514   761058
```

### Population Labels

**igsr-chb.tsv.tsv** (1000 Genomes format)
```
Sample name    Sex    Biosample ID    Population code    Population name    ...
NA18531        female    SAME124483    CHB               Han Chinese        ...
NA18536        male      SAME124486    CHB               Han Chinese        ...
```

### Cluster Assignments

**clusters/chromosome_6_region_0.tsv** (no header)
```
representative_sample_1    individual_1
representative_sample_1    individual_2
representative_sample_2    individual_3
```

## Dependencies

- numpy >= 1.20
- scipy >= 1.7
- pandas >= 1.3
- bcftools (for VCF processing)

Optional:
- xgboost >= 1.5 (for XGBoost integration)
- torch >= 1.10 (for PyTorch integration)

## Performance Considerations

### Memory Efficiency

The sparse matrix representation provides significant memory savings:

```python
# Dense matrix: 10,000 samples x 1,000,000 SNPs = 10 GB (float32)
# Sparse matrix: ~1-2 GB (depending on sparsity)

sparse_matrix, _, _ = SparseGenotypeMatrix.from_vcf(vcf_path)
print(f"Sparsity: {100 * (1 - sparse_matrix.nnz / sparse_matrix.size):.2f}%")
```

### VCF Loading

For large VCF files:
- Use region filtering to load specific genomic regions
- Adjust MAF threshold to reduce number of variants
- Use sample filtering to load specific individuals

```python
# Load only high-frequency variants in specific region
sparse_matrix, _, _ = SparseGenotypeMatrix.from_vcf(
    vcf_path="large_file.vcf.gz",
    region="chr6:25000000-35000000",  # 10 Mb region
    maf_threshold=0.05,  # Only common variants
    samples=["NA18531", "NA18536"],  # Specific samples
)
```

## Error Handling

All functions include comprehensive error handling:

```python
from snp_deconvolution.data_integration import HaploblockMLDataLoader

try:
    loader = HaploblockMLDataLoader("/nonexistent/path")
except ValueError as e:
    print(f"Error: {e}")
    # Error: Pipeline output directory does not exist: /nonexistent/path

try:
    dataset = loader.create_ml_dataset(
        population_files=["/nonexistent/file.tsv"]
    )
except FileNotFoundError as e:
    print(f"Error: {e}")
    # Error: Population file not found: /nonexistent/file.tsv
```

## Logging

All modules use Python's logging framework:

```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run your analysis
loader = HaploblockMLDataLoader("/data/pipeline_output")
# INFO - __main__ - Initialized HaploblockMLDataLoader with dir: /data/pipeline_output
```
