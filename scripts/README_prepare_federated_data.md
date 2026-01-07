# 联邦学习数据准备脚本

## 概述

`prepare_federated_data.py` 是一个用于将Haploblock Pipeline输出的数据转换为NVFlare联邦学习格式的脚本。

## 功能特性

1. **多种特征模式支持**
   - Cluster模式: 使用聚类ID特征（隐私保护）
   - SNP模式: 使用原始SNP特征（基线对比）
   - Both模式: 同时准备两种特征

2. **智能数据划分**
   - 使用分层抽样保证各站点标签分布一致
   - 自动处理样本数量不均衡的情况
   - 为每个站点创建独立的训练/验证集

3. **多格式输出**
   - PyTorch格式 (.pt): 用于Lightning训练
   - NumPy格式 (.npz): 用于XGBoost训练
   - JSON元信息: 记录数据集配置和统计信息

4. **完整的验证机制**
   - 文件存在性检查
   - 数据一致性验证
   - 标签分布统计

## 依赖要求

确保已安装以下依赖:

```bash
pip install numpy torch scikit-learn
```

同时需要项目的以下模块:
- `snp_deconvolution.data_integration.unified_loader`
- `snp_deconvolution.nvflare_real.data.data_splitter`

## 使用方法

### 基础用法

使用默认配置准备cluster特征:

```bash
python scripts/prepare_federated_data.py \
    --pipeline_output data/out_dir/TNFa \
    --num_sites 3 \
    --output_dir data/federated/TNFa
```

### 高级用法

#### 1. 同时准备cluster和SNP特征

```bash
python scripts/prepare_federated_data.py \
    --pipeline_output data/out_dir/TNFa \
    --vcf_path data/chr6.vcf.gz \
    --mode both \
    --num_sites 3 \
    --output_dir data/federated/TNFa \
    --verify
```

#### 2. 自定义人群标签文件

```bash
python scripts/prepare_federated_data.py \
    --pipeline_output data/out_dir/TNFa \
    --population_files data/igsr-chb.tsv.tsv data/igsr-gbr.tsv.tsv \
    --num_sites 2 \
    --output_dir data/federated/TNFa_2pop
```

#### 3. 调整验证集比例和随机种子

```bash
python scripts/prepare_federated_data.py \
    --pipeline_output data/out_dir/TNFa \
    --num_sites 5 \
    --val_ratio 0.2 \
    --seed 123 \
    --output_dir data/federated/TNFa_5sites
```

#### 4. 使用详细日志

```bash
python scripts/prepare_federated_data.py \
    --pipeline_output data/out_dir/TNFa \
    --num_sites 3 \
    --output_dir data/federated/TNFa \
    --verbose \
    --verify
```

## 参数说明

### 输入数据参数

- `--pipeline_output`: Haploblock pipeline输出目录
  - 默认: `out_dir/TNFa`
  - 必须包含 `clusters/` 目录（cluster模式）

- `--population_files`: 人群标签文件列表
  - 默认: `data/igsr-chb.tsv.tsv data/igsr-gbr.tsv.tsv data/igsr-pur.tsv.tsv`
  - 格式: TSV文件，需包含 "Sample name" 列

- `--vcf_path`: VCF文件路径
  - SNP模式时必需
  - 支持压缩格式 (.vcf.gz)

### 特征模式

- `--mode`: 特征模式
  - `cluster`: 仅准备cluster特征（默认，推荐）
  - `snp`: 仅准备SNP特征
  - `both`: 同时准备两种特征

### 联邦学习配置

- `--num_sites`: 联邦学习站点数量
  - 默认: 3
  - 建议范围: 2-10

- `--output_dir`: 输出目录
  - 默认: `data/federated`
  - 将创建 `site-1/`, `site-2/`, ... 子目录

### 数据划分参数

- `--val_ratio`: 验证集比例
  - 默认: 0.15 (15%)
  - 范围: (0, 1)

- `--seed`: 随机种子
  - 默认: 42
  - 用于保证可重复性

### 其他选项

- `--verify`: 验证数据划分的正确性
  - 检查文件完整性和数据一致性

- `--verbose`: 显示详细日志
  - 包括DEBUG级别信息

## 输出文件结构

运行脚本后，将生成以下文件结构:

```
data/federated/TNFa/
├── dataset_metadata.json          # 数据集元信息
├── site-1/
│   ├── train_cluster.pt           # PyTorch训练集
│   ├── train_cluster.npz          # NumPy训练集
│   ├── val_cluster.pt             # PyTorch验证集
│   └── val_cluster.npz            # NumPy验证集
├── site-2/
│   ├── train_cluster.pt
│   ├── train_cluster.npz
│   ├── val_cluster.pt
│   └── val_cluster.npz
└── site-3/
    ├── train_cluster.pt
    ├── train_cluster.npz
    ├── val_cluster.pt
    └── val_cluster.npz
```

如果使用 `--mode both`，将生成两个子目录:

```
data/federated/TNFa/
├── cluster/
│   ├── dataset_metadata.json
│   ├── site-1/
│   ├── site-2/
│   └── site-3/
└── snp/
    ├── dataset_metadata.json
    ├── site-1/
    ├── site-2/
    └── site-3/
```

## 数据格式说明

### PyTorch格式 (.pt)

使用 `torch.save()` 保存的字典:

```python
{
    'X': torch.Tensor,  # shape: (n_samples, n_features)
    'y': torch.Tensor   # shape: (n_samples,)
}
```

加载方式:

```python
import torch

data = torch.load('site-1/train_cluster.pt')
X_train = data['X']
y_train = data['y']
```

### NumPy格式 (.npz)

使用 `np.savez()` 保存:

```python
{
    'X': np.ndarray,  # shape: (n_samples, n_features)
    'y': np.ndarray   # shape: (n_samples,)
}
```

加载方式:

```python
import numpy as np

data = np.load('site-1/train_cluster.npz')
X_train = data['X']
y_train = data['y']
```

### 元信息文件 (dataset_metadata.json)

JSON格式，包含:

```json
{
  "dataset": {
    "mode": "cluster",
    "n_samples": 300,
    "n_classes": 3,
    "n_features": 1234,
    "class_distribution": {
      "0": 100,
      "1": 100,
      "2": 100
    },
    "vocab_sizes": [10, 15, 8, ...],
    "n_haploblocks": 1234
  },
  "split_config": {
    "num_sites": 3,
    "val_ratio": 0.15,
    "seed": 42
  },
  "source": {
    "pipeline_output": "data/out_dir/TNFa",
    "population_files": [...],
    "vcf_path": null
  }
}
```

## 常见问题

### 1. Pipeline输出目录不存在

**错误信息**: `Pipeline输出目录不存在`

**解决方案**: 确保已运行Haploblock Pipeline并生成输出:

```bash
# 检查目录是否存在
ls -la data/out_dir/TNFa/

# 如果不存在，需要先运行pipeline
```

### 2. 缺少cluster文件

**错误信息**: `No cluster files found in clusters/`

**解决方案**:
1. 确保运行了Pipeline的聚类步骤 (step 4)
2. 检查clusters目录:
   ```bash
   ls data/out_dir/TNFa/clusters/
   ```

### 3. 人群标签文件格式错误

**错误信息**: 标签加载失败

**解决方案**:
- 确保TSV文件包含 "Sample name" 列
- 检查文件编码（应为UTF-8）
- 验证制表符分隔

### 4. 样本数量太少无法分层

**警告信息**: `无法使用分层抽样 (某些类别样本太少)`

**说明**: 这是正常的，脚本会自动降级为随机抽样

**改进方案**:
- 增加样本数量
- 减少站点数量
- 调整验证集比例

### 5. VCF文件路径错误

**错误信息**: `VCF文件不存在`

**解决方案**:
- 使用绝对路径或相对于项目根目录的路径
- 确保文件存在且可读

## 性能优化建议

### 1. 内存优化

对于大型数据集，考虑:
- 分批处理
- 使用稀疏矩阵格式
- 减少站点数量

### 2. 速度优化

- 使用SSD存储
- 关闭验证选项 (去掉 `--verify`)
- 使用更少的站点

### 3. 存储优化

- 仅生成需要的特征模式
- 压缩输出文件 (后处理)

## 集成到联邦学习流程

### 1. 准备数据

```bash
python scripts/prepare_federated_data.py \
    --pipeline_output data/out_dir/TNFa \
    --num_sites 3 \
    --output_dir data/federated/TNFa \
    --verify
```

### 2. 配置NVFlare

在NVFlare配置中指定数据路径:

```json
{
  "site-1": {
    "data_path": "data/federated/TNFa/site-1"
  },
  "site-2": {
    "data_path": "data/federated/TNFa/site-2"
  },
  "site-3": {
    "data_path": "data/federated/TNFa/site-3"
  }
}
```

### 3. 训练模型

使用Lightning或XGBoost训练器加载数据:

```python
# Lightning
from snp_deconvolution.nvflare_real.lightning import FederatedDataModule

data_module = FederatedDataModule(
    data_dir="data/federated/TNFa/site-1",
    feature_type="cluster"
)

# XGBoost
import numpy as np

data = np.load("data/federated/TNFa/site-1/train_cluster.npz")
X_train, y_train = data['X'], data['y']
```

## 开发说明

### 代码结构

脚本主要包含以下功能模块:

1. **参数验证** (`validate_arguments`): 检查输入参数合法性
2. **数据加载** (`load_and_prepare_data`): 使用UnifiedFeatureLoader加载数据
3. **元信息保存** (`save_metadata`): 保存数据集配置
4. **统计汇总** (`print_summary`): 打印详细统计信息
5. **主流程** (`prepare_federated_data`): 协调各模块执行

### 扩展建议

1. **支持更多特征格式**:
   - 添加对其他特征类型的支持
   - 实现自定义特征提取器

2. **高级划分策略**:
   - 非均匀划分（模拟数据异构性）
   - 基于地理位置的划分
   - 标签分布偏斜

3. **数据增强**:
   - 添加噪声注入
   - 样本重采样
   - 特征选择

## 贡献指南

欢迎提交Issue和Pull Request！

提交前请确保:
1. 代码符合PEP 8规范
2. 添加了相应的单元测试
3. 更新了文档

## 许可证

与项目主仓库保持一致。

## 联系方式

如有问题，请在项目仓库提交Issue。
