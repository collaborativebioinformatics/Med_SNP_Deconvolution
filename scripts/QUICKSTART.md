# 快速开始: 联邦学习数据准备

## 5分钟快速上手

### 1. 验证环境

确保已安装依赖:

```bash
# 检查Python版本 (需要 3.8+)
python --version

# 安装依赖
pip install numpy torch scikit-learn
```

### 2. 运行测试

验证脚本正常工作:

```bash
python scripts/test_prepare_federated_data.py
```

如果看到 "所有测试通过!"，说明环境配置正确。

### 3. 准备数据

#### 选项A: 使用cluster特征 (推荐)

```bash
python scripts/prepare_federated_data.py \
    --pipeline_output data/out_dir/TNFa \
    --num_sites 3 \
    --output_dir data/federated/TNFa
```

#### 选项B: 使用SNP特征 (基线对比)

```bash
python scripts/prepare_federated_data.py \
    --pipeline_output data/out_dir/TNFa \
    --vcf_path data/chr6.vcf.gz \
    --mode snp \
    --num_sites 3 \
    --output_dir data/federated/TNFa_snp
```

#### 选项C: 同时准备两种特征

```bash
python scripts/prepare_federated_data.py \
    --pipeline_output data/out_dir/TNFa \
    --vcf_path data/chr6.vcf.gz \
    --mode both \
    --num_sites 3 \
    --output_dir data/federated/TNFa_both
```

### 4. 查看结果

```bash
# 查看元信息
cat data/federated/TNFa/dataset_metadata.json

# 查看文件结构
tree data/federated/TNFa -L 2

# 或使用 ls
ls -R data/federated/TNFa/
```

### 5. 在代码中使用

#### PyTorch/Lightning

```python
import torch

# 加载站点1的训练数据
data = torch.load('data/federated/TNFa/site-1/train_cluster.pt')
X_train = data['X']  # shape: (n_samples, n_features)
y_train = data['y']  # shape: (n_samples,)

print(f"训练集大小: {X_train.shape}")
print(f"标签分布: {torch.unique(y_train, return_counts=True)}")
```

#### XGBoost

```python
import numpy as np

# 加载站点1的训练数据
data = np.load('data/federated/TNFa/site-1/train_cluster.npz')
X_train = data['X']
y_train = data['y']

print(f"训练集大小: {X_train.shape}")
print(f"标签分布: {np.unique(y_train, return_counts=True)}")
```

## 常用命令

### 查看帮助

```bash
python scripts/prepare_federated_data.py --help
```

### 验证数据划分

```bash
python scripts/prepare_federated_data.py \
    --pipeline_output data/out_dir/TNFa \
    --num_sites 3 \
    --output_dir data/federated/TNFa \
    --verify
```

### 使用详细日志

```bash
python scripts/prepare_federated_data.py \
    --pipeline_output data/out_dir/TNFa \
    --num_sites 3 \
    --output_dir data/federated/TNFa \
    --verbose
```

### 调整配置

```bash
python scripts/prepare_federated_data.py \
    --pipeline_output data/out_dir/TNFa \
    --num_sites 5 \                    # 5个站点
    --val_ratio 0.2 \                 # 20%验证集
    --seed 123 \                      # 自定义随机种子
    --output_dir data/federated/TNFa
```

## 输出文件说明

每个站点包含4个文件:

- `train_cluster.pt`: PyTorch训练集
- `train_cluster.npz`: NumPy训练集
- `val_cluster.pt`: PyTorch验证集
- `val_cluster.npz`: NumPy验证集

根目录还包含:
- `dataset_metadata.json`: 数据集配置和统计信息

## 故障排查

### 问题1: Pipeline输出目录不存在

```bash
# 检查目录
ls -la data/out_dir/TNFa/

# 如果不存在，需要先运行Haploblock Pipeline
```

### 问题2: 缺少cluster文件

```bash
# 检查clusters目录
ls data/out_dir/TNFa/clusters/

# 如果为空，需要运行Pipeline的聚类步骤
```

### 问题3: 人群标签文件不存在

```bash
# 检查人群文件
ls data/igsr-*.tsv.tsv

# 如果不存在，需要下载或指定正确路径
```

### 问题4: 内存不足

```bash
# 减少站点数量
python scripts/prepare_federated_data.py \
    --num_sites 2 \
    ...

# 或者分批处理不同区域
```

## 下一步

数据准备完成后:

1. **配置NVFlare**: 在配置文件中指定各站点数据路径
2. **训练模型**: 使用Lightning或XGBoost训练器
3. **评估结果**: 比较cluster和SNP特征的性能

详细文档请参考:
- `README_prepare_federated_data.md`: 完整文档
- `example_prepare_federated_data.sh`: 示例脚本

## 需要帮助?

- 查看完整文档: `README_prepare_federated_data.md`
- 运行测试: `test_prepare_federated_data.py`
- 查看示例: `example_prepare_federated_data.sh`

---

最后更新: 2026-01-07
