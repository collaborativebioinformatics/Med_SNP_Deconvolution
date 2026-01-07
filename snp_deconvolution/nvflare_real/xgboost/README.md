# XGBoost Data Loader for NVFlare

NVFlare联邦学习框架的XGBoost数据加载器实现，专为SNP基因型数据设计。

## 概述

本模块实现了符合NVFlare要求的XGBoost数据加载器，支持：
- SNP特征加载
- 可选的聚类特征集成
- 自动训练/验证集划分
- 标签分层采样
- DMatrix创建和验证

## 文件结构

```
xgboost/
├── __init__.py              # 模块初始化
├── data_loader.py           # 核心数据加载器（NVFlare required）
├── test_data_loader.py      # 测试和验证工具
├── example_usage.py         # 使用示例
└── README.md                # 本文件
```

## 核心组件

### SNPXGBDataLoader

NVFlare要求的数据加载器类，必须实现`load_data()`方法。

**关键特性**：
- 符合NVFlare FedXGBHistogramExecutor要求
- 自动加载站点特定数据
- 支持多种文件命名格式
- 内置数据验证和错误处理
- 返回XGBoost DMatrix格式

## 使用方法

### 1. 基本用法

```python
from snp_deconvolution.nvflare_real.xgboost import SNPXGBDataLoader

# 初始化
data_loader = SNPXGBDataLoader(
    data_dir="/path/to/data",
    site_name="site1",
    use_cluster_features=True,
    validation_split=0.2
)

# 加载数据（NVFlare会自动调用）
train_dmatrix, val_dmatrix = data_loader.load_data()
```

### 2. 在NVFlare中使用

数据加载器会被NVFlare的`FedXGBHistogramExecutor`自动调用。

配置示例（`config_fed_client.json`）：

```json
{
  "components": [
    {
      "id": "snp_data_loader",
      "path": "snp_deconvolution.nvflare_real.xgboost.data_loader.SNPXGBDataLoader",
      "args": {
        "data_dir": "/path/to/data",
        "site_name": "site1",
        "use_cluster_features": true,
        "validation_split": 0.2
      }
    }
  ]
}
```

### 3. 测试数据加载

```bash
# 验证数据文件存在
python -m snp_deconvolution.nvflare_real.xgboost.test_data_loader \
  --data_dir /path/to/data \
  --site_names site1 site2 site3 \
  --validate_only

# 测试单个站点加载
python -m snp_deconvolution.nvflare_real.xgboost.test_data_loader \
  --data_dir /path/to/data \
  --site_name site1
```

## 数据格式要求

### 训练数据文件

文件名格式（按优先级尝试）：
1. `{site_name}_train.csv`
2. `{site_name}.csv`
3. `site_{site_name}.csv`

CSV格式：
```csv
snp_0,snp_1,snp_2,...,snp_n,label
0,1,2,...,1,0
1,0,1,...,2,1
2,0,2,...,0,2
```

要求：
- SNP特征列以`snp_`开头（可配置）
- 标签列名为`label`（可配置）
- 标签值：0, 1, 2（三分类）

### 聚类特征文件（可选）

文件名：`{site_name}_cluster_features.csv`

CSV格式：
```csv
cluster_0,cluster_1,cluster_2,...
0.5,0.3,0.2
0.7,0.1,0.2
```

要求：
- 行数必须与训练数据一致
- 列名以`cluster_`开头

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_dir` | str | - | 数据目录路径（必需） |
| `site_name` | str | - | 站点名称（必需） |
| `use_cluster_features` | bool | True | 是否使用聚类特征 |
| `validation_split` | float | 0.2 | 验证集比例（0-1） |
| `random_seed` | int | 42 | 随机种子 |
| `feature_prefix` | str | "snp_" | 特征列前缀 |
| `label_column` | str | "label" | 标签列名 |
| `enable_categorical` | bool | False | 启用分类特征支持 |

## API参考

### `load_data()`

NVFlare要求的核心方法。

**返回值**：
- 单个DMatrix：`xgb.DMatrix`（无验证集）
- 元组：`(train_dmatrix, val_dmatrix)`（有验证集）

**异常**：
- `FileNotFoundError`：找不到数据文件
- `ValueError`：数据格式错误或标签缺失

### `get_data_info()`

获取数据集统计信息。

**返回值**：
```python
{
    "site_name": "site1",
    "num_samples": 1000,
    "num_features": 150,
    "num_classes": 3,
    "label_distribution": [300, 400, 300],
    "use_cluster_features": True,
    "validation_split": 0.2
}
```

## 测试工具

### test_data_loader.py

全面的测试和验证工具。

**功能**：
1. 验证数据文件存在性
2. 测试数据加载
3. 验证DMatrix创建
4. 测试基本XGBoost训练
5. 检查标签分布

**命令行选项**：
```bash
# 完整测试
python -m snp_deconvolution.nvflare_real.xgboost.test_data_loader \
  --data_dir /path/to/data \
  --site_name site1 \
  --validation_split 0.2

# 只验证文件
python -m snp_deconvolution.nvflare_real.xgboost.test_data_loader \
  --data_dir /path/to/data \
  --site_names site1 site2 site3 \
  --validate_only

# 不使用聚类特征
python -m snp_deconvolution.nvflare_real.xgboost.test_data_loader \
  --data_dir /path/to/data \
  --site_name site1 \
  --no_cluster_features
```

## 示例代码

参见 `example_usage.py` 了解10个完整示例：

1. 基本用法
2. 带验证集训练
3. 不使用聚类特征
4. 自定义验证集比例
5. 不使用验证集
6. 获取数据信息
7. 手动K折交叉验证
8. GPU训练
9. 特征重要性分析
10. 模拟多站点训练

## 数据流程

```
1. 初始化数据加载器
   ├── 设置数据路径
   └── 配置参数

2. load_data() 被调用
   ├── 加载训练数据 CSV
   ├── 加载聚类特征 CSV（可选）
   ├── 提取SNP特征
   ├── 提取标签
   ├── 合并特征
   ├── 训练/验证划分
   └── 创建DMatrix

3. 返回给NVFlare
   ├── 训练DMatrix
   └── 验证DMatrix（可选）
```

## 性能优化

### 内存优化
- 使用numpy数组而非DataFrame
- 及时释放中间变量
- 支持稀疏矩阵（未来）

### 速度优化
- 使用pandas高效读取CSV
- 向量化操作
- 避免不必要的数据复制

## NVFlare集成

### 客户端配置

```json
{
  "executors": [
    {
      "tasks": ["*"],
      "executor": {
        "path": "nvflare.app_opt.xgboost.histogram_based_v2.fed_executor.FedXGBHistogramExecutor",
        "args": {
          "data_loader_id": "snp_data_loader"
        }
      }
    }
  ],
  "components": [
    {
      "id": "snp_data_loader",
      "path": "snp_deconvolution.nvflare_real.xgboost.data_loader.SNPXGBDataLoader",
      "args": {
        "data_dir": "{DATASET_PATH}",
        "site_name": "{SITE_NAME}"
      }
    }
  ]
}
```

### 执行流程

1. NVFlare启动客户端
2. 初始化`FedXGBHistogramExecutor`
3. Executor调用`SNPXGBDataLoader.load_data()`
4. 返回DMatrix用于训练
5. 开始联邦学习

## 故障排查

### 常见问题

**Q: FileNotFoundError**
```python
# 检查文件存在性
import os
assert os.path.exists("/path/to/data/site1_train.csv")
```

**Q: 内存错误**
```python
# 减少验证集比例
validation_split=0.1  # 从0.2减少到0.1
```

**Q: 标签缺失**
```python
# 检查CSV文件
import pandas as pd
df = pd.read_csv("data.csv")
assert "label" in df.columns
```

**Q: 聚类特征长度不匹配**
```python
# 确保行数一致
train_df = pd.read_csv("site1_train.csv")
cluster_df = pd.read_csv("site1_cluster_features.csv")
assert len(train_df) == len(cluster_df)
```

## 依赖项

```txt
numpy>=1.20.0
pandas>=1.3.0
xgboost>=2.0.0
scikit-learn>=1.0.0
nvflare>=2.5.1
```

## 版本历史

- **v1.0.0**: 初始版本
  - 基本数据加载功能
  - 支持SNP和聚类特征
  - 自动训练/验证划分
  - 完整测试工具

## 相关文档

- [NVFlare XGBoost文档](https://nvflare.readthedocs.io/en/2.5.1/user_guide/federated_xgboost/)
- [XGBoost DMatrix API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix)
- [Job配置文档](../../nvflare_jobs/xgboost_fedavg/README.md)
- [快速开始指南](../../nvflare_jobs/xgboost_fedavg/QUICKSTART.md)

## 许可证

参见项目根目录LICENSE文件。
