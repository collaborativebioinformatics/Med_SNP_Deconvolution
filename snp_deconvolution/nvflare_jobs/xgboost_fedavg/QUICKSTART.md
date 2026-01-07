# Quick Start Guide - XGBoost Federated Learning

快速开始使用NVFlare进行XGBoost联邦学习的SNP分类任务。

## 前置要求

### 1. 安装依赖

```bash
# 安装NVFlare
pip install nvflare>=2.5.1

# 安装XGBoost
pip install xgboost>=2.0.0

# 安装scikit-learn（用于数据划分）
pip install scikit-learn

# GPU支持（可选）
pip install xgboost[gpu]
```

### 2. 准备数据

确保每个站点的数据文件存在：

```bash
data/
├── site1_train.csv              # 站点1训练数据
├── site1_cluster_features.csv   # 站点1聚类特征（可选）
├── site2_train.csv              # 站点2训练数据
└── site2_cluster_features.csv   # 站点2聚类特征（可选）
```

**数据格式要求**：

训练数据CSV格式：
```csv
snp_0,snp_1,snp_2,...,snp_n,label
0,1,2,...,1,0
1,0,1,...,2,1
```

- SNP特征列以 `snp_` 开头
- 标签列名为 `label`
- 标签值为 0, 1, 2（三分类）

## 快速测试（3步）

### Step 1: 验证数据文件

```bash
python -m snp_deconvolution.nvflare_real.xgboost.test_data_loader \
  --data_dir /path/to/data \
  --site_names site1 site2 site3 \
  --validate_only
```

预期输出：
```
✓ Training data found: /path/to/data/site1_train.csv
✓ Training data found: /path/to/data/site2_train.csv
✓ Training data found: /path/to/data/site3_train.csv
```

### Step 2: 测试单个站点数据加载

```bash
python -m snp_deconvolution.nvflare_real.xgboost.test_data_loader \
  --data_dir /path/to/data \
  --site_name site1
```

预期输出：
```
✓ Data loader initialized successfully
✓ Training DMatrix created: 800 samples, 150 features
✓ Validation DMatrix created: 200 samples, 150 features
✓ Basic training completed successfully
All tests passed successfully!
```

### Step 3: 运行联邦训练（模拟器）

```bash
cd /Users/saltfish/Files/Coding/Med_SNP_Deconvolution

# CPU版本（3个客户端）
nvflare simulator snp_deconvolution/nvflare_jobs/xgboost_fedavg \
  -w /tmp/nvflare/xgb_workspace \
  -n 3 \
  -t 3

# GPU版本（需要CUDA支持）
nvflare simulator snp_deconvolution/nvflare_jobs/xgboost_fedavg \
  -w /tmp/nvflare/xgb_workspace \
  -n 3 \
  -t 3 \
  -gpu 0,1,2
```

## 自定义配置

### 方法1: 使用配置生成器

```bash
cd snp_deconvolution/nvflare_jobs/xgboost_fedavg

# 生成自定义配置
python generate_job_config.py \
  --output_dir ./custom_job \
  --job_name my_xgboost_job \
  --num_rounds 200 \
  --min_clients 3 \
  --gpu
```

### 方法2: 手动编辑配置文件

编辑 `app/config/config_fed_server.json`:

```json
{
  "xgb_params": {
    "eta": 0.05,           // 调整学习率
    "max_depth": 8,        // 调整树深度
    "num_rounds": 200      // 调整训练轮数
  }
}
```

编辑 `app/config/config_fed_client.json`:

```json
{
  "args": {
    "data_dir": "/your/data/path",           // 修改数据路径
    "site_name": "your_site_name",           // 修改站点名
    "use_cluster_features": false,           // 禁用聚类特征
    "validation_split": 0.3                  // 调整验证集比例
  }
}
```

## 模拟器运行参数

```bash
nvflare simulator [job_path] [options]

选项：
  -w, --workspace PATH    工作空间路径
  -n, --clients NUM       客户端数量
  -t, --threads NUM       线程数量
  -gpu GPU_IDS           GPU ID列表（如 0,1,2）
```

示例：

```bash
# 基本运行（2客户端，CPU）
nvflare simulator xgboost_fedavg -w /tmp/workspace -n 2 -t 2

# 多客户端（5客户端，CPU）
nvflare simulator xgboost_fedavg -w /tmp/workspace -n 5 -t 5

# GPU加速（3客户端，3个GPU）
nvflare simulator xgboost_fedavg -w /tmp/workspace -n 3 -t 3 -gpu 0,1,2
```

## 查看结果

### 训练日志

```bash
# 查看服务器日志
cat /tmp/workspace/xgboost_fedavg/simulate_job/app_server/log.txt

# 查看客户端日志
cat /tmp/workspace/xgboost_fedavg/simulate_job/app_site-1/log.txt
```

### 训练指标

```bash
# 训练损失
cat /tmp/workspace/xgboost_fedavg/simulate_job/app_server/cross_site_val/metrics.json

# 或使用Python
python -c "
import json
with open('/tmp/workspace/xgboost_fedavg/simulate_job/app_server/cross_site_val/metrics.json') as f:
    metrics = json.load(f)
    print(json.dumps(metrics, indent=2))
"
```

### 加载训练好的模型

```python
import xgboost as xgb
import numpy as np

# 加载模型
model = xgb.Booster()
model.load_model("/tmp/workspace/xgboost_fedavg/simulate_job/app_server/model.json")

# 预测
test_data = xgb.DMatrix(X_test)
predictions = model.predict(test_data)

# 获取类别
predicted_classes = np.argmax(predictions, axis=1)
```

## 常见问题

### Q1: 找不到数据文件

```
FileNotFoundError: Could not find data file for site site1
```

**解决方案**：
1. 检查文件命名是否正确：`{site_name}_train.csv`
2. 检查数据路径是否正确
3. 使用 `--validate_only` 验证文件存在性

### Q2: 内存不足

```
MemoryError: Unable to allocate array
```

**解决方案**：
- 减少 `validation_split` (如 0.1)
- 减少客户端数量
- 使用特征选择减少特征数

### Q3: GPU错误

```
XGBoostError: gpu_id is set but XGBoost is not compiled with CUDA
```

**解决方案**：
```bash
# 重新安装GPU版本
pip uninstall xgboost
pip install xgboost[gpu]

# 或使用CPU配置
使用 config_fed_server.json 而不是 config_fed_server_gpu.json
```

### Q4: 模型不收敛

```
WARNING: Model not converging after 100 rounds
```

**解决方案**：
- 增加 `num_rounds` (如 200, 300)
- 降低 `eta` 学习率 (如 0.05, 0.01)
- 检查数据质量和标签分布

## 生产环境部署

参考 [NVFlare Production Deployment](https://nvflare.readthedocs.io/en/latest/getting_started.html) 文档。

基本步骤：
1. 设置NVFlare服务器和客户端
2. 配置认证和加密
3. 提交Job到Admin Console
4. 监控训练进度
5. 下载结果模型

## 下一步

- 阅读完整文档：[README.md](./README.md)
- 了解参数调优：[XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- 探索隐私计算：启用 `secure_training: true`
- 生产部署：参考NVFlare官方文档

## 技术支持

- NVFlare文档：https://nvflare.readthedocs.io/
- XGBoost文档：https://xgboost.readthedocs.io/
- GitHub Issues：提交问题到项目仓库
