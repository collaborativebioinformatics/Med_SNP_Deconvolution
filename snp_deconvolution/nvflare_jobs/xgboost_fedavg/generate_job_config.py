#!/usr/bin/env python3
"""
生成NVFlare XGBoost Job配置文件

用于快速创建和自定义job配置
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List


def generate_server_config(
    num_rounds: int = 100,
    min_clients: int = 2,
    use_gpu: bool = False,
    xgb_params: Dict = None,
    output_file: str = None
) -> Dict:
    """
    生成Server配置文件

    Args:
        num_rounds: 训练轮数
        min_clients: 最小客户端数量
        use_gpu: 是否使用GPU
        xgb_params: XGBoost参数（可选）
        output_file: 输出文件路径

    Returns:
        配置字典
    """
    # 默认XGBoost参数
    if xgb_params is None:
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eta": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "lambda": 1,
            "alpha": 0,
            "tree_method": "gpu_hist" if use_gpu else "hist",
            "eval_metric": "mlogloss",
            "seed": 42,
            "verbosity": 1
        }

        # GPU特定参数
        if use_gpu:
            xgb_params.update({
                "predictor": "gpu_predictor",
                "gpu_id": 0
            })
        else:
            xgb_params["nthread"] = -1

    config = {
        "format_version": 2,
        "min_clients": min_clients,
        "num_rounds": num_rounds,
        "server": {
            "heart_beat_timeout": 600
        },
        "task_data_filters": [],
        "task_result_filters": [],
        "components": [
            {
                "id": "xgb_controller",
                "path": "nvflare.app_opt.xgboost.histogram_based_v2.fed_controller.XGBFedController",
                "args": {
                    "num_rounds": "{num_rounds}",
                    "data_split_mode": 0,
                    "secure_training": False,
                    "xgb_params": xgb_params,
                    "xgb_options": {
                        "early_stopping_rounds": 10,
                        "verbose_eval": True
                    }
                }
            },
            {
                "id": "persistor",
                "path": "nvflare.app_opt.xgboost.histogram_based_v2.model_persistor.XGBModelPersistor",
                "args": {
                    "save_name": "model.json"
                }
            },
            {
                "id": "metric_logger",
                "path": "nvflare.app_common.widgets.streaming.AnalyticsReceiver",
                "args": {
                    "events": ["fed.analytix_log_stats"]
                }
            }
        ],
        "workflows": [
            {
                "id": "xgb_controller",
                "path": "nvflare.app_common.workflows.controller.ControllerWorkflow",
                "args": {
                    "controller_id": "xgb_controller",
                    "persistor_id": "persistor",
                    "shareable_generator_id": "shareable_generator",
                    "train_task_name": "train",
                    "min_clients": "{min_clients}",
                    "num_rounds": "{num_rounds}"
                }
            }
        ]
    }

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Server config written to: {output_path}")

    return config


def generate_client_config(
    data_dir: str = "{DATASET_PATH}",
    site_name: str = "{SITE_NAME}",
    use_cluster_features: bool = True,
    validation_split: float = 0.2,
    use_gpu: bool = False,
    output_file: str = None
) -> Dict:
    """
    生成Client配置文件

    Args:
        data_dir: 数据目录
        site_name: 站点名称
        use_cluster_features: 是否使用聚类特征
        validation_split: 验证集比例
        use_gpu: 是否使用GPU
        output_file: 输出文件路径

    Returns:
        配置字典
    """
    config = {
        "format_version": 2,
        "executors": [
            {
                "tasks": ["*"],
                "executor": {
                    "path": "nvflare.app_opt.xgboost.histogram_based_v2.fed_executor.FedXGBHistogramExecutor",
                    "args": {
                        "data_loader_id": "snp_data_loader",
                        "early_stopping_rounds": 10,
                        "verbose_eval": True,
                        "use_gpus": use_gpu,
                        "metrics_writer_id": "metrics_writer"
                    }
                }
            }
        ],
        "task_result_filters": [],
        "task_data_filters": [],
        "components": [
            {
                "id": "snp_data_loader",
                "path": "snp_deconvolution.nvflare_real.xgboost.data_loader.SNPXGBDataLoader",
                "args": {
                    "data_dir": data_dir,
                    "site_name": site_name,
                    "use_cluster_features": use_cluster_features,
                    "validation_split": validation_split,
                    "random_seed": 42,
                    "feature_prefix": "snp_",
                    "label_column": "label",
                    "enable_categorical": False
                }
            },
            {
                "id": "metrics_writer",
                "path": "nvflare.app_common.widgets.streaming.AnalyticsSender",
                "args": {
                    "event_type": "fed.analytix_log_stats"
                }
            },
            {
                "id": "event_to_fed",
                "path": "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
                "args": {
                    "events_to_convert": ["analytix_log_stats"],
                    "fed_event_prefix": "fed."
                }
            }
        ]
    }

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Client config written to: {output_path}")

    return config


def generate_meta_json(
    job_name: str = "xgboost_fedavg_snp",
    min_clients: int = 2,
    output_file: str = None
) -> Dict:
    """
    生成meta.json文件

    Args:
        job_name: Job名称
        min_clients: 最小客户端数量
        output_file: 输出文件路径

    Returns:
        配置字典
    """
    config = {
        "name": job_name,
        "description": "Federated XGBoost training for SNP-based population deconvolution using histogram-based aggregation",
        "version": "1.0.0",
        "resource_spec": {},
        "deploy_map": {
            "app": ["@ALL"]
        },
        "min_clients": min_clients,
        "mandatory_clients": []
    }

    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Meta config written to: {output_path}")

    return config


def generate_complete_job(
    output_dir: str,
    job_name: str = "xgboost_fedavg_snp",
    num_rounds: int = 100,
    min_clients: int = 2,
    use_gpu: bool = False,
    **kwargs
):
    """
    生成完整的Job配置目录

    Args:
        output_dir: 输出目录
        job_name: Job名称
        num_rounds: 训练轮数
        min_clients: 最小客户端数量
        use_gpu: 是否使用GPU
        **kwargs: 其他参数
    """
    output_path = Path(output_dir)
    app_config_dir = output_path / "app" / "config"
    app_config_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating complete job configuration in: {output_path}")
    print("=" * 80)

    # 生成server配置
    server_config_file = app_config_dir / "config_fed_server.json"
    generate_server_config(
        num_rounds=num_rounds,
        min_clients=min_clients,
        use_gpu=use_gpu,
        output_file=str(server_config_file)
    )

    # 生成client配置
    client_config_file = app_config_dir / "config_fed_client.json"
    generate_client_config(
        use_gpu=use_gpu,
        output_file=str(client_config_file)
    )

    # 生成meta.json
    meta_file = output_path / "meta.json"
    generate_meta_json(
        job_name=job_name,
        min_clients=min_clients,
        output_file=str(meta_file)
    )

    print("=" * 80)
    print(f"Job configuration generated successfully!")
    print(f"\nJob structure:")
    print(f"{output_path}/")
    print(f"├── app/")
    print(f"│   └── config/")
    print(f"│       ├── config_fed_server.json")
    print(f"│       └── config_fed_client.json")
    print(f"└── meta.json")


def main():
    parser = argparse.ArgumentParser(
        description="Generate NVFlare XGBoost job configuration"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./custom_xgboost_job",
        help="Output directory for job configuration"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="xgboost_fedavg_snp",
        help="Job name"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=100,
        help="Number of training rounds"
    )
    parser.add_argument(
        "--min_clients",
        type=int,
        default=2,
        help="Minimum number of clients"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for training"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="XGBoost learning rate"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=6,
        help="XGBoost max tree depth"
    )

    args = parser.parse_args()

    generate_complete_job(
        output_dir=args.output_dir,
        job_name=args.job_name,
        num_rounds=args.num_rounds,
        min_clients=args.min_clients,
        use_gpu=args.gpu
    )


if __name__ == "__main__":
    main()
