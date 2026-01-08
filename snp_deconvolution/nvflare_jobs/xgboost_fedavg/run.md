
# Create My job foulder and files as below:
  my_job/
  ├── meta.json
  └── app/
      └── config/
          ├── config_fed_server.json
          └── config_fed_client.json

  1. meta.json

  {
      "name": "xgboost_fedavg_snp",
      "resource_spec": {},
      "deploy_map": {
          "app": ["@ALL"]
      },
      "min_clients": 3
  }

  2. config_fed_server.json

```json
  {
      "format_version": 2,
      "num_rounds": 10,
      "workflows": [
          {
              "id": "xgb_controller",
              "path": "nvflare.app_opt.xgboost.histogram_based_v2.fed_controller.XGBFedController",
              "args": {
                  "num_rounds": 10,
                  "data_split_mode": 0,
                  "secure_training": false,
                  "xgb_params": {
                      "max_depth": 6,
                      "eta": 0.1,
                      "objective": "multi:softprob",
                      "num_class": 3,
                      "eval_metric": "mlogloss",
                      "tree_method": "hist",
                      "nthread": 4
                  },
                  "in_process": true
              }
          }
      ]
  }
```

  3. config_fed_client.json

```json
  {
      "format_version": 2,
      "executors": [
          {
              "tasks": ["*"],
              "executor": {
                  "path": "nvflare.app_opt.xgboost.histogram_based_v2.fed_executor.FedXGBHistogramExecutor",
                  "args": {
                      "data_loader_id": "dataloader",
                      "in_process": true
                  }
              }
          }
      ],
      "components": [
          {
              "id": "dataloader",
              "path": "snp_deconvolution.nvflare_real.xgboost.data_loader.SNPXGBDataLoader",
              "args": {
                  "data_dir": "/home/shadeform/Med_SNP_Deconvolution/data/federated",
                  "site_name": "{SITE_NAME}",
                  "use_cluster_features": false,
                  "validation_split": 0.2
              }
          }
      ]
  }

```

cd ~/Med_SNP_Deconvolution
PYTHONPATH=$PWD:$PYTHONPATH nvflare simulator -w workspace -n 3 -t 3 snp_deconvolution/nvflare_jobs/xgboost_fedavg/my_job
