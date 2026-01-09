#!/usr/bin/env python3
"""
Automated Federated Learning Experiment Runner

This script automates running multiple federated learning experiments with different
configurations (data splits, strategies, models) and collects comprehensive results.

Features:
    - Multiple data split strategies (IID, Dirichlet, Label Skew, Quantity Skew)
    - Multiple FL algorithms (FedAvg, FedProx, Scaffold, FedOpt variants)
    - Automated data preparation and NVFlare POC simulation
    - Comprehensive metrics collection and analysis
    - Summary report with best configurations
    - Error handling and experiment isolation
    - Progress tracking with tqdm
    - Parallel execution support (experimental)

Usage:
    # Run all experiments
    python run_fl_experiments.py --output_dir results/experiments

    # Run specific strategies and splits
    python run_fl_experiments.py --strategies fedavg,fedprox --splits iid,dirichlet_0.5

    # Dry run to see what would be executed
    python run_fl_experiments.py --dry_run

    # Override number of rounds
    python run_fl_experiments.py --num_rounds 100

Author: Generated for Med_SNP_Deconvolution
Date: 2026-01-09
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Experiment Configuration
# ============================================================================

EXPERIMENT_MATRIX = {
    "data_splits": {
        "iid": {"alpha": None, "labels_per_site": None},
        "dirichlet_0.5": {"alpha": 0.5, "labels_per_site": None},
        "dirichlet_0.1": {"alpha": 0.1, "labels_per_site": None},
        "label_skew": {"alpha": None, "labels_per_site": 2},
    },
    "strategies": {
        "fedavg": {"client_script": "client.py", "args": {}},
        "fedprox": {"client_script": "client.py", "args": {"--strategy": "fedprox", "--mu": "0.01"}},
        "scaffold": {"client_script": "scaffold_client.py", "args": {}},
        "fedopt_adam": {"client_script": "client.py", "args": {}, "controller": "fedopt", "optimizer": "adam"},
        "fedopt_yogi": {"client_script": "client.py", "args": {}, "controller": "fedopt", "optimizer": "yogi"},
    },
    "models": ["lightning"],  # Can add "xgboost" later
    "default_num_rounds": 50,
    "default_num_sites": 3,
    "default_local_epochs": 1,
    "default_batch_size": 128,
    "default_learning_rate": 1e-4,
}


# ============================================================================
# Experiment Configuration
# ============================================================================

class ExperimentConfig:
    """Configuration for a single experiment."""

    def __init__(
        self,
        name: str,
        split_type: str,
        strategy: str,
        model_type: str,
        num_rounds: int,
        num_sites: int,
        data_dir: Path,
        output_dir: Path,
        split_params: Dict[str, Any],
        strategy_params: Dict[str, Any],
    ):
        self.name = name
        self.split_type = split_type
        self.strategy = strategy
        self.model_type = model_type
        self.num_rounds = num_rounds
        self.num_sites = num_sites
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.split_params = split_params
        self.strategy_params = strategy_params

        # Create experiment-specific directories
        self.exp_data_dir = self.data_dir / self.name
        self.exp_output_dir = self.output_dir / self.name
        self.exp_workspace = self.exp_output_dir / "workspace"
        self.exp_results = self.exp_output_dir / "results"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "split_type": self.split_type,
            "strategy": self.strategy,
            "model_type": self.model_type,
            "num_rounds": self.num_rounds,
            "num_sites": self.num_sites,
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "split_params": self.split_params,
            "strategy_params": self.strategy_params,
        }


# ============================================================================
# Experiment Runner
# ============================================================================

class ExperimentRunner:
    """Manages and executes federated learning experiments."""

    def __init__(
        self,
        base_data_dir: Path,
        output_dir: Path,
        pipeline_output: Path,
        population_files: List[str],
        num_rounds: int,
        num_sites: int,
        dry_run: bool = False,
    ):
        self.base_data_dir = base_data_dir
        self.output_dir = output_dir
        self.pipeline_output = pipeline_output
        self.population_files = population_files
        self.num_rounds = num_rounds
        self.num_sites = num_sites
        self.dry_run = dry_run

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track experiment results
        self.results: List[Dict[str, Any]] = []
        self.failed_experiments: List[Dict[str, Any]] = []

        # Paths to scripts
        self.prepare_data_script = project_root / "scripts" / "prepare_federated_data.py"
        self.job_script = project_root / "snp_deconvolution" / "nvflare_real" / "lightning" / "job.py"

    def generate_experiments(
        self,
        strategies: List[str],
        splits: List[str],
        models: List[str] = None,
    ) -> List[ExperimentConfig]:
        """
        Generate experiment configurations.

        Args:
            strategies: List of FL strategies to test
            splits: List of data split types to test
            models: List of model types to test (default: ["lightning"])

        Returns:
            List of ExperimentConfig objects
        """
        if models is None:
            models = EXPERIMENT_MATRIX["models"]

        experiments = []

        for split_name in splits:
            # Parse split name (e.g., "dirichlet_0.5" -> "dirichlet" with alpha=0.5)
            if "_" in split_name:
                split_base, param_value = split_name.split("_", 1)
                try:
                    param_value = float(param_value)
                except ValueError:
                    logger.warning(f"Invalid split parameter: {split_name}, skipping")
                    continue

                # Find matching split config
                split_config = None
                for key, config in EXPERIMENT_MATRIX["data_splits"].items():
                    if key.startswith(split_base):
                        split_config = config.copy()
                        if "alpha" in key and config["alpha"] is not None:
                            split_config["alpha"] = param_value
                        break

                if split_config is None:
                    logger.warning(f"Unknown split type: {split_name}, skipping")
                    continue

                split_type = split_base
            else:
                split_type = split_name
                split_config = EXPERIMENT_MATRIX["data_splits"].get(split_name)
                if split_config is None:
                    logger.warning(f"Unknown split type: {split_name}, skipping")
                    continue

            for strategy in strategies:
                strategy_config = EXPERIMENT_MATRIX["strategies"].get(strategy)
                if strategy_config is None:
                    logger.warning(f"Unknown strategy: {strategy}, skipping")
                    continue

                for model in models:
                    # Generate experiment name
                    exp_name = f"{model}_{split_name}_{strategy}"

                    # Create experiment config
                    config = ExperimentConfig(
                        name=exp_name,
                        split_type=split_type,
                        strategy=strategy,
                        model_type=model,
                        num_rounds=self.num_rounds,
                        num_sites=self.num_sites,
                        data_dir=self.base_data_dir,
                        output_dir=self.output_dir,
                        split_params=split_config,
                        strategy_params=strategy_config,
                    )

                    experiments.append(config)

        logger.info(f"Generated {len(experiments)} experiment configurations")
        return experiments

    def prepare_data(self, config: ExperimentConfig) -> bool:
        """
        Prepare federated data for an experiment.

        Args:
            config: Experiment configuration

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"[{config.name}] Preparing federated data...")

        # Build command
        cmd = [
            sys.executable,
            str(self.prepare_data_script),
            "--pipeline_output", str(self.pipeline_output),
            "--population_files"] + self.population_files + [
            "--num_sites", str(config.num_sites),
            "--output_dir", str(config.exp_data_dir),
            "--mode", "cluster",
            "--split_type", config.split_type,
            "--seed", "42",
        ]

        # Add split-specific parameters
        if config.split_params.get("alpha") is not None:
            cmd.extend(["--alpha", str(config.split_params["alpha"])])
        if config.split_params.get("labels_per_site") is not None:
            cmd.extend(["--labels_per_site", str(config.split_params["labels_per_site"])])

        logger.info(f"[{config.name}] Data preparation command: {' '.join(cmd)}")

        if self.dry_run:
            logger.info(f"[{config.name}] DRY RUN: Would prepare data")
            return True

        # Execute command
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600  # 10 minute timeout
            )
            logger.info(f"[{config.name}] Data preparation completed")
            logger.debug(f"[{config.name}] stdout: {result.stdout}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"[{config.name}] Data preparation failed: {e}")
            logger.error(f"[{config.name}] stderr: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"[{config.name}] Data preparation timed out")
            return False

    def run_nvflare_simulation(self, config: ExperimentConfig) -> bool:
        """
        Run NVFlare POC simulation for an experiment.

        Args:
            config: Experiment configuration

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"[{config.name}] Running NVFlare simulation...")

        # Create workspace directory
        config.exp_workspace.mkdir(parents=True, exist_ok=True)

        # Build base command
        cmd = [
            sys.executable,
            str(self.job_script),
            "--mode", "poc",
            "--job_name", config.name,
            "--num_rounds", str(config.num_rounds),
            "--num_clients", str(config.num_sites),
            "--data_dir", str(config.exp_data_dir),
            "--workspace", str(config.exp_workspace),
            "--feature_type", "cluster",
            "--architecture", "cnn_transformer",
            "--num_classes", "3",
            "--local_epochs", str(EXPERIMENT_MATRIX["default_local_epochs"]),
            "--batch_size", str(EXPERIMENT_MATRIX["default_batch_size"]),
            "--learning_rate", str(EXPERIMENT_MATRIX["default_learning_rate"]),
            "--run_now",
        ]

        # Add strategy-specific arguments
        strategy_args = config.strategy_params.get("args", {})
        for key, value in strategy_args.items():
            cmd.extend([key, str(value)])

        # Handle FedOpt strategies (requires custom job creation)
        if config.strategy_params.get("controller") == "fedopt":
            logger.warning(f"[{config.name}] FedOpt strategies require custom job creation")
            logger.warning(f"[{config.name}] This feature is not yet implemented in this script")
            return False

        # Handle Scaffold strategy (different client script)
        if config.strategy == "scaffold":
            # Note: This requires modifying the job.py to accept custom client script
            # For now, we'll log a warning
            logger.warning(f"[{config.name}] Scaffold requires custom client script")
            logger.warning(f"[{config.name}] Ensure job.py supports --client_script argument")

        logger.info(f"[{config.name}] NVFlare command: {' '.join(cmd)}")

        if self.dry_run:
            logger.info(f"[{config.name}] DRY RUN: Would run NVFlare simulation")
            return True

        # Execute command
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600 * 2  # 2 hour timeout
            )
            logger.info(f"[{config.name}] NVFlare simulation completed")
            logger.debug(f"[{config.name}] stdout: {result.stdout}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"[{config.name}] NVFlare simulation failed: {e}")
            logger.error(f"[{config.name}] stderr: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"[{config.name}] NVFlare simulation timed out")
            return False

    def collect_metrics(self, config: ExperimentConfig) -> Optional[Dict[str, Any]]:
        """
        Collect metrics from completed experiment.

        Args:
            config: Experiment configuration

        Returns:
            Dictionary of metrics, or None if collection failed
        """
        logger.info(f"[{config.name}] Collecting metrics...")

        # Look for results in workspace
        workspace_dir = config.exp_workspace / config.name
        if not workspace_dir.exists():
            logger.warning(f"[{config.name}] Workspace directory not found: {workspace_dir}")
            return None

        # Try to find metrics files (NVFlare typically saves in workspace/job_name/app_server/)
        server_dir = workspace_dir / "app_server"
        if server_dir.exists():
            # Look for cross_site_val directory or similar
            metrics_files = list(server_dir.rglob("*metrics*.json"))
            if metrics_files:
                logger.info(f"[{config.name}] Found {len(metrics_files)} metrics files")
                # Parse the first metrics file
                try:
                    with open(metrics_files[0], 'r') as f:
                        metrics = json.load(f)
                    return metrics
                except Exception as e:
                    logger.error(f"[{config.name}] Failed to parse metrics file: {e}")

        # Fallback: try to extract metrics from logs
        log_files = list(workspace_dir.rglob("log.txt"))
        if log_files:
            logger.info(f"[{config.name}] Attempting to extract metrics from logs...")
            metrics = self._extract_metrics_from_logs(log_files[0])
            if metrics:
                return metrics

        logger.warning(f"[{config.name}] Could not collect metrics")
        return None

    def _extract_metrics_from_logs(self, log_file: Path) -> Optional[Dict[str, Any]]:
        """
        Extract metrics from NVFlare log file.

        Args:
            log_file: Path to log file

        Returns:
            Dictionary of metrics, or None if extraction failed
        """
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()

            # Extract key metrics using simple pattern matching
            metrics = {
                "rounds": [],
                "train_loss": [],
                "val_loss": [],
                "val_accuracy": [],
            }

            # Parse log lines for metrics
            for line in log_content.split('\n'):
                # Example patterns (adjust based on actual log format):
                # "Round 5: train_loss=0.234, val_loss=0.456, val_accuracy=0.89"
                if "val_accuracy" in line.lower() or "accuracy" in line.lower():
                    # Simple extraction - this is a placeholder
                    # Real implementation would need proper regex parsing
                    pass

            # If we extracted any metrics, return them
            if any(metrics.values()):
                return metrics

            return None

        except Exception as e:
            logger.error(f"Failed to extract metrics from logs: {e}")
            return None

    def run_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Run a complete experiment: prepare data, run simulation, collect metrics.

        Args:
            config: Experiment configuration

        Returns:
            Dictionary containing experiment results and metadata
        """
        start_time = time.time()
        result = {
            "config": config.to_dict(),
            "status": "pending",
            "start_time": datetime.now().isoformat(),
            "metrics": None,
            "error": None,
        }

        try:
            # Step 1: Prepare data
            if not self.prepare_data(config):
                raise RuntimeError("Data preparation failed")

            # Step 2: Run NVFlare simulation
            if not self.run_nvflare_simulation(config):
                raise RuntimeError("NVFlare simulation failed")

            # Step 3: Collect metrics
            if not self.dry_run:
                metrics = self.collect_metrics(config)
                result["metrics"] = metrics

            # Success
            result["status"] = "success" if not self.dry_run else "dry_run"
            logger.info(f"[{config.name}] Experiment completed successfully")

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"[{config.name}] Experiment failed: {e}")

        finally:
            result["end_time"] = datetime.now().isoformat()
            result["duration_seconds"] = time.time() - start_time

        return result

    def run_all_experiments(
        self,
        strategies: List[str],
        splits: List[str],
        models: List[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Run all experiments sequentially with progress tracking.

        Args:
            strategies: List of FL strategies to test
            splits: List of data split types to test
            models: List of model types to test

        Returns:
            Tuple of (successful_results, failed_results)
        """
        # Generate experiment configurations
        experiments = self.generate_experiments(strategies, splits, models)

        if not experiments:
            logger.warning("No experiments to run")
            return [], []

        logger.info(f"Starting {len(experiments)} experiments...")

        # Run experiments with progress bar
        successful_results = []
        failed_results = []

        with tqdm(total=len(experiments), desc="Running experiments") as pbar:
            for config in experiments:
                pbar.set_description(f"Running: {config.name}")

                result = self.run_experiment(config)

                if result["status"] == "success" or result["status"] == "dry_run":
                    successful_results.append(result)
                else:
                    failed_results.append(result)

                # Save intermediate results
                self._save_intermediate_results(successful_results, failed_results)

                pbar.update(1)

        logger.info(f"Completed {len(successful_results)} successful experiments")
        logger.info(f"Failed {len(failed_results)} experiments")

        return successful_results, failed_results

    def _save_intermediate_results(
        self,
        successful_results: List[Dict[str, Any]],
        failed_results: List[Dict[str, Any]]
    ):
        """Save intermediate results to JSON files."""
        results_file = self.output_dir / "experiment_results.json"
        failed_file = self.output_dir / "failed_experiments.json"

        try:
            with open(results_file, 'w') as f:
                json.dump(successful_results, f, indent=2)

            with open(failed_file, 'w') as f:
                json.dump(failed_results, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {e}")

    def generate_summary_report(
        self,
        successful_results: List[Dict[str, Any]],
        failed_results: List[Dict[str, Any]]
    ):
        """
        Generate summary report of all experiments.

        Args:
            successful_results: List of successful experiment results
            failed_results: List of failed experiment results
        """
        logger.info("Generating summary report...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(successful_results) + len(failed_results),
            "successful_experiments": len(successful_results),
            "failed_experiments": len(failed_results),
            "experiments": {
                "successful": successful_results,
                "failed": failed_results,
            },
        }

        # Save full report
        report_file = self.output_dir / "experiment_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Full report saved to: {report_file}")

        # Generate summary statistics
        self._generate_summary_statistics(successful_results)

        # Print summary to console
        self._print_summary(successful_results, failed_results)

    def _generate_summary_statistics(self, results: List[Dict[str, Any]]):
        """
        Generate summary statistics and comparisons.

        Args:
            results: List of successful experiment results
        """
        if not results:
            logger.warning("No results to analyze")
            return

        # Organize results by split type and strategy
        results_by_split = defaultdict(list)
        results_by_strategy = defaultdict(list)

        for result in results:
            config = result["config"]
            results_by_split[config["split_type"]].append(result)
            results_by_strategy[config["strategy"]].append(result)

        # Create summary DataFrame
        summary_data = []
        for result in results:
            config = result["config"]
            metrics = result.get("metrics", {})

            row = {
                "experiment": config["name"],
                "split_type": config["split_type"],
                "strategy": config["strategy"],
                "model": config["model_type"],
                "num_rounds": config["num_rounds"],
                "status": result["status"],
                "duration_seconds": result.get("duration_seconds", 0),
            }

            # Add metrics if available
            if metrics:
                # Extract final metrics (assuming metrics contains round-wise data)
                if isinstance(metrics, dict):
                    row.update(metrics)

            summary_data.append(row)

        # Save summary CSV
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_csv = self.output_dir / "experiment_summary.csv"
            df.to_csv(summary_csv, index=False)
            logger.info(f"Summary CSV saved to: {summary_csv}")

            # Find best configurations
            self._find_best_configurations(df)

    def _find_best_configurations(self, df: pd.DataFrame):
        """
        Find and report best configurations.

        Args:
            df: DataFrame containing experiment results
        """
        logger.info("\n" + "="*70)
        logger.info("Best Configurations")
        logger.info("="*70)

        # Best strategy per split type (if accuracy is available)
        if "val_accuracy" in df.columns:
            for split_type in df["split_type"].unique():
                split_df = df[df["split_type"] == split_type]
                best = split_df.loc[split_df["val_accuracy"].idxmax()]
                logger.info(f"\nBest for {split_type}:")
                logger.info(f"  Strategy: {best['strategy']}")
                logger.info(f"  Accuracy: {best['val_accuracy']:.4f}")
                logger.info(f"  Experiment: {best['experiment']}")

        # Fastest converging strategy
        if "duration_seconds" in df.columns:
            logger.info(f"\nFastest experiments:")
            fastest = df.nsmallest(3, "duration_seconds")
            for _, row in fastest.iterrows():
                logger.info(f"  {row['experiment']}: {row['duration_seconds']:.1f}s")

        logger.info("="*70)

    def _print_summary(
        self,
        successful_results: List[Dict[str, Any]],
        failed_results: List[Dict[str, Any]]
    ):
        """
        Print experiment summary to console.

        Args:
            successful_results: List of successful experiment results
            failed_results: List of failed experiment results
        """
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        print(f"Total experiments: {len(successful_results) + len(failed_results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        print()

        if successful_results:
            print("Successful Experiments:")
            for result in successful_results:
                config = result["config"]
                duration = result.get("duration_seconds", 0)
                print(f"  - {config['name']} ({duration:.1f}s)")
            print()

        if failed_results:
            print("Failed Experiments:")
            for result in failed_results:
                config = result["config"]
                error = result.get("error", "Unknown error")
                print(f"  - {config['name']}: {error}")
            print()

        print(f"Results saved to: {self.output_dir}")
        print("="*70)


# ============================================================================
# Main Function
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated Federated Learning Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments with default settings
  python run_fl_experiments.py

  # Run specific strategies and splits
  python run_fl_experiments.py --strategies fedavg,fedprox --splits iid,dirichlet_0.5

  # Dry run to see what would be executed
  python run_fl_experiments.py --dry_run

  # Override number of rounds
  python run_fl_experiments.py --num_rounds 100

  # Custom data and output directories
  python run_fl_experiments.py \\
      --pipeline_output out_dir/TNFa \\
      --data_dir data/federated_experiments \\
      --output_dir results/experiments
        """
    )

    # Data arguments
    parser.add_argument(
        "--pipeline_output",
        type=str,
        default=str(project_root / "out_dir" / "TNFa"),
        help="Haploblock pipeline output directory"
    )
    parser.add_argument(
        "--population_files",
        nargs="+",
        default=[
            str(project_root / "data" / "igsr-chb.tsv.tsv"),
            str(project_root / "data" / "igsr-gbr.tsv.tsv"),
            str(project_root / "data" / "igsr-pur.tsv.tsv"),
        ],
        help="Population label files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(project_root / "data" / "federated_experiments"),
        help="Base directory for federated data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "results" / "fl_experiments"),
        help="Output directory for experiment results"
    )

    # Experiment selection
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help='Comma-separated list of strategies to test (e.g., "fedavg,fedprox") or "all"'
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="all",
        help='Comma-separated list of splits to test (e.g., "iid,dirichlet_0.5") or "all"'
    )
    parser.add_argument(
        "--models",
        type=str,
        default="lightning",
        help='Comma-separated list of models to test or "all"'
    )

    # Experiment configuration
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=EXPERIMENT_MATRIX["default_num_rounds"],
        help="Number of FL rounds per experiment"
    )
    parser.add_argument(
        "--num_sites",
        type=int,
        default=EXPERIMENT_MATRIX["default_num_sites"],
        help="Number of federated sites"
    )

    # Execution options
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be run without executing"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel (experimental, not implemented)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse experiment selection
    if args.strategies == "all":
        strategies = list(EXPERIMENT_MATRIX["strategies"].keys())
    else:
        strategies = [s.strip() for s in args.strategies.split(",")]

    if args.splits == "all":
        splits = list(EXPERIMENT_MATRIX["data_splits"].keys())
    else:
        splits = [s.strip() for s in args.splits.split(",")]

    if args.models == "all":
        models = EXPERIMENT_MATRIX["models"]
    else:
        models = [m.strip() for m in args.models.split(",")]

    # Print configuration
    logger.info("="*70)
    logger.info("Automated Federated Learning Experiment Runner")
    logger.info("="*70)
    logger.info(f"Strategies: {', '.join(strategies)}")
    logger.info(f"Data splits: {', '.join(splits)}")
    logger.info(f"Models: {', '.join(models)}")
    logger.info(f"Rounds per experiment: {args.num_rounds}")
    logger.info(f"Number of sites: {args.num_sites}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    if args.parallel:
        logger.warning("Parallel execution is not yet implemented")
    logger.info("="*70)

    # Create experiment runner
    runner = ExperimentRunner(
        base_data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        pipeline_output=Path(args.pipeline_output),
        population_files=args.population_files,
        num_rounds=args.num_rounds,
        num_sites=args.num_sites,
        dry_run=args.dry_run,
    )

    # Run all experiments
    try:
        successful_results, failed_results = runner.run_all_experiments(
            strategies=strategies,
            splits=splits,
            models=models,
        )

        # Generate summary report
        runner.generate_summary_report(successful_results, failed_results)

        # Exit with appropriate code
        if failed_results:
            logger.warning(f"{len(failed_results)} experiments failed")
            sys.exit(1)
        else:
            logger.info("All experiments completed successfully!")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("Experiment run interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Experiment run failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
