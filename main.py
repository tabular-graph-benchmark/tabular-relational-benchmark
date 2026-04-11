"""Main entry for running tabular model experiments with optional GNN injection.

This file is an English-only port based primarily on TaBLEau/main.py.
"""

# Example usages:
#   python main.py --dataset_size all --task_type all --feature_type all --models all --gnn_stages all
#   python main.py --dataset kaggle_Audit_Data --models all --gnn_stages all
#   python main.py --dataset kaggle_Audit_Data --models excelformer --gnn_stages none
#   python main.py --dataset_size all --task_type all --feature_type all --models all --gnn_stages none
#   python main.py --dataset_size all --task_type all --feature_type all --models excelformer --gnn_stages none

import argparse
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns


def _patch_torch_frame_api() -> None:
    """Patch torch_frame API differences across versions.

    Some torch_frame versions define stypes in `torch_frame._stype` but expect
    convenient aliases (e.g., `torch_frame.multicategorical`) to exist at the
    top-level module. In this repo we rely on `torch_frame.data.*`, which may
    import those aliases.
    """

    try:
        import torch_frame

        if not hasattr(torch_frame, "multicategorical"):
            try:
                import torch_frame._stype as _stype

                if hasattr(_stype, "multicategorical"):
                    torch_frame.multicategorical = _stype.multicategorical
            except Exception:
                pass

        if not hasattr(torch_frame, "categorical"):
            try:
                import torch_frame._stype as _stype

                if hasattr(_stype, "categorical"):
                    torch_frame.categorical = _stype.categorical
            except Exception:
                pass

        if not hasattr(torch_frame, "numerical"):
            try:
                import torch_frame._stype as _stype

                if hasattr(_stype, "numerical"):
                    torch_frame.numerical = _stype.numerical
            except Exception:
                pass
    except Exception:
        # torch_frame isn't a hard dependency for all baselines.
        return


_patch_torch_frame_api()

# First, patch/adapter official models before importing model code (if available).
try:
    from gnn_injection import adapt_official_models

    patcher = adapt_official_models()
except Exception:
    patcher = None

try:
    from gnn_injection import GNNInjector, STAGE_TO_FUNCTION
except Exception:
    GNNInjector = None
    STAGE_TO_FUNCTION = {
        "start": "start",
        "materialize": "materialize",
        "encoding": "encode",
        "columnwise": "column_interact",
        "decoding": "decode",
    }

from utils.data_utils import DatasetLoader
from model_runner import ModelRunner


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiment.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="GNN-injected tabular model experiments")

    # Dataset category / selection
    parser.add_argument(
        "--dataset_size",
        type=str,
        default="small_datasets",
        choices=["small_datasets", "large_datasets", "all"],
        help="Dataset size bucket",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="binclass",
        choices=["binclass", "multiclass", "regression", "all"],
        help="Task type",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="numerical",
        choices=["numerical", "categorical", "balanced", "all"],
        help="Feature type",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset name (optional)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset directory")

    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        default=["excelformer"],
        help="Models to run (space-separated)",
    )
    parser.add_argument(
        "--gnn_stages",
        nargs="+",
        default=["none", "materialization", "encoding", "columnwise", "decoding"],
        help="GNN injection stages to test (space-separated)",
    )

    # Graph construction / graphify domain
    parser.add_argument(
        "--graphify",
        type=str,
        default="row",
        choices=["row", "feature", "all"],
        help=(
            "Graphify domain for GNN: 'row' builds a sample graph (row-level GNN); "
            "'feature' builds a column/feature graph (feature-level GNN); "
            "'all' runs both row and feature."
        ),
    )

    # Experiment setup
    parser.add_argument("--train_ratio", type=float, default=0.80, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Test split ratio")
    parser.add_argument("--few_shot", action="store_true", help="Enable few-shot setting")
    parser.add_argument(
        "--few_shot_ratio",
        type=float,
        default=0.05,
        help="Training ratio used under few-shot",
    )

    # GNN hyperparameters
    parser.add_argument("--gnn_hidden_dim", type=int, default=256, help="GNN hidden dimension")
    parser.add_argument("--gnn_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--gnn_dropout", type=float, default=0.2, help="GNN dropout")

    # DGM / dynamic-graph hyperparameters (used by ExcelFormer row-GNN injection)
    parser.add_argument(
        "--gnn_dgm_k",
        type=int,
        default=10,
        help="DGM_d max k (top-k candidates per node)",
    )
    parser.add_argument(
        "--gnn_dgm_gate",
        type=str,
        default="none",
        choices=["none", "sigmoid"],
        help="Enable variable-K behavior by gating top-k edges using DGM logprobs",
    )
    parser.add_argument(
        "--gnn_dgm_gate_sharpness",
        type=float,
        default=1.0,
        help="Sigmoid sharpness for the logprob gate (larger => harder gating)",
    )
    parser.add_argument(
        "--gnn_dgm_gate_threshold_init",
        type=float,
        default=-10.0,
        help="Initial value for the learnable logprob gating threshold",
    )
    parser.add_argument(
        "--gnn_dgm_eval_prune",
        nargs="?",
        const=1,
        default=0,
        type=int,
        help="If set (or set to 1), hard-prune gated edges during eval",
    )
    parser.add_argument(
        "--gnn_dgm_eval_prune_threshold",
        type=float,
        default=0.5,
        help="Eval-time prune threshold applied to gated edge weights",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=300, help="Max training epochs")
    parser.add_argument(
        "--gnn_epochs",
        type=int,
        default=None,
        help="Epochs for offline GNN training (start/materialize stages). Default: same as --epochs.",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience")
    parser.add_argument(
        "--debug_metrics",
        action="store_true",
        help="Enable lightweight debug prints (regression y stats, naive baseline, etc.)",
    )

    parser.add_argument(
        "--restore_best",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to snapshot and restore the best-val checkpoint within a run (1=yes, 0=no).",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id (-1 for CPU)")
    parser.add_argument("--output_dir", type=str, default="./result", help="Output directory")
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name (default: dataset_type_timestamp)",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_experiment_name(args) -> str:
    """Create an experiment name from CLI args."""
    if args.exp_name:
        return args.exp_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dataset:
        prefix = f"{args.dataset}"
    else:
        size_prefix = args.dataset_size if args.dataset_size != "all" else "all_sizes"
        task_prefix = args.task_type if args.task_type != "all" else "all_tasks"
        feature_prefix = args.feature_type if args.feature_type != "all" else "all_features"
        prefix = f"{size_prefix}_{task_prefix}_{feature_prefix}"

    if args.few_shot:
        prefix += f"_fewshot{args.few_shot_ratio}"

    return f"{prefix}_{timestamp}"


def expand_all_option(option_list, all_options):
    """Expand an option list containing 'all' into the full option set."""
    if "all" in option_list:
        return all_options
    return option_list


def resolve_model_aliases(models, available_models):
    """Resolve user-friendly model aliases into concrete model names."""
    alias_map = {
        "idgl": "idgl_gnn",
        "lds": "lds_gnn",
    }

    resolved = []
    for name in models:
        if name in alias_map and alias_map[name] in available_models:
            resolved.append(alias_map[name])
        else:
            resolved.append(name)

    unknown = [m for m in resolved if m not in available_models]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available models: {sorted(available_models)}")

    return resolved


def run_experiment(args):
    """Run the experiment sweep.

    Returns:
        A dict-like results object (or None in the current flow).
    """
    set_seed(args.seed)

    # Backward-compatible fallback: older scripts used './datasets', while this repo uses './data'.
    if args.data_dir == "./datasets" and not Path(args.data_dir).exists() and Path("./data").exists():
        args.data_dir = "./data"

    loader = DatasetLoader(args.data_dir)

    categories = loader.get_dataset_categories()
    print("Dataset category stats:", categories)

    model_runner = ModelRunner("./models")
    gnn_injector = GNNInjector(model_runner) if GNNInjector is not None else None

    available_models = []
    model_type_mapping = {}
    for model_type, models in model_runner.available_models.items():
        for model_name in models.keys():
            available_models.append(model_name)
            model_type_mapping[model_name] = model_type

    models_to_test = expand_all_option(args.models, available_models)
    models_to_test = resolve_model_aliases(models_to_test, available_models)
    print(f"models_to_test: {models_to_test}")

    valid_stages = ["none"] + list(STAGE_TO_FUNCTION.keys())
    gnn_stages_to_test = expand_all_option(args.gnn_stages, valid_stages)

    if args.dataset:
        datasets_to_test = [args.dataset]
    else:
        datasets_to_test = loader.get_datasets_by_category(
            None if args.dataset_size == "all" else args.dataset_size,
            None if args.task_type == "all" else args.task_type,
            None if args.feature_type == "all" else args.feature_type,
        )

    logger.info(
        "Will test %s datasets, %s models, and %s GNN stages",
        len(datasets_to_test),
        len(models_to_test),
        len(gnn_stages_to_test),
    )

    experiment_config = {
        "train_val_test_split_ratio": [args.train_ratio, args.val_ratio, args.test_ratio],
        "val_ratio": args.val_ratio,
        "few_shot": args.few_shot,
        "few_shot_ratio": args.few_shot_ratio,
        "graphify": args.graphify,
        "epochs": args.epochs,
            "gnn_epochs": args.epochs if args.gnn_epochs is None else int(args.gnn_epochs),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "device": torch.device(
            f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
        ),
        "gpu": args.gpu,
        "seed": args.seed,
        "debug_metrics": args.debug_metrics,
        "restore_best": bool(int(args.restore_best)),
        "dgm_k": int(args.gnn_dgm_k),

        # Align ExcelFormer config keys (keeps backward compatibility)
        "gnn_hidden": int(args.gnn_hidden_dim),
        "gnn_dropout": float(args.gnn_dropout),
        "gnn_dgm_k": int(args.gnn_dgm_k),
        "gnn_dgm_gate": str(args.gnn_dgm_gate),
        "gnn_dgm_gate_sharpness": float(args.gnn_dgm_gate_sharpness),
        "gnn_dgm_gate_threshold_init": float(args.gnn_dgm_gate_threshold_init),
        "gnn_dgm_eval_prune": bool(int(args.gnn_dgm_eval_prune)),
        "gnn_dgm_eval_prune_threshold": float(args.gnn_dgm_eval_prune_threshold),
    }

    gnn_config = {
        "hidden_dim": args.gnn_hidden_dim,
        "num_layers": args.gnn_layers,
        "dropout": args.gnn_dropout,
    }

    all_results = []

    for dataset_name in datasets_to_test:
        logger.info("Processing dataset: %s", dataset_name)

        try:
            dataset_info = loader.load_dataset(dataset_name)
            df = dataset_info["df"]
            print(df.shape)

            from sklearn.model_selection import train_test_split
            y = df["target"] if "target" in df.columns else df.iloc[:, -1]
            task_type = loader.get_dataset_info(dataset_name).get("task_type", "binclass")

            if experiment_config.get("few_shot", False):
                split_ratio = [
                    experiment_config["few_shot_ratio"],
                    experiment_config["val_ratio"],
                    1
                    - experiment_config["few_shot_ratio"]
                    - experiment_config["val_ratio"],
                ]
            else:
                split_ratio = [
                    experiment_config["train_val_test_split_ratio"][0],
                    experiment_config["train_val_test_split_ratio"][1],
                    experiment_config["train_val_test_split_ratio"][2],
                ]

            stratify_y = y if "class" in task_type or "binclass" in task_type else None
            train_val_df, test_df, train_val_y, test_y = train_test_split(
                df,
                y,
                test_size=split_ratio[2],
                stratify=stratify_y,
                random_state=experiment_config["seed"],
            )

            val_ratio = split_ratio[1] / (split_ratio[0] + split_ratio[1])
            stratify_train_val = (
                train_val_y if "class" in task_type or "binclass" in task_type else None
            )
            train_df, val_df, train_y, val_y = train_test_split(
                train_val_df,
                train_val_y,
                test_size=val_ratio,
                stratify=stratify_train_val,
                random_state=experiment_config["seed"],
            )
        except Exception as e:
            logger.error("Failed to load dataset %s: %s", dataset_name, str(e))
            continue

        dataset_results = {
            "dataset": dataset_name,
            "info": loader.get_dataset_info(dataset_name),
            "models": {},
        }

        for model_name in models_to_test:
            logger.info("Processing model: %s", model_name)

            model_results = {}
            model_type = model_type_mapping[model_name]

            if str(args.graphify).lower() == "all" and model_type != "baseline":
                graphify_to_test = ["row", "feature"]
            else:
                graphify_to_test = [str(args.graphify).lower()]

            if model_type == "baseline":
                logger.info("Model %s is a baseline model; skipping GNN stages", model_name)
                try:
                    set_seed(int(experiment_config.get("seed", args.seed)))
                    result = model_runner.run_model(
                        model_name,
                        train_df,
                        val_df,
                        test_df,
                        dataset_results,
                        experiment_config,
                        model_type,
                        "none",
                    )
                    model_results["none"] = result
                    if "best_test_metric" in result:
                        print(f"[RESULT] Best test metric: {result['best_test_metric']}")
                except Exception as e:
                    logger.error("Error running model %s: %s", model_name, str(e))
                    model_results["none"] = {"error": str(e)}
            else:
                for graphify_mode in graphify_to_test:
                    graphify_mode = str(graphify_mode).lower()
                    model_results.setdefault(graphify_mode, {})

                    exp_cfg = dict(experiment_config)
                    exp_cfg["graphify"] = graphify_mode

                    for gnn_stage in gnn_stages_to_test:
                        logger.info(
                            "Testing model %s with graphify=%s at stage %s",
                            model_name,
                            graphify_mode,
                            gnn_stage,
                        )
                        print("\n")

                        # GNN injection is optional; keep it disabled by default (matches upstream).
                        # if gnn_stage != 'none' and gnn_injector is not None:
                        #     gnn_injector.inject(model_name, gnn_stage, gnn_config)

                        try:
                            # Optimization: avoid rerunning identical configurations when
                            # `--graphify all` is used.
                            #
                            # For ExcelFormer in this port, the following are equivalent:
                            #   - (graphify=row, gnn_stage=none)
                            #   - (graphify=feature, gnn_stage in {none,start,materialize})
                            #
                            # Paper rationale: feature-level graphification is inactive at
                            # start/materialize (no feature tokens yet), and `none` is the
                            # baseline. Therefore we can safely alias metrics to reduce
                            # wall time when sweeping stages.
                            if (
                                str(graphify_mode).lower() == "feature"
                                and str(gnn_stage).lower() in {"none", "start", "materialize"}
                                and "row" in model_results
                                and isinstance(model_results.get("row"), dict)
                                and "none" in model_results["row"]
                                and isinstance(model_results["row"].get("none"), dict)
                                and ("best_test_metric" in model_results["row"]["none"]
                                     or "error" in model_results["row"]["none"])
                            ):
                                src = model_results["row"]["none"]
                                aliased = {
                                    "best_val_metric": src.get("best_val_metric"),
                                    "best_test_metric": src.get("best_test_metric"),
                                    "early_stop_epochs": src.get("early_stop_epochs"),
                                    "gnn_early_stop_epochs": src.get("gnn_early_stop_epochs", 0),
                                    "elapsed_time": 0.0,
                                    "copied_from": "row/none",
                                }
                                # Propagate errors as-is (rare, but keeps behavior consistent).
                                if "error" in src:
                                    aliased["error"] = src.get("error")

                                logger.info(
                                    "Aliasing excelformer metrics for graphify=feature stage %s from row/none",
                                    str(gnn_stage).lower(),
                                )
                                model_results[graphify_mode][gnn_stage] = aliased
                                continue

                            set_seed(int(exp_cfg.get("seed", args.seed)))
                            result = model_runner.run_model(
                                model_name,
                                train_df,
                                val_df,
                                test_df,
                                dataset_results,
                                exp_cfg,
                                model_type,
                                gnn_stage,
                            )
                            model_results[graphify_mode][gnn_stage] = result
                            if "best_test_metric" in result:
                                print(f"[RESULT] Best test metric: {result['best_test_metric']}")
                        except Exception as e:
                            logger.error(
                                "Error running model %s (graphify=%s) at stage %s: %s",
                                model_name,
                                graphify_mode,
                                gnn_stage,
                                str(e),
                            )
                            model_results[graphify_mode][gnn_stage] = {"error": str(e)}

            dataset_results["models"][model_name] = model_results

        all_results.append(dataset_results)

    # Save a compact text summary (avoid collisions and overly long names).
    exp_name = get_experiment_name(args)
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(f"dataset: {result['dataset']}\n")
            for model_name, model_results in result["models"].items():
                f.write(f"  model: {model_name}\n")

                # Baselines keep the old shape: {stage -> res}
                if "none" in model_results and isinstance(model_results.get("none"), dict) and (
                    "best_test_metric" in model_results.get("none", {}) or "error" in model_results.get("none", {})
                ):
                    for gnn_stage, res in model_results.items():
                        f.write(f"    gnn_stage: {gnn_stage}\n")
                        if "best_val_metric" in res:
                            f.write(f"          Best val metric: {res['best_val_metric']}\n")
                        if "best_test_metric" in res:
                            f.write(f"          Best test metric: {res['best_test_metric']}\n")
                        if "early_stop_epochs" in res:
                            f.write(f"          Early-stop epochs: {res['early_stop_epochs']}\n")
                        if "gnn_early_stop_epochs" in res:
                            f.write(
                                f"          GNN early-stop epochs: {res['gnn_early_stop_epochs']}\n"
                            )
                        if "error" in res:
                            f.write(f"          Error: {res['error']}\n")
                        if "elapsed_time" in res:
                            f.write(f"          Elapsed: {res['elapsed_time']:.2f} seconds\n")
                else:
                    # Backbones with graphify loop: {graphify -> {stage -> res}}
                    for graphify_mode, stage_dict in model_results.items():
                        f.write(f"    graphify: {graphify_mode}\n")
                        for gnn_stage, res in stage_dict.items():
                            f.write(f"      gnn_stage: {gnn_stage}\n")
                            if "best_val_metric" in res:
                                f.write(f"            Best val metric: {res['best_val_metric']}\n")
                            if "best_test_metric" in res:
                                f.write(f"            Best test metric: {res['best_test_metric']}\n")
                            if "early_stop_epochs" in res:
                                f.write(f"            Early-stop epochs: {res['early_stop_epochs']}\n")
                            if "gnn_early_stop_epochs" in res:
                                f.write(
                                    f"            GNN early-stop epochs: {res['gnn_early_stop_epochs']}\n"
                                )
                            if "error" in res:
                                f.write(f"            Error: {res['error']}\n")
                            if "elapsed_time" in res:
                                f.write(f"            Elapsed: {res['elapsed_time']:.2f} seconds\n")

    logger.info("Saved summary to %s", summary_path)

    # The upstream flow returns None; keep it unchanged.
    return None









def main():
    """Main function."""
    args = parse_args()
    exp_name = get_experiment_name(args)

    logger.info("Starting experiment: %s", exp_name)

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    logger.info("Using device: %s", device)

    start_time = time.time()
    results = run_experiment(args)
    elapsed_time = time.time() - start_time
    logger.info("Experiment finished. Total elapsed: %.2f seconds", elapsed_time)

    


if __name__ == "__main__":
    main()