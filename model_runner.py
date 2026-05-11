import importlib.util
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict


logger = logging.getLogger(__name__)


class ModelRunner:
	"""Model execution interface for running different tabular models.

	In TaBLEau, models were categorized under:
	  - models/pytorch_frame
	  - models/custom
	  - models/comparison

	In this tabular-relational-benchmark workspace, they are re-categorized under:
	  - models/backbone
	  - models/baseline
	"""

	def __init__(self, base_dir: str = "./models"):
		self.base_dir = Path(base_dir)
		self.available_models = self._scan_models()
		self.gnn_hooks: Dict[str, Dict[str, Callable]] = {}

	def _scan_dir_py_files(self, directory: Path) -> Dict[str, str]:
		if not directory.exists():
			return {}
		return {p.stem: str(p) for p in directory.glob("*.py")}

	def _scan_models(self) -> Dict[str, Dict[str, str]]:
		"""Scan and index available model files.

		Returns:
			dict: Mapping from model group -> {model_name: model_path}
				  Where group is typically 'backbone' or 'baseline'.
		"""

		backbone_dir = self.base_dir / "backbone"
		baseline_dir = self.base_dir / "baseline"

		if backbone_dir.exists() or baseline_dir.exists():
			available = {
				"backbone": self._scan_dir_py_files(backbone_dir),
				"baseline": self._scan_dir_py_files(baseline_dir),
			}
			logger.info(
				"Found %s backbone model(s): %s",
				len(available["backbone"]),
				sorted(available["backbone"].keys()),
			)
			logger.info(
				"Found %s baseline model(s): %s",
				len(available["baseline"]),
				sorted(available["baseline"].keys()),
			)
			return available

		# Legacy fallback: map old categories into the new ones.
		pytorch_frame = self._scan_dir_py_files(self.base_dir / "pytorch_frame")
		custom = self._scan_dir_py_files(self.base_dir / "custom")
		comparison = self._scan_dir_py_files(self.base_dir / "comparison")

		backbone = {}
		backbone.update(pytorch_frame)
		backbone.update(custom)

		available = {
			"backbone": backbone,
			"baseline": comparison,
		}

		logger.info(
			"Legacy layout detected. Mapped %s backbone model(s) from pytorch_frame+custom: %s",
			len(available["backbone"]),
			sorted(available["backbone"].keys()),
		)
		logger.info(
			"Legacy layout detected. Mapped %s baseline model(s) from comparison: %s",
			len(available["baseline"]),
			sorted(available["baseline"].keys()),
		)

		return available

	def _load_model_module(self, model_name: str, model_group: str):
		"""Dynamically load a model module."""

		if model_group not in self.available_models:
			raise ValueError(
				f"Unknown model group: {model_group}. Available: {sorted(self.available_models.keys())}"
			)

		if model_name not in self.available_models[model_group]:
			raise ValueError(f"Model not found: {model_name} under group: {model_group}")

		model_path = self.available_models[model_group][model_name]
		spec = importlib.util.spec_from_file_location(model_name, model_path)
		module = importlib.util.module_from_spec(spec)
		# Register module in sys.modules before execution to support @dataclass and other introspective decorators
		sys.modules[model_name] = module
		spec.loader.exec_module(module)
		return module

	def register_gnn_hook(self, model_name: str, stage: str, hook_fn: Callable):
		"""Register a hook function for a model and a specific stage."""

		self.gnn_hooks.setdefault(model_name, {})[stage] = hook_fn
		logger.info("Registered GNN hook: model=%s stage=%s", model_name, stage)

	def _apply_gnn_hooks(self, model_name: str, module):
		"""Apply hooks by wrapping stage functions at runtime."""

		hooks = self.gnn_hooks.get(model_name)
		if not hooks:
			return module

		for stage, hook_fn in hooks.items():
			attr = f"{stage}_fn"
			if not hasattr(module, attr):
				logger.warning(
					"Cannot apply hook: model=%s has no function %s",
					model_name,
					attr,
				)
				continue

			original_fn = getattr(module, attr)

			def hooked_fn(*args, _original_fn=original_fn, _hook_fn=hook_fn, **kwargs):
				result = _original_fn(*args, **kwargs)
				return _hook_fn(result, *args, **kwargs)

			setattr(module, attr, hooked_fn)
			logger.info("Applied GNN hook: model=%s stage=%s", model_name, stage)

		return module

	def run_model(
		self,
		model_name,
		train_df,
		val_df,
		test_df,
		dataset_results,
		config,
		model_group,
		gnn_stage,
	):
		"""Run a specific model."""

		logger.info("Running model: %s", model_name)
		start_time = time.time()

		original_argv = sys.argv
		sys.argv = [sys.argv[0]]

		try:
			module = self._load_model_module(model_name, model_group)
			module = self._apply_gnn_hooks(model_name, module)

			if hasattr(module, "main"):
				results = module.main(train_df, val_df, test_df, dataset_results, config, gnn_stage)
			elif hasattr(module, "run_experiment"):
				results = module.run_experiment(train_df, val_df, test_df, config)
			else:
				for func_name in ["train_and_evaluate", "run", "execute"]:
					if hasattr(module, func_name):
						results = getattr(module, func_name)(train_df, val_df, test_df, config)
						break
				else:
					raise ValueError(f"No runnable function found in model: {model_name}")
		except Exception as e:
			logger.error("Error while running model %s: %s", model_name, str(e))
			logger.error("Full traceback:\n%s", traceback.format_exc())
			return {
				"model": model_name,
				"error": str(e),
				"elapsed_time": time.time() - start_time,
			}
		finally:
			sys.argv = original_argv

		elapsed_time = time.time() - start_time

		if isinstance(results, dict):
			results["model"] = model_name
			results["elapsed_time"] = elapsed_time
		else:
			results = {
				"model": model_name,
				"raw_results": results,
				"elapsed_time": elapsed_time,
			}

		logger.info("Finished model %s. Elapsed: %.2f seconds", model_name, elapsed_time)
		return results
