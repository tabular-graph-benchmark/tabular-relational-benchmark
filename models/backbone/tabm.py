from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
	import pandas as pd
except ModuleNotFoundError as e:  # pragma: no cover
	pd = None
	_PANDAS_IMPORT_ERROR = e

try:
	from sklearn.preprocessing import QuantileTransformer
except ModuleNotFoundError as e:  # pragma: no cover
	QuantileTransformer = None
	_SKLEARN_IMPORT_ERROR = e

try:
	from torch_geometric.nn import GCNConv
	from torch_geometric.utils import coalesce
except ModuleNotFoundError as e:  # pragma: no cover
	GCNConv = None
	coalesce = None
	_PYG_IMPORT_ERROR = e


def resolve_device(config: dict) -> torch.device:
	gpu_id = None
	try:
		gpu_id = config.get("gpu", None)
	except Exception:
		gpu_id = None

	if torch.cuda.is_available():
		if gpu_id is not None:
			try:
				device = torch.device(f"cuda:{int(gpu_id)}")
				print(f"[DEVICE] Using cuda:{int(gpu_id)} (from config['gpu']={gpu_id})")
				return device
			except Exception:
				pass
		print("[DEVICE] Using cuda")
		return torch.device("cuda")

	print("[DEVICE] Using cpu")
	return torch.device("cpu")


def _ensure_tabm_importable() -> None:
	modelcomparison_root = Path(__file__).resolve().parents[3]
	tabm_dir = modelcomparison_root / "tabm"
	# Insert the ModelComparison root (not the tabm subdir) so that
	# `import tabm` resolves to the external package directory
	# (/home/shangyuan/ModelComparison/tabm) instead of this module
	# (models/backbone/tabm.py), avoiding a circular/self-import.
	if tabm_dir.exists() and str(modelcomparison_root) not in sys.path:
		sys.path.insert(0, str(modelcomparison_root))


def _import_dgm_d():
	try:
		from DGMlib.layers import DGM_d as _DGM_d

		return _DGM_d
	except Exception as first_exc:
		modelcomparison_root = Path(__file__).resolve().parents[3]
		dgm_path = modelcomparison_root / "DGM_pytorch"
		if dgm_path.exists() and str(dgm_path) not in sys.path:
			sys.path.insert(0, str(dgm_path))
		try:
			from DGMlib.layers import DGM_d as _DGM_d

			return _DGM_d
		except Exception as second_exc:
			raise second_exc from first_exc


def _standardize(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
	mean = x.mean(dim=dim, keepdim=True)
	std = x.std(dim=dim, keepdim=True).clamp_min(eps)
	return (x - mean) / std


class SimpleGCN(nn.Module):
	def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
		super().__init__()
		if GCNConv is None:  # pragma: no cover
			raise ModuleNotFoundError(
				"torch_geometric is required for TabM row graphification (GCNConv)."
			) from _PYG_IMPORT_ERROR
		layers = max(2, int(num_layers))
		dims = [int(in_dim)] + [int(hidden_dim)] * (layers - 1) + [int(out_dim)]
		self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(layers)])

	def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None):
		for i, conv in enumerate(self.convs):
			x = conv(x, edge_index, edge_weight=edge_weight)
			if i + 1 < len(self.convs):
				x = torch.relu(x)
		return x


def _dgm_gate_and_prune_edges(
	edge_index_dgm: torch.Tensor,
	logprobs_dgm: torch.Tensor | None,
	num_nodes: int,
	*,
	gate: str,
	gate_sharpness: float,
	gate_threshold_param: torch.Tensor | None,
	is_training: bool,
	eval_prune: bool,
	eval_prune_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
	"""TRB-aligned DGM gating/prune + undirected/self-loop + dedup."""

	if edge_index_dgm.numel() == 0:
		sl = torch.arange(num_nodes, device=edge_index_dgm.device)
		ei = torch.stack([sl, sl], dim=0)
		return ei, None

	gate = str(gate or "none").lower()
	if gate not in {"none", "sigmoid"}:
		raise ValueError(f"Unknown gnn_dgm_gate={gate!r}")

	edge_weight = None
	if gate != "none":
		if logprobs_dgm is None:
			raise ValueError("logprobs_dgm is required when gnn_dgm_gate is enabled")
		lp = logprobs_dgm.reshape(-1)
		thr = gate_threshold_param
		if thr is None:
			thr = torch.tensor(0.0, device=lp.device, dtype=lp.dtype)
		edge_weight = torch.sigmoid((lp - thr) * float(gate_sharpness)).to(torch.float32)

	rev = torch.stack([edge_index_dgm[1], edge_index_dgm[0]], dim=0)
	sl = torch.arange(num_nodes, device=edge_index_dgm.device)
	self_edges = torch.stack([sl, sl], dim=0)
	ei = torch.cat([edge_index_dgm, rev, self_edges], dim=1)
	if edge_weight is not None:
		ew = torch.cat(
			[
				edge_weight,
				edge_weight,
				torch.ones(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype),
			],
			dim=0,
		)
		if (not is_training) and bool(eval_prune):
			keep = ew >= float(eval_prune_threshold)
			ei = ei[:, keep]
			ew = ew[keep]
	else:
		ew = None

	if coalesce is not None:
		ei, ew = coalesce(ei, ew, num_nodes=num_nodes, reduce="max")

	return ei, ew


def _init_feature_topology_prior(feature_names: list[str], init_mode: str = "chain") -> dict:
	feature_names = list(feature_names)
	F_ = int(len(feature_names))
	if F_ <= 0:
		raise ValueError("Cannot initialize feature topology with zero features")
	init_mode = str(init_mode).lower()
	if init_mode == "full":
		adj = torch.ones((F_, F_), dtype=torch.bool)
	elif init_mode == "identity":
		adj = torch.eye(F_, dtype=torch.bool)
	elif init_mode == "chain":
		adj = torch.eye(F_, dtype=torch.bool)
		if F_ > 1:
			idx = torch.arange(F_ - 1)
			adj[idx, idx + 1] = True
			adj[idx + 1, idx] = True
	else:
		raise ValueError(f"Unknown feature_topology_init={init_mode!r}")

	edge_index = adj.nonzero(as_tuple=False).t().contiguous()
	return {
		"init_mode": init_mode,
		"num_features": F_,
		"feature_names": feature_names,
		"adjacency_mask": adj,
		"edge_index": edge_index,
	}


class FeatureTopologyMaskedMixing(nn.Module):
	def __init__(self, allowed_mask: torch.Tensor):
		super().__init__()
		allowed_mask = allowed_mask.to(dtype=torch.bool)
		if allowed_mask.dim() != 2 or allowed_mask.shape[0] != allowed_mask.shape[1]:
			raise ValueError(f"allowed_mask must be square [F,F], got {tuple(allowed_mask.shape)}")
		self.register_buffer("allowed_mask", allowed_mask)
		F_ = int(allowed_mask.shape[0])
		logits = torch.zeros((F_, F_), dtype=torch.float32, device=allowed_mask.device)
		logits = logits.masked_fill(~allowed_mask, -20.0)
		logits = logits.masked_fill(torch.eye(F_, dtype=torch.bool, device=allowed_mask.device), 5.0)
		self.logits = nn.Parameter(logits)

	def forward(self, tokens: torch.Tensor) -> torch.Tensor:
		# tokens: [B,F,C]
		w = torch.sigmoid(self.logits)
		w = w * self.allowed_mask.to(dtype=w.dtype)
		w = 0.5 * (w + w.t())
		w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
		return torch.einsum("ij,bjd->bid", w, tokens)


class PerFeatureBackbone(nn.Module):
	"""A lightweight per-feature backbone that preserves feature channels.

	Input: tokens [B, F, E]
	Output: per-feature embeddings [B, F, d_block]
	"""
	def __init__(self, *, in_dim: int, d_block: int, n_blocks: int, dropout: float, activation: str = "ReLU"):
		super().__init__()
		act = getattr(nn, activation) if isinstance(activation, str) else activation
		layers = []
		for i in range(n_blocks):
			in_f = in_dim if i == 0 else d_block
			layers.append(nn.Linear(in_f, d_block))
			layers.append(act())
			if dropout and dropout > 0:
				layers.append(nn.Dropout(dropout))
		self.net = nn.Sequential(*layers)

	def forward(self, tokens: torch.Tensor) -> torch.Tensor:
		# tokens: [B,F,E] -> flatten last dim transform applied to each feature independently
		B, F, E = tokens.shape
		x = tokens.reshape(B * F, E)
		x = self.net(x)
		x = x.reshape(B, F, -1)
		return x


def start_fn(train_df, val_df, test_df):
	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError("pandas is required") from _PANDAS_IMPORT_ERROR
	return train_df, val_df, test_df


def feature_gnn_after_start_fn(train_df, val_df, test_df, config: dict, task_type: str):
	feature_names = [c for c in train_df.columns if c != "target"]
	init_mode = str(config.get("feature_topology_init", "chain"))
	topology = _init_feature_topology_prior(feature_names, init_mode=init_mode)
	config["feature_topology_meta"] = {
		"init_mode": topology["init_mode"],
		"num_features": topology["num_features"],
		"feature_names": topology["feature_names"],
		"edge_index": topology["edge_index"].tolist(),
	}
	print(
		f"[START-GNN] Feature-level topology initialized (inactive). "
		f"init_mode={topology['init_mode']}, num_features={topology['num_features']}"
	)
	return train_df, val_df, test_df, 0


def _factorize_object_columns(
	train_df: pd.DataFrame,
	val_df: pd.DataFrame,
	test_df: pd.DataFrame,
	feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
	train_feat = train_df[feature_cols].copy()
	val_feat = val_df[feature_cols].copy()
	test_feat = test_df[feature_cols].copy()

	for col in feature_cols:
		if train_feat[col].dtype == object or str(train_feat[col].dtype).startswith("category"):
			codes, uniques = pd.factorize(train_feat[col], sort=True)
			mapping = {k: int(v) for v, k in enumerate(uniques.tolist())}
			train_feat[col] = codes.astype(np.float32)
			val_feat[col] = val_feat[col].map(mapping).fillna(-1).astype(np.float32)
			test_feat[col] = test_feat[col].map(mapping).fillna(-1).astype(np.float32)
		else:
			train_feat[col] = pd.to_numeric(train_feat[col], errors="coerce").astype(np.float32)
			val_feat[col] = pd.to_numeric(val_feat[col], errors="coerce").astype(np.float32)
			test_feat[col] = pd.to_numeric(test_feat[col], errors="coerce").astype(np.float32)

	train_feat = train_feat.fillna(0.0)
	val_feat = val_feat.fillna(0.0)
	test_feat = test_feat.fillna(0.0)

	return (
		train_feat.values.astype(np.float32, copy=False),
		val_feat.values.astype(np.float32, copy=False),
		test_feat.values.astype(np.float32, copy=False),
		feature_cols,
	)


def materialize_fn(train_df, val_df, test_df, dataset_results, config: dict):
	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError("pandas is required") from _PANDAS_IMPORT_ERROR
	if QuantileTransformer is None:  # pragma: no cover
		raise ModuleNotFoundError("scikit-learn is required") from _SKLEARN_IMPORT_ERROR

	try:
		import rtdl_num_embeddings
		from rtdl_num_embeddings import PiecewiseLinearEmbeddings
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError("rtdl_num_embeddings is required") from e

	device = resolve_device(config)
	task_type = dataset_results.get("info", {}).get("task_type", "binclass")
	feature_cols = [c for c in train_df.columns if c != "target"]
	X_train_np, X_val_np, X_test_np, feature_names = _factorize_object_columns(
		train_df, val_df, test_df, feature_cols
	)

	# Remove constants based on train split.
	unique_counts = np.array([len(np.unique(X_train_np[:, i])) for i in range(X_train_np.shape[1])])
	valid_cols = unique_counts > 1
	if not bool(valid_cols.all()):
		X_train_np = X_train_np[:, valid_cols]
		X_val_np = X_val_np[:, valid_cols]
		X_test_np = X_test_np[:, valid_cols]
		feature_names = [n for i, n in enumerate(feature_names) if bool(valid_cols[i])]

	# Quantile normalize.
	noise = np.random.default_rng(0).normal(0.0, 1e-5, X_train_np.shape).astype(np.float32)
	n_quantiles = max(min(len(X_train_np) // 30, 1000), 10)
	n_quantiles = min(int(n_quantiles), int(len(X_train_np)))
	qt = QuantileTransformer(
		n_quantiles=n_quantiles,
		output_distribution="normal",
		subsample=10**9,
	).fit(X_train_np + noise)
	X_train_np = qt.transform(X_train_np).astype(np.float32)
	X_val_np = qt.transform(X_val_np).astype(np.float32)
	X_test_np = qt.transform(X_test_np).astype(np.float32)

	y_train_np = train_df["target"].values
	y_val_np = val_df["target"].values
	y_test_np = test_df["target"].values

	label_stats = None
	n_classes = None
	if task_type == "regression":
		y_mean = float(np.mean(y_train_np))
		y_std = float(np.std(y_train_np) if np.std(y_train_np) > 1e-12 else 1.0)
		label_stats = {"mean": y_mean, "std": y_std}
		y_train_np = ((y_train_np - y_mean) / y_std).astype(np.float32)
		y_val_np = ((y_val_np - y_mean) / y_std).astype(np.float32)
		y_test_np = ((y_test_np - y_mean) / y_std).astype(np.float32)
	else:
		y_train_np = y_train_np.astype(np.int64)
		y_val_np = y_val_np.astype(np.int64)
		y_test_np = y_test_np.astype(np.int64)
		n_classes = int(len(np.unique(y_train_np)))
		if task_type == "binclass" and n_classes != 2:
			n_classes = 2

	X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
	X_val = torch.tensor(X_val_np, dtype=torch.float32, device=device)
	X_test = torch.tensor(X_test_np, dtype=torch.float32, device=device)
	if task_type == "regression":
		y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)
		y_val = torch.tensor(y_val_np, dtype=torch.float32, device=device)
		y_test = torch.tensor(y_test_np, dtype=torch.float32, device=device)
	else:
		y_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
		y_val = torch.tensor(y_val_np, dtype=torch.long, device=device)
		y_test = torch.tensor(y_test_np, dtype=torch.long, device=device)

	n_bins = max(2, min(48, int(X_train.shape[0]) - 1))
	num_embeddings = PiecewiseLinearEmbeddings(
		rtdl_num_embeddings.compute_bins(X_train, n_bins=n_bins),
		d_embedding=int(config.get("tabm_d_embedding", 16)),
		activation=False,
		version="B",
	).to(device)

	return {
		"X_train": X_train,
		"y_train": y_train,
		"X_val": X_val,
		"y_val": y_val,
		"X_test": X_test,
		"y_test": y_test,
		"n_num_features": int(X_train.shape[1]),
		"n_classes": n_classes,
		"task_type": task_type,
		"label_stats": label_stats,
		"num_embeddings": num_embeddings,
		"feature_names": feature_names,
		"device": device,
	}


def feature_gnn_after_materialize_fn(material_outputs: dict, config: dict, dataset_name: str, task_type: str):
	feature_names = list(material_outputs.get("feature_names", []))
	if len(feature_names) == 0:
		raise ValueError("No feature_names in material_outputs")
	init_mode = str(config.get("feature_topology_init", "chain"))
	topology = _init_feature_topology_prior(feature_names, init_mode=init_mode)
	topology["dataset_name"] = str(dataset_name)
	topology["task_type"] = str(task_type)
	material_outputs["feature_topology"] = topology
	config["feature_topology_meta"] = {
		"init_mode": topology["init_mode"],
		"num_features": topology["num_features"],
		"feature_names": topology["feature_names"],
		"edge_index": topology["edge_index"].tolist(),
	}
	print(
		f"[MATERIALIZE-GNN] Feature-level topology initialized (inactive). "
		f"init_mode={topology['init_mode']}, num_features={topology['num_features']}"
	)
	return material_outputs


class _RowGNNConfig:
	def __init__(
		self,
		*,
		k: int,
		dgm_k: int,
		dgm_distance: str,
		gate: str,
		gate_sharpness: float,
		gate_threshold_init: float,
		eval_prune: bool,
		eval_prune_threshold: float,
		gnn_hidden: int,
		gnn_layers: int,
		gnn_dropout: float,
		dgm_reg: float,
		attn_heads: int,
	):
		self.k = int(k)
		self.dgm_k = int(dgm_k)
		self.dgm_distance = str(dgm_distance)
		self.gate = str(gate)
		self.gate_sharpness = float(gate_sharpness)
		self.gate_threshold_init = float(gate_threshold_init)
		self.eval_prune = bool(eval_prune)
		self.eval_prune_threshold = float(eval_prune_threshold)
		self.gnn_hidden = int(gnn_hidden)
		self.gnn_layers = int(gnn_layers)
		self.gnn_dropout = float(gnn_dropout)
		self.dgm_reg = float(dgm_reg)
		self.attn_heads = int(attn_heads)


def _read_row_gnn_config(config: dict) -> _RowGNNConfig:
	return _RowGNNConfig(
		k=int(config.get("tabm_k", 32)),
		dgm_k=int(config.get("gnn_dgm_k", config.get("dgm_k", 10))),
		dgm_distance=str(config.get("gnn_dgm_distance", config.get("dgm_distance", "euclidean"))),
		gate=str(config.get("gnn_dgm_gate", "none")).lower(),
		gate_sharpness=float(config.get("gnn_dgm_gate_sharpness", 1.0)),
		gate_threshold_init=float(config.get("gnn_dgm_gate_threshold_init", -10.0)),
		eval_prune=bool(config.get("gnn_dgm_eval_prune", False)),
		eval_prune_threshold=float(config.get("gnn_dgm_eval_prune_threshold", 0.5)),
		gnn_hidden=int(config.get("gnn_hidden", 64)),
		gnn_layers=int(config.get("gnn_layers", 2)),
		gnn_dropout=float(config.get("gnn_dropout", 0.1)),
		dgm_reg=float(config.get("gnn_dgm_reg", 0.01)),
		attn_heads=max(1, int(config.get("gnn_num_heads", 4))),
	)


class RowGraphifier(nn.Module):
	def __init__(self, d_block: int, out_dim: int, *, stage: str, cfg: _RowGNNConfig, device: torch.device):
		super().__init__()
		DGM_d_local = _import_dgm_d()

		class _Id(nn.Module):
			def forward(self, x, A=None):
				return x

		self.stage = stage
		self.cfg = cfg
		self.d_block = d_block
		self.out_dim = out_dim
		self.row_token_embed = nn.Parameter(torch.randn(cfg.k, d_block, device=device))
		self.pool_query = nn.Parameter(torch.randn(d_block, device=device))
		self.attn_in = nn.MultiheadAttention(embed_dim=d_block, num_heads=cfg.attn_heads, batch_first=True)
		self.norm_in = nn.LayerNorm(d_block)
		self.dgm = DGM_d_local(_Id(), k=int(cfg.dgm_k), distance=cfg.dgm_distance)
		self.gate_threshold = None
		if cfg.gate != "none":
			self.gate_threshold = nn.Parameter(
				torch.tensor(float(cfg.gate_threshold_init), device=device, dtype=torch.float32)
			)

		if stage == "decoding":
			self.gcn = SimpleGCN(d_block, cfg.gnn_hidden, out_dim, num_layers=max(2, cfg.gnn_layers))
			self.attn_out = None
			self.norm_out = None
			self.ffn_pre = None
			self.ffn_post = None
			self.ctx_proj = None
			self.fusion_alpha = None
		else:
			self.gcn = SimpleGCN(d_block, cfg.gnn_hidden, d_block, num_layers=max(2, cfg.gnn_layers))
			self.attn_out = nn.MultiheadAttention(embed_dim=d_block, num_heads=cfg.attn_heads, batch_first=True)
			self.norm_out = nn.LayerNorm(d_block)
			self.ctx_proj = nn.Linear(d_block, d_block)
			self.ffn_pre = nn.Sequential(
				nn.Linear(d_block, d_block * 2),
				nn.GELU(),
				nn.Dropout(cfg.gnn_dropout),
				nn.Linear(d_block * 2, d_block),
			)
			self.ffn_post = nn.Sequential(
				nn.Linear(d_block, d_block * 2),
				nn.GELU(),
				nn.Dropout(cfg.gnn_dropout),
				nn.Linear(d_block * 2, d_block),
			)
			self.fusion_alpha = nn.Parameter(torch.tensor(-0.847, device=device))

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
		# x: [B,k,d_block]
		B = int(x.size(0))
		x0 = x
		tokens = x + self.row_token_embed.unsqueeze(0)
		attn1, _ = self.attn_in(self.norm_in(tokens), self.norm_in(tokens), self.norm_in(tokens))
		tokens = tokens + attn1
		if self.stage != "decoding":
			tokens = tokens + self.ffn_pre(self.norm_in(tokens))

		pool_logits = (tokens * self.pool_query).sum(dim=-1) / math.sqrt(self.d_block)
		pool_w = torch.softmax(pool_logits, dim=1)
		x_pool = (pool_w.unsqueeze(-1) * tokens).sum(dim=1)
		x_pool = _standardize(x_pool, dim=0)

		x_batched = x_pool.unsqueeze(0)
		if hasattr(self.dgm, "k"):
			self.dgm.k = int(min(int(self.dgm.k), max(1, B - 1)))
		x_dgm, edge_index_dgm, logprobs = self.dgm(x_batched, A=None)
		x_dgm = x_dgm.squeeze(0)

		edge_index, edge_weight = _dgm_gate_and_prune_edges(
			edge_index_dgm,
			logprobs,
			B,
			gate=self.cfg.gate,
			gate_sharpness=self.cfg.gate_sharpness,
			gate_threshold_param=self.gate_threshold,
			is_training=bool(self.training),
			eval_prune=self.cfg.eval_prune,
			eval_prune_threshold=self.cfg.eval_prune_threshold,
		)

		out = self.gcn(x_dgm, edge_index, edge_weight=edge_weight)
		if self.stage == "decoding":
			return out, logprobs

		ctx = self.ctx_proj(out).unsqueeze(1)
		tokens2 = tokens + ctx
		attn2, _ = self.attn_out(self.norm_out(tokens2), self.norm_out(tokens2), self.norm_out(tokens2))
		tokens2 = tokens2 + attn2
		tokens2 = tokens2 + self.ffn_post(self.norm_out(tokens2))
		alpha = torch.sigmoid(self.fusion_alpha)
		return x0 + alpha * tokens2, logprobs


def _build_metric(task_type: str, out_dim: int, device: torch.device):
	try:
		from torchmetrics import AUROC, Accuracy, MeanSquaredError
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError("torchmetrics is required") from e

	is_classification = task_type != "regression"
	is_binary = bool(is_classification and out_dim == 2 and task_type == "binclass")
	if is_binary:
		metric = AUROC(task="binary")
		name = "AUC"
	elif is_classification:
		metric = Accuracy(task="multiclass", num_classes=int(out_dim))
		name = "Acc"
	else:
		metric = MeanSquaredError()
		name = "RMSE"
	return metric.to(device), name, is_classification, is_binary


def row_gnn_after_start_fn(train_df, val_df, test_df, config: dict, task_type: str):
	"""Offline supervised row embedding extraction (train-only loss, val early stop)."""
	device = resolve_device(config)
	feature_cols = [c for c in train_df.columns if c != "target"]
	X_train_np, X_val_np, X_test_np, _ = _factorize_object_columns(train_df, val_df, test_df, feature_cols)
	X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
	X_val = torch.tensor(X_val_np, dtype=torch.float32, device=device)
	X_test = torch.tensor(X_test_np, dtype=torch.float32, device=device)

	if task_type in {"binclass", "multiclass"}:
		out_dim = int(len(np.unique(train_df["target"].values)))
		if task_type == "binclass" and out_dim != 2:
			out_dim = 2
		y_train = torch.tensor(train_df["target"].values, dtype=torch.long, device=device)
		y_val = torch.tensor(val_df["target"].values, dtype=torch.long, device=device)
	else:
		out_dim = 1
		y_train = torch.tensor(train_df["target"].values, dtype=torch.float32, device=device)
		y_val = torch.tensor(val_df["target"].values, dtype=torch.float32, device=device)

	DGM_d_local = _import_dgm_d()

	class _Id(nn.Module):
		def forward(self, x, A=None):
			return x

	cfg = _read_row_gnn_config(config)
	dgm = DGM_d_local(_Id(), k=int(min(cfg.dgm_k, max(1, int(X_train.size(0)) - 1))), distance=cfg.dgm_distance).to(device)
	gate_thr = None
	if cfg.gate != "none":
		gate_thr = nn.Parameter(torch.tensor(cfg.gate_threshold_init, device=device))

	gnn = SimpleGCN(int(X_train.size(1)), cfg.gnn_hidden, cfg.gnn_hidden, num_layers=max(2, cfg.gnn_layers)).to(device)
	head = nn.Linear(cfg.gnn_hidden, out_dim).to(device)
	params = list(dgm.parameters()) + list(gnn.parameters()) + list(head.parameters())
	if gate_thr is not None:
		params += [gate_thr]
	opt = torch.optim.Adam(params, lr=float(config.get("gnn_lr", 1e-3)))

	def _forward(x: torch.Tensor):
		B = int(x.size(0))
		x_std = _standardize(x, dim=0)
		x_dgm, ei_dgm, lp = dgm(x_std.unsqueeze(0), A=None)
		x_dgm = x_dgm.squeeze(0)
		ei, ew = _dgm_gate_and_prune_edges(
			ei_dgm,
			lp,
			B,
			gate=cfg.gate,
			gate_sharpness=cfg.gate_sharpness,
			gate_threshold_param=gate_thr,
			is_training=bool(dgm.training),
			eval_prune=cfg.eval_prune,
			eval_prune_threshold=cfg.eval_prune_threshold,
		)
		emb = gnn(x_dgm, ei, edge_weight=ew)
		logits = head(emb)
		return logits, emb, lp

	epochs = int(config.get("gnn_epochs", config.get("epochs", 200)))
	patience = int(config.get("gnn_patience", 10))
	loss_threshold = float(config.get("gnn_loss_threshold", 1e-4))
	best_val = float("inf")
	best_state = None
	pat = 0
	stop_epoch = 0
	for epoch in range(1, epochs + 1):
		dgm.train(); gnn.train(); head.train()
		opt.zero_grad()
		logits, _, lp = _forward(X_train)
		if task_type in {"binclass", "multiclass"}:
			loss = F.cross_entropy(logits, y_train)
		else:
			loss = F.mse_loss(logits.squeeze(-1), y_train)
		if lp is not None:
			loss = loss + (-lp.mean()) * float(cfg.dgm_reg)
		loss.backward(); opt.step()

		dgm.eval(); gnn.eval(); head.eval()
		with torch.no_grad():
			logits_v, _, _ = _forward(X_val)
			if task_type in {"binclass", "multiclass"}:
				val_loss = F.cross_entropy(logits_v, y_val)
			else:
				val_loss = F.mse_loss(logits_v.squeeze(-1), y_val)
		v = float(val_loss.item())
		if v < best_val - loss_threshold:
			best_val = v
			pat = 0
			best_state = {
				"dgm": dgm.state_dict(),
				"gnn": gnn.state_dict(),
				"head": head.state_dict(),
				"gate_thr": gate_thr.detach().clone() if gate_thr is not None else None,
			}
		else:
			pat += 1
			if pat >= patience:
				stop_epoch = epoch
				break

	if stop_epoch == 0:
		stop_epoch = epochs
	if best_state is not None:
		dgm.load_state_dict(best_state["dgm"])
		gnn.load_state_dict(best_state["gnn"])
		head.load_state_dict(best_state["head"])
		if best_state["gate_thr"] is not None and gate_thr is not None:
			with torch.no_grad():
				gate_thr.copy_(best_state["gate_thr"])

	dgm.eval(); gnn.eval(); head.eval()
	with torch.no_grad():
		_, e_tr, _ = _forward(X_train)
		_, e_va, _ = _forward(X_val)
		_, e_te, _ = _forward(X_test)

	cols = [f"N_feature_emb_{i + 1}" for i in range(int(e_tr.size(1)))]
	train_out = pd.DataFrame(e_tr.detach().cpu().numpy(), columns=cols, index=train_df.index)
	val_out = pd.DataFrame(e_va.detach().cpu().numpy(), columns=cols, index=val_df.index)
	test_out = pd.DataFrame(e_te.detach().cpu().numpy(), columns=cols, index=test_df.index)
	train_out["target"] = train_df["target"].values
	val_out["target"] = val_df["target"].values
	test_out["target"] = test_df["target"].values
	return train_out, val_out, test_out, int(stop_epoch)


def row_gnn_after_materialize_fn(material_outputs: dict, config: dict, task_type: str):
	"""Offline row embedding extraction after materialization; replaces X_* tensors."""
	device = material_outputs["device"]
	X_train = material_outputs["X_train"]
	X_val = material_outputs["X_val"]
	X_test = material_outputs["X_test"]
	y_train = material_outputs["y_train"]
	y_val = material_outputs["y_val"]

	if task_type in {"binclass", "multiclass"}:
		out_dim = int(material_outputs.get("n_classes") or int(torch.unique(y_train).numel()))
	else:
		out_dim = 1

	DGM_d_local = _import_dgm_d()

	class _Id(nn.Module):
		def forward(self, x, A=None):
			return x

	cfg = _read_row_gnn_config(config)
	dgm = DGM_d_local(_Id(), k=int(min(cfg.dgm_k, max(1, int(X_train.size(0)) - 1))), distance=cfg.dgm_distance).to(device)
	gate_thr = None
	if cfg.gate != "none":
		gate_thr = nn.Parameter(torch.tensor(cfg.gate_threshold_init, device=device))

	gnn = SimpleGCN(int(X_train.size(1)), cfg.gnn_hidden, cfg.gnn_hidden, num_layers=max(2, cfg.gnn_layers)).to(device)
	head = nn.Linear(cfg.gnn_hidden, out_dim).to(device)
	params = list(dgm.parameters()) + list(gnn.parameters()) + list(head.parameters())
	if gate_thr is not None:
		params += [gate_thr]
	opt = torch.optim.Adam(params, lr=float(config.get("gnn_lr", 1e-3)))

	def _forward(x: torch.Tensor):
		B = int(x.size(0))
		x_std = _standardize(x, dim=0)
		x_dgm, ei_dgm, lp = dgm(x_std.unsqueeze(0), A=None)
		x_dgm = x_dgm.squeeze(0)
		ei, ew = _dgm_gate_and_prune_edges(
			ei_dgm,
			lp,
			B,
			gate=cfg.gate,
			gate_sharpness=cfg.gate_sharpness,
			gate_threshold_param=gate_thr,
			is_training=bool(dgm.training),
			eval_prune=cfg.eval_prune,
			eval_prune_threshold=cfg.eval_prune_threshold,
		)
		emb = gnn(x_dgm, ei, edge_weight=ew)
		logits = head(emb)
		return logits, emb, lp

	epochs = int(config.get("gnn_epochs", config.get("epochs", 200)))
	patience = int(config.get("gnn_patience", 10))
	loss_threshold = float(config.get("gnn_loss_threshold", 1e-4))
	best_val = float("inf")
	best_state = None
	pat = 0
	stop_epoch = 0
	for epoch in range(1, epochs + 1):
		dgm.train(); gnn.train(); head.train()
		opt.zero_grad()
		logits, _, lp = _forward(X_train)
		if task_type in {"binclass", "multiclass"}:
			loss = F.cross_entropy(logits, y_train)
		else:
			loss = F.mse_loss(logits.squeeze(-1), y_train)
		if lp is not None:
			loss = loss + (-lp.mean()) * float(cfg.dgm_reg)
		loss.backward(); opt.step()

		dgm.eval(); gnn.eval(); head.eval()
		with torch.no_grad():
			logits_v, _, _ = _forward(X_val)
			if task_type in {"binclass", "multiclass"}:
				val_loss = F.cross_entropy(logits_v, y_val)
			else:
				val_loss = F.mse_loss(logits_v.squeeze(-1), y_val)
		v = float(val_loss.item())
		if v < best_val - loss_threshold:
			best_val = v
			pat = 0
			best_state = {
				"dgm": dgm.state_dict(),
				"gnn": gnn.state_dict(),
				"head": head.state_dict(),
				"gate_thr": gate_thr.detach().clone() if gate_thr is not None else None,
			}
		else:
			pat += 1
			if pat >= patience:
				stop_epoch = epoch
				break

	if stop_epoch == 0:
		stop_epoch = epochs
	if best_state is not None:
		dgm.load_state_dict(best_state["dgm"])
		gnn.load_state_dict(best_state["gnn"])
		head.load_state_dict(best_state["head"])
		if best_state["gate_thr"] is not None and gate_thr is not None:
			with torch.no_grad():
				gate_thr.copy_(best_state["gate_thr"])

	dgm.eval(); gnn.eval(); head.eval()
	with torch.no_grad():
		_, e_tr, _ = _forward(X_train)
		_, e_va, _ = _forward(X_val)
		_, e_te, _ = _forward(X_test)

	material_outputs["X_train"] = e_tr.detach()
	material_outputs["X_val"] = e_va.detach()
	material_outputs["X_test"] = e_te.detach()
	material_outputs["n_num_features"] = int(e_tr.size(1))

	# Recompute bins for embeddings.
	try:
		import rtdl_num_embeddings
		from rtdl_num_embeddings import PiecewiseLinearEmbeddings

		n_bins = max(2, min(48, int(material_outputs["X_train"].shape[0]) - 1))
		material_outputs["num_embeddings"] = PiecewiseLinearEmbeddings(
			rtdl_num_embeddings.compute_bins(material_outputs["X_train"], n_bins=n_bins),
			d_embedding=int(config.get("tabm_d_embedding", 16)),
			activation=False,
			version="B",
		).to(device)
	except Exception as e:
		print(f"[MATERIALIZE-GNN-DGM][WARNING] Failed to recompute bins: {e}")

	return material_outputs, int(stop_epoch)


def tabm_core_fn(material_outputs: dict, config: dict, task_type: str, *, gnn_stage: str, graphify: str):
	device = material_outputs["device"]
	stage = "none" if gnn_stage is None else str(gnn_stage).lower()
	graphify = str(graphify or "row").lower()

	# Load external tabm package explicitly via importlib to avoid circular import
	# (local file is also named tabm.py, causing resolution ambiguity)
	modelcomparison_root = Path(__file__).resolve().parents[3]
	tabm_module_path = modelcomparison_root / "tabm" / "tabm.py"
	if not tabm_module_path.exists():
		raise FileNotFoundError(f"Cannot find external tabm module at {tabm_module_path}")
	spec = importlib.util.spec_from_file_location("_external_tabm", str(tabm_module_path))
	_external_tabm = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(_external_tabm)
	EnsembleView = _external_tabm.EnsembleView
	LinearEnsemble = _external_tabm.LinearEnsemble
	MLPBackboneBatchEnsemble = _external_tabm.MLPBackboneBatchEnsemble

	X_train = material_outputs["X_train"]
	y_train = material_outputs["y_train"]
	X_val = material_outputs["X_val"]
	y_val = material_outputs["y_val"]
	X_test = material_outputs["X_test"]
	y_test = material_outputs["y_test"]
	n_classes = material_outputs.get("n_classes")
	label_stats = material_outputs.get("label_stats")
	num_embeddings = material_outputs["num_embeddings"]

	# Hyperparameters
	k = int(config.get("tabm_k", 32))
	n_blocks = int(config.get("tabm_n_blocks", 2))
	d_block = int(config.get("tabm_d_block", 512))
	dropout = float(config.get("tabm_dropout", 0.1))

	with torch.no_grad():
		tokens = num_embeddings(X_train[:1])
		F_ = int(tokens.size(1))
		E_ = int(tokens.size(2))
		d_in = int(F_ * E_)

	ensemble_view = EnsembleView(k=k)
	out_dim = 1 if task_type == "regression" else int(n_classes or int(torch.unique(y_train).numel()))

	# Optionally preserve per-feature channels through the backbone.
	# Default: enabled to provide topology-aware columnwise/decoding behavior.
	preserve_feature_dims = bool(config.get("tabm_preserve_feature_dims", True)) or os.environ.get("TABM_PRESERVE_FEATURE_DIMS", "") == "1"

	if preserve_feature_dims:
		# Per-feature backbone: input tokens shape [B,F,E_] -> output [B,F,d_block]
		backbone = PerFeatureBackbone(in_dim=E_, d_block=d_block, n_blocks=n_blocks, dropout=dropout).to(device)
		# When using row_graph with per-feature backbone, we need a projection layer to
		# convert from [B, k, F*d_block] to [B, k, d_block] before row_graph input.
		# This layer is separate from the final output_layer.
		projection_for_row_graph = None
		if graphify == "row" and stage in {"columnwise", "decoding"}:
			projection_for_row_graph = LinearEnsemble(F_ * d_block, d_block, k=k).to(device)
		# Final output_layer: None if row_graph produces logits, else project to out_dim
		output_layer = None if (stage == "decoding" and graphify == "row") else LinearEnsemble(F_ * d_block, out_dim, k=k).to(device)
	else:
		backbone = MLPBackboneBatchEnsemble(
			d_in=d_in,
			n_blocks=n_blocks,
			d_block=d_block,
			dropout=dropout,
			k=k,
			tabm_init=True,
			scaling_init="normal",
			start_scaling_init_chunks=None,
		).to(device)
		# If decoding is performed via the row graph, the `row_graph` will
		# produce logits itself so we skip the local `output_layer`. But when
		# decoding with feature-graph (no row_graph), we need an explicit
		# `output_layer` to project backbone features to `out_dim` logits.
		projection_for_row_graph = None
		output_layer = None if (stage == "decoding" and graphify == "row") else LinearEnsemble(d_block, out_dim, k=k).to(device)

	metric, metric_name, is_cls, is_bin = _build_metric(task_type, out_dim, device)

	# Feature topology mixing (active only in core stages)
	topology_mixer = None
	if graphify == "feature" and stage in {"encoding", "columnwise", "decoding"}:
		topology = material_outputs.get("feature_topology")
		if topology is None:
			feature_names = list(material_outputs.get("feature_names", []))
			init_mode = str(config.get("feature_topology_init", "chain"))
			topology = _init_feature_topology_prior(feature_names, init_mode=init_mode)
			material_outputs["feature_topology"] = topology
			config["feature_topology_meta"] = {
				"init_mode": topology["init_mode"],
				"num_features": topology["num_features"],
				"feature_names": topology["feature_names"],
				"edge_index": topology["edge_index"].tolist(),
			}
			print(
				f"[CORE] Feature topology prior initialized in-core (inactive earlier). "
				f"init_mode={topology['init_mode']}, num_features={topology['num_features']}"
			)
		allowed = topology["adjacency_mask"].to(device=device)
		if int(allowed.size(0)) != F_:
			raise ValueError("feature_topology mask size mismatch")
		topology_mixer = FeatureTopologyMaskedMixing(allowed).to(device)

	# Row graphifier (active only in core stages)
	row_graph = None
	if graphify == "row" and stage in {"encoding", "columnwise", "decoding"}:
		cfg = _read_row_gnn_config(config)
		graph_dim = d_in if stage == "encoding" else d_block
		row_graph = RowGraphifier(graph_dim, out_dim, stage=stage, cfg=cfg, device=device).to(device)
		dgm_reg_scale = float(cfg.dgm_reg)
	else:
		dgm_reg_scale = 0.0

	def _forward_features(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
		# tokens: [B,F,E]
		tok = num_embeddings(X)
		if topology_mixer is not None:
			tok = topology_mixer(tok)

		dgm_reg = None

		if preserve_feature_dims:
			# Per-feature backbone path: backbone(tok) -> [B,F,d_block]
			per_feat = backbone(tok)
			# columnwise: allow applying topology mixer after backbone
			if stage == "columnwise" and topology_mixer is not None:
				per_feat = topology_mixer(per_feat)
			# Flatten per-feature embeddings and convert to ensemble view
			z = per_feat.flatten(1, -1)  # [B, F * d_block]
			x = ensemble_view(z)  # [B, k, F * d_block]
			
			# For decoding with row_graph, project from [B,k,F*d_block] to [B,k,d_block]
			# before row_graph processes it to produce logits.
			if projection_for_row_graph is not None and stage == "decoding":
				x_proj = projection_for_row_graph(x)
				logits, lp = row_graph(x_proj)
				if lp is not None:
					dgm_reg = -lp.mean()
				return logits, dgm_reg
		
			return x, dgm_reg

		# Original flattened backbone path
		x = ensemble_view(tok.flatten(1, -1))

		if row_graph is not None and stage == "encoding":
			x, lp = row_graph(x)
			if lp is not None:
				dgm_reg = -lp.mean()

		x = backbone(x)
		if row_graph is not None and stage == "columnwise":
			x, lp = row_graph(x)
			if lp is not None:
				dgm_reg = -lp.mean()

		if row_graph is not None and stage == "decoding":
			logits, lp = row_graph(x)
			if lp is not None:
				dgm_reg = -lp.mean()
			return logits, dgm_reg

		return x, dgm_reg

	def _loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		if task_type == "regression":
			if logits.dim() == 3:
				return F.mse_loss(logits.flatten(0, 1).squeeze(-1), y.repeat_interleave(k))
			return F.mse_loss(logits.squeeze(-1), y)
		if logits.dim() == 3:
			return F.cross_entropy(logits.flatten(0, 1), y.repeat_interleave(k))
		return F.cross_entropy(logits, y)

	def _eval_split(X: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
		metric.reset()
		with torch.no_grad():
			z, _ = _forward_features(X)
			logits = z if output_layer is None else output_layer(z)
			loss = float(_loss(logits, y).item())
			if task_type == "regression":
				pred = (logits.mean(dim=1) if logits.dim() == 3 else logits).squeeze(-1)
				if label_stats is not None:
					pred = pred * float(label_stats["std"]) + float(label_stats["mean"])
					yd = y * float(label_stats["std"]) + float(label_stats["mean"])
				else:
					yd = y
				metric.update(pred.view(-1), yd.view(-1))
				mse = float(metric.compute().item())
				return loss, float(math.sqrt(max(0.0, mse)))
			prob = torch.softmax(logits, dim=-1)
			prob = prob.mean(dim=1) if prob.dim() == 3 else prob
			if is_bin:
				metric.update(prob[:, 1], y)
			else:
				metric.update(prob.argmax(dim=-1), y)
			return loss, float(metric.compute().item())

	params = list(num_embeddings.parameters()) + list(ensemble_view.parameters()) + list(backbone.parameters())
	if output_layer is not None:
		params += list(output_layer.parameters())
	if topology_mixer is not None:
		params += list(topology_mixer.parameters())
	if row_graph is not None:
		params += list(row_graph.parameters())

	def _get_state_dict():
		return {
			"num_embeddings": num_embeddings.state_dict(),
			"ensemble_view": ensemble_view.state_dict(),
			"backbone": backbone.state_dict(),
			"output_layer": output_layer.state_dict() if output_layer is not None else None,
			"topology_mixer": topology_mixer.state_dict() if topology_mixer is not None else None,
			"row_graph": row_graph.state_dict() if row_graph is not None else None,
		}

	def _load_state_dict(state: dict):
		num_embeddings.load_state_dict(state["num_embeddings"])
		ensemble_view.load_state_dict(state["ensemble_view"])
		backbone.load_state_dict(state["backbone"])
		if output_layer is not None and state.get("output_layer") is not None:
			output_layer.load_state_dict(state["output_layer"])
		if topology_mixer is not None and state.get("topology_mixer") is not None:
			topology_mixer.load_state_dict(state["topology_mixer"])
		if row_graph is not None and state.get("row_graph") is not None:
			row_graph.load_state_dict(state["row_graph"])

	optimizer = torch.optim.AdamW(
		params,
		lr=float(config.get("lr", 0.002)),
		weight_decay=float(config.get("weight_decay", 3e-4)),
	)

	batch_size = int(config.get("batch_size", 256))
	epochs = int(config.get("epochs", 200))
	patience = int(config.get("patience", 10))
	loss_threshold = float(config.get("loss_threshold", 1e-6))

	best_val = float("inf")
	best_val_metric = float("-inf") if task_type != "regression" else float("inf")
	best_test_metric = float("-inf") if task_type != "regression" else float("inf")
	pat = 0
	best_state = None
	stop_epoch = 0

	for epoch in range(1, epochs + 1):
		for m in [num_embeddings, ensemble_view, backbone]:
			m.train()
		if output_layer is not None:
			output_layer.train()
		if topology_mixer is not None:
			topology_mixer.train()
		if row_graph is not None:
			row_graph.train()

		idx = torch.randperm(int(X_train.size(0)), device=device)
		epoch_loss = 0.0
		nb = 0
		for i in range(0, int(X_train.size(0)), batch_size):
			b = idx[i : i + batch_size]
			optimizer.zero_grad()
			z, dgm_reg = _forward_features(X_train[b])
			logits = z if output_layer is None else output_layer(z)
			loss = _loss(logits, y_train[b])
			if dgm_reg is not None:
				loss = loss + dgm_reg * dgm_reg_scale
			loss.backward()
			torch.nn.utils.clip_grad_norm_(params, 1.0)
			optimizer.step()
			epoch_loss += float(loss.item())
			nb += 1

		for m in [num_embeddings, ensemble_view, backbone]:
			m.eval()
		if output_layer is not None:
			output_layer.eval()
		if topology_mixer is not None:
			topology_mixer.eval()
		if row_graph is not None:
			row_graph.eval()

		val_loss, val_metric = _eval_split(X_val, y_val)
		_, test_metric = _eval_split(X_test, y_test)
		improved = val_loss < best_val - loss_threshold
		if improved:
			best_val = val_loss
			best_val_metric = val_metric
			best_test_metric = test_metric
			pat = 0
			best_state = _get_state_dict()
		else:
			pat += 1
			if pat >= patience:
				stop_epoch = epoch
				break

		if epoch == 1 or epoch % 10 == 0:
			print(
				f"[TABM][CORE] Epoch {epoch}/{epochs} "
				f"Train Loss {epoch_loss/max(1,nb):.4f} Val Loss {val_loss:.4f} Val {metric_name} {val_metric:.4f}"
			)

	if stop_epoch == 0:
		stop_epoch = epochs

	if best_state is not None:
		_load_state_dict(best_state)

	return {
		"gnn_early_stop_epochs": int(material_outputs.get("gnn_early_stop_epochs", 0)),
		"best_val_loss": float(best_val),
		"best_val_metric": float(best_val_metric),
		"final_metric": float(best_val_metric),
		"best_test_metric": float(best_test_metric),
		"metric": metric_name,
		"early_stop_epochs": int(stop_epoch),
		"gnn_stage": stage,
		"graphify": graphify,
	}


def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
	print("TabM - staged execution")
	stage = "none" if gnn_stage is None else str(gnn_stage).lower()
	if stage == "materialization":
		stage = "materialize"
	print(f"gnn_stage: {stage}")

	task_type = dataset_results.get("info", {}).get("task_type", "binclass")
	graphify = str(config.get("graphify", "row")).lower()

	try:
		train_df, val_df, test_df = start_fn(train_df, val_df, test_df)
		gnn_early_stop_epochs = 0

		if graphify == "row":
			if stage == "start":
				train_df, val_df, test_df, gnn_early_stop_epochs = row_gnn_after_start_fn(
					train_df, val_df, test_df, config, task_type
				)
			material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
			material_outputs["gnn_early_stop_epochs"] = int(gnn_early_stop_epochs)
			if stage == "materialize":
				material_outputs, gnn_early_stop_epochs = row_gnn_after_materialize_fn(
					material_outputs, config, task_type
				)
				material_outputs["gnn_early_stop_epochs"] = int(gnn_early_stop_epochs)
			return tabm_core_fn(material_outputs, config, task_type, gnn_stage=stage, graphify="row")

		if graphify == "feature":
			if stage == "start":
				train_df, val_df, test_df, gnn_early_stop_epochs = feature_gnn_after_start_fn(
					train_df, val_df, test_df, config, task_type
				)
			material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
			material_outputs["gnn_early_stop_epochs"] = int(gnn_early_stop_epochs)
			if stage == "materialize":
				material_outputs = feature_gnn_after_materialize_fn(
					material_outputs,
					config,
					dataset_results.get("dataset", "unknown"),
					task_type,
				)
			return tabm_core_fn(material_outputs, config, task_type, gnn_stage=stage, graphify="feature")

		raise ValueError(f"Unknown graphify mode: {graphify}")
	except Exception as e:
		return {"gnn_stage": stage, "graphify": graphify, "error": str(e)}


#  small+binclass
#  python main.py --models tabm --dataset kaggle_Audit_Data --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  small+regression
#  python main.py --models tabm --dataset openml_The_Office_Dataset --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+binclass
#  python main.py --models tabm --dataset credit --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+multiclass
#  python main.py --models tabm --dataset eye --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  python main.py --models tabm --dataset helena --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+regression
#  python main.py --models tabm --dataset house --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
