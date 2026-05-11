from __future__ import annotations

"""VIME backbone with staged execution + row/feature graphification.

This file is an English-only refactor ported and adapted from:
  - ModelComparison/TaBLEau/models/custom/vime.py

Key alignment rules (matching ExcelFormer/FTTransformer ports in this repo):
  - Row graphification at `start`/`materialize` is *offline* and uses raw row
	feature vectors as node features (no self-attention/pooling projection).
  - Feature graphification at `start`/`materialize` is *inactive* (only
	initializes a schema-level topology prior; token-level mixing starts later).

The project runner calls:
  main(train_df, val_df, test_df, dataset_results, config, gnn_stage)
"""

import copy
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

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
	from sklearn.linear_model import LogisticRegression, Ridge
	from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
except ModuleNotFoundError as e:  # pragma: no cover
	LogisticRegression = None
	Ridge = None
	accuracy_score = None
	mean_squared_error = None
	roc_auc_score = None
	_SKLEARN_IMPORT_ERROR = e

try:
	from torch_geometric.nn import GCNConv
except ModuleNotFoundError as e:  # pragma: no cover
	GCNConv = None
	_PYG_IMPORT_ERROR = e


def _import_dgm_d():
	"""Import DGM_d with a robust path fallback for this workspace."""

	try:
		from DGMlib.layers import DGM_d as _DGM_d

		return _DGM_d
	except Exception as first_exc:
		# Fallback: search upward for a sibling `DGM_pytorch/` directory.
		# Workspace layout (common):
		#   ModelComparison/
		#     DGM_pytorch/
		#     tabular-relational-benchmark/
		this_file = Path(__file__).resolve()
		dgm_path = None
		for parent in this_file.parents:
			cand = parent / "DGM_pytorch"
			if cand.exists():
				dgm_path = cand
				break
		if dgm_path is not None:
			sys.path.insert(0, str(dgm_path))
		try:
			from DGMlib.layers import DGM_d as _DGM_d

			return _DGM_d
		except Exception as second_exc:
			raise second_exc from first_exc


def resolve_device(config: dict) -> torch.device:
	"""Resolve the torch device from `config`.

	Conventions:
	- gpu=-1 forces CPU.
	- Otherwise uses CUDA if available.
	"""

	gpu_id = None
	try:
		gpu_id = config.get("gpu", None)
	except Exception:
		gpu_id = None

	if gpu_id is not None:
		try:
			if int(gpu_id) < 0:
				print(f"[DEVICE] Using cpu (from config['gpu']={gpu_id})")
				return torch.device("cpu")
		except Exception:
			pass

	if torch.cuda.is_available():
		if gpu_id is not None:
			try:
				device = torch.device(f"cuda:{int(gpu_id)}")
				print(f"[DEVICE] Using cuda:{int(gpu_id)} (from config['gpu']={gpu_id})")
				return device
			except Exception:
				print(f"[DEVICE] Using cuda (invalid config['gpu']={gpu_id})")
				return torch.device("cuda")

		print("[DEVICE] Using cuda (default)")
		return torch.device("cuda")

	print("[DEVICE] Using cpu (CUDA not available)")
	return torch.device("cpu")


class SimpleGCN(torch.nn.Module):
	"""A minimal GCN used by row-level graphification/injection."""

	def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
		super().__init__()
		if GCNConv is None:  # pragma: no cover
			raise ModuleNotFoundError(
				"torch_geometric is required for GNN injection. Install torch-geometric to use this feature."
			) from _PYG_IMPORT_ERROR
		self.layers = torch.nn.ModuleList()
		dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
		for i in range(len(dims) - 1):
			self.layers.append(GCNConv(dims[i], dims[i + 1]))

	def forward(
		self,
		x: torch.Tensor,
		edge_index: torch.Tensor,
		edge_weight: torch.Tensor | None = None,
	) -> torch.Tensor:
		for i, layer in enumerate(self.layers):
			x = layer(x, edge_index, edge_weight=edge_weight)
			if i < len(self.layers) - 1:
				x = torch.relu(x)
		return x


def _standardize(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
	mean = x.mean(dim=dim, keepdim=True)
	std = x.std(dim=dim, keepdim=True).clamp_min(eps)
	return (x - mean) / std


def _symmetrize_and_self_loop(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
	device = edge_index.device
	rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
	self_loops = torch.arange(num_nodes, device=device)
	self_edge = torch.stack([self_loops, self_loops], dim=0)
	ei = torch.cat([edge_index, rev, self_edge], dim=1)

	edge_ids = ei[0] * num_nodes + ei[1]
	unique_ids = torch.unique(edge_ids, sorted=False)
	ei0 = unique_ids // num_nodes
	ei1 = unique_ids % num_nodes
	return torch.stack([ei0, ei1], dim=0)


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
	"""Convert DGM_d top-k edges into an undirected self-looped graph.

	If `gate == 'none'`, returns an unweighted graph (edge_weight=None).
	Otherwise returns `edge_weight` in [0,1] via a sigmoid gate on logprobs.
	Optionally prunes edges at eval time using `eval_prune_threshold`.
	"""

	if edge_index_dgm.numel() == 0:
		return _symmetrize_and_self_loop(edge_index_dgm, num_nodes), None

	gate = str(gate or "none").lower()
	if gate not in {"none", "sigmoid"}:
		raise ValueError(f"Unknown gnn_dgm_gate={gate!r}. Expected 'none' or 'sigmoid'.")

	edge_weight = None
	if gate != "none":
		if logprobs_dgm is None:
			raise ValueError("logprobs_dgm is required when gnn_dgm_gate is enabled.")

		lp = logprobs_dgm
		if lp.dim() == 3:
			lp = lp.reshape(-1)
		elif lp.dim() == 2:
			lp = lp.reshape(-1)
		elif lp.dim() == 1:
			pass
		else:
			raise ValueError(f"Unexpected logprobs_dgm shape: {tuple(logprobs_dgm.shape)}")

		if lp.numel() != edge_index_dgm.shape[1]:
			raise ValueError(
				f"logprobs_dgm numel ({lp.numel()}) does not match edges ({edge_index_dgm.shape[1]})."
			)

		thr = gate_threshold_param
		if thr is None:
			thr = torch.tensor(0.0, device=lp.device, dtype=lp.dtype)

		s = float(gate_sharpness)
		edge_weight = torch.sigmoid((lp - thr) * s).to(dtype=torch.float32)

	rev = torch.stack([edge_index_dgm[1], edge_index_dgm[0]], dim=0)
	self_loops = torch.arange(num_nodes, device=edge_index_dgm.device)
	self_edge = torch.stack([self_loops, self_loops], dim=0)
	ei = torch.cat([edge_index_dgm, rev, self_edge], dim=1)

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

	try:
		from torch_geometric.utils import coalesce

		ei, ew = coalesce(ei, ew, num_nodes, num_nodes, reduce="max")
	except Exception:
		edge_ids = ei[0] * num_nodes + ei[1]
		unique_ids, inv = torch.unique(edge_ids, sorted=False, return_inverse=True)
		ei0 = unique_ids // num_nodes
		ei1 = unique_ids % num_nodes
		ei = torch.stack([ei0, ei1], dim=0)
		if ew is not None:
			out = torch.zeros(unique_ids.shape[0], device=ew.device, dtype=ew.dtype)
			if hasattr(out, "scatter_reduce_"):
				out.scatter_reduce_(0, inv, ew, reduce="amax", include_self=False)
				ew = out
			else:  # pragma: no cover
				for idx in range(inv.numel()):
					j = int(inv[idx].item())
					out[j] = torch.maximum(out[j], ew[idx])
				ew = out

	return ei, ew


def _init_feature_topology_prior(feature_names, init_mode: str = "chain") -> dict:
	feature_names = list(feature_names)
	n = int(len(feature_names))
	if n <= 0:
		raise ValueError("Cannot initialize feature topology with zero features.")

	init_mode = str(init_mode).lower()
	if init_mode == "full":
		adj = torch.ones((n, n), dtype=torch.bool, device="cpu")
	elif init_mode == "identity":
		adj = torch.eye(n, dtype=torch.bool, device="cpu")
	elif init_mode == "chain":
		adj = torch.eye(n, dtype=torch.bool, device="cpu")
		if n >= 2:
			idx = torch.arange(n - 1)
			adj[idx, idx + 1] = True
			adj[idx + 1, idx] = True
	else:
		raise ValueError(
			"Unknown feature_topology_init mode: "
			f"{init_mode!r}. Expected one of: 'chain', 'full', 'identity'."
		)

	edge_index = adj.nonzero(as_tuple=False).t().contiguous()
	return {
		"init_mode": init_mode,
		"num_features": n,
		"feature_names": feature_names,
		"adjacency_mask": adj,
		"edge_index": edge_index,
	}


def _feature_topology_to_meta(topology: dict) -> dict:
	edge_index = topology.get("edge_index")
	if isinstance(edge_index, torch.Tensor):
		edge_index = edge_index.detach().cpu().tolist()

	return {
		"init_mode": topology.get("init_mode"),
		"num_features": int(topology.get("num_features", 0)),
		"feature_names": list(topology.get("feature_names", [])),
		"edge_index": edge_index,
	}


class FeatureTopologyMaskedMixing(torch.nn.Module):
	"""Topology-masked feature mixing shared across samples."""

	def __init__(
		self,
		allowed_mask: torch.Tensor,
		*,
		init_strength: float = 0.0,
		symmetric: bool = True,
		force_self_loops: bool = True,
	):
		super().__init__()
		if allowed_mask.dim() != 2 or allowed_mask.shape[0] != allowed_mask.shape[1]:
			raise ValueError(
				"allowed_mask must be a square [F,F] tensor; "
				f"got shape={tuple(allowed_mask.shape)}"
			)
		allowed_mask = allowed_mask.to(dtype=torch.bool)
		self.register_buffer("allowed_mask", allowed_mask)
		self.symmetric = bool(symmetric)
		self.force_self_loops = bool(force_self_loops)

		f = int(allowed_mask.shape[0])
		logits = torch.full(
			(f, f),
			float(init_strength),
			dtype=torch.float32,
			device=allowed_mask.device,
		)
		logits = logits.masked_fill(~allowed_mask, -20.0)
		if self.force_self_loops:
			eye = torch.eye(f, dtype=torch.bool, device=allowed_mask.device)
			logits = logits.masked_fill(eye, 5.0)

		self.logits = torch.nn.Parameter(logits)

	def _normalized_weights(self) -> torch.Tensor:
		w = torch.sigmoid(self.logits)
		w = w * self.allowed_mask.to(dtype=w.dtype)

		if self.force_self_loops:
			f = w.shape[0]
			eye = torch.eye(f, device=w.device, dtype=w.dtype)
			w = w * (1.0 - eye) + eye

		if self.symmetric:
			w = 0.5 * (w + w.t())

		row_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
		return w / row_sum

	def forward(self, tokens: torch.Tensor) -> torch.Tensor:
		if tokens.dim() != 3:
			raise ValueError(f"Expected tokens [B,F,C], got shape={tuple(tokens.shape)}")
		p = self._normalized_weights()
		return torch.einsum("ij,bjd->bid", p, tokens)


def _split_xy_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError("pandas is required for VIME.") from _PANDAS_IMPORT_ERROR
	if "target" not in df.columns:
		raise ValueError("Expected a 'target' column in the input DataFrame.")
	feature_cols = [c for c in df.columns if c != "target"]
	X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
	y = df["target"].to_numpy(copy=True)
	return X, y


def _task_type_from_info(dataset_results: dict) -> str:
	return str(dataset_results.get("info", {}).get("task_type", "binclass")).lower()


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	mean = X.mean(axis=0, keepdims=True)
	std = X.std(axis=0, keepdims=True)
	std = np.maximum(std, 1e-6)
	Xn = (X - mean) / std
	return Xn.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
	return ((X - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def mask_generator(p_m: float, x: torch.Tensor) -> torch.Tensor:
	return (torch.rand_like(x) < p_m).float()


def pretext_generator(m: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
	"""VIME pretext: shuffle rows for masked entries."""

	x_bar = x[torch.randperm(x.size(0))]
	x_tilde = x * (1 - m) + x_bar * m
	m_new = (x != x_tilde).float()
	return m_new, x_tilde


class VIMEEncoder(nn.Module):
	"""Minimal VIME encoder: (net -> mask head + recon head)."""

	def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
		super().__init__()
		h = int(hidden_dim)
		d = float(dropout)
		self.net = nn.Sequential(
			nn.Linear(in_dim, h),
			nn.ReLU(),
			nn.Dropout(d),
			nn.Linear(h, h),
			nn.ReLU(),
		)
		self.mask_head = nn.Linear(h, in_dim)
		self.recon_head = nn.Linear(h, in_dim)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		h = self.net(x)
		mask_logits = self.mask_head(h)
		recon = self.recon_head(h)
		return h, mask_logits, recon


def _evaluate_baseline(
	Z_train: np.ndarray,
	y_train: np.ndarray,
	Z_val: np.ndarray,
	y_val: np.ndarray,
	Z_test: np.ndarray,
	y_test: np.ndarray,
	*,
	task_type: str,
	config: dict,
) -> Tuple[float, float]:
	"""Train a simple linear readout on frozen embeddings."""

	if LogisticRegression is None or Ridge is None:  # pragma: no cover
		raise ModuleNotFoundError(
			"scikit-learn is required for VIME evaluation. Install scikit-learn to use this model."
		) from _SKLEARN_IMPORT_ERROR

	task_type = str(task_type).lower()
	if task_type in ["binclass", "multiclass"]:
		clf = LogisticRegression(max_iter=1000)
		clf.fit(Z_train, y_train)
		if task_type == "binclass":
			val_prob = clf.predict_proba(Z_val)[:, 1]
			test_prob = clf.predict_proba(Z_test)[:, 1]
			val_metric = float(roc_auc_score(y_val, val_prob))
			test_metric = float(roc_auc_score(y_test, test_prob))
		else:
			val_pred = clf.predict(Z_val)
			test_pred = clf.predict(Z_test)
			val_metric = float(accuracy_score(y_val, val_pred))
			test_metric = float(accuracy_score(y_test, test_pred))
		return val_metric, test_metric

	# regression
	y_mean = float(np.mean(y_train))
	y_std = float(np.std(y_train) + 1e-8)
	standardize_y = bool(config.get("standardize_y", True))
	if standardize_y:
		y_train_fit = (y_train - y_mean) / y_std
		y_val_fit = (y_val - y_mean) / y_std
		y_test_fit = (y_test - y_mean) / y_std
	else:
		y_train_fit = y_train
		y_val_fit = y_val
		y_test_fit = y_test

	reg = Ridge(alpha=float(config.get("baseline_ridge_alpha", 1.0)))
	reg.fit(Z_train, y_train_fit)
	val_pred = reg.predict(Z_val)
	test_pred = reg.predict(Z_test)
	if standardize_y:
		val_pred = val_pred * y_std + y_mean
		test_pred = test_pred * y_std + y_mean
	val_rmse = float(math.sqrt(mean_squared_error(y_val, val_pred)))
	test_rmse = float(math.sqrt(mean_squared_error(y_test, test_pred)))
	return val_rmse, test_rmse


def row_gnn_after_start_fn(
	train_df: pd.DataFrame,
	val_df: pd.DataFrame,
	test_df: pd.DataFrame,
	config: dict,
	task_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
	"""Offline row-embedding extraction between start and materialize.

	Paper-aligned fairness rule: this hook uses raw row feature vectors as node
	features (no self-attention/pooling projection).

	Output schema:
	- Features are replaced by embedding columns: N_feature_emb_1..N_feature_emb_d
	- Target is preserved in 'target'.
	"""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError(
			"pandas is required for the VIME start-stage row-GNN graphification."
		) from _PANDAS_IMPORT_ERROR

	try:
		DGM_d_local = _import_dgm_d()
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"DGM_d (from DGM_pytorch/DGMlib) is required for row-level graphification. "
			"Ensure the DGM_pytorch module is present and importable."
		) from e

	device = resolve_device(config)
	print(f"[START-GNN-DGM] Using device: {device}")

	dgm_k = int(config.get("gnn_dgm_k", config.get("dgm_k", 10)))
	dgm_distance = str(config.get("gnn_dgm_distance", config.get("dgm_distance", "euclidean")))
	dgm_gate = str(config.get("gnn_dgm_gate", "none")).lower()
	dgm_gate_sharpness = float(config.get("gnn_dgm_gate_sharpness", 1.0))
	dgm_gate_threshold_init = float(config.get("gnn_dgm_gate_threshold_init", -10.0))
	dgm_eval_prune = bool(config.get("gnn_dgm_eval_prune", False))
	dgm_eval_prune_threshold = float(config.get("gnn_dgm_eval_prune_threshold", 0.5))
	gnn_epochs = int(config.get("gnn_epochs", config.get("epochs", 200)))
	patience = int(config.get("gnn_patience", 10))
	loss_threshold = float(config.get("gnn_loss_threshold", 1e-4))
	gnn_hidden = int(config.get("gnn_hidden", 64))
	gnn_out_dim = int(config.get("gnn_out_dim", gnn_hidden))
	lr = float(config.get("gnn_lr", 1e-3))

	all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
	feature_cols = [c for c in all_df.columns if c != "target"]
	if len(feature_cols) == 0:
		raise ValueError("No feature columns found (expected non-'target' columns).")

	try:
		x_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32, device=device)
		x_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32, device=device)
		x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32, device=device)
	except Exception as e:
		raise ValueError(
			"Failed to convert DataFrame features into a float tensor. "
			"Ensure feature columns are numeric and do not contain raw strings."
		) from e

	y_train_np = train_df["target"].values
	y_val_np = val_df["target"].values
	y_test_np = test_df["target"].values

	if task_type in ["binclass", "multiclass"]:
		y_all_np = np.concatenate([y_train_np, y_val_np, y_test_np])
		num_classes = len(pd.unique(y_all_np))
		if task_type == "binclass" and num_classes != 2:
			num_classes = 2
		out_dim = int(num_classes)
	else:
		out_dim = 1

	n_train = int(len(train_df))

	class DGMEmbedWrapper(torch.nn.Module):
		def forward(self, x, A=None):
			return x

	dgm_embed_f = DGMEmbedWrapper()
	dgm_k_train = int(min(dgm_k, max(1, n_train - 1)))
	dgm_module = DGM_d_local(dgm_embed_f, k=dgm_k_train, distance=dgm_distance).to(device)

	dgm_gate_threshold_param = None
	if dgm_gate != "none":
		dgm_gate_threshold_param = torch.nn.Parameter(
			torch.tensor(dgm_gate_threshold_init, device=device, dtype=torch.float32)
		)

	gnn = SimpleGCN(x_train.shape[1], gnn_hidden, gnn_out_dim).to(device)
	pred_head = torch.nn.Linear(gnn_out_dim, out_dim).to(device)

	params = list(gnn.parameters()) + list(pred_head.parameters()) + list(dgm_module.parameters())
	if dgm_gate_threshold_param is not None:
		params += [dgm_gate_threshold_param]
	optimizer = torch.optim.Adam(params, lr=lr)

	best_val_loss = float("inf")
	early_stop_counter = 0
	gnn_early_stop_epochs = 0
	best_states = None

	standardize_mode = str(config.get("gnn_row_standardize", "train")).lower()
	if standardize_mode not in {"split", "train"}:
		raise ValueError(
			"Unknown gnn_row_standardize mode. Expected 'split' or 'train'. "
			f"Got: {standardize_mode!r}"
		)

	train_mean = x_train.mean(dim=0, keepdim=True)
	train_std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)

	def _normalize(x_tensor: torch.Tensor) -> torch.Tensor:
		if standardize_mode == "train":
			return (x_tensor - train_mean) / train_std
		return _standardize(x_tensor, dim=0)

	def forward_pass(x_tensor, dgm_module_inst):
		ns = x_tensor.shape[0]
		row_emb_std = _normalize(x_tensor)
		row_emb_batched = row_emb_std.contiguous().unsqueeze(0)
		if hasattr(dgm_module_inst, "k"):
			dgm_module_inst.k = int(min(int(dgm_module_inst.k), max(1, ns - 1)))
		x_dgm, edge_index_dgm, logprobs_dgm = dgm_module_inst(row_emb_batched, A=None)
		x_dgm = x_dgm.squeeze(0)

		edge_index_dgm, edge_weight_dgm = _dgm_gate_and_prune_edges(
			edge_index_dgm,
			logprobs_dgm,
			ns,
			gate=dgm_gate,
			gate_sharpness=dgm_gate_sharpness,
			gate_threshold_param=dgm_gate_threshold_param,
			is_training=bool(dgm_module_inst.training),
			eval_prune=dgm_eval_prune,
			eval_prune_threshold=dgm_eval_prune_threshold,
		)
		gcn_out = gnn(x_dgm, edge_index_dgm, edge_weight=edge_weight_dgm)
		logits = pred_head(gcn_out)
		return logits, gcn_out, logprobs_dgm

	for epoch in range(gnn_epochs):
		dgm_module.train()
		gnn.train()
		pred_head.train()

		optimizer.zero_grad()
		logits, _, logprobs_dgm = forward_pass(x_train, dgm_module)

		if task_type in ["binclass", "multiclass"]:
			y_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
			train_loss = F.cross_entropy(logits, y_train)
		else:
			y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)
			train_loss = F.mse_loss(logits.squeeze(), y_train)

		dgm_reg = -logprobs_dgm.mean() * 0.01
		train_loss = train_loss + dgm_reg
		train_loss.backward()
		optimizer.step()

		dgm_module.eval()
		gnn.eval()
		pred_head.eval()
		with torch.no_grad():
			logits_val, _, _ = forward_pass(x_val, dgm_module)
			if task_type in ["binclass", "multiclass"]:
				y_val = torch.tensor(y_val_np, dtype=torch.long, device=device)
				val_loss = F.cross_entropy(logits_val, y_val)
			else:
				y_val = torch.tensor(y_val_np, dtype=torch.float32, device=device)
				val_loss = F.mse_loss(logits_val.squeeze(), y_val)

		train_loss_val = float(train_loss.item())
		val_loss_val = float(val_loss.item())
		improved = val_loss_val < best_val_loss - loss_threshold
		if improved:
			best_val_loss = val_loss_val
			early_stop_counter = 0
			best_states = {
				"gnn": gnn.state_dict(),
				"pred_head": pred_head.state_dict(),
				"dgm_module": dgm_module.state_dict(),
			}
			if dgm_gate_threshold_param is not None:
				best_states["dgm_gate_threshold"] = dgm_gate_threshold_param.detach().clone()
		else:
			early_stop_counter += 1

		suffix = " (improved)" if improved else ""
		print(
			f"[START-GNN-DGM] Epoch {epoch+1}/{gnn_epochs}, "
			f"Train Loss: {train_loss_val:.4f}, Val Loss: {val_loss_val:.4f}{suffix}"
		)

		if early_stop_counter >= patience:
			gnn_early_stop_epochs = epoch + 1
			print(
				f"[START-GNN-DGM] Early stopping at epoch {epoch+1} (Val Loss: {val_loss_val:.4f})"
			)
			break

	if best_states is not None:
		gnn.load_state_dict(best_states["gnn"])
		pred_head.load_state_dict(best_states["pred_head"])
		dgm_module.load_state_dict(best_states["dgm_module"])
		if dgm_gate_threshold_param is not None and best_states.get("dgm_gate_threshold") is not None:
			with torch.no_grad():
				dgm_gate_threshold_param.copy_(best_states["dgm_gate_threshold"])

	dgm_module.eval()
	gnn.eval()
	pred_head.eval()
	with torch.no_grad():
		_, emb_train, _ = forward_pass(x_train, dgm_module)
		_, emb_val, _ = forward_pass(x_val, dgm_module)
		_, emb_test, _ = forward_pass(x_test, dgm_module)

	row_emb_cols = [f"N_feature_emb_{i + 1}" for i in range(int(gnn_out_dim))]
	train_df_emb = pd.DataFrame(emb_train.detach().cpu().numpy(), columns=row_emb_cols, index=train_df.index)
	val_df_emb = pd.DataFrame(emb_val.detach().cpu().numpy(), columns=row_emb_cols, index=val_df.index)
	test_df_emb = pd.DataFrame(emb_test.detach().cpu().numpy(), columns=row_emb_cols, index=test_df.index)

	train_df_emb["target"] = train_df["target"].values
	val_df_emb["target"] = val_df["target"].values
	test_df_emb["target"] = test_df["target"].values

	return train_df_emb, val_df_emb, test_df_emb, int(gnn_early_stop_epochs)


def row_gnn_after_materialize_arrays(
	mat: Dict[str, Any],
	config: Dict[str, Any],
	task_type: str,
) -> Tuple[Dict[str, Any], int]:
	"""Offline row-embedding extraction after `materialize_fn`.

	Returns updated `mat` where X_* are replaced by N_feature_emb_* arrays.
	"""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError(
			"pandas is required for the materialize-stage row-GNN graphification."
		) from _PANDAS_IMPORT_ERROR

	# Use raw (pre-standardization) features for paper-aligned row graphification.
	X_train = mat.get("X_train_raw", mat["X_train"])
	X_val = mat.get("X_val_raw", mat["X_val"])
	X_test = mat.get("X_test_raw", mat["X_test"])

	in_dim = int(X_train.shape[1])
	feature_cols = [f"N_feature_{i + 1}" for i in range(in_dim)]

	train_df = pd.DataFrame(X_train, columns=feature_cols)
	val_df = pd.DataFrame(X_val, columns=feature_cols)
	test_df = pd.DataFrame(X_test, columns=feature_cols)
	train_df["target"] = mat["y_train"]
	val_df["target"] = mat["y_val"]
	test_df["target"] = mat["y_test"]

	train_df_emb, val_df_emb, test_df_emb, gnn_early_stop_epochs = row_gnn_after_start_fn(
		train_df,
		val_df,
		test_df,
		config,
		task_type,
	)

	emb_cols = [c for c in train_df_emb.columns if c != "target"]
	mat["X_train"] = train_df_emb[emb_cols].values.astype(np.float32)
	mat["X_val"] = val_df_emb[emb_cols].values.astype(np.float32)
	mat["X_test"] = test_df_emb[emb_cols].values.astype(np.float32)
	mat["row_emb_cols"] = emb_cols
	return mat, int(gnn_early_stop_epochs)


def feature_gnn_after_start_fn(
	train_df: pd.DataFrame,
	val_df: pd.DataFrame,
	test_df: pd.DataFrame,
	config: dict,
	task_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
	"""Feature-level topology init after `start` (inactive)."""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError(
			"pandas is required for the VIME feature-level placeholder."
		) from _PANDAS_IMPORT_ERROR

	feature_names = [c for c in train_df.columns if c != "target"]
	init_mode = str(config.get("feature_topology_init", "chain")).lower()
	topology = _init_feature_topology_prior(feature_names, init_mode=init_mode)
	config["feature_topology_meta"] = _feature_topology_to_meta(topology)
	print(
		f"[START-GNN] Feature-level topology initialized (inactive). "
		f"init_mode={topology['init_mode']}, num_features={topology['num_features']}"
	)
	return train_df, val_df, test_df, 0


def feature_gnn_after_materialize_fn(
	mat: Dict[str, Any],
	config: dict,
	*,
	stage_tag: str,
) -> Tuple[Dict[str, Any], int]:
	"""Feature-level topology init after `materialize_fn` (inactive)."""

	feature_names = list(mat.get("feature_names", []))
	if not feature_names:
		# Fallback: use synthetic names when called from array-only paths.
		in_dim = int(mat["X_train"].shape[1])
		feature_names = [f"N_feature_{i + 1}" for i in range(in_dim)]

	init_mode = str(config.get("feature_topology_init", "chain")).lower()
	topology = _init_feature_topology_prior(feature_names, init_mode=init_mode)
	config["feature_topology_meta"] = _feature_topology_to_meta(topology)
	print(
		f"[{stage_tag}-GNN] Feature-level topology initialized (inactive). "
		f"init_mode={topology['init_mode']}, num_features={topology['num_features']}"
	)
	return mat, 0


def start_fn(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError("pandas is required to run this model.") from _PANDAS_IMPORT_ERROR
	return train_df, val_df, test_df


def materialize_fn(
	train_df: pd.DataFrame,
	val_df: pd.DataFrame,
	test_df: pd.DataFrame,
	dataset_results: dict,
	config: dict,
) -> Dict[str, Any]:
	"""Stage 1: materialize.

	Convert split DataFrames into standardized numpy arrays.
	"""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError("pandas is required for VIME materialize_fn.") from _PANDAS_IMPORT_ERROR

	X_train_raw, y_train = _split_xy_from_df(train_df)
	X_val_raw, y_val = _split_xy_from_df(val_df)
	X_test_raw, y_test = _split_xy_from_df(test_df)

	X_train, mean, std = _standardize_fit(X_train_raw)
	X_val = _standardize_apply(X_val_raw, mean, std)
	X_test = _standardize_apply(X_test_raw, mean, std)

	task_type = _task_type_from_info(dataset_results)
	y_mean = None
	y_std = None
	if task_type == "regression" and bool(config.get("standardize_y", True)):
		y_mean = float(np.mean(y_train))
		y_std = float(np.std(y_train) + 1e-8)

	feature_names = [c for c in train_df.columns if c != "target"]

	return {
		"X_train": X_train,
		"y_train": y_train,
		"X_val": X_val,
		"y_val": y_val,
		"X_test": X_test,
		"y_test": y_test,
		"X_train_raw": X_train_raw,
		"X_val_raw": X_val_raw,
		"X_test_raw": X_test_raw,
		"mean": mean,
		"std": std,
		"y_mean": y_mean,
		"y_std": y_std,
		"feature_names": feature_names,
	}


def _coerce_stage_for_core(stage: str) -> str:
	stage = "none" if stage is None else str(stage).lower()
	# Offline graphification happens outside core.
	if stage in {"start", "materialize", "materialization"}:
		return "none"
	return stage


def vime_core_fn_row_gnn(mat: Dict[str, Any], config: Dict[str, Any], task_type: str, *, gnn_stage: str):
	"""VIME core under row-graphify mode."""

	device = resolve_device(config)
	X_train = torch.tensor(mat["X_train"], dtype=torch.float32, device=device)
	X_val = torch.tensor(mat["X_val"], dtype=torch.float32, device=device)
	X_test = torch.tensor(mat["X_test"], dtype=torch.float32, device=device)

	in_dim = int(X_train.size(1))
	hidden = int(config.get("hidden_dim", 256))
	dropout = float(config.get("dropout", 0.1))
	epochs = int(config.get("epochs", 200))
	batch_size = int(config.get("batch_size", 256))
	lr = float(config.get("lr", 1e-3))
	patience = int(config.get("patience", 10))
	alpha = float(config.get("vime_alpha", 2.0))
	p_m = float(config.get("vime_p_m", 0.3))
	weight_decay = float(config.get("weight_decay", 1e-5))

	stage = _coerce_stage_for_core(gnn_stage)
	enc = VIMEEncoder(in_dim, hidden, dropout).to(device)

	# decoding: joint supervised training + VIME pretext.
	if stage == "decoding":
		try:
			DGM_d = _import_dgm_d()
		except Exception as e:  # pragma: no cover
			raise ModuleNotFoundError(
				"DGM_d (from DGM_pytorch/DGMlib) is required for row-level decoding injection."
			) from e

		tag = "[VIME][ROW][DECODING-JOINT]"
		is_class = task_type in ["binclass", "multiclass"]

		y_train = mat["y_train"]
		y_val = mat["y_val"]
		y_test = mat["y_test"]

		if is_class:
			y_all = np.concatenate([y_train, y_val, y_test])
			num_classes = int(len(np.unique(y_all)))
			if task_type == "binclass" and num_classes != 2:
				num_classes = 2
			out_dim = 2 if task_type == "binclass" else int(num_classes)
			y_train_t = torch.tensor(y_train, dtype=torch.long)
			y_val_t = torch.tensor(y_val, dtype=torch.long)
			y_test_t = torch.tensor(y_test, dtype=torch.long)
			y_mean = None
			y_std = None
		else:
			out_dim = 1
			y_train_t = torch.tensor(y_train, dtype=torch.float32)
			y_val_t = torch.tensor(y_val, dtype=torch.float32)
			y_test_t = torch.tensor(y_test, dtype=torch.float32)
			y_mean = mat.get("y_mean", float(np.mean(y_train)))
			y_std = mat.get("y_std", float(np.std(y_train) + 1e-8))

		x_train = X_train.detach().cpu()
		x_val = X_val.detach().cpu()
		x_test = X_test.detach().cpu()

		train_dl = torch.utils.data.DataLoader(
			torch.utils.data.TensorDataset(x_train, y_train_t),
			batch_size=batch_size,
			shuffle=True,
		)
		val_dl = torch.utils.data.DataLoader(
			torch.utils.data.TensorDataset(x_val, y_val_t),
			batch_size=batch_size,
			shuffle=False,
		)
		test_dl = torch.utils.data.DataLoader(
			torch.utils.data.TensorDataset(x_test, y_test_t),
			batch_size=batch_size,
			shuffle=False,
		)

		hidden_dim = int(config.get("gnn_hidden", 64))
		patience_dec = int(config.get("gnn_patience", 10))
		loss_threshold = float(config.get("gnn_loss_threshold", 1e-4))
		gnn_lr = float(config.get("gnn_lr", lr))
		attn_dim = int(config.get("gnn_attn_dim", hidden_dim))
		attn_heads = int(config.get("gnn_num_heads", config.get("gnn_attn_heads", 4)))
		dgm_k = int(config.get("gnn_dgm_k", config.get("dgm_k", config.get("gnn_knn", 5))))
		dgm_distance = str(config.get("gnn_dgm_distance", config.get("dgm_distance", "euclidean")))
		dgm_gate = str(config.get("gnn_dgm_gate", "none")).lower()
		dgm_gate_sharpness = float(config.get("gnn_dgm_gate_sharpness", 1.0))
		dgm_gate_threshold_init = float(config.get("gnn_dgm_gate_threshold_init", -10.0))
		dgm_eval_prune = bool(config.get("gnn_dgm_eval_prune", False))
		dgm_eval_prune_threshold = float(config.get("gnn_dgm_eval_prune_threshold", 0.5))

		num_cols = int(in_dim)
		self_attn = torch.nn.MultiheadAttention(
			embed_dim=attn_dim,
			num_heads=attn_heads,
			batch_first=True,
		).to(device)
		attn_norm = torch.nn.LayerNorm(attn_dim).to(device)
		input_proj = torch.nn.Linear(1, attn_dim).to(device)
		column_embed = torch.nn.Parameter(torch.randn(num_cols, attn_dim, device=device))
		pool_query = torch.nn.Parameter(torch.randn(attn_dim, device=device))
		row_ctx_proj = torch.nn.Linear(hidden, attn_dim).to(device)

		class DGMEmbedWrapper(torch.nn.Module):
			def forward(self, x, A=None):
				return x

		dgm_module = DGM_d(DGMEmbedWrapper(), k=dgm_k, distance=dgm_distance).to(device)

		dgm_gate_threshold_param = None
		if dgm_gate != "none":
			dgm_gate_threshold_param = torch.nn.Parameter(
				torch.tensor(dgm_gate_threshold_init, device=device, dtype=torch.float32)
			)

		gnn = SimpleGCN(attn_dim, hidden_dim, out_dim, num_layers=2).to(device)

		params = (
			list(enc.parameters())
			+ list(self_attn.parameters())
			+ list(attn_norm.parameters())
			+ list(input_proj.parameters())
			+ list(row_ctx_proj.parameters())
			+ list(dgm_module.parameters())
			+ list(gnn.parameters())
			+ [column_embed, pool_query]
		)
		if dgm_gate_threshold_param is not None:
			params += [dgm_gate_threshold_param]

		optimizer = torch.optim.Adam(params, lr=gnn_lr, weight_decay=weight_decay)

		def forward_pass(xb: torch.Tensor):
			B = int(xb.shape[0])
			if B <= 1:
				raise ValueError(f"{tag} Need batch size >= 2 for DGM (got B={B})")
			xb = xb.to(device)
			z_row = enc.net(xb)
			row_ctx = row_ctx_proj(z_row).unsqueeze(1)

			tokens = input_proj(xb.unsqueeze(-1)) + column_embed.unsqueeze(0) + row_ctx
			tokens_norm = attn_norm(tokens)
			attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
			tokens_attn = tokens + attn_out1

			pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(attn_dim)
			pool_weights = torch.softmax(pool_logits, dim=1)
			x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)

			x_std = _standardize(x_pooled, dim=0)
			x_batched = x_std.unsqueeze(0)
			if hasattr(dgm_module, "k"):
				dgm_module.k = int(min(int(dgm_module.k), max(1, B - 1)))
			x_dgm, edge_index_dgm, logprobs_dgm = dgm_module(x_batched, A=None)
			x_dgm = x_dgm.squeeze(0)

			edge_index_dgm, edge_weight_dgm = _dgm_gate_and_prune_edges(
				edge_index_dgm,
				logprobs_dgm,
				B,
				gate=dgm_gate,
				gate_sharpness=dgm_gate_sharpness,
				gate_threshold_param=dgm_gate_threshold_param,
				is_training=bool(dgm_module.training),
				eval_prune=dgm_eval_prune,
				eval_prune_threshold=dgm_eval_prune_threshold,
			)
			logits = gnn(x_dgm, edge_index_dgm, edge_weight=edge_weight_dgm)
			return logits, logprobs_dgm

		best_val_loss = float("inf")
		best_val_metric = float("nan")
		best_test_metric = float("nan")
		best_state = None
		early_stop_counter = 0
		total_epochs = epochs

		decoding_vime_lambda = float(config.get("decoding_vime_lambda", 0.1))

		for epoch in range(epochs):
			enc.train()
			self_attn.train()
			attn_norm.train()
			input_proj.train()
			row_ctx_proj.train()
			dgm_module.train()
			gnn.train()
			for xb, yb in train_dl:
				optimizer.zero_grad()
				xb = xb.to(device)
				yb = yb.to(device)
				if xb.size(0) <= 1:
					continue

				logits, logprobs_dgm = forward_pass(xb)
				if is_class:
					sup_loss = F.cross_entropy(logits, yb.long())
				else:
					yb_loss = (yb.float() - float(y_mean)) / float(y_std)
					sup_loss = F.mse_loss(logits.squeeze(), yb_loss)

				m = mask_generator(p_m, xb)
				m_label, x_tilde = pretext_generator(m, xb)
				_, mask_logits, recon = enc(x_tilde)
				mask_loss = F.binary_cross_entropy_with_logits(mask_logits, m_label)
				recon_loss = F.mse_loss(recon, xb)
				vime_loss = mask_loss + alpha * recon_loss

				loss = sup_loss + decoding_vime_lambda * vime_loss
				if logprobs_dgm is not None:
					loss = loss + (-logprobs_dgm.mean() * 0.01)
				loss.backward()
				optimizer.step()

			enc.eval()
			self_attn.eval()
			attn_norm.eval()
			input_proj.eval()
			row_ctx_proj.eval()
			dgm_module.eval()
			gnn.eval()
			with torch.no_grad():
				val_losses = []
				val_logits_all = []
				val_y_all = []
				for xb, yb in val_dl:
					xb = xb.to(device)
					yb = yb.to(device)
					if xb.size(0) <= 1:
						continue
					logits, _ = forward_pass(xb)
					if is_class:
						vloss = F.cross_entropy(logits, yb.long())
					else:
						yb_loss = (yb.float() - float(y_mean)) / float(y_std)
						vloss = F.mse_loss(logits.squeeze(), yb_loss)
					val_losses.append(float(vloss.item()) * xb.size(0))
					val_logits_all.append(logits.detach().cpu())
					val_y_all.append(yb.detach().cpu())

				if not val_losses:
					continue

				val_loss_val = float(sum(val_losses) / max(1, int(len(y_val))))
				val_logits_cat = torch.cat(val_logits_all, dim=0)
				val_y_cat = torch.cat(val_y_all, dim=0)
				if task_type == "binclass":
					val_prob = torch.softmax(val_logits_cat, dim=-1)[:, 1]
					val_metric = float(roc_auc_score(val_y_cat.numpy(), val_prob.numpy()))
				elif task_type == "multiclass":
					val_pred = val_logits_cat.argmax(dim=-1)
					val_metric = float(accuracy_score(val_y_cat.numpy(), val_pred.numpy()))
				else:
					val_pred = val_logits_cat.squeeze().numpy()
					val_pred = val_pred * float(y_std) + float(y_mean)
					val_metric = float(math.sqrt(mean_squared_error(val_y_cat.numpy(), val_pred)))

				improved = float(val_loss_val) < best_val_loss - loss_threshold
				if improved:
					best_val_loss = float(val_loss_val)
					best_val_metric = float(val_metric)
					best_state = {
						"enc": copy.deepcopy(enc.state_dict()),
						"self_attn": copy.deepcopy(self_attn.state_dict()),
						"attn_norm": copy.deepcopy(attn_norm.state_dict()),
						"input_proj": copy.deepcopy(input_proj.state_dict()),
						"row_ctx_proj": copy.deepcopy(row_ctx_proj.state_dict()),
						"dgm_module": copy.deepcopy(dgm_module.state_dict()),
						"gnn": copy.deepcopy(gnn.state_dict()),
						"column_embed": column_embed.detach().clone(),
						"pool_query": pool_query.detach().clone(),
					}
					if dgm_gate_threshold_param is not None:
						best_state["dgm_gate_threshold"] = dgm_gate_threshold_param.detach().clone()
					early_stop_counter = 0

					test_logits_all = []
					test_y_all = []
					for xb, yb in test_dl:
						xb = xb.to(device)
						yb = yb.to(device)
						if xb.size(0) <= 1:
							continue
						logits, _ = forward_pass(xb)
						test_logits_all.append(logits.detach().cpu())
						test_y_all.append(yb.detach().cpu())
					if test_logits_all:
						test_logits_cat = torch.cat(test_logits_all, dim=0)
						test_y_cat = torch.cat(test_y_all, dim=0)
						if task_type == "binclass":
							te_prob = torch.softmax(test_logits_cat, dim=-1)[:, 1]
							best_test_metric = float(roc_auc_score(test_y_cat.numpy(), te_prob.numpy()))
						elif task_type == "multiclass":
							te_pred = test_logits_cat.argmax(dim=-1)
							best_test_metric = float(accuracy_score(test_y_cat.numpy(), te_pred.numpy()))
						else:
							te_pred = test_logits_cat.squeeze().numpy()
							te_pred = te_pred * float(y_std) + float(y_mean)
							best_test_metric = float(math.sqrt(mean_squared_error(test_y_cat.numpy(), te_pred)))
				else:
					early_stop_counter += 1

			print(
				f"{tag} epoch {epoch+1}/{epochs} "
				f"val_loss={best_val_loss:.4f} best_val_metric={best_val_metric:.4f}"
			)

			if early_stop_counter >= patience_dec:
				total_epochs = epoch + 1
				print(f"{tag} Early stopping at epoch {total_epochs}")
				break

		if best_state is not None:
			enc.load_state_dict(best_state["enc"])
			self_attn.load_state_dict(best_state["self_attn"])
			attn_norm.load_state_dict(best_state["attn_norm"])
			input_proj.load_state_dict(best_state["input_proj"])
			row_ctx_proj.load_state_dict(best_state["row_ctx_proj"])
			dgm_module.load_state_dict(best_state["dgm_module"])
			gnn.load_state_dict(best_state["gnn"])
			with torch.no_grad():
				column_embed.copy_(best_state["column_embed"])
				pool_query.copy_(best_state["pool_query"])
				if dgm_gate_threshold_param is not None and best_state.get("dgm_gate_threshold") is not None:
					dgm_gate_threshold_param.copy_(best_state["dgm_gate_threshold"])

		return {
			"best_val_metric": float(best_val_metric),
			"best_test_metric": float(best_test_metric),
			"early_stop_epochs": int(total_epochs),
		}

	# encoding/columnwise: mini-batch attention + DGM + GCN feature transform (ExcelFormer-style)
	use_injection = stage in {"encoding", "columnwise"}
	feature_gnn = None
	if use_injection:
		try:
			DGM_d = _import_dgm_d()
		except Exception as e:  # pragma: no cover
			raise ModuleNotFoundError(
				"DGM_d (from DGM_pytorch/DGMlib) is required for row-level injection."
			) from e

		dgm_k = int(config.get("gnn_dgm_k", config.get("dgm_k", config.get("gnn_knn", 5))))
		dgm_distance = str(config.get("gnn_dgm_distance", config.get("dgm_distance", "euclidean")))
		dgm_gate = str(config.get("gnn_dgm_gate", "none")).lower()
		dgm_gate_sharpness = float(config.get("gnn_dgm_gate_sharpness", 1.0))
		dgm_gate_threshold_init = float(config.get("gnn_dgm_gate_threshold_init", -10.0))
		dgm_eval_prune = bool(config.get("gnn_dgm_eval_prune", False))
		dgm_eval_prune_threshold = float(config.get("gnn_dgm_eval_prune_threshold", 0.5))
		attn_dim = int(config.get("gnn_attn_dim", int(config.get("gnn_hidden", 64))))
		attn_heads = int(config.get("gnn_num_heads", config.get("gnn_attn_heads", 4)))
		gnn_hidden = int(config.get("gnn_hidden", 64))
		gnn_out_dim = int(config.get("gnn_out_dim", attn_dim))
		gnn_dropout = float(config.get("gnn_dropout", 0.1))

		class FeatureGNN(nn.Module):
			def __init__(self, num_cols: int):
				super().__init__()
				self.num_cols = int(num_cols)
				self.attn_in = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True)
				self.attn_norm = nn.LayerNorm(attn_dim)
				self.attn_out = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True)
				self.attn_out_norm = nn.LayerNorm(attn_dim)
				self.input_proj = nn.Linear(1, attn_dim)
				self.gnn = SimpleGCN(attn_dim, gnn_hidden, gnn_out_dim, num_layers=2)
				self.gcn_to_attn = nn.Linear(gnn_out_dim, attn_dim)
				self.out_proj = nn.Linear(attn_dim, 1)
				self.column_embed = nn.Parameter(torch.randn(self.num_cols, attn_dim))
				self.pool_query = nn.Parameter(torch.randn(attn_dim))

				self.ffn_pre = nn.Sequential(
					nn.Linear(attn_dim, attn_dim * 2),
					nn.GELU(),
					nn.Dropout(gnn_dropout),
					nn.Linear(attn_dim * 2, attn_dim),
				)
				self.ffn_post = nn.Sequential(
					nn.Linear(attn_dim, attn_dim * 2),
					nn.GELU(),
					nn.Dropout(gnn_dropout),
					nn.Linear(attn_dim * 2, attn_dim),
				)
				self.fusion_alpha_param = nn.Parameter(torch.tensor(-0.847))

				class DGMEmbedWrapper(nn.Module):
					def forward(self, x, A=None):
						return x

				self.dgm_module = DGM_d(DGMEmbedWrapper(), k=dgm_k, distance=dgm_distance)
				self.dropout = nn.Dropout(gnn_dropout)

				self.dgm_gate_threshold_param = None
				if dgm_gate != "none":
					self.dgm_gate_threshold_param = nn.Parameter(
						torch.tensor(dgm_gate_threshold_init, dtype=torch.float32)
					)

			def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
				B = int(x.shape[0])
				tokens = self.input_proj(x.unsqueeze(-1)) + self.column_embed.unsqueeze(0)
				tokens_norm = self.attn_norm(tokens)
				attn_out1, _ = self.attn_in(tokens_norm, tokens_norm, tokens_norm)
				tokens_attn = tokens + attn_out1
				tokens_attn = tokens_attn + self.ffn_pre(self.attn_norm(tokens_attn))

				pool_logits = (tokens_attn * self.pool_query).sum(dim=2) / math.sqrt(attn_dim)
				pool_weights = torch.softmax(pool_logits, dim=1)
				row_emb = (pool_weights.unsqueeze(2) * tokens_attn).sum(dim=1)
				row_emb_std = _standardize(row_emb, dim=0)

				row_emb_batched = row_emb_std.unsqueeze(0)
				if hasattr(self.dgm_module, "k"):
					self.dgm_module.k = int(min(int(self.dgm_module.k), max(1, B - 1)))
				row_emb_dgm, edge_index_dgm, logprobs_dgm = self.dgm_module(row_emb_batched, A=None)
				row_emb_dgm = row_emb_dgm.squeeze(0)

				edge_index_dgm, edge_weight_dgm = _dgm_gate_and_prune_edges(
					edge_index_dgm,
					logprobs_dgm,
					B,
					gate=dgm_gate,
					gate_sharpness=dgm_gate_sharpness,
					gate_threshold_param=self.dgm_gate_threshold_param,
					is_training=bool(self.dgm_module.training),
					eval_prune=dgm_eval_prune,
					eval_prune_threshold=dgm_eval_prune_threshold,
				)

				gcn_out = self.gnn(row_emb_dgm, edge_index_dgm, edge_weight=edge_weight_dgm)
				gcn_ctx = self.gcn_to_attn(gcn_out).unsqueeze(1)
				tokens_with_ctx = tokens_attn + gcn_ctx
				tokens_ctx_norm = self.attn_out_norm(tokens_with_ctx)
				attn_out2, _ = self.attn_out(tokens_ctx_norm, tokens_ctx_norm, tokens_ctx_norm)
				tokens_mid = tokens_with_ctx + attn_out2
				tokens_out = tokens_mid + self.ffn_post(self.attn_out_norm(tokens_mid))
				recon = self.out_proj(self.dropout(tokens_out)).squeeze(-1)

				fusion_alpha = torch.sigmoid(self.fusion_alpha_param)
				x_fused = x + fusion_alpha * (recon - x)
				return x_fused, logprobs_dgm

		feature_gnn = FeatureGNN(in_dim).to(device)

	params = list(enc.parameters())
	if use_injection and feature_gnn is not None:
		params += list(feature_gnn.parameters())
	opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

	def run_epoch(split: str) -> float:
		if split == "train":
			enc.train()
			X = X_train
		elif split == "val":
			enc.eval()
			X = X_val
		else:
			enc.eval()
			X = X_test

		if feature_gnn is not None:
			feature_gnn.train(split == "train")

		total = 0.0
		n = 0
		for i in range(0, X.size(0), batch_size):
			xb = X[i : i + batch_size]
			m = mask_generator(p_m, xb)
			m_label, x_tilde = pretext_generator(m, xb)
			if split == "train":
				opt.zero_grad()
			logprobs_dgm = None
			if feature_gnn is not None:
				x_tilde, logprobs_dgm = feature_gnn(x_tilde)

			_, mask_logits, recon = enc(x_tilde)
			mask_loss = F.binary_cross_entropy_with_logits(mask_logits, m_label)
			recon_loss = F.mse_loss(recon, xb)
			loss = mask_loss + alpha * recon_loss
			if logprobs_dgm is not None:
				loss = loss + (-logprobs_dgm.mean() * 0.01)
			if split == "train":
				loss.backward()
				opt.step()
			total += float(loss.item()) * xb.size(0)
			n += xb.size(0)
		return float(total / max(1, n))

	best_val = float("inf")
	no_improve = 0
	best_state = None
	min_epochs = patience + 1
	total_epochs = epochs
	for ep in range(epochs):
		_ = run_epoch("train")
		val_loss = run_epoch("val")
		if val_loss + 1e-8 < best_val:
			best_val = val_loss
			no_improve = 0
			best_state = {
				"enc": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
				"feature_gnn": copy.deepcopy(feature_gnn.state_dict()) if feature_gnn is not None else None,
			}
		else:
			no_improve += 1
			if no_improve >= patience and (ep + 1) >= min_epochs:
				total_epochs = ep + 1
				break

	if best_state is not None:
		enc.load_state_dict(best_state["enc"])
		if feature_gnn is not None and best_state.get("feature_gnn") is not None:
			feature_gnn.load_state_dict(best_state["feature_gnn"])

	enc.eval()
	with torch.no_grad():
		Z_train = enc.net(X_train).detach().cpu().numpy()
		Z_val = enc.net(X_val).detach().cpu().numpy()
		Z_test = enc.net(X_test).detach().cpu().numpy()

	val_metric, test_metric = _evaluate_baseline(
		Z_train,
		mat["y_train"],
		Z_val,
		mat["y_val"],
		Z_test,
		mat["y_test"],
		task_type=task_type,
		config=config,
	)
	return {
		"best_val_metric": float(val_metric),
		"best_test_metric": float(test_metric),
		"early_stop_epochs": int(total_epochs),
	}


def vime_core_fn_feature_gnn(
	mat: Dict[str, Any],
	config: Dict[str, Any],
	task_type: str,
	*,
	gnn_stage: str,
):
	"""VIME core under feature-graphify mode.

	Feature graphification is token-level mixing over feature tokens and is
	activated only at encoding/columnwise/decoding.
	"""

	device = resolve_device(config)
	X_train = torch.tensor(mat["X_train"], dtype=torch.float32, device=device)
	X_val = torch.tensor(mat["X_val"], dtype=torch.float32, device=device)
	X_test = torch.tensor(mat["X_test"], dtype=torch.float32, device=device)

	in_dim = int(X_train.size(1))
	hidden = int(config.get("hidden_dim", 256))
	dropout = float(config.get("dropout", 0.1))
	epochs = int(config.get("epochs", 200))
	batch_size = int(config.get("batch_size", 256))
	lr = float(config.get("lr", 1e-3))
	patience = int(config.get("patience", 10))
	alpha = float(config.get("vime_alpha", 2.0))
	p_m = float(config.get("vime_p_m", 0.3))
	weight_decay = float(config.get("weight_decay", 1e-5))

	stage = _coerce_stage_for_core(gnn_stage)
	use_mixing = stage in {"encoding", "columnwise", "decoding"}

	meta = config.get("feature_topology_meta")
	if meta is None:
		feature_names = list(mat.get("feature_names", []))
		if not feature_names:
			feature_names = [f"N_feature_{i + 1}" for i in range(in_dim)]
		init_mode = str(config.get("feature_topology_init", "chain")).lower()
		topo = _init_feature_topology_prior(feature_names, init_mode=init_mode)
		meta = _feature_topology_to_meta(topo)
		config["feature_topology_meta"] = meta

	# Build allowed mask from edge_index
	allowed_mask = torch.eye(in_dim, dtype=torch.bool)
	try:
		ei = meta.get("edge_index")
		if isinstance(ei, list) and len(ei) == 2:
			src = torch.tensor(ei[0], dtype=torch.long)
			dst = torch.tensor(ei[1], dtype=torch.long)
			allowed_mask[dst, src] = True
	except Exception:
		pass

	mixer = FeatureTopologyMaskedMixing(
		allowed_mask.to(device),
		init_strength=float(config.get("feature_mixing_init_strength", 0.0)),
		symmetric=bool(config.get("feature_mixing_symmetric", True)),
		force_self_loops=True,
	).to(device)

	enc = VIMEEncoder(in_dim, hidden, dropout).to(device)

	if stage == "decoding":
		is_class = task_type in ["binclass", "multiclass"]
		y_train = mat["y_train"]
		y_val = mat["y_val"]
		y_test = mat["y_test"]
		if is_class:
			y_all = np.concatenate([y_train, y_val, y_test])
			num_classes = int(len(np.unique(y_all)))
			if task_type == "binclass" and num_classes != 2:
				num_classes = 2
			out_dim = 2 if task_type == "binclass" else int(num_classes)
			y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
			y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
			y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)
			y_mean = None
			y_std = None
		else:
			out_dim = 1
			y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
			y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
			y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)
			y_mean = mat.get("y_mean", float(np.mean(y_train)))
			y_std = mat.get("y_std", float(np.std(y_train) + 1e-8))

		head = nn.Linear(hidden, out_dim).to(device)
		params = list(enc.parameters()) + list(mixer.parameters()) + list(head.parameters())
		opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

		def iter_batches(X: torch.Tensor, y: torch.Tensor, shuffle: bool):
			idx = torch.randperm(X.size(0), device=X.device) if shuffle else torch.arange(X.size(0), device=X.device)
			for s in range(0, X.size(0), batch_size):
				j = idx[s : s + batch_size]
				yb = y[j]
				xb = X[j]
				yield xb, yb

		best_val_loss = float("inf")
		best_val_metric = float("nan")
		best_test_metric = float("nan")
		best_state = None
		no_improve = 0
		total_epochs = epochs
		min_epochs = patience + 1

		for ep in range(epochs):
			enc.train()
			mixer.train()
			head.train()
			for xb, yb in iter_batches(X_train, y_train_t, shuffle=True):
				opt.zero_grad()
				m = mask_generator(p_m, xb)
				m_label, x_tilde = pretext_generator(m, xb)
				x_mixed = mixer(x_tilde.unsqueeze(-1)).squeeze(-1)
				h, mask_logits, recon = enc(x_mixed)
				mask_loss = F.binary_cross_entropy_with_logits(mask_logits, m_label)
				recon_loss = F.mse_loss(recon, xb)
				vime_loss = mask_loss + alpha * recon_loss

				logits = head(h)
				if is_class:
					sup_loss = F.cross_entropy(logits, yb.long())
				else:
					yb_loss = (yb.float() - float(y_mean)) / float(y_std)
					sup_loss = F.mse_loss(logits.squeeze(), yb_loss)
				loss = sup_loss + float(config.get("decoding_vime_lambda", 0.1)) * vime_loss
				loss.backward()
				opt.step()

			enc.eval()
			mixer.eval()
			head.eval()
			with torch.no_grad():
				val_logits_all = []
				val_y_all = []
				val_losses = []
				for xb, yb in iter_batches(X_val, y_val_t, shuffle=False):
					m = mask_generator(p_m, xb)
					m_label, x_tilde = pretext_generator(m, xb)
					x_mixed = mixer(x_tilde.unsqueeze(-1)).squeeze(-1)
					h, _, _ = enc(x_mixed)
					logits = head(h)
					if is_class:
						vloss = F.cross_entropy(logits, yb.long())
					else:
						yb_loss = (yb.float() - float(y_mean)) / float(y_std)
						vloss = F.mse_loss(logits.squeeze(), yb_loss)
					val_losses.append(float(vloss.item()) * xb.size(0))
					val_logits_all.append(logits.detach().cpu())
					val_y_all.append(yb.detach().cpu())

				val_loss = float(sum(val_losses) / max(1, int(X_val.size(0)))) if val_losses else float("inf")
				val_logits_cat = torch.cat(val_logits_all, dim=0) if val_logits_all else None
				val_y_cat = torch.cat(val_y_all, dim=0) if val_y_all else None
				if val_logits_cat is None:
					continue
				if task_type == "binclass":
					val_prob = torch.softmax(val_logits_cat, dim=-1)[:, 1]
					val_metric = float(roc_auc_score(val_y_cat.numpy(), val_prob.numpy()))
				elif task_type == "multiclass":
					val_pred = val_logits_cat.argmax(dim=-1)
					val_metric = float(accuracy_score(val_y_cat.numpy(), val_pred.numpy()))
				else:
					val_pred = val_logits_cat.squeeze().numpy()
					val_pred = val_pred * float(y_std) + float(y_mean)
					val_metric = float(math.sqrt(mean_squared_error(val_y_cat.numpy(), val_pred)))

				if val_loss + 1e-8 < best_val_loss:
					best_val_loss = val_loss
					best_val_metric = float(val_metric)
					best_state = {
						"enc": copy.deepcopy(enc.state_dict()),
						"mixer": copy.deepcopy(mixer.state_dict()),
						"head": copy.deepcopy(head.state_dict()),
					}
					no_improve = 0

					# update best test
					test_logits_all = []
					test_y_all = []
					for xb, yb in iter_batches(X_test, y_test_t, shuffle=False):
						m = mask_generator(p_m, xb)
						m_label, x_tilde = pretext_generator(m, xb)
						x_mixed = mixer(x_tilde.unsqueeze(-1)).squeeze(-1)
						h, _, _ = enc(x_mixed)
						logits = head(h)
						test_logits_all.append(logits.detach().cpu())
						test_y_all.append(yb.detach().cpu())
					if test_logits_all:
						test_logits_cat = torch.cat(test_logits_all, dim=0)
						test_y_cat = torch.cat(test_y_all, dim=0)
						if task_type == "binclass":
							te_prob = torch.softmax(test_logits_cat, dim=-1)[:, 1]
							best_test_metric = float(roc_auc_score(test_y_cat.numpy(), te_prob.numpy()))
						elif task_type == "multiclass":
							te_pred = test_logits_cat.argmax(dim=-1)
							best_test_metric = float(accuracy_score(test_y_cat.numpy(), te_pred.numpy()))
						else:
							te_pred = test_logits_cat.squeeze().numpy()
							te_pred = te_pred * float(y_std) + float(y_mean)
							best_test_metric = float(math.sqrt(mean_squared_error(test_y_cat.numpy(), te_pred)))
				else:
					no_improve += 1

				if no_improve >= patience and (ep + 1) >= min_epochs:
					total_epochs = ep + 1
					break

		if best_state is not None:
			enc.load_state_dict(best_state["enc"])
			mixer.load_state_dict(best_state["mixer"])
			head.load_state_dict(best_state["head"])

		return {
			"best_val_metric": float(best_val_metric),
			"best_test_metric": float(best_test_metric),
			"early_stop_epochs": int(total_epochs),
		}

	# non-decoding stages: pretext training (optionally with mixing), then baseline on embeddings
	params = list(enc.parameters())
	if use_mixing:
		params += list(mixer.parameters())
	opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

	def run_epoch(split: str) -> float:
		if split == "train":
			enc.train()
			mixer.train(use_mixing)
			X = X_train
		elif split == "val":
			enc.eval()
			mixer.eval()
			X = X_val
		else:
			enc.eval()
			mixer.eval()
			X = X_test
		# mixer is only applied when active
		total = 0.0
		n = 0
		for i in range(0, X.size(0), batch_size):
			xb = X[i : i + batch_size]
			m = mask_generator(p_m, xb)
			m_label, x_tilde = pretext_generator(m, xb)
			if use_mixing:
				x_tilde = mixer(x_tilde.unsqueeze(-1)).squeeze(-1)
			if split == "train":
				opt.zero_grad()
			_, mask_logits, recon = enc(x_tilde)
			mask_loss = F.binary_cross_entropy_with_logits(mask_logits, m_label)
			recon_loss = F.mse_loss(recon, xb)
			loss = mask_loss + alpha * recon_loss
			if split == "train":
				loss.backward()
				opt.step()
			total += float(loss.item()) * xb.size(0)
			n += xb.size(0)
		return float(total / max(1, n))

	best_val = float("inf")
	no_improve = 0
	best_state = None
	min_epochs = patience + 1
	total_epochs = epochs
	for ep in range(epochs):
		_ = run_epoch("train")
		val_loss = run_epoch("val")
		if val_loss + 1e-8 < best_val:
			best_val = val_loss
			no_improve = 0
			best_state = {
				"enc": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
				"mixer": copy.deepcopy(mixer.state_dict()) if use_mixing else None,
			}
		else:
			no_improve += 1
			if no_improve >= patience and (ep + 1) >= min_epochs:
				total_epochs = ep + 1
				break

	if best_state is not None:
		enc.load_state_dict(best_state["enc"])
		if use_mixing and best_state.get("mixer") is not None:
			mixer.load_state_dict(best_state["mixer"])

	enc.eval()
	with torch.no_grad():
		Z_train = enc.net(X_train).detach().cpu().numpy()
		Z_val = enc.net(X_val).detach().cpu().numpy()
		Z_test = enc.net(X_test).detach().cpu().numpy()

	val_metric, test_metric = _evaluate_baseline(
		Z_train,
		mat["y_train"],
		Z_val,
		mat["y_val"],
		Z_test,
		mat["y_test"],
		task_type=task_type,
		config=config,
	)
	return {
		"best_val_metric": float(val_metric),
		"best_test_metric": float(test_metric),
		"early_stop_epochs": int(total_epochs),
	}


def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
	"""Model entrypoint called by the project runner."""

	print("VIME - staged execution")
	print(f"gnn_stage: {gnn_stage}")
	task_type = dataset_results.get("info", {}).get("task_type", "binclass")

	stage = "none" if gnn_stage is None else str(gnn_stage).lower()
	if stage == "materialization":
		stage = "materialize"

	train_df, val_df, test_df = start_fn(train_df, val_df, test_df)

	graphify = str(config.get("graphify", "row")).lower()
	gnn_early_stop_epochs = 0

	if graphify == "row":
		if stage == "start":
			train_df, val_df, test_df, gnn_early_stop_epochs = row_gnn_after_start_fn(
				train_df,
				val_df,
				test_df,
				config,
				task_type,
			)

		mat = materialize_fn(train_df, val_df, test_df, dataset_results, config)
		mat["gnn_early_stop_epochs"] = int(gnn_early_stop_epochs)

		if stage == "materialize":
			mat, gnn_early_stop_epochs = row_gnn_after_materialize_arrays(mat, config, task_type)
			mat["gnn_early_stop_epochs"] = int(gnn_early_stop_epochs)

		res = vime_core_fn_row_gnn(mat, config, task_type, gnn_stage=stage)
		res["gnn_early_stop_epochs"] = int(mat.get("gnn_early_stop_epochs", 0))
		return res

	if graphify == "feature":
		if stage == "start":
			train_df, val_df, test_df, _ = feature_gnn_after_start_fn(
				train_df,
				val_df,
				test_df,
				config,
				task_type,
			)

		mat = materialize_fn(train_df, val_df, test_df, dataset_results, config)
		mat["gnn_early_stop_epochs"] = 0
		if stage == "materialize":
			mat, _ = feature_gnn_after_materialize_fn(mat, config, stage_tag="MATERIALIZE")

		res = vime_core_fn_feature_gnn(mat, config, task_type, gnn_stage=stage)
		res["gnn_early_stop_epochs"] = 0
		return res

	raise ValueError(f"Unknown graphify={graphify!r}. Expected 'row' or 'feature'.")


#  small+binclass
#  python main.py --models vime --dataset kaggle_Audit_Data --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  small+regression
#  python main.py --models vime --dataset openml_The_Office_Dataset --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+binclass
#  python main.py --models vime --dataset credit --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+multiclass
#  python main.py --models vime --dataset eye --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  python main.py --models vime --dataset helena --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+regression
#  python main.py --models vime --dataset house --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0

