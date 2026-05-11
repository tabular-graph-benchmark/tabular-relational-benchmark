from __future__ import annotations

"""ResNet backbone with optional row/feature graphification.

This file is an English-only refactor ported from:
  - ModelComparison/TaBLEau/models/pytorch_frame/resnet.py

Key paper-aligned behavior (matching the ExcelFormer/FTTransformer ports in
this repo):
  - Row graphification at `start`/`materialize` is *offline* and uses raw row
	feature vectors as node features (no self-attention/pooling projection).
  - Feature graphification at `start`/`materialize` is *inactive* (only
	initializes a schema-level topology prior; token-level MP starts later).

The project runner calls `main(train_df, val_df, test_df, dataset_results, config, gnn_stage)`.
"""

import math
import os
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
	import pandas as pd
except ModuleNotFoundError as e:  # pragma: no cover
	pd = None
	_PANDAS_IMPORT_ERROR = e

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
		repo_root = Path(__file__).resolve().parents[3]
		dgm_path = repo_root / "DGM_pytorch"
		if dgm_path.exists():
			sys.path.insert(0, str(dgm_path))
		try:
			from DGMlib.layers import DGM_d as _DGM_d

			return _DGM_d
		except Exception as second_exc:
			raise second_exc from first_exc


def resolve_device(config: dict) -> torch.device:
	"""Resolve the torch device from `config`."""

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
				print(f"[DEVICE] Using cuda (invalid config['gpu']={gpu_id})")
				return torch.device("cuda")

		print("[DEVICE] Using cuda (default)")
		return torch.device("cuda")

	print("[DEVICE] Using cpu (CUDA not available)")
	return torch.device("cpu")


class SimpleGCN(torch.nn.Module):
	"""A minimal GCN for joint training."""

	def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
		super().__init__()
		if GCNConv is None:  # pragma: no cover
			raise ModuleNotFoundError(
				"torch_geometric is required for the offline row-graphification. "
				"Install torch-geometric to use this feature."
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
	"""Z-score standardization along a given dimension."""

	mean = x.mean(dim=dim, keepdim=True)
	std = x.std(dim=dim, keepdim=True).clamp_min(eps)
	return (x - mean) / std


def _symmetrize_and_self_loop(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
	"""Make edges undirected, add self-loops, and deduplicate."""

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

	- `DGM_d` returns `edge_index` as (neighbor -> node) edges.
	- `logprobs_dgm` is shaped like [B, N, K] (often B=1), and flattening it
	  matches the flattening used to build `edge_index`.

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


def row_gnn_after_start_fn(
	train_df: pd.DataFrame,
	val_df: pd.DataFrame,
	test_df: pd.DataFrame,
	config: dict,
	task_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
	"""Offline row-embedding extraction between start and materialize.

	Paper-aligned behavior: use raw row feature vectors as node features.
	No self-attention or attention pooling is used in row graphification.

	Output schema:
	  - Numerical embedding columns: N_feature_emb_1..N_feature_emb_d
	  - Target preserved in `target`
	"""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError(
			"pandas is required for the ResNet start-stage row-level graphification. Install pandas to use this feature."
		) from _PANDAS_IMPORT_ERROR

	try:
		DGM_d_local = _import_dgm_d()
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"DGM_d (from DGM_pytorch/DGMlib) is required for the start-stage row-level injection. "
			"Ensure the DGM_pytorch module is present and importable."
		) from e

	device = resolve_device(config)
	print(f"[START-GNN-DGM] Using device: {device}")

	dgm_k = int(config.get("gnn_dgm_k", config.get("dgm_k", 10)))
	print(f"[DGM-K] Using dgm_k={dgm_k}")

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
	num_cols = len(feature_cols)

	x_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32, device=device)
	x_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32, device=device)
	x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32, device=device)
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

	n_train = len(train_df)

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

	gnn = SimpleGCN(num_cols, gnn_hidden, gnn_out_dim).to(device)
	pred_head = torch.nn.Linear(gnn_out_dim, out_dim).to(device)

	params = list(gnn.parameters()) + list(pred_head.parameters()) + list(dgm_module.parameters())
	if dgm_gate_threshold_param is not None:
		params += [dgm_gate_threshold_param]
	optimizer = torch.optim.Adam(params, lr=lr)

	best_val_loss = float("inf")
	early_stop_counter = 0
	gnn_early_stop_epochs = 0
	best_states = None

	def forward_pass(x_tensor, dgm_module_inst):
		ns = x_tensor.shape[0]
		row_emb_std = _standardize(x_tensor, dim=0)
		row_emb_batched = row_emb_std.contiguous().unsqueeze(0)
		if hasattr(dgm_module_inst, "k"):
			dgm_module_inst.k = int(min(int(dgm_module_inst.k), max(1, ns - 1)))
		row_emb_dgm, edge_index_dgm, logprobs_dgm = dgm_module_inst(row_emb_batched, A=None)
		row_emb_dgm = row_emb_dgm.squeeze(0)

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

		gcn_out = gnn(row_emb_dgm, edge_index_dgm, edge_weight=edge_weight_dgm)
		logits = pred_head(gcn_out)

		e = edge_index_dgm.shape[1]
		avg_deg = e / max(1, ns)
		print(f"[START-GNN-DGM] split ns={ns}, edges={e}, avg_deg={avg_deg:.2f}")

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

		train_emb = emb_train.detach().cpu().numpy()
		val_emb = emb_val.detach().cpu().numpy()
		test_emb = emb_test.detach().cpu().numpy()

	row_emb_cols = [f"N_feature_emb_{i + 1}" for i in range(int(gnn_out_dim))]
	train_df_emb = pd.DataFrame(train_emb, columns=row_emb_cols, index=train_df.index)
	val_df_emb = pd.DataFrame(val_emb, columns=row_emb_cols, index=val_df.index)
	test_df_emb = pd.DataFrame(test_emb, columns=row_emb_cols, index=test_df.index)
	train_df_emb["target"] = train_df["target"].values
	val_df_emb["target"] = val_df["target"].values
	test_df_emb["target"] = test_df["target"].values

	return train_df_emb, val_df_emb, test_df_emb, gnn_early_stop_epochs


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
			"pandas is required for the ResNet feature-level placeholder. Install pandas to use this model."
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


def _init_feature_topology_prior(feature_names, init_mode: str = "chain") -> dict:
	"""Initialize a schema-level feature topology prior."""

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
	"""Return a JSON-serializable subset of the topology dict."""

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


def row_gnn_after_materialize_fn(
	train_tensor_frame,
	val_tensor_frame,
	test_tensor_frame,
	config: dict,
	dataset_name: str,
	task_type: str,
):
	"""Offline row-embedding extraction after `materialize_fn`.

	This is a direct port of the ExcelFormer paper-aligned row graphification.
	"""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError(
			"pandas is required for the materialize-stage row-level graphification. Install pandas to use this feature."
		) from _PANDAS_IMPORT_ERROR

	try:
		DGM_d_local = _import_dgm_d()
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"DGM_d (from DGM_pytorch/DGMlib) is required for the materialize-stage row-level injection. "
			"Ensure the DGM_pytorch module is present and importable."
		) from e

	repo_root = Path(__file__).resolve().parents[2]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

	try:
		from torch_frame import stype
		from torch_frame.data.loader import DataLoader
		from torch_frame.datasets.yandex import Yandex
		from torch_frame.transforms import CatToNumTransform, MutualInformationSort
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"Failed to import local torch_frame components required by the materialize-stage graphification."
		) from e

	device = resolve_device(config)
	print(f"[MATERIALIZE-GNN-DGM] Using device: {device}")

	def tensor_frame_to_df(tensor_frame):
		col_names = tensor_frame.col_names_dict[stype.numerical]
		features = tensor_frame.feat_dict[stype.numerical].detach().cpu().numpy()
		df = pd.DataFrame(features, columns=col_names)
		if hasattr(tensor_frame, "y") and tensor_frame.y is not None:
			df["target"] = tensor_frame.y.detach().cpu().numpy()
		return df

	train_df = tensor_frame_to_df(train_tensor_frame)
	val_df = tensor_frame_to_df(val_tensor_frame)
	test_df = tensor_frame_to_df(test_tensor_frame)

	all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
	feature_cols = [c for c in all_df.columns if c != "target"]
	num_cols = len(feature_cols)

	if "gnn_dgm_k" in config:
		dgm_k = int(config["gnn_dgm_k"])
		print(f"[DGM-K] Using user-specified gnn_dgm_k={dgm_k}")
	elif "dgm_k" in config:
		dgm_k = int(config["dgm_k"])
		print(f"[DGM-K] Using user-specified dgm_k={dgm_k}")
	else:
		dgm_k = 10

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

	x_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32, device=device)
	x_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32, device=device)
	x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32, device=device)
	y_train_np = train_df["target"].values
	y_val_np = val_df["target"].values
	y_test_np = test_df["target"].values

	if task_type in ["binclass", "multiclass"]:
		y_all_np = np.concatenate([y_train_np, y_val_np, y_test_np])
		num_classes = len(pd.unique(y_all_np))
		if task_type == "binclass" and num_classes != 2:
			num_classes = 2
		print(f"[MATERIALIZE-GNN-DGM] Detected num_classes: {int(num_classes)}")
		out_dim = int(num_classes)
	else:
		out_dim = 1

	n_train = len(train_df)

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

	gnn = SimpleGCN(num_cols, gnn_hidden, gnn_out_dim).to(device)
	pred_head = torch.nn.Linear(gnn_out_dim, out_dim).to(device)

	params = list(gnn.parameters()) + list(pred_head.parameters()) + list(dgm_module.parameters())
	if dgm_gate_threshold_param is not None:
		params += [dgm_gate_threshold_param]
	optimizer = torch.optim.Adam(params, lr=lr)

	best_val_loss = float("inf")
	early_stop_counter = 0
	gnn_early_stop_epochs = 0
	best_states = None

	def forward_pass(x_tensor, dgm_module_inst):
		ns = x_tensor.shape[0]
		row_emb_std = _standardize(x_tensor, dim=0)
		row_emb_batched = row_emb_std.contiguous().unsqueeze(0)
		if hasattr(dgm_module_inst, "k"):
			dgm_module_inst.k = int(min(int(dgm_module_inst.k), max(1, ns - 1)))
		row_emb_dgm, edge_index_dgm, logprobs_dgm = dgm_module_inst(row_emb_batched, A=None)
		row_emb_dgm = row_emb_dgm.squeeze(0)

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

		gcn_out = gnn(row_emb_dgm, edge_index_dgm, edge_weight=edge_weight_dgm)
		logits = pred_head(gcn_out)

		e = edge_index_dgm.shape[1]
		avg_deg = e / max(1, ns)
		print(f"[MATERIALIZE-GNN-DGM] split Ns={ns}, edges={e}, avg_deg={avg_deg:.2f}")
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

		suffix = " ↓ (improved)" if improved else ""
		print(
			f"[MATERIALIZE-GNN-DGM] Epoch {epoch + 1}/{gnn_epochs}, "
			f"Train Loss: {train_loss_val:.4f}, Val Loss: {val_loss_val:.4f}{suffix}"
		)
		if early_stop_counter >= patience:
			gnn_early_stop_epochs = epoch + 1
			print(
				f"[MATERIALIZE-GNN-DGM] Early stopping at epoch {epoch + 1} (Val Loss: {val_loss_val:.4f})"
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
		train_emb = emb_train.detach().cpu().numpy()
		val_emb = emb_val.detach().cpu().numpy()
		test_emb = emb_test.detach().cpu().numpy()

	row_emb_cols = [f"N_feature_emb_{i + 1}" for i in range(int(gnn_out_dim))]
	train_df_emb = pd.DataFrame(train_emb, columns=row_emb_cols, index=train_df.index)
	val_df_emb = pd.DataFrame(val_emb, columns=row_emb_cols, index=val_df.index)
	test_df_emb = pd.DataFrame(test_emb, columns=row_emb_cols, index=test_df.index)
	train_df_emb["target"] = train_df["target"].values
	val_df_emb["target"] = val_df["target"].values
	test_df_emb["target"] = test_df["target"].values

	dataset = Yandex(train_df_emb, val_df_emb, test_df_emb, name=dataset_name, task_type=task_type)
	try:
		dataset.materialize(device=device)
	except TypeError:
		dataset.materialize()

	train_tensor_frame = dataset.tensor_frame[dataset.df["split_col"] == 0]
	val_tensor_frame = dataset.tensor_frame[dataset.df["split_col"] == 1]
	test_tensor_frame = dataset.tensor_frame[dataset.df["split_col"] == 2]

	categorical_transform = CatToNumTransform()
	categorical_transform.fit(train_tensor_frame, dataset.col_stats)
	train_tensor_frame = categorical_transform(train_tensor_frame)
	val_tensor_frame = categorical_transform(val_tensor_frame)
	test_tensor_frame = categorical_transform(test_tensor_frame)
	col_stats = categorical_transform.transformed_stats

	mutual_info_sort = MutualInformationSort(task_type=dataset.task_type)
	mutual_info_sort.fit(train_tensor_frame, col_stats)
	train_tensor_frame = mutual_info_sort(train_tensor_frame)
	val_tensor_frame = mutual_info_sort(val_tensor_frame)
	test_tensor_frame = mutual_info_sort(test_tensor_frame)

	batch_size = int(config.get("batch_size", 512))
	train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
	test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

	return (
		train_loader,
		val_loader,
		test_loader,
		col_stats,
		mutual_info_sort,
		dataset,
		train_tensor_frame,
		val_tensor_frame,
		test_tensor_frame,
		gnn_early_stop_epochs,
	)


def feature_gnn_after_materialize_fn(material_outputs: dict, config: dict, dataset_name: str, task_type: str):
	"""Feature-level topology init after `materialize` (inactive)."""

	repo_root = Path(__file__).resolve().parents[2]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

	try:
		from torch_frame import stype
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError("Failed to import torch_frame.stype for feature topology init.") from e

	train_tensor_frame = material_outputs["train_tensor_frame"]
	feature_names = list(train_tensor_frame.col_names_dict.get(stype.numerical, []))
	if len(feature_names) == 0:
		raise ValueError(
			"No numerical features found in TensorFrame for feature-level topology init. "
			"(Expected numerical stype after CatToNumTransform.)"
		)

	init_mode = str(config.get("feature_topology_init", "chain")).lower()
	topology = _init_feature_topology_prior(feature_names, init_mode=init_mode)
	topology["dataset_name"] = str(dataset_name)
	topology["task_type"] = str(task_type)

	material_outputs["feature_topology"] = topology
	config["feature_topology_meta"] = _feature_topology_to_meta(topology)

	print(
		f"[MATERIALIZE-GNN] Feature-level topology initialized (inactive). "
		f"init_mode={topology['init_mode']}, num_features={topology['num_features']}"
	)

	return material_outputs


def start_fn(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
	"""Stage 0: start (pass-through)."""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError("pandas is required to run this model.") from _PANDAS_IMPORT_ERROR
	return train_df, val_df, test_df


def materialize_fn(train_df, val_df, test_df, dataset_results, config):
	"""Stage 1: Materialization."""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError("pandas is required to run materialize_fn.") from _PANDAS_IMPORT_ERROR

	print("Executing materialize_fn")
	print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

	repo_root = Path(__file__).resolve().parents[2]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

	try:
		from torch_frame.data.loader import DataLoader
		from torch_frame.datasets.yandex import Yandex
		from torch_frame import stype
		from torch_frame.nn.encoder.stype_encoder import (
			EmbeddingEncoder,
			LinearBucketEncoder,
			LinearEncoder,
			LinearPeriodicEncoder,
		)
		from torch_frame.transforms import CatToNumTransform, MutualInformationSort
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"Failed to import local torch_frame components required by ResNet materialize_fn."
		) from e

	try:
		from torchmetrics import AUROC, Accuracy, MeanSquaredError
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"torchmetrics is required for ResNet metrics. Install torchmetrics to use this model."
		) from e

	dataset_name = dataset_results["dataset"]
	task_type = dataset_results["info"]["task_type"]

	device = resolve_device(config)
	print(f"[MATERIALIZE] Device: {device}")

	dataset = Yandex(train_df, val_df, test_df, name=dataset_name, task_type=task_type)
	dataset.materialize(device=device)

	train_tensor_frame = dataset.tensor_frame[dataset.df["split_col"] == 0]
	val_tensor_frame = dataset.tensor_frame[dataset.df["split_col"] == 1]
	test_tensor_frame = dataset.tensor_frame[dataset.df["split_col"] == 2]

	categorical_transform = CatToNumTransform()
	categorical_transform.fit(train_tensor_frame, dataset.col_stats)
	train_tensor_frame = categorical_transform(train_tensor_frame)
	val_tensor_frame = categorical_transform(val_tensor_frame)
	test_tensor_frame = categorical_transform(test_tensor_frame)
	col_stats = categorical_transform.transformed_stats

	mutual_info_sort = MutualInformationSort(task_type=dataset.task_type)
	mutual_info_sort.fit(train_tensor_frame, col_stats)
	train_tensor_frame = mutual_info_sort(train_tensor_frame)
	val_tensor_frame = mutual_info_sort(val_tensor_frame)
	test_tensor_frame = mutual_info_sort(test_tensor_frame)

	batch_size = int(config.get("batch_size", 512))
	train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
	test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

	is_classification = bool(dataset.task_type.is_classification)
	out_channels = int(dataset.num_classes) if is_classification else 1
	is_binary_class = bool(is_classification and out_channels == 2)

	if is_binary_class:
		metric_computer = AUROC(task="binary")
		metric = "AUC"
	elif is_classification:
		metric_computer = Accuracy(task="multiclass", num_classes=out_channels)
		metric = "Acc"
	else:
		metric_computer = MeanSquaredError()
		metric = "RMSE"
	metric_computer = metric_computer.to(device)

	numerical_encoder_type = str(config.get("numerical_encoder", "linear")).lower()
	if numerical_encoder_type == "linear":
		numerical_encoder = LinearEncoder()
	elif numerical_encoder_type == "linearbucket":
		numerical_encoder = LinearBucketEncoder()
	elif numerical_encoder_type == "linearperiodic":
		numerical_encoder = LinearPeriodicEncoder()
	else:
		raise ValueError(
			"Unsupported numerical_encoder. Expected one of: 'linear', 'linearbucket', 'linearperiodic'."
		)

	stype_encoder_dict = {
		stype.categorical: EmbeddingEncoder(),
		stype.numerical: numerical_encoder,
	}

	return {
		"dataset": dataset,
		"train_tensor_frame": train_tensor_frame,
		"val_tensor_frame": val_tensor_frame,
		"test_tensor_frame": test_tensor_frame,
		"train_loader": train_loader,
		"val_loader": val_loader,
		"test_loader": test_loader,
		"col_stats": col_stats,
		"mutual_info_sort": mutual_info_sort,
		"stype_encoder_dict": stype_encoder_dict,
		"metric_computer": metric_computer,
		"metric": metric,
		"is_classification": is_classification,
		"is_binary_class": is_binary_class,
		"out_channels": out_channels,
		"device": device,
	}


class FCResidualBlock(torch.nn.Module):
	"""Two-layer MLP residual block (Revisiting tabular DL ResNet)."""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		*,
		normalization: str | None = "layer_norm",
		dropout_prob: float = 0.2,
	) -> None:
		super().__init__()
		self.lin1 = torch.nn.Linear(in_channels, out_channels)
		self.lin2 = torch.nn.Linear(out_channels, out_channels)
		self.relu = torch.nn.ReLU()
		self.dropout = torch.nn.Dropout(p=float(dropout_prob))

		normalization = None if normalization is None else str(normalization).lower()
		if normalization == "batch_norm":
			self.norm1 = torch.nn.BatchNorm1d(out_channels)
			self.norm2 = torch.nn.BatchNorm1d(out_channels)
		elif normalization == "layer_norm":
			self.norm1 = torch.nn.LayerNorm(out_channels)
			self.norm2 = torch.nn.LayerNorm(out_channels)
		elif normalization in {"none", ""}:
			self.norm1 = None
			self.norm2 = None
		else:
			raise ValueError(
				"Unknown normalization for ResNet: "
				f"{normalization!r}. Expected 'batch_norm', 'layer_norm', or None."
			)

		if in_channels != out_channels:
			self.shortcut = torch.nn.Linear(in_channels, out_channels)
		else:
			self.shortcut = None

	def reset_parameters(self) -> None:
		self.lin1.reset_parameters()
		self.lin2.reset_parameters()
		if self.norm1 is not None:
			self.norm1.reset_parameters()
		if self.norm2 is not None:
			self.norm2.reset_parameters()
		if self.shortcut is not None:
			self.shortcut.reset_parameters()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.lin1(x)
		out = self.norm1(out) if self.norm1 is not None else out
		out = self.relu(out)
		out = self.dropout(out)

		out = self.lin2(out)
		out = self.norm2(out) if self.norm2 is not None else out
		out = self.relu(out)
		out = self.dropout(out)

		if self.shortcut is not None:
			x = self.shortcut(x)
		return out + x


def resnet_core_fn_row_gnn(material_outputs, config, task_type: str, gnn_stage=None):
	"""ResNet core function (row-GNN capable).

	For parity with `excelformer.py` and `fttransformer.py`, this core supports an
	optional row-level dynamic-graph injection when `gnn_stage` is one of:
	  - "encoding": inject before the ResNet backbone.
	  - "columnwise": same injection point (ResNet has no columnwise blocks).
	  - "decoding": replace the decoder with a (DGM_d + GCN) predictor.
	"""

	print("Executing resnet_core_fn_row_gnn")
	print(f"[CORE] gnn_stage={gnn_stage}")

	repo_root = Path(__file__).resolve().parents[2]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

	try:
		import copy
		from torch.optim.lr_scheduler import ExponentialLR
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError("PyTorch is required to run ResNet.") from e

	try:
		from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError("Failed to import torch_frame encoder components.") from e

	try:
		from tqdm import tqdm
	except Exception:  # pragma: no cover
		def tqdm(x, **kwargs):
			return x

	train_tensor_frame = material_outputs["train_tensor_frame"]
	train_loader = material_outputs["train_loader"]
	val_loader = material_outputs["val_loader"]
	test_loader = material_outputs["test_loader"]
	col_stats = material_outputs["col_stats"]
	stype_encoder_dict = material_outputs["stype_encoder_dict"]
	device = material_outputs["device"]
	out_channels = int(material_outputs["out_channels"])
	is_classification = bool(material_outputs["is_classification"])
	is_binary_class = bool(material_outputs["is_binary_class"])
	metric_computer = material_outputs["metric_computer"]
	metric = material_outputs["metric"]

	stage = "none" if gnn_stage is None else str(gnn_stage).lower()
	if stage == "materialization":
		stage = "materialize"
	if stage == "columnwise":
		print(
			"[CORE][WARN] ResNet has no explicit columnwise-interaction blocks; "
			"`gnn_stage=columnwise` uses the same injection point as `encoding` "
			"(after feature encoding, before flatten/backbone)."
		)

	channels = int(config.get("channels", 256))
	num_layers = int(config.get("num_layers", 6))
	dropout = float(config.get("dropout", 0.2))
	normalization = config.get("normalization", "layer_norm")
	patience = int(config.get("patience", 10))
	loss_threshold = float(config.get("loss_threshold", 1e-4))
	restore_best = bool(config.get("restore_best", True))

	use_gnn = stage in ["encoding", "columnwise", "decoding"]
	gnn_hidden = int(config.get("gnn_hidden", 64))

	DGM_d_local = None
	if use_gnn:
		try:
			DGM_d_local = _import_dgm_d()
		except Exception as e:  # pragma: no cover
			raise ModuleNotFoundError(
				"DGM_d (from DGM_pytorch/DGMlib) is required for encoding/columnwise/decoding GNN stages."
			) from e

	col_names_dict = train_tensor_frame.col_names_dict
	feature_encoder = StypeWiseFeatureEncoder(
		out_channels=channels,
		col_stats=col_stats,
		col_names_dict=col_names_dict,
		stype_encoder_dict=stype_encoder_dict,
	).to(device)

	num_cols = int(sum(len(v) for v in col_names_dict.values()))
	in_channels = int(channels * num_cols)

	backbone = torch.nn.Sequential(
		*[
			FCResidualBlock(
				in_channels if i == 0 else channels,
				channels,
				normalization=normalization,
				dropout_prob=dropout,
			)
			for i in range(num_layers)
		]
	).to(device)

	decoder = None
	if stage != "decoding":
		decoder = torch.nn.Sequential(
			torch.nn.LayerNorm(channels),
			torch.nn.ReLU(),
			torch.nn.Linear(channels, out_channels),
		).to(device)

	# Optional GNN components (mirrors FTTransformer implementation).
	gnn = None
	dgm_module = None
	self_attn = None
	attn_norm = None
	column_embed = None
	pool_query = None
	self_attn_out = None
	gcn_to_attn = None
	attn_out_norm = None
	ffn_pre = None
	ffn_post = None
	fusion_alpha_param = None
	dgm_gate_threshold_param = None

	if use_gnn:
		attn_heads = int(config.get("gnn_num_heads", 4))
		dgm_k = int(config.get("gnn_dgm_k", config.get("dgm_k", config.get("gnn_knn", 5))))
		dgm_distance = str(config.get("gnn_dgm_distance", config.get("dgm_distance", "euclidean")))
		dgm_gate = str(config.get("gnn_dgm_gate", "none")).lower()
		dgm_gate_sharpness = float(config.get("gnn_dgm_gate_sharpness", 1.0))
		dgm_gate_threshold_init = float(config.get("gnn_dgm_gate_threshold_init", -10.0))
		dgm_eval_prune = bool(config.get("gnn_dgm_eval_prune", False))
		dgm_eval_prune_threshold = float(config.get("gnn_dgm_eval_prune_threshold", 0.5))

		class DGMEmbedWrapper(torch.nn.Module):
			def forward(self, x, A=None):
				return x

		dgm_embed_f = DGMEmbedWrapper()
		dgm_module = DGM_d_local(dgm_embed_f, k=dgm_k, distance=dgm_distance).to(device)
		if dgm_gate != "none":
			dgm_gate_threshold_param = torch.nn.Parameter(
				torch.tensor(dgm_gate_threshold_init, device=device, dtype=torch.float32)
			)

		self_attn = torch.nn.MultiheadAttention(
			embed_dim=channels, num_heads=attn_heads, batch_first=True
		).to(device)
		attn_norm = torch.nn.LayerNorm(channels).to(device)
		column_embed = torch.nn.Parameter(torch.randn(num_cols, channels, device=device))
		pool_query = torch.nn.Parameter(torch.randn(channels, device=device))

		if stage == "decoding":
			gnn = SimpleGCN(channels, gnn_hidden, out_channels, num_layers=2).to(device)
			print("[CORE] Decoding stage replaces decoder with Self-Attention + DGM_d + GCN.")
			print(f"[CORE]  - attention_heads={attn_heads}, dgm_k={dgm_k}, dgm_distance={dgm_distance}")
			print(f"[CORE]  - gnn_hidden={gnn_hidden}, out_channels={out_channels}")
		else:
			gnn = SimpleGCN(channels, gnn_hidden, channels, num_layers=2).to(device)
			self_attn_out = torch.nn.MultiheadAttention(
				embed_dim=channels, num_heads=attn_heads, batch_first=True
			).to(device)
			gcn_to_attn = torch.nn.Linear(channels, channels).to(device)
			attn_out_norm = torch.nn.LayerNorm(channels).to(device)

			gnn_dropout = float(config.get("gnn_dropout", 0.1))
			ffn_pre = torch.nn.Sequential(
				torch.nn.Linear(channels, channels * 2),
				torch.nn.GELU(),
				torch.nn.Dropout(gnn_dropout),
				torch.nn.Linear(channels * 2, channels),
			).to(device)
			ffn_post = torch.nn.Sequential(
				torch.nn.Linear(channels, channels * 2),
				torch.nn.GELU(),
				torch.nn.Dropout(gnn_dropout),
				torch.nn.Linear(channels * 2, channels),
			).to(device)
			fusion_alpha_param = torch.nn.Parameter(torch.tensor(-0.847, device=device))

			label = "encoding" if stage == "encoding" else "columnwise"
			print(f"[CORE] {label} stage uses Self-Attention + DGM_d + GCN, then fuses back into tokens.")
			print(f"[CORE]  - attention_heads={attn_heads}, dgm_k={dgm_k}, dgm_distance={dgm_distance}")
			print(f"[CORE]  - gnn_hidden={gnn_hidden}, channels={channels}")

	def _dgm_forward(x_pooled: torch.Tensor):
		x_pooled_std = _standardize(x_pooled, dim=0)
		x_pooled_batched = x_pooled_std.contiguous().unsqueeze(0)
		if dgm_module is not None and hasattr(dgm_module, "k"):
			ns = x_pooled_batched.shape[1]
			dgm_module.k = int(min(int(dgm_module.k), max(1, ns - 1)))
		x_dgm, edge_index_dgm, logprobs_dgm = dgm_module(x_pooled_batched, A=None)
		x_dgm = x_dgm.squeeze(0)
		edge_index_dgm, edge_weight_dgm = _dgm_gate_and_prune_edges(
			edge_index_dgm,
			logprobs_dgm,
			x_dgm.shape[0],
			gate=dgm_gate,
			gate_sharpness=dgm_gate_sharpness,
			gate_threshold_param=dgm_gate_threshold_param,
			is_training=bool(dgm_module.training),
			eval_prune=dgm_eval_prune,
			eval_prune_threshold=dgm_eval_prune_threshold,
		)
		return x_dgm, edge_index_dgm, edge_weight_dgm

	def model_forward(tf) -> torch.Tensor:
		x, _ = feature_encoder(tf)  # [B, F, C]
		batch_size, num_cols_local, channels_local = x.shape

		# Encoding/columnwise-stage injection (ResNet shares the same injection point).
		if stage in ["encoding", "columnwise"] and dgm_module is not None:
			tokens = x + column_embed.unsqueeze(0)
			tokens_norm = attn_norm(tokens)
			attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
			tokens_attn = tokens + attn_out1
			ffn_out1 = ffn_pre(attn_norm(tokens_attn))
			tokens_attn = tokens_attn + ffn_out1

			pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(float(channels_local))
			pool_weights = torch.softmax(pool_logits, dim=1)
			x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)

			x_dgm, edge_index_dgm, edge_weight_dgm = _dgm_forward(x_pooled)
			x_gnn_out = gnn(x_dgm, edge_index_dgm, edge_weight=edge_weight_dgm)

			gcn_ctx = gcn_to_attn(x_gnn_out).unsqueeze(1)
			tokens_with_ctx = tokens_attn + gcn_ctx
			tokens_ctx_norm = attn_out_norm(tokens_with_ctx)
			attn_out2, _ = self_attn_out(tokens_ctx_norm, tokens_ctx_norm, tokens_ctx_norm)
			tokens_mid = tokens_with_ctx + attn_out2
			ffn_out2 = ffn_post(attn_out_norm(tokens_mid))
			tokens_out = tokens_mid + ffn_out2

			fusion_alpha = torch.sigmoid(fusion_alpha_param)
			x = x + fusion_alpha * tokens_out

		x_flat = x.view(x.size(0), math.prod(x.shape[1:]))
		x_back = backbone(x_flat)

		# Decoding-stage injection (replaces decoder).
		if stage == "decoding" and dgm_module is not None:
			tokens = x + column_embed.unsqueeze(0)
			tokens_norm = attn_norm(tokens)
			attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
			tokens_attn = tokens + attn_out1

			pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(float(channels_local))
			pool_weights = torch.softmax(pool_logits, dim=1)
			x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)

			# Combine ResNet representation with pooled token context.
			x_pooled = x_pooled + x_back
			x_dgm, edge_index_dgm, edge_weight_dgm = _dgm_forward(x_pooled)
			out = gnn(x_dgm, edge_index_dgm, edge_weight=edge_weight_dgm)
			return out

		return decoder(x_back)

	lr = float(config.get("lr", 1e-3))
	gamma = float(config.get("gamma", 0.95))

	all_params = list(feature_encoder.parameters()) + list(backbone.parameters())
	if decoder is not None:
		all_params += list(decoder.parameters())
	if gnn is not None:
		all_params += list(gnn.parameters())
	if dgm_module is not None:
		all_params += list(dgm_module.parameters())
	if self_attn is not None:
		all_params += list(self_attn.parameters())
	if attn_norm is not None:
		all_params += list(attn_norm.parameters())
	if column_embed is not None:
		all_params += [column_embed]
	if pool_query is not None:
		all_params += [pool_query]
	if self_attn_out is not None:
		all_params += list(self_attn_out.parameters())
	if gcn_to_attn is not None:
		all_params += list(gcn_to_attn.parameters())
	if attn_out_norm is not None:
		all_params += list(attn_out_norm.parameters())
	if ffn_pre is not None:
		all_params += list(ffn_pre.parameters())
	if ffn_post is not None:
		all_params += list(ffn_post.parameters())
	if fusion_alpha_param is not None:
		all_params += [fusion_alpha_param]
	if use_gnn and dgm_gate_threshold_param is not None:
		all_params += [dgm_gate_threshold_param]

	optimizer = torch.optim.Adam(all_params, lr=lr)
	lr_scheduler = ExponentialLR(optimizer, gamma=gamma)

	def _set_train_mode():
		feature_encoder.train()
		backbone.train()
		if decoder is not None:
			decoder.train()
		if gnn is not None:
			gnn.train()
		if dgm_module is not None:
			dgm_module.train()
		if self_attn is not None:
			self_attn.train()
		if attn_norm is not None:
			attn_norm.train()
		if self_attn_out is not None:
			self_attn_out.train()
		if gcn_to_attn is not None:
			gcn_to_attn.train()
		if attn_out_norm is not None:
			attn_out_norm.train()
		if ffn_pre is not None:
			ffn_pre.train()
		if ffn_post is not None:
			ffn_post.train()

	def _set_eval_mode():
		feature_encoder.eval()
		backbone.eval()
		if decoder is not None:
			decoder.eval()
		if gnn is not None:
			gnn.eval()
		if dgm_module is not None:
			dgm_module.eval()
		if self_attn is not None:
			self_attn.eval()
		if attn_norm is not None:
			attn_norm.eval()
		if self_attn_out is not None:
			self_attn_out.eval()
		if gcn_to_attn is not None:
			gcn_to_attn.eval()
		if attn_out_norm is not None:
			attn_out_norm.eval()
		if ffn_pre is not None:
			ffn_pre.eval()
		if ffn_post is not None:
			ffn_post.eval()

	def _loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		if is_classification:
			return F.cross_entropy(pred, y)
		return F.mse_loss(pred.view(-1), y.view(-1))

	@torch.no_grad()
	def eval_loader(loader):
		_set_eval_mode()
		metric_computer.reset()
		total_loss = 0.0
		total_count = 0

		for tf in loader:
			tf = tf.to(device)
			pred = model_forward(tf)
			y = tf.y
			bs = int(len(tf))
			loss = _loss(pred, y)
			total_loss += float(loss) * bs
			total_count += bs

			if is_binary_class:
				metric_computer.update(pred[:, 1], y)
			elif is_classification:
				metric_computer.update(pred.argmax(dim=-1), y)
			else:
				metric_computer.update(pred.view(-1), y.view(-1))

		metric_val = metric_computer.compute().item()
		if not is_classification:
			metric_val = float(metric_val) ** 0.5

		avg_loss = total_loss / max(1, total_count)
		return float(metric_val), float(avg_loss)

	best_val_loss = float("inf")
	best_val_metric = -float("inf") if is_classification else float("inf")
	best_epoch = 0
	early_stop_counter = 0
	early_stop_epochs = int(config.get("epochs", 300))
	best_states = None

	train_losses = []
	train_metrics = []
	val_metrics = []
	val_losses = []

	epochs = int(config.get("epochs", 300))
	for epoch in range(1, epochs + 1):
		_set_train_mode()
		loss_accum = 0.0
		count = 0

		for tf in tqdm(train_loader, desc=f"Epoch: {epoch}"):
			tf = tf.to(device)
			pred = model_forward(tf)
			y = tf.y
			loss = _loss(pred, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			bs = int(len(tf))
			loss_accum += float(loss) * bs
			count += bs

		train_metric, _train_loss_eval = eval_loader(train_loader)
		val_metric, val_loss = eval_loader(val_loader)

		train_losses.append(loss_accum / max(1, count))
		train_metrics.append(train_metric)
		val_metrics.append(val_metric)
		val_losses.append(val_loss)

		improved = val_loss < best_val_loss - loss_threshold
		if improved:
			best_val_loss = val_loss
			best_val_metric = val_metric
			best_epoch = epoch
			early_stop_counter = 0
			best_states = {
				"encoder": copy.deepcopy(feature_encoder.state_dict()),
				"backbone": copy.deepcopy(backbone.state_dict()),
				"decoder": copy.deepcopy(decoder.state_dict()) if decoder is not None else None,
				"gnn": copy.deepcopy(gnn.state_dict()) if gnn is not None else None,
				"dgm_module": copy.deepcopy(dgm_module.state_dict()) if dgm_module is not None else None,
				"self_attn": copy.deepcopy(self_attn.state_dict()) if self_attn is not None else None,
				"attn_norm": copy.deepcopy(attn_norm.state_dict()) if attn_norm is not None else None,
				"self_attn_out": copy.deepcopy(self_attn_out.state_dict()) if self_attn_out is not None else None,
				"gcn_to_attn": copy.deepcopy(gcn_to_attn.state_dict()) if gcn_to_attn is not None else None,
				"attn_out_norm": copy.deepcopy(attn_out_norm.state_dict()) if attn_out_norm is not None else None,
				"ffn_pre": copy.deepcopy(ffn_pre.state_dict()) if ffn_pre is not None else None,
				"ffn_post": copy.deepcopy(ffn_post.state_dict()) if ffn_post is not None else None,
				"column_embed": column_embed.detach().clone() if column_embed is not None else None,
				"pool_query": pool_query.detach().clone() if pool_query is not None else None,
				"fusion_alpha": fusion_alpha_param.detach().clone() if fusion_alpha_param is not None else None,
				"dgm_gate_threshold": dgm_gate_threshold_param.detach().clone() if dgm_gate_threshold_param is not None else None,
			}
		else:
			early_stop_counter += 1

		marker = "✓" if improved else "✗"
		print(
			f"[CORE] Epoch {epoch:3d} {marker} | "
			f"Train Loss: {train_losses[-1]:.4f}, Train {metric}: {train_metric:.4f}, "
			f"Val {metric}: {val_metric:.4f}, Val Loss: {val_loss:.4f} | "
			f"Early Stop: {early_stop_counter}/{patience}"
		)

		lr_scheduler.step()
		if early_stop_counter >= patience:
			early_stop_epochs = epoch
			print(
				f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f})"
			)
			break

	if restore_best and best_states is not None:
		feature_encoder.load_state_dict(best_states["encoder"])
		backbone.load_state_dict(best_states["backbone"])
		if decoder is not None and best_states.get("decoder") is not None:
			decoder.load_state_dict(best_states["decoder"])
		if gnn is not None and best_states.get("gnn") is not None:
			gnn.load_state_dict(best_states["gnn"])
		if dgm_module is not None and best_states.get("dgm_module") is not None:
			dgm_module.load_state_dict(best_states["dgm_module"])
		if self_attn is not None and best_states.get("self_attn") is not None:
			self_attn.load_state_dict(best_states["self_attn"])
		if attn_norm is not None and best_states.get("attn_norm") is not None:
			attn_norm.load_state_dict(best_states["attn_norm"])
		if self_attn_out is not None and best_states.get("self_attn_out") is not None:
			self_attn_out.load_state_dict(best_states["self_attn_out"])
		if gcn_to_attn is not None and best_states.get("gcn_to_attn") is not None:
			gcn_to_attn.load_state_dict(best_states["gcn_to_attn"])
		if attn_out_norm is not None and best_states.get("attn_out_norm") is not None:
			attn_out_norm.load_state_dict(best_states["attn_out_norm"])
		if ffn_pre is not None and best_states.get("ffn_pre") is not None:
			ffn_pre.load_state_dict(best_states["ffn_pre"])
		if ffn_post is not None and best_states.get("ffn_post") is not None:
			ffn_post.load_state_dict(best_states["ffn_post"])
		if column_embed is not None and best_states.get("column_embed") is not None:
			with torch.no_grad():
				column_embed.copy_(best_states["column_embed"])
		if pool_query is not None and best_states.get("pool_query") is not None:
			with torch.no_grad():
				pool_query.copy_(best_states["pool_query"])
		if fusion_alpha_param is not None and best_states.get("fusion_alpha") is not None:
			with torch.no_grad():
				fusion_alpha_param.copy_(best_states["fusion_alpha"])
		if dgm_gate_threshold_param is not None and best_states.get("dgm_gate_threshold") is not None:
			with torch.no_grad():
				dgm_gate_threshold_param.copy_(best_states["dgm_gate_threshold"])
		print(f"[CORE] Restored best weights from epoch {best_epoch}")

	print(
		f"[CORE] Training complete. Best Val Loss: {best_val_loss:.4f}, "
		f"Best Val {metric}: {best_val_metric:.4f}"
	)
	test_metric, test_loss = eval_loader(test_loader)
	print(f"[CORE] Final Test {metric}: {test_metric:.4f}, Test Loss: {test_loss:.4f}")

	return {
		"gnn_early_stop_epochs": material_outputs.get("gnn_early_stop_epochs", 0),
		"train_losses": train_losses,
		"train_metrics": train_metrics,
		"val_metrics": val_metrics,
		"val_losses": val_losses,
		"best_val_loss": best_val_loss,
		"best_val_metric": best_val_metric,
		"final_metric": best_val_metric,
		"best_test_metric": test_metric,
		"encoder": feature_encoder,
		"backbone": backbone,
		"decoder": decoder,
		"gnn": gnn,
		"model_forward": model_forward,
		"train_loader": train_loader,
		"val_loader": val_loader,
		"test_loader": test_loader,
		"is_classification": is_classification,
		"is_binary_class": is_binary_class,
		"metric_computer": metric_computer,
		"metric": metric,
		"early_stop_epochs": early_stop_epochs,
		"gnn_stage": stage,
	}


def resnet_core_fn_feature_gnn(material_outputs, config, task_type: str, gnn_stage=None):
	"""ResNet core function implementing paper-aligned feature graphification."""

	stage = "none" if gnn_stage is None else str(gnn_stage).lower()
	if stage in {"none", "start", "materialize", "materialization"}:
		return resnet_core_fn_row_gnn(material_outputs, config, task_type, gnn_stage="none")

	if stage not in {"encoding", "columnwise", "decoding"}:
		raise ValueError(
			f"Unknown gnn_stage={stage!r} for graphify=feature. "
			"Expected one of: none/start/materialize/encoding/columnwise/decoding."
		)

	if stage == "columnwise":
		print(
			"[CORE][WARN] ResNet has no explicit columnwise-interaction blocks; "
			"`gnn_stage=columnwise` uses the same injection point as `encoding` "
			"(after feature encoding, before flatten/backbone)."
		)

	repo_root = Path(__file__).resolve().parents[2]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

	try:
		from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError("Failed to import torch_frame encoder components.") from e

	try:
		from torch.optim.lr_scheduler import ExponentialLR
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError("PyTorch is required to run ResNet.") from e

	try:
		from tqdm import tqdm
	except Exception:  # pragma: no cover
		def tqdm(x, **kwargs):
			return x

	train_tensor_frame = material_outputs["train_tensor_frame"]
	train_loader = material_outputs["train_loader"]
	val_loader = material_outputs["val_loader"]
	test_loader = material_outputs["test_loader"]
	col_stats = material_outputs["col_stats"]
	stype_encoder_dict = material_outputs["stype_encoder_dict"]
	device = material_outputs["device"]
	out_channels = int(material_outputs["out_channels"])
	is_classification = bool(material_outputs["is_classification"])
	is_binary_class = bool(material_outputs["is_binary_class"])
	metric_computer = material_outputs["metric_computer"]
	metric = material_outputs["metric"]

	feature_topology = material_outputs.get("feature_topology")
	if feature_topology is None:
		# Initialize a topology prior in-core if not created earlier.
		try:
			from torch_frame import stype
		except Exception as e:  # pragma: no cover
			raise ModuleNotFoundError("Failed to import torch_frame.stype for topology init.") from e

		feature_names = list(train_tensor_frame.col_names_dict.get(stype.numerical, []))
		init_mode = str(config.get("feature_topology_init", "chain")).lower()
		feature_topology = _init_feature_topology_prior(feature_names, init_mode=init_mode)
		material_outputs["feature_topology"] = feature_topology
		config["feature_topology_meta"] = _feature_topology_to_meta(feature_topology)
		print(
			f"[CORE] Feature topology prior initialized in-core (inactive earlier). "
			f"init_mode={feature_topology['init_mode']}, num_features={feature_topology['num_features']}"
		)

	allowed_mask = feature_topology["adjacency_mask"].to(device=device)
	topology_mixing = FeatureTopologyMaskedMixing(allowed_mask).to(device)
	fusion_alpha_param = torch.nn.Parameter(torch.tensor(-0.847, device=device))

	channels = int(config.get("channels", 256))
	num_layers = int(config.get("num_layers", 6))
	dropout = float(config.get("dropout", 0.2))
	normalization = config.get("normalization", "layer_norm")
	lr = float(config.get("lr", 1e-3))
	gamma = float(config.get("gamma", 0.95))
	patience = int(config.get("patience", 10))
	loss_threshold = float(config.get("loss_threshold", 1e-4))
	restore_best = bool(config.get("restore_best", True))

	col_names_dict = train_tensor_frame.col_names_dict
	feature_encoder = StypeWiseFeatureEncoder(
		out_channels=channels,
		col_stats=col_stats,
		col_names_dict=col_names_dict,
		stype_encoder_dict=stype_encoder_dict,
	).to(device)

	num_cols = sum(len(v) for v in col_names_dict.values())
	in_channels = int(channels * num_cols)
	backbone = torch.nn.Sequential(
		*[
			FCResidualBlock(
				in_channels if i == 0 else channels,
				channels,
				normalization=normalization,
				dropout_prob=dropout,
			)
			for i in range(num_layers)
		]
	).to(device)
	decoder = torch.nn.Sequential(
		torch.nn.LayerNorm(channels),
		torch.nn.ReLU(),
		torch.nn.Linear(channels, out_channels),
	).to(device)
	decoder_readout = None
	if stage == "decoding":
		# Match the staged semantics in transformer backbones:
		# decoding-stage feature graphification changes the readout, not the backbone.
		decoder_readout = torch.nn.Linear(channels, out_channels).to(device)

	def _feature_inject(x_tokens: torch.Tensor) -> torch.Tensor:
		x_mp = topology_mixing(x_tokens)
		alpha = torch.sigmoid(fusion_alpha_param)
		return x_tokens + alpha * (x_mp - x_tokens)

	def model_forward(tf) -> torch.Tensor:
		x_tokens, _ = feature_encoder(tf)  # [B, F, C]
		# ResNet has a single encoder->flatten backbone; feature-level injection happens on tokens.
		if stage in {"encoding", "columnwise"}:
			x_tokens = _feature_inject(x_tokens)
		x_flat = x_tokens.view(x_tokens.size(0), math.prod(x_tokens.shape[1:]))
		x_back = backbone(x_flat)

		if stage == "decoding":
			# Decode from feature-topology-mixed tokens (pooled) plus ResNet backbone representation.
			x_mp = topology_mixing(x_tokens)
			pooled = x_mp.mean(dim=1)
			pooled = pooled + x_back
			return decoder_readout(pooled)

		return decoder(x_back)

	params = (
		list(feature_encoder.parameters())
		+ list(backbone.parameters())
		+ list(decoder.parameters())
		+ list(topology_mixing.parameters())
		+ [fusion_alpha_param]
	)
	if decoder_readout is not None:
		params += list(decoder_readout.parameters())
	optimizer = torch.optim.Adam(params, lr=lr)
	lr_scheduler = ExponentialLR(optimizer, gamma=gamma)

	def _loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		if is_classification:
			return F.cross_entropy(pred, y)
		return F.mse_loss(pred.view(-1), y.view(-1))

	@torch.no_grad()
	def eval_loader(loader):
		feature_encoder.eval()
		backbone.eval()
		decoder.eval()
		if decoder_readout is not None:
			decoder_readout.eval()
		topology_mixing.eval()
		metric_computer.reset()

		total_loss = 0.0
		total_count = 0
		for tf in loader:
			tf = tf.to(device)
			pred = model_forward(tf)
			y = tf.y
			bs = int(len(tf))
			loss = _loss(pred, y)
			total_loss += float(loss) * bs
			total_count += bs

			if is_binary_class:
				metric_computer.update(pred[:, 1], y)
			elif is_classification:
				metric_computer.update(pred.argmax(dim=-1), y)
			else:
				metric_computer.update(pred.view(-1), y.view(-1))

		metric_val = metric_computer.compute().item()
		if not is_classification:
			metric_val = float(metric_val) ** 0.5
		avg_loss = total_loss / max(1, total_count)
		return float(metric_val), float(avg_loss)

	best_val_loss = float("inf")
	best_val_metric = -float("inf") if is_classification else float("inf")
	best_epoch = 0
	early_stop_counter = 0
	early_stop_epochs = int(config.get("epochs", 300))
	best_states = None

	train_losses = []
	train_metrics = []
	val_metrics = []
	val_losses = []

	epochs = int(config.get("epochs", 300))
	for epoch in range(1, epochs + 1):
		feature_encoder.train()
		backbone.train()
		decoder.train()
		if decoder_readout is not None:
			decoder_readout.train()
		topology_mixing.train()

		loss_accum = 0.0
		count = 0
		for tf in tqdm(train_loader, desc=f"Epoch: {epoch}"):
			tf = tf.to(device)
			pred = model_forward(tf)
			y = tf.y
			loss = _loss(pred, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			bs = int(len(tf))
			loss_accum += float(loss) * bs
			count += bs

		train_metric, _train_loss_eval = eval_loader(train_loader)
		val_metric, val_loss = eval_loader(val_loader)

		train_losses.append(loss_accum / max(1, count))
		train_metrics.append(train_metric)
		val_metrics.append(val_metric)
		val_losses.append(val_loss)

		improved = val_loss < best_val_loss - loss_threshold
		if improved:
			best_val_loss = val_loss
			best_val_metric = val_metric
			best_epoch = epoch
			early_stop_counter = 0
			best_states = {
				"encoder": feature_encoder.state_dict(),
				"backbone": backbone.state_dict(),
				"decoder": decoder.state_dict(),
				"decoder_readout": decoder_readout.state_dict() if decoder_readout is not None else None,
				"topology_mixing": topology_mixing.state_dict(),
				"fusion_alpha": fusion_alpha_param.detach().clone(),
			}
		else:
			early_stop_counter += 1

		alpha_raw = float(fusion_alpha_param.detach().cpu().item())
		alpha_sig = float(torch.sigmoid(fusion_alpha_param).detach().cpu().item())
		marker = "✓" if improved else "✗"
		print(
			f"[CORE] Epoch {epoch:3d} {marker} | "
			f"Train Loss: {train_losses[-1]:.4f}, Train {metric}: {train_metric:.4f}, "
			f"Val {metric}: {val_metric:.4f}, Val Loss: {val_loss:.4f} | "
			f"Early Stop: {early_stop_counter}/{patience} | "
			f"fusion_alpha_raw={alpha_raw:.3f} fusion_alpha_sigmoid={alpha_sig:.3f}"
		)

		lr_scheduler.step()
		if early_stop_counter >= patience:
			early_stop_epochs = epoch
			print(
				f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f})"
			)
			break

	if restore_best and best_states is not None:
		feature_encoder.load_state_dict(best_states["encoder"])
		backbone.load_state_dict(best_states["backbone"])
		decoder.load_state_dict(best_states["decoder"])
		if decoder_readout is not None and best_states.get("decoder_readout") is not None:
			decoder_readout.load_state_dict(best_states["decoder_readout"])
		topology_mixing.load_state_dict(best_states["topology_mixing"])
		with torch.no_grad():
			fusion_alpha_param.copy_(best_states["fusion_alpha"])
		print(f"[CORE] Restored best weights from epoch {best_epoch}")

	print(f"[CORE] Training complete. Best Val Loss: {best_val_loss:.4f}, Best Val {metric}: {best_val_metric:.4f}")
	test_metric, test_loss = eval_loader(test_loader)
	print(f"[CORE] Final Test {metric}: {test_metric:.4f}, Test Loss: {test_loss:.4f}")

	return {
		"gnn_early_stop_epochs": material_outputs.get("gnn_early_stop_epochs", 0),
		"train_losses": train_losses,
		"train_metrics": train_metrics,
		"val_metrics": val_metrics,
		"val_losses": val_losses,
		"best_val_loss": best_val_loss,
		"best_val_metric": best_val_metric,
		"final_metric": best_val_metric,
		"best_test_metric": test_metric,
		"encoder": feature_encoder,
		"backbone": backbone,
		"decoder": decoder,
		"decoder_readout": decoder_readout,
		"gnn": None,
		"model_forward": model_forward,
		"train_loader": train_loader,
		"val_loader": val_loader,
		"test_loader": test_loader,
		"is_classification": is_classification,
		"is_binary_class": is_binary_class,
		"metric_computer": metric_computer,
		"metric": metric,
		"early_stop_epochs": early_stop_epochs,
		"gnn_stage": stage,
		"feature_topology_meta": config.get("feature_topology_meta"),
		"fusion_alpha_raw": float(fusion_alpha_param.detach().cpu().item()),
		"fusion_alpha_sigmoid": float(torch.sigmoid(fusion_alpha_param).detach().cpu().item()),
	}


def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
	"""Model entrypoint called by the project runner."""

	print("ResNet - staged execution")
	print(f"gnn_stage: {gnn_stage}")
	task_type = dataset_results.get("info", {}).get("task_type", "binclass")

	stage = "none" if gnn_stage is None else str(gnn_stage).lower()
	if stage == "materialization":
		stage = "materialize"

	try:
		train_df, val_df, test_df = start_fn(train_df, val_df, test_df)

		graphify = str(config.get("graphify", "row")).lower()
		gnn_early_stop_epochs = 0

		if graphify == "row":
			if stage == "start":
				train_df, val_df, test_df, gnn_early_stop_epochs = row_gnn_after_start_fn(
					train_df, val_df, test_df, config, task_type
				)

			material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
			material_outputs["gnn_early_stop_epochs"] = gnn_early_stop_epochs

			if stage == "materialize":
				(
					train_loader,
					val_loader,
					test_loader,
					col_stats,
					mutual_info_sort,
					dataset,
					train_tensor_frame,
					val_tensor_frame,
					test_tensor_frame,
					gnn_early_stop_epochs,
				) = row_gnn_after_materialize_fn(
					material_outputs["train_tensor_frame"],
					material_outputs["val_tensor_frame"],
					material_outputs["test_tensor_frame"],
					config,
					dataset_results["dataset"],
					task_type,
				)
				material_outputs.update(
					{
						"train_loader": train_loader,
						"val_loader": val_loader,
						"test_loader": test_loader,
						"col_stats": col_stats,
						"mutual_info_sort": mutual_info_sort,
						"dataset": dataset,
						"train_tensor_frame": train_tensor_frame,
						"val_tensor_frame": val_tensor_frame,
						"test_tensor_frame": test_tensor_frame,
						"gnn_early_stop_epochs": gnn_early_stop_epochs,
					}
				)

				# Rebuild stype_encoder_dict for embedding-only dataset (numerical only).
				try:
					from torch_frame import stype
					from torch_frame.nn.encoder.stype_encoder import LinearEncoder

					material_outputs["stype_encoder_dict"] = {stype.numerical: LinearEncoder()}
				except Exception:
					pass

			results = resnet_core_fn_row_gnn(material_outputs, config, task_type, gnn_stage=stage)

		elif graphify == "feature":
			if stage == "start":
				train_df, val_df, test_df, gnn_early_stop_epochs = feature_gnn_after_start_fn(
					train_df, val_df, test_df, config, task_type
				)

			material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
			material_outputs["gnn_early_stop_epochs"] = gnn_early_stop_epochs

			if stage in {"materialize"}:
				material_outputs = feature_gnn_after_materialize_fn(
					material_outputs,
					config,
					dataset_results["dataset"],
					task_type,
				)

			results = resnet_core_fn_feature_gnn(material_outputs, config, task_type, gnn_stage=stage)

		else:
			raise ValueError(f"Unknown graphify mode: {graphify}. Expected 'row' or 'feature'.")

		return results
	except Exception as e:
		return {
			"gnn_stage": stage,
			"graphify": str(config.get("graphify", "row")),
			"error": str(e),
		}


#  small+binclass
#  python main.py --models resnet --dataset kaggle_Audit_Data --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  small+regression
#  python main.py --models resnet --dataset openml_The_Office_Dataset --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+binclass
#  python main.py --models resnet --dataset credit --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+multiclass
#  python main.py --models resnet --dataset eye --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  python main.py --models resnet --dataset helena --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+regression
#  python main.py --models resnet --dataset house --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0

