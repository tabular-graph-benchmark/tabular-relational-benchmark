from __future__ import annotations


import math
import sys
from pathlib import Path
from typing import Tuple

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
	"""Resolve the torch device from config."""

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


def _standardize(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
	"""Z-score standardization along a given dimension."""

	mean = x.mean(dim=dim, keepdim=True)
	std = x.std(dim=dim, keepdim=True, unbiased=False).clamp_min(eps)
	return (x - mean) / std


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

	If `gate == 'none'`, returns (edge_index, None).
	If `gate == 'sigmoid'`, returns (edge_index, edge_weight in [0,1]).
	"""

	if edge_index_dgm.numel() == 0:
		return edge_index_dgm, None

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

	# Symmetrize + add self-loops.
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
		# Best-effort dedup fallback.
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


def _unique_parameters(params):
	"""Deduplicate parameters while preserving order."""

	unique = []
	seen = set()
	for p in params:
		if p is None:
			continue
		pid = id(p)
		if pid in seen:
			continue
		seen.add(pid)
		unique.append(p)
	return unique


class SimpleGCN(torch.nn.Module):
	"""A minimal GCN for joint training."""

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
		"""tokens: [B, F, C] -> [B, F, C]"""

		if tokens.dim() != 3:
			raise ValueError(f"Expected tokens [B,F,C], got shape={tuple(tokens.shape)}")
		p = self._normalized_weights()
		return torch.einsum("ij,bjd->bid", p, tokens)


def row_gnn_after_start_fn(
	train_df: pd.DataFrame,
	val_df: pd.DataFrame,
	test_df: pd.DataFrame,
	config: dict,
	task_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
	"""Offline row-embedding extraction between start and materialize.

	Paper-aligned fairness rule: this hook uses *raw row feature vectors* as
	node features. No self-attention, input projection, or pooling is applied.

	Output schema:
	  - Features are replaced by numerical embedding columns: N_feature_emb_1 .. N_feature_emb_{d}
	  - Target is preserved in the `target` column.
	"""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError(
			"pandas is required for the TabNet start-stage row-GNN injection. Install pandas to use this feature."
		) from _PANDAS_IMPORT_ERROR

	try:
		DGM_d_local = _import_dgm_d()
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"DGM_d (from DGM_pytorch/DGMlib) is required for row-level injection. "
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

	standardize_mode = str(config.get("gnn_row_standardize", "split")).lower()
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
		x_std = _normalize(x_tensor)
		x_batched = x_std.contiguous().unsqueeze(0)
		if hasattr(dgm_module_inst, "k"):
			dgm_module_inst.k = int(min(int(dgm_module_inst.k), max(1, ns - 1)))
		x_dgm, edge_index_dgm, logprobs_dgm = dgm_module_inst(x_batched, A=None)
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
			print(f"[START-GNN-DGM] Early stopping at epoch {epoch+1} (Val Loss: {val_loss_val:.4f})")
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

	train_df_emb["target"] = y_train_np
	val_df_emb["target"] = y_val_np
	test_df_emb["target"] = y_test_np

	return train_df_emb, val_df_emb, test_df_emb, gnn_early_stop_epochs


def row_gnn_after_materialize_fn(
	train_tensor_frame,
	val_tensor_frame,
	test_tensor_frame,
	config: dict,
	dataset_name: str,
	task_type: str,
):
	"""Offline row-embedding extraction after `materialize_fn`.

	Pipeline (paper-aligned; matches ExcelFormer/FTTransformer in this repo):
	  1) Convert TensorFrames back into split DataFrames.
	  2) Use each row's raw feature vector as the node feature.
	  3) Train DGM_d dynamic graph -> GCN with supervised loss.
	  4) Replace features by numerical embedding columns N_feature_emb_1..N_feature_emb_{d}.
	  5) Re-wrap into a Yandex dataset, materialize, then apply CatToNum +
		 MutualInformationSort, and rebuild loaders.
	"""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError(
			"pandas is required for the materialize-stage row-level GNN injection. Install pandas to use this feature."
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
			"Failed to import local torch_frame components required by the materialize-stage GNN injection."
		) from e

	device = resolve_device(config)
	print(f"[MATERIALIZE-GNN-DGM] Using device: {device}")

	def tensor_frame_to_df(tensor_frame):
		# After CatToNum+sorting in this benchmark, usable features are numerical.
		col_names = list(tensor_frame.col_names_dict.get(stype.numerical, []))
		if len(col_names) == 0:
			raise ValueError("TensorFrame has no numerical features to export.")
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
			"pandas is required for the TabNet feature-level placeholder. Install pandas to use this model."
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


def start_fn(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
	"""Stage 0: start (pass-through)."""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError("pandas is required to run this model.") from _PANDAS_IMPORT_ERROR
	return train_df, val_df, test_df


def materialize_fn(train_df, val_df, test_df, dataset_results, config):
	"""Stage 1: materialization (DataFrame -> TensorFrame + loaders + metrics)."""

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
		from torch_frame.transforms import CatToNumTransform, MutualInformationSort
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"Failed to import local torch_frame components required by TabNet materialize_fn."
		) from e

	try:
		from torchmetrics import AUROC, Accuracy, MeanSquaredError
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"torchmetrics is required for TabNet metrics. Install torchmetrics to use this model."
		) from e

	dataset_name = dataset_results["dataset"]
	task_type = dataset_results["info"]["task_type"]

	device = resolve_device(config)
	print(f"[MATERIALIZE] Device: {device}")

	dataset = Yandex(train_df, val_df, test_df, name=dataset_name, task_type=task_type)
	try:
		dataset.materialize(device=device)
	except TypeError:
		dataset.materialize()

	train_tensor_frame = dataset.tensor_frame[dataset.df["split_col"] == 0]
	val_tensor_frame = dataset.tensor_frame[dataset.df["split_col"] == 1]
	test_tensor_frame = dataset.tensor_frame[dataset.df["split_col"] == 2]

	# Align preprocessing with ExcelFormer/FTTransformer: CatToNum + MutualInformationSort.
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
		"metric_computer": metric_computer,
		"metric": metric,
		"is_classification": is_classification,
		"is_binary_class": is_binary_class,
		"out_channels": out_channels,
		"device": device,
	}


def tabnet_core_fn_row_gnn(material_outputs, config, task_type: str, gnn_stage=None):
	"""TabNet core function supporting row-graphification (online stages)."""

	repo_root = Path(__file__).resolve().parents[2]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

	try:
		import torch_frame
		from torch_frame import stype
		from torch_frame.nn.encoder.stype_encoder import EmbeddingEncoder, StackEncoder
		from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
		from torch_frame.typing import NAStrategy
		from torch_frame.nn.models.tabnet import AttentiveTransformer, FeatureTransformer, GLUBlock
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError("Failed to import local torch_frame TabNet components.") from e

	try:
		from torch.optim.lr_scheduler import ExponentialLR
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError("PyTorch is required to run TabNet.") from e

	try:
		from tqdm import tqdm
	except Exception:  # pragma: no cover
		def tqdm(x, **kwargs):
			return x

	try:
		DGM_d_local = _import_dgm_d()
	except Exception:
		DGM_d_local = None

	train_tensor_frame = material_outputs["train_tensor_frame"]
	train_loader = material_outputs["train_loader"]
	val_loader = material_outputs["val_loader"]
	test_loader = material_outputs["test_loader"]
	col_stats = material_outputs["col_stats"]
	col_names_dict = train_tensor_frame.col_names_dict
	device = material_outputs["device"]
	out_channels = int(material_outputs["out_channels"])
	is_classification = bool(material_outputs["is_classification"])
	is_binary_class = bool(material_outputs["is_binary_class"])
	metric_computer = material_outputs["metric_computer"]
	metric = material_outputs["metric"]

	stage = "none" if gnn_stage is None else str(gnn_stage).lower()
	if stage == "materialization":
		stage = "materialize"
	if stage == "decoding":
		print(
			"[CORE] Row-graphification (decoding): online graph-aware readout is enabled "
			"(node features = TabNet representations; dynamic per-batch DGM_d graph)."
		)

	cat_emb_channels = int(config.get("cat_emb_channels", 2))
	print(f"[CORE] cat_emb_channels={cat_emb_channels}")

	stype_encoder_dict = {
		stype.categorical: EmbeddingEncoder(na_strategy=NAStrategy.MOST_FREQUENT),
		stype.numerical: StackEncoder(na_strategy=NAStrategy.MEAN),
	}

	feature_encoder = StypeWiseFeatureEncoder(
		out_channels=cat_emb_channels,
		col_stats=col_stats,
		col_names_dict=col_names_dict,
		stype_encoder_dict=stype_encoder_dict,
	).to(device)

	num_cols = sum(len(v) for v in col_names_dict.values())
	in_channels = int(cat_emb_channels * num_cols)
	# NOTE: This benchmark often uses extremely small train splits (e.g., 5% of ~200 rows),
	# where BatchNorm running statistics can become degenerate (zero-variance features)
	# and cause eval-time explosions. Default to using batch statistics in both train/eval.
	use_running_stats = bool(config.get("tabnet_bn_running_stats", False))
	bn = torch.nn.BatchNorm1d(in_channels, track_running_stats=use_running_stats).to(device)

	def encode_batch(tf, *, debug: bool = False) -> torch.Tensor:
		x_tokens, _ = feature_encoder(tf)  # [B, F, C]
		if debug:
			print(
				f"[TabNet] feature_encoder: B={x_tokens.shape[0]} F={x_tokens.shape[1]} C={x_tokens.shape[2]}"
			)
		bsz = x_tokens.shape[0]
		x_flat = x_tokens.view(bsz, math.prod(x_tokens.shape[1:]))
		x_flat = bn(x_flat)
		return x_flat

	split_feat_channels = int(config.get("channels", 128))
	split_attn_channels = int(config.get("channels", 128))
	num_layers = int(config.get("num_layers", 6))
	gamma = float(config.get("gamma", 1.2))
	num_shared_glu_layers = int(config.get("num_shared_glu_layers", 2))
	num_dependent_glu_layers = int(config.get("num_dependent_glu_layers", 2))
	patience = int(config.get("patience", 10))
	loss_threshold = float(config.get("loss_threshold", 1e-4))
	restore_best = bool(config.get("restore_best", True))

	# Optional online row-graphification components (encoding/columnwise), kept for parity
	# with the original TaBLEau implementation.
	gnn_encoding_components = None
	gnn_columnwise_components = None
	gnn_decoding_components = None

	if stage == "encoding":
		if DGM_d_local is None:  # pragma: no cover
			raise ModuleNotFoundError(
				"DGM_d is required for row-graphification at encoding stage. "
				"Ensure DGM_pytorch is importable."
			)

		attn_heads = int(config.get("gnn_num_heads", 2))
		gnn_hidden = int(config.get("gnn_hidden", 64))
		dgm_k = int(config.get("gnn_dgm_k", config.get("dgm_k", 10)))
		dgm_distance = str(config.get("gnn_dgm_distance", config.get("dgm_distance", "euclidean")))
		gnn_dropout = float(config.get("gnn_dropout", 0.1))

		self_attn = torch.nn.MultiheadAttention(
			embed_dim=cat_emb_channels,
			num_heads=attn_heads,
			batch_first=True,
		).to(device)
		attn_norm = torch.nn.LayerNorm(cat_emb_channels).to(device)
		column_embed = torch.nn.Parameter(torch.randn(num_cols, cat_emb_channels, device=device))
		pool_query = torch.nn.Parameter(torch.randn(cat_emb_channels, device=device))

		class DGMEmbedWrapper(torch.nn.Module):
			def forward(self, x, A=None):
				return x

		dgm_module = DGM_d_local(DGMEmbedWrapper(), k=dgm_k, distance=dgm_distance).to(device)
		gnn = SimpleGCN(cat_emb_channels, gnn_hidden, cat_emb_channels, num_layers=2).to(device)

		self_attn_out = torch.nn.MultiheadAttention(
			embed_dim=cat_emb_channels,
			num_heads=attn_heads,
			batch_first=True,
		).to(device)
		gcn_to_attn = torch.nn.Linear(cat_emb_channels, cat_emb_channels).to(device)
		attn_out_norm = torch.nn.LayerNorm(cat_emb_channels).to(device)
		ffn_pre = torch.nn.Sequential(
			torch.nn.Linear(cat_emb_channels, cat_emb_channels * 2),
			torch.nn.GELU(),
			torch.nn.Dropout(gnn_dropout),
			torch.nn.Linear(cat_emb_channels * 2, cat_emb_channels),
		).to(device)
		ffn_post = torch.nn.Sequential(
			torch.nn.Linear(cat_emb_channels, cat_emb_channels * 2),
			torch.nn.GELU(),
			torch.nn.Dropout(gnn_dropout),
			torch.nn.Linear(cat_emb_channels * 2, cat_emb_channels),
		).to(device)
		fusion_alpha_param = torch.nn.Parameter(torch.tensor(-0.847, device=device))

		gnn_encoding_components = {
			"self_attn": self_attn,
			"attn_norm": attn_norm,
			"column_embed": column_embed,
			"pool_query": pool_query,
			"dgm_module": dgm_module,
			"gnn": gnn,
			"self_attn_out": self_attn_out,
			"gcn_to_attn": gcn_to_attn,
			"attn_out_norm": attn_out_norm,
			"ffn_pre": ffn_pre,
			"ffn_post": ffn_post,
			"fusion_alpha_param": fusion_alpha_param,
		}

		print(
			"[CORE] Row-graphification (encoding) pipeline created: "
			f"self-attn heads={attn_heads}, dgm_k={dgm_k}, gnn_hidden={gnn_hidden}"
		)

	if stage == "columnwise":
		if DGM_d_local is None:  # pragma: no cover
			raise ModuleNotFoundError(
				"DGM_d is required for row-graphification at columnwise stage. "
				"Ensure DGM_pytorch is importable."
			)

		attn_heads = int(config.get("gnn_num_heads", 4))
		gnn_hidden = int(config.get("gnn_hidden", 64))
		dgm_k = int(config.get("gnn_dgm_k", config.get("dgm_k", 10)))
		dgm_distance = str(config.get("gnn_dgm_distance", config.get("dgm_distance", "euclidean")))
		gnn_dropout = float(config.get("gnn_dropout", 0.1))

		self_attn = torch.nn.MultiheadAttention(
			embed_dim=split_feat_channels,
			num_heads=attn_heads,
			batch_first=True,
		).to(device)
		attn_norm = torch.nn.LayerNorm(split_feat_channels).to(device)
		pool_query = torch.nn.Parameter(torch.randn(split_feat_channels, device=device))

		class DGMEmbedWrapper(torch.nn.Module):
			def forward(self, x, A=None):
				return x

		dgm_module = DGM_d_local(DGMEmbedWrapper(), k=dgm_k, distance=dgm_distance).to(device)
		gnn = SimpleGCN(split_feat_channels, gnn_hidden, split_feat_channels, num_layers=2).to(device)

		self_attn_out = torch.nn.MultiheadAttention(
			embed_dim=split_feat_channels,
			num_heads=attn_heads,
			batch_first=True,
		).to(device)
		gcn_to_attn = torch.nn.Linear(split_feat_channels, split_feat_channels).to(device)
		attn_out_norm = torch.nn.LayerNorm(split_feat_channels).to(device)
		ffn_pre = torch.nn.Sequential(
			torch.nn.Linear(split_feat_channels, split_feat_channels * 2),
			torch.nn.GELU(),
			torch.nn.Dropout(gnn_dropout),
			torch.nn.Linear(split_feat_channels * 2, split_feat_channels),
		).to(device)
		ffn_post = torch.nn.Sequential(
			torch.nn.Linear(split_feat_channels, split_feat_channels * 2),
			torch.nn.GELU(),
			torch.nn.Dropout(gnn_dropout),
			torch.nn.Linear(split_feat_channels * 2, split_feat_channels),
		).to(device)
		fusion_alpha_param = torch.nn.Parameter(torch.tensor(-0.847, device=device))

		gnn_columnwise_components = {
			"self_attn": self_attn,
			"attn_norm": attn_norm,
			"pool_query": pool_query,
			"dgm_module": dgm_module,
			"gnn": gnn,
			"self_attn_out": self_attn_out,
			"gcn_to_attn": gcn_to_attn,
			"attn_out_norm": attn_out_norm,
			"ffn_pre": ffn_pre,
			"ffn_post": ffn_post,
			"fusion_alpha_param": fusion_alpha_param,
		}
		print(
			"[CORE] Row-graphification (columnwise) pipeline created: "
			f"self-attn heads={attn_heads}, dgm_k={dgm_k}, gnn_hidden={gnn_hidden}"
		)

	if stage == "decoding":
		if DGM_d_local is None:  # pragma: no cover
			raise ModuleNotFoundError(
				"DGM_d is required for row-graphification at decoding stage. "
				"Ensure DGM_pytorch is importable."
			)

		gnn_hidden = int(config.get("gnn_hidden", 64))
		dgm_k = int(config.get("gnn_dgm_k", config.get("dgm_k", 10)))
		dgm_distance = str(config.get("gnn_dgm_distance", config.get("dgm_distance", "euclidean")))

		class DGMEmbedWrapper(torch.nn.Module):
			def forward(self, x, A=None):
				return x

		dgm_module = DGM_d_local(DGMEmbedWrapper(), k=dgm_k, distance=dgm_distance).to(device)
		gnn = SimpleGCN(split_feat_channels, gnn_hidden, out_channels, num_layers=2).to(device)
		gnn_decoding_components = {
			"dgm_module": dgm_module,
			"gnn": gnn,
		}
		print(
			"[CORE] Row-graphification (decoding) pipeline created: "
			f"dgm_k={dgm_k}, dgm_distance={dgm_distance}, gnn_hidden={gnn_hidden}, out_channels={out_channels}"
		)

	# Shared GLU block
	if num_shared_glu_layers > 0:
		shared_glu_block = GLUBlock(
			in_channels=in_channels,
			out_channels=split_feat_channels + split_attn_channels,
			num_glu_layers=num_shared_glu_layers,
			no_first_residual=True,
		).to(device)
	else:
		shared_glu_block = torch.nn.Identity().to(device)

	feat_transformers = torch.nn.ModuleList(
		[
			FeatureTransformer(
				in_channels,
				split_feat_channels + split_attn_channels,
				num_dependent_glu_layers=num_dependent_glu_layers,
				shared_glu_block=shared_glu_block,
			).to(device)
			for _ in range(num_layers + 1)
		]
	)

	attn_transformers = torch.nn.ModuleList(
		[
			AttentiveTransformer(
				in_channels=split_attn_channels,
				out_channels=in_channels,
			).to(device)
			for _ in range(num_layers)
		]
	)

	lin = torch.nn.Linear(split_feat_channels, out_channels).to(device)
	lin.reset_parameters()

	def process_batch_interaction(x: torch.Tensor, *, return_reg: bool = False):
		prior = torch.ones_like(x)
		reg = torch.tensor(0.0, device=x.device)

		attention_x = feat_transformers[0](x)
		attention_x = attention_x[:, split_feat_channels:]

		feature_outputs = []
		for i in range(num_layers):
			attention_mask = attn_transformers[i](attention_x, prior)
			masked_x = attention_mask * x
			out = feat_transformers[i + 1](masked_x)

			feature_x = F.relu(out[:, :split_feat_channels])
			feature_outputs.append(feature_x)
			attention_x = out[:, split_feat_channels:]
			prior = (gamma - attention_mask) * prior

			if return_reg and x.shape[0] > 0:
				entropy = -torch.sum(attention_mask * torch.log(attention_mask + 1e-15), dim=1).mean()
				reg = reg + entropy

		if return_reg:
			return feature_outputs, reg / max(1, num_layers)
		return feature_outputs

	def forward(tf, *, return_reg: bool = False, debug: bool = False):
		x = encode_batch(tf, debug=debug)
		bsz = x.shape[0]

		# Online row-graphification at encoding stage (mini-batch graph).
		if stage == "encoding" and gnn_encoding_components is not None and bsz > 1:
			comp = gnn_encoding_components
			x_tokens = x.view(bsz, num_cols, cat_emb_channels)
			tokens = x_tokens + comp["column_embed"].unsqueeze(0)
			tokens_norm = comp["attn_norm"](tokens)
			attn_out1, _ = comp["self_attn"](tokens_norm, tokens_norm, tokens_norm)
			tokens_attn = tokens + attn_out1
			ffn_out1 = comp["ffn_pre"](comp["attn_norm"](tokens_attn))
			tokens_attn = tokens_attn + ffn_out1

			pool_logits = (tokens_attn * comp["pool_query"]).sum(dim=-1) / math.sqrt(cat_emb_channels)
			pool_weights = torch.softmax(pool_logits, dim=1)
			x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)

			x_pooled_std = _standardize(x_pooled, dim=0)
			x_batched = x_pooled_std.unsqueeze(0)
			if hasattr(comp["dgm_module"], "k"):
				comp["dgm_module"].k = int(min(int(comp["dgm_module"].k), max(1, bsz - 1)))
			x_dgm, edge_index_dgm, _ = comp["dgm_module"](x_batched, A=None)
			x_dgm = x_dgm.squeeze(0)

			# Make undirected + self-loops (cheap fallback).
			rev = torch.stack([edge_index_dgm[1], edge_index_dgm[0]], dim=0)
			self_nodes = torch.arange(bsz, device=edge_index_dgm.device)
			self_edge = torch.stack([self_nodes, self_nodes], dim=0)
			edge_index_dgm = torch.cat([edge_index_dgm, rev, self_edge], dim=1)

			x_gnn_out = comp["gnn"](x_dgm, edge_index_dgm)
			gcn_ctx = comp["gcn_to_attn"](x_gnn_out).unsqueeze(1)
			tokens_with_ctx = tokens_attn + gcn_ctx
			tokens_ctx_norm = comp["attn_out_norm"](tokens_with_ctx)
			attn_out2, _ = comp["self_attn_out"](tokens_ctx_norm, tokens_ctx_norm, tokens_ctx_norm)
			tokens_mid = tokens_with_ctx + attn_out2
			ffn_out2 = comp["ffn_post"](comp["attn_out_norm"](tokens_mid))
			tokens_out = tokens_mid + ffn_out2
			fusion_alpha = torch.sigmoid(comp["fusion_alpha_param"])
			x_tokens = x_tokens + fusion_alpha * tokens_out
			x = x_tokens.view(bsz, in_channels)

		if return_reg:
			feature_outputs, reg = process_batch_interaction(x, return_reg=True)
		else:
			feature_outputs = process_batch_interaction(x, return_reg=False)

		# Online row-graphification at columnwise stage.
		if stage == "columnwise" and gnn_columnwise_components is not None and bsz > 1:
			comp = gnn_columnwise_components
			feat_stack = torch.stack(feature_outputs, dim=1)  # [B, L, C]
			tokens_norm = comp["attn_norm"](feat_stack)
			attn_out1, _ = comp["self_attn"](tokens_norm, tokens_norm, tokens_norm)
			tokens_attn = feat_stack + attn_out1
			ffn_out1 = comp["ffn_pre"](comp["attn_norm"](tokens_attn))
			tokens_attn = tokens_attn + ffn_out1

			pool_logits = (tokens_attn * comp["pool_query"]).sum(dim=-1) / math.sqrt(split_feat_channels)
			pool_weights = torch.softmax(pool_logits, dim=1)
			x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)

			x_pooled_std = _standardize(x_pooled, dim=0)
			x_batched = x_pooled_std.unsqueeze(0)
			if hasattr(comp["dgm_module"], "k"):
				comp["dgm_module"].k = int(min(int(comp["dgm_module"].k), max(1, bsz - 1)))
			x_dgm, edge_index_dgm, _ = comp["dgm_module"](x_batched, A=None)
			x_dgm = x_dgm.squeeze(0)

			rev = torch.stack([edge_index_dgm[1], edge_index_dgm[0]], dim=0)
			self_nodes = torch.arange(bsz, device=edge_index_dgm.device)
			self_edge = torch.stack([self_nodes, self_nodes], dim=0)
			edge_index_dgm = torch.cat([edge_index_dgm, rev, self_edge], dim=1)

			x_gnn_out = comp["gnn"](x_dgm, edge_index_dgm)
			gcn_ctx = comp["gcn_to_attn"](x_gnn_out).unsqueeze(1)
			tokens_with_ctx = tokens_attn + gcn_ctx
			tokens_ctx_norm = comp["attn_out_norm"](tokens_with_ctx)
			attn_out2, _ = comp["self_attn_out"](tokens_ctx_norm, tokens_ctx_norm, tokens_ctx_norm)
			tokens_mid = tokens_with_ctx + attn_out2
			ffn_out2 = comp["ffn_post"](comp["attn_out_norm"](tokens_mid))
			tokens_out = tokens_mid + ffn_out2
			fusion_alpha = torch.sigmoid(comp["fusion_alpha_param"])
			feat_stack = feat_stack + fusion_alpha * tokens_out
			feature_outputs = [feat_stack[:, i, :] for i in range(feat_stack.shape[1])]

		out_pre = sum(feature_outputs)
		if stage == "decoding" and gnn_decoding_components is not None and bsz > 1:
			comp = gnn_decoding_components
			x_pooled_std = _standardize(out_pre, dim=0)
			x_batched = x_pooled_std.unsqueeze(0)
			if hasattr(comp["dgm_module"], "k"):
				comp["dgm_module"].k = int(min(int(comp["dgm_module"].k), max(1, bsz - 1)))
			x_dgm, edge_index_dgm, _ = comp["dgm_module"](x_batched, A=None)
			x_dgm = x_dgm.squeeze(0)

			rev = torch.stack([edge_index_dgm[1], edge_index_dgm[0]], dim=0)
			self_nodes = torch.arange(bsz, device=edge_index_dgm.device)
			self_edge = torch.stack([self_nodes, self_nodes], dim=0)
			edge_index_dgm = torch.cat([edge_index_dgm, rev, self_edge], dim=1)

			out = comp["gnn"](x_dgm, edge_index_dgm)
		else:
			out = lin(out_pre)
		if return_reg:
			return out, reg
		return out

	lr = float(config.get("lr", 0.001))
	gamma_lr = float(config.get("gamma_lr", 0.95))

	# Collect parameters without using `set()` (preserves determinism).
	params = list(feature_encoder.parameters())
	params += list(bn.parameters())
	params += [p for m in feat_transformers for p in m.parameters()]
	params += [p for m in attn_transformers for p in m.parameters()]
	params += list(lin.parameters())
	if gnn_encoding_components is not None:
		comp = gnn_encoding_components
		params += list(comp["self_attn"].parameters())
		params += list(comp["attn_norm"].parameters())
		params += [comp["column_embed"], comp["pool_query"], comp["fusion_alpha_param"]]
		params += list(comp["dgm_module"].parameters())
		params += list(comp["gnn"].parameters())
		params += list(comp["self_attn_out"].parameters())
		params += list(comp["gcn_to_attn"].parameters())
		params += list(comp["attn_out_norm"].parameters())
		params += list(comp["ffn_pre"].parameters())
		params += list(comp["ffn_post"].parameters())
	if gnn_columnwise_components is not None:
		comp = gnn_columnwise_components
		params += list(comp["self_attn"].parameters())
		params += list(comp["attn_norm"].parameters())
		params += [comp["pool_query"], comp["fusion_alpha_param"]]
		params += list(comp["dgm_module"].parameters())
		params += list(comp["gnn"].parameters())
		params += list(comp["self_attn_out"].parameters())
		params += list(comp["gcn_to_attn"].parameters())
		params += list(comp["attn_out_norm"].parameters())
		params += list(comp["ffn_pre"].parameters())
		params += list(comp["ffn_post"].parameters())
	if gnn_decoding_components is not None:
		comp = gnn_decoding_components
		params += list(comp["dgm_module"].parameters())
		params += list(comp["gnn"].parameters())

	params = _unique_parameters(params)
	optimizer = torch.optim.Adam(params, lr=lr)
	lr_scheduler = ExponentialLR(optimizer, gamma=gamma_lr)

	def _set_train_mode():
		feature_encoder.train()
		bn.train()
		for m in feat_transformers:
			m.train()
		for m in attn_transformers:
			m.train()
		lin.train()
		if gnn_encoding_components is not None:
			for k, v in gnn_encoding_components.items():
				if hasattr(v, "train"):
					v.train()
		if gnn_columnwise_components is not None:
			for k, v in gnn_columnwise_components.items():
				if hasattr(v, "train"):
					v.train()
		if gnn_decoding_components is not None:
			for k, v in gnn_decoding_components.items():
				if hasattr(v, "train"):
					v.train()

	def _set_eval_mode():
		feature_encoder.eval()
		bn.eval()
		for m in feat_transformers:
			m.eval()
		for m in attn_transformers:
			m.eval()
		lin.eval()
		if gnn_encoding_components is not None:
			for k, v in gnn_encoding_components.items():
				if hasattr(v, "eval"):
					v.eval()
		if gnn_columnwise_components is not None:
			for k, v in gnn_columnwise_components.items():
				if hasattr(v, "eval"):
					v.eval()
		if gnn_decoding_components is not None:
			for k, v in gnn_decoding_components.items():
				if hasattr(v, "eval"):
					v.eval()

	def train_one_epoch(epoch: int) -> float:
		_set_train_mode()
		loss_accum = 0.0
		total_count = 0
		first_batch = True
		for tf in tqdm(train_loader, desc=f"Epoch: {epoch}"):
			tf = tf.to(device)
			debug = bool(epoch == 1 and first_batch)
			pred, reg = forward(tf, return_reg=True, debug=debug)
			first_batch = False

			if is_classification:
				loss = F.cross_entropy(pred, tf.y)
			else:
				loss = F.mse_loss(pred.view(-1), tf.y.view(-1))

			loss = loss + 0.01 * reg

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			loss_accum += float(loss) * len(tf.y)
			total_count += len(tf.y)

		return loss_accum / max(1, total_count)

	@torch.no_grad()
	def eval_loader(loader):
		_set_eval_mode()
		metric_computer.reset()
		for tf in loader:
			tf = tf.to(device)
			pred = forward(tf, return_reg=False, debug=False)

			if is_binary_class:
				metric_computer.update(pred[:, 1], tf.y)
			elif is_classification:
				metric_computer.update(pred.argmax(dim=-1), tf.y)
			else:
				metric_computer.update(pred.view(-1), tf.y.view(-1))

		computed = float(metric_computer.compute().item())
		if is_classification:
			return computed
		return computed**0.5

	best_val_metric = -float("inf") if is_classification else float("inf")
	best_test_metric = -float("inf") if is_classification else float("inf")
	best_epoch = 0
	early_stop_counter = 0
	early_stop_epochs = 0

	train_losses = []
	train_metrics = []
	val_metrics = []

	best_states = None
	epochs = int(config.get("epochs", 200))
	for epoch in range(1, epochs + 1):
		train_loss = train_one_epoch(epoch)
		train_metric = eval_loader(train_loader)
		val_metric = eval_loader(val_loader)
		test_metric = eval_loader(test_loader)

		train_losses.append(train_loss)
		train_metrics.append(train_metric)
		val_metrics.append(val_metric)

		improved = (val_metric > best_val_metric) if is_classification else (val_metric < best_val_metric)
		if improved:
			best_val_metric = val_metric
			best_test_metric = test_metric
			best_epoch = epoch
			early_stop_counter = 0
			if restore_best:
				import copy

				best_states = {
					"feature_encoder": copy.deepcopy(feature_encoder.state_dict()),
					"bn": copy.deepcopy(bn.state_dict()),
					"feat_transformers": [copy.deepcopy(m.state_dict()) for m in feat_transformers],
					"attn_transformers": [copy.deepcopy(m.state_dict()) for m in attn_transformers],
					"lin": copy.deepcopy(lin.state_dict()),
				}
				if gnn_encoding_components is not None:
					best_states["gnn_encoding"] = {
						k: (copy.deepcopy(v.state_dict()) if hasattr(v, "state_dict") else None)
						for k, v in gnn_encoding_components.items()
						if k not in {"column_embed", "pool_query", "fusion_alpha_param"}
					}
					best_states["gnn_encoding_embed"] = gnn_encoding_components["column_embed"].detach().clone()
					best_states["gnn_encoding_pool"] = gnn_encoding_components["pool_query"].detach().clone()
					best_states["gnn_encoding_alpha"] = gnn_encoding_components["fusion_alpha_param"].detach().clone()
				if gnn_columnwise_components is not None:
					best_states["gnn_columnwise"] = {
						k: (copy.deepcopy(v.state_dict()) if hasattr(v, "state_dict") else None)
						for k, v in gnn_columnwise_components.items()
						if k not in {"pool_query", "fusion_alpha_param"}
					}
					best_states["gnn_columnwise_pool"] = gnn_columnwise_components["pool_query"].detach().clone()
					best_states["gnn_columnwise_alpha"] = gnn_columnwise_components["fusion_alpha_param"].detach().clone()
				if gnn_decoding_components is not None:
					best_states["gnn_decoding"] = {
						k: (copy.deepcopy(v.state_dict()) if hasattr(v, "state_dict") else None)
						for k, v in gnn_decoding_components.items()
					}
		else:
			early_stop_counter += 1

		suffix = "✓" if improved else "✗"
		print(
			f"Epoch {epoch:3d} {suffix} | Train Loss: {train_loss:.4f}, "
			f"Train {metric}: {train_metric:.4f}, Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f} "
			f"| EarlyStop {early_stop_counter}/{patience}"
		)

		lr_scheduler.step()
		if early_stop_counter >= patience:
			early_stop_epochs = epoch
			print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
			break

	if restore_best and best_states is not None:
		feature_encoder.load_state_dict(best_states["feature_encoder"])
		bn.load_state_dict(best_states["bn"])
		for i, m in enumerate(feat_transformers):
			m.load_state_dict(best_states["feat_transformers"][i])
		for i, m in enumerate(attn_transformers):
			m.load_state_dict(best_states["attn_transformers"][i])
		lin.load_state_dict(best_states["lin"])
		if gnn_encoding_components is not None and best_states.get("gnn_encoding") is not None:
			for k, sd in best_states["gnn_encoding"].items():
				if sd is not None:
					gnn_encoding_components[k].load_state_dict(sd)
			with torch.no_grad():
				gnn_encoding_components["column_embed"].copy_(best_states["gnn_encoding_embed"])
				gnn_encoding_components["pool_query"].copy_(best_states["gnn_encoding_pool"])
				gnn_encoding_components["fusion_alpha_param"].copy_(best_states["gnn_encoding_alpha"])
		if gnn_columnwise_components is not None and best_states.get("gnn_columnwise") is not None:
			for k, sd in best_states["gnn_columnwise"].items():
				if sd is not None:
					gnn_columnwise_components[k].load_state_dict(sd)
			with torch.no_grad():
				gnn_columnwise_components["pool_query"].copy_(best_states["gnn_columnwise_pool"])
				gnn_columnwise_components["fusion_alpha_param"].copy_(best_states["gnn_columnwise_alpha"])
		if gnn_decoding_components is not None and best_states.get("gnn_decoding") is not None:
			for k, sd in best_states["gnn_decoding"].items():
				if sd is not None:
					gnn_decoding_components[k].load_state_dict(sd)
		print(f"[CORE] Restored best weights from epoch {best_epoch}")

	return {
		"gnn_early_stop_epochs": material_outputs.get("gnn_early_stop_epochs", 0),
		"train_losses": train_losses,
		"train_metrics": train_metrics,
		"val_metrics": val_metrics,
		"best_val_metric": best_val_metric,
		"final_metric": best_val_metric,
		"best_test_metric": best_test_metric,
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


def tabnet_core_fn_feature_gnn(material_outputs, config, task_type: str, gnn_stage=None):
	"""TabNet core function implementing paper-aligned feature graphification."""

	stage = "none" if gnn_stage is None else str(gnn_stage).lower()
	if stage in {"none", "start", "materialize", "materialization"}:
		return tabnet_core_fn_row_gnn(material_outputs, config, task_type, gnn_stage="none")

	if stage not in {"encoding", "columnwise", "decoding"}:
		raise ValueError(
			f"Unknown gnn_stage={stage!r} for graphify=feature. "
			"Expected one of: none/start/materialize/encoding/columnwise/decoding."
		)

	repo_root = Path(__file__).resolve().parents[2]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

	try:
		from torch_frame import stype
		from torch_frame.nn.encoder.stype_encoder import EmbeddingEncoder, StackEncoder
		from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
		from torch_frame.typing import NAStrategy
		from torch_frame.nn.models.tabnet import AttentiveTransformer, FeatureTransformer, GLUBlock
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError("Failed to import local torch_frame TabNet components.") from e

	try:
		from torch.optim.lr_scheduler import ExponentialLR
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError("PyTorch is required to run TabNet.") from e

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
	col_names_dict = train_tensor_frame.col_names_dict
	device = material_outputs["device"]
	out_channels = int(material_outputs["out_channels"])
	is_classification = bool(material_outputs["is_classification"])
	is_binary_class = bool(material_outputs["is_binary_class"])
	metric_computer = material_outputs["metric_computer"]
	metric = material_outputs["metric"]

	# Ensure a topology prior exists (inactive earlier, required now).
	feature_topology = material_outputs.get("feature_topology")
	if feature_topology is None:
		# Align feature order to the encoder token order.
		# Paper-aligned ports (ExcelFormer/FTTransformer) define feature topology over
		# numerical tokens after CatToNumTransform.
		feature_names = list(train_tensor_frame.col_names_dict.get(stype.numerical, []))
		if len(feature_names) == 0:
			raise ValueError(
				"No numerical features found in TensorFrame for feature-level topology init. "
				"(Expected numerical stype after CatToNumTransform.)"
			)
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

	cat_emb_channels = int(config.get("cat_emb_channels", 2))
	stype_encoder_dict = {
		stype.categorical: EmbeddingEncoder(na_strategy=NAStrategy.MOST_FREQUENT),
		stype.numerical: StackEncoder(na_strategy=NAStrategy.MEAN),
	}

	feature_encoder = StypeWiseFeatureEncoder(
		out_channels=cat_emb_channels,
		col_stats=col_stats,
		col_names_dict=col_names_dict,
		stype_encoder_dict=stype_encoder_dict,
	).to(device)

	num_cols = sum(len(v) for v in col_names_dict.values())
	in_channels = int(cat_emb_channels * num_cols)
	use_running_stats = bool(config.get("tabnet_bn_running_stats", False))
	bn = torch.nn.BatchNorm1d(in_channels, track_running_stats=use_running_stats).to(device)

	def _feature_inject(x_tokens: torch.Tensor) -> torch.Tensor:
		x_mp = topology_mixing(x_tokens)
		alpha = torch.sigmoid(fusion_alpha_param)
		return x_tokens + alpha * (x_mp - x_tokens)

	split_feat_channels = int(config.get("channels", 128))
	split_attn_channels = int(config.get("channels", 128))
	num_layers = int(config.get("num_layers", 6))
	gamma = float(config.get("gamma", 1.2))
	num_shared_glu_layers = int(config.get("num_shared_glu_layers", 2))
	num_dependent_glu_layers = int(config.get("num_dependent_glu_layers", 2))
	patience = int(config.get("patience", 10))
	loss_threshold = float(config.get("loss_threshold", 1e-4))
	restore_best = bool(config.get("restore_best", True))

	if num_shared_glu_layers > 0:
		shared_glu_block = GLUBlock(
			in_channels=in_channels,
			out_channels=split_feat_channels + split_attn_channels,
			num_glu_layers=num_shared_glu_layers,
			no_first_residual=True,
		).to(device)
	else:
		shared_glu_block = torch.nn.Identity().to(device)

	feat_transformers = torch.nn.ModuleList(
		[
			FeatureTransformer(
				in_channels,
				split_feat_channels + split_attn_channels,
				num_dependent_glu_layers=num_dependent_glu_layers,
				shared_glu_block=shared_glu_block,
			).to(device)
			for _ in range(num_layers + 1)
		]
	)

	attn_transformers = torch.nn.ModuleList(
		[
			AttentiveTransformer(
				in_channels=split_attn_channels,
				out_channels=in_channels,
			).to(device)
			for _ in range(num_layers)
		]
	)

	# Decoder behavior matches the paper-aligned ports:
	# - encoding/columnwise: keep standard TabNet readout.
	# - decoding: replace readout by a graph-aware (topology-mixed) readout.
	lin = torch.nn.Linear(split_feat_channels, out_channels).to(device)
	lin.reset_parameters()
	decoder_readout = torch.nn.Linear(in_channels, out_channels).to(device)
	decoder_readout.reset_parameters()

	def forward(tf, *, return_reg: bool = False):
		x_tokens, _ = feature_encoder(tf)  # [B, F, C]
		if stage == "encoding":
			x_tokens = _feature_inject(x_tokens)

		bsz = x_tokens.shape[0]
		x = x_tokens.reshape(bsz, math.prod(x_tokens.shape[1:]))
		x = bn(x)

		prior = torch.ones_like(x)
		reg = torch.tensor(0.0, device=x.device)

		attention_x = feat_transformers[0](x)
		attention_x = attention_x[:, split_feat_channels:]

		outs = []
		last_masked_x = None
		for i in range(num_layers):
			attention_mask = attn_transformers[i](attention_x, prior)
			masked_x = attention_mask * x
			last_masked_x = masked_x

			# Columnwise injection: after feature selection, apply topology-masked mixing.
			if stage == "columnwise":
				masked_tokens = masked_x.reshape(bsz, num_cols, cat_emb_channels)
				masked_tokens = _feature_inject(masked_tokens)
				masked_x = masked_tokens.reshape(bsz, in_channels)

			out = feat_transformers[i + 1](masked_x)
			feature_x = F.relu(out[:, :split_feat_channels])
			outs.append(feature_x)
			attention_x = out[:, split_feat_channels:]
			prior = (gamma - attention_mask) * prior

			if return_reg and bsz > 0:
				entropy = -torch.sum(attention_mask * torch.log(attention_mask + 1e-15), dim=1).mean()
				reg = reg + entropy

		if stage == "decoding":
			if last_masked_x is None:
				last_masked_x = x
			last_tokens = last_masked_x.reshape(bsz, num_cols, cat_emb_channels)
			last_tokens = topology_mixing(last_tokens)
			out = decoder_readout(last_tokens.reshape(bsz, in_channels))
		else:
			out = sum(outs)
			out = lin(out)

		if return_reg:
			return out, reg / max(1, num_layers)
		return out

	lr = float(config.get("lr", 0.001))
	gamma_lr = float(config.get("gamma_lr", 0.95))

	params = list(feature_encoder.parameters())
	params += list(bn.parameters())
	params += [p for m in feat_transformers for p in m.parameters()]
	params += [p for m in attn_transformers for p in m.parameters()]
	params += list(topology_mixing.parameters()) + [fusion_alpha_param]
	params += list(lin.parameters())
	if stage == "decoding":
		params += list(decoder_readout.parameters())

	params = _unique_parameters(params)
	optimizer = torch.optim.Adam(params, lr=lr)
	lr_scheduler = ExponentialLR(optimizer, gamma=gamma_lr)

	def _set_train_mode():
		feature_encoder.train()
		bn.train()
		for m in feat_transformers:
			m.train()
		for m in attn_transformers:
			m.train()
		topology_mixing.train()
		lin.train()
		decoder_readout.train()

	def _set_eval_mode():
		feature_encoder.eval()
		bn.eval()
		for m in feat_transformers:
			m.eval()
		for m in attn_transformers:
			m.eval()
		topology_mixing.eval()
		lin.eval()
		decoder_readout.eval()

	def train_one_epoch(epoch: int) -> float:
		_set_train_mode()
		loss_accum = 0.0
		total_count = 0
		for tf in tqdm(train_loader, desc=f"Epoch: {epoch}"):
			tf = tf.to(device)
			pred, reg = forward(tf, return_reg=True)
			if is_classification:
				loss = F.cross_entropy(pred, tf.y)
			else:
				loss = F.mse_loss(pred.view(-1), tf.y.view(-1))
			loss = loss + 0.01 * reg
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			loss_accum += float(loss) * len(tf.y)
			total_count += len(tf.y)
		return loss_accum / max(1, total_count)

	@torch.no_grad()
	def eval_metric(loader):
		_set_eval_mode()
		metric_computer.reset()
		for tf in loader:
			tf = tf.to(device)
			pred = forward(tf, return_reg=False)
			if is_binary_class:
				metric_computer.update(pred[:, 1], tf.y)
			elif is_classification:
				metric_computer.update(pred.argmax(dim=-1), tf.y)
			else:
				metric_computer.update(pred.view(-1), tf.y.view(-1))
		computed = float(metric_computer.compute().item())
		if is_classification:
			return computed
		return computed**0.5

	best_val_metric = -float("inf") if is_classification else float("inf")
	best_test_metric = -float("inf") if is_classification else float("inf")
	best_epoch = 0
	early_stop_counter = 0
	early_stop_epochs = 0

	train_losses = []
	train_metrics = []
	val_metrics = []
	best_states = None

	epochs = int(config.get("epochs", 200))
	for epoch in range(1, epochs + 1):
		train_loss = train_one_epoch(epoch)
		train_metric = eval_metric(train_loader)
		val_metric = eval_metric(val_loader)
		test_metric = eval_metric(test_loader)

		train_losses.append(train_loss)
		train_metrics.append(train_metric)
		val_metrics.append(val_metric)

		improved = (val_metric > best_val_metric) if is_classification else (val_metric < best_val_metric)
		if improved:
			best_val_metric = val_metric
			best_test_metric = test_metric
			best_epoch = epoch
			early_stop_counter = 0
			if restore_best:
				import copy

				best_states = {
					"feature_encoder": copy.deepcopy(feature_encoder.state_dict()),
					"bn": copy.deepcopy(bn.state_dict()),
					"feat_transformers": [copy.deepcopy(m.state_dict()) for m in feat_transformers],
					"attn_transformers": [copy.deepcopy(m.state_dict()) for m in attn_transformers],
					"topology_mixing": copy.deepcopy(topology_mixing.state_dict()),
					"fusion_alpha": fusion_alpha_param.detach().clone(),
					"lin": copy.deepcopy(lin.state_dict()),
					"decoder_readout": copy.deepcopy(decoder_readout.state_dict()),
				}
		else:
			early_stop_counter += 1

		suffix = "✓" if improved else "✗"
		print(
			f"Epoch {epoch:3d} {suffix} | Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
			f"Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f} | EarlyStop {early_stop_counter}/{patience}"
		)

		lr_scheduler.step()
		if early_stop_counter >= patience:
			early_stop_epochs = epoch
			print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
			break

	if restore_best and best_states is not None:
		feature_encoder.load_state_dict(best_states["feature_encoder"])
		bn.load_state_dict(best_states["bn"])
		for i, m in enumerate(feat_transformers):
			m.load_state_dict(best_states["feat_transformers"][i])
		for i, m in enumerate(attn_transformers):
			m.load_state_dict(best_states["attn_transformers"][i])
		topology_mixing.load_state_dict(best_states["topology_mixing"])
		with torch.no_grad():
			fusion_alpha_param.copy_(best_states["fusion_alpha"])
		lin.load_state_dict(best_states["lin"])
		decoder_readout.load_state_dict(best_states["decoder_readout"])
		print(f"[CORE] Restored best weights from epoch {best_epoch}")

	return {
		"gnn_early_stop_epochs": material_outputs.get("gnn_early_stop_epochs", 0),
		"train_losses": train_losses,
		"train_metrics": train_metrics,
		"val_metrics": val_metrics,
		"best_val_metric": best_val_metric,
		"final_metric": best_val_metric,
		"best_test_metric": best_test_metric,
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


def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
	"""Model entrypoint called by the project runner."""

	print("TabNet - staged execution")
	print(f"gnn_stage: {gnn_stage}")
	task_type = dataset_results.get("info", {}).get("task_type", "binclass")

	stage = "none" if gnn_stage is None else str(gnn_stage).lower()
	if stage == "materialization":
		stage = "materialize"

	graphify = str(config.get("graphify", "row")).lower()
	if graphify not in {"row", "feature", "all"}:
		graphify = "row"

	gnn_early_stop_epochs = 0

	try:
		train_df, val_df, test_df = start_fn(train_df, val_df, test_df)

		if graphify in {"row", "all"} and stage == "start":
			train_df, val_df, test_df, gnn_early_stop_epochs = row_gnn_after_start_fn(
				train_df, val_df, test_df, config, task_type
			)

		if graphify in {"feature", "all"} and stage == "start":
			train_df, val_df, test_df, _ = feature_gnn_after_start_fn(train_df, val_df, test_df, config, task_type)

		material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
		material_outputs["gnn_early_stop_epochs"] = gnn_early_stop_epochs

		if graphify in {"row", "all"} and stage == "materialize":
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
				gnn_early_stop_epochs2,
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
					"gnn_early_stop_epochs": gnn_early_stop_epochs2,
				}
			)

		if graphify in {"feature", "all"} and stage == "materialize":
			material_outputs = feature_gnn_after_materialize_fn(
				material_outputs, config, dataset_results["dataset"], task_type
			)

		# Run core
		if graphify == "feature":
			results = tabnet_core_fn_feature_gnn(material_outputs, config, task_type, gnn_stage=stage)
		elif graphify == "all":
			# For simplicity: online stages follow feature graphify definition.
			results = tabnet_core_fn_feature_gnn(material_outputs, config, task_type, gnn_stage=stage)
		else:
			results = tabnet_core_fn_row_gnn(material_outputs, config, task_type, gnn_stage=stage)

	except Exception as e:
		is_cls = task_type in {"binclass", "multiclass"}
		results = {
			"train_losses": [],
			"train_metrics": [],
			"val_metrics": [],
			"best_val_metric": float("-inf") if is_cls else float("inf"),
			"best_test_metric": float("-inf") if is_cls else float("inf"),
			"early_stop_epochs": 0,
			"gnn_early_stop_epochs": gnn_early_stop_epochs,
			"error": str(e),
		}

	return results

#  small+binclass
#  python main.py --models tabnet --dataset kaggle_Audit_Data --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  small+regression
#  python main.py --models tabnet --dataset openml_The_Office_Dataset --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+binclass
#  python main.py --models tabnet --dataset credit --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+multiclass
#  python main.py --models tabnet --dataset eye --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  python main.py --models tabnet --dataset helena --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+regression
#  python main.py --models tabnet --dataset house --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0

