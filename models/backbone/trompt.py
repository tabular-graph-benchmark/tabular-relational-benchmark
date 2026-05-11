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
	"""Resolve the torch device from `config`.

	- If CUDA is available and `config['gpu']` is provided, use that GPU.
	- Otherwise fall back to default CUDA or CPU.
	"""

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


class SimpleGCN(torch.nn.Module):
	"""A minimal GCN for joint training."""

	def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
		super().__init__()
		if GCNConv is None:  # pragma: no cover
			raise ModuleNotFoundError(
				"torch_geometric is required for row/online GNN injection. Install torch-geometric to use this feature."
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

	Paper-aligned fairness rule: this hook uses raw row feature vectors as
	node features. No self-attention, input projection, or pooling is applied.

	Output schema:
	  - Features are replaced by numerical embedding columns: N_feature_emb_1 .. N_feature_emb_{d}
	  - Target is preserved in the `target` column.
	"""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError(
			"pandas is required for TromPT start-stage row-GNN injection. Install pandas to use this feature."
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

	feature_cols = [c for c in train_df.columns if c != "target"]
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

		e = edge_index_dgm.shape[1]
		avg_deg = e / max(1, ns)
		print(f"[START-GNN-DGM] split ns={ns}, edges={e}, avg_deg={avg_deg:.2f}")

		return logits, gcn_out, logprobs_dgm

	# Label tensors
	if task_type in ["binclass", "multiclass"]:
		y_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
		y_val = torch.tensor(y_val_np, dtype=torch.long, device=device)
	else:
		y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)
		y_val = torch.tensor(y_val_np, dtype=torch.float32, device=device)

	for epoch in range(gnn_epochs):
		dgm_module.train()
		gnn.train()
		pred_head.train()

		optimizer.zero_grad()
		logits, _, logprobs_dgm = forward_pass(x_train, dgm_module)
		if task_type in ["binclass", "multiclass"]:
			train_loss = F.cross_entropy(logits, y_train)
		else:
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
				val_loss = F.cross_entropy(logits_val, y_val)
			else:
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

	row_emb_cols = [f"N_feature_emb_{i + 1}" for i in range(int(gnn_out_dim))]
	train_df_emb = pd.DataFrame(emb_train.detach().cpu().numpy(), columns=row_emb_cols, index=train_df.index)
	val_df_emb = pd.DataFrame(emb_val.detach().cpu().numpy(), columns=row_emb_cols, index=val_df.index)
	test_df_emb = pd.DataFrame(emb_test.detach().cpu().numpy(), columns=row_emb_cols, index=test_df.index)

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
	"""Feature-level topology init after `start` (inactive).

	Paper-aligned behavior: feature-level graphification is inactive at `start`.
	Only a topology prior is initialized and stored into `config`.
	"""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError(
			"pandas is required for TromPT feature-level placeholder. Install pandas to use this model."
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
	"""Stage 0: start (currently a pass-through)."""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError("pandas is required to run this model.") from _PANDAS_IMPORT_ERROR
	return train_df, val_df, test_df


def materialize_fn(train_df, val_df, test_df, dataset_results, config):
	"""Stage 1: Materialization.

	Convert split DataFrames into TensorFrames and build loaders/metric helpers.
	"""

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
			"Failed to import local torch_frame components required by TromPT materialize_fn."
		) from e

	try:
		from torchmetrics import AUROC, Accuracy, MeanSquaredError
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"torchmetrics is required for TromPT metrics. Install torchmetrics to use this model."
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

	channels = int(config.get("channels", 128))
	num_prompts = int(config.get("num_prompts", 128))
	num_layers = int(config.get("num_layers", 6))

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
		"channels": channels,
		"num_prompts": num_prompts,
		"num_layers": num_layers,
	}


def _tensor_frame_to_float_matrix(tensor_frame) -> torch.Tensor:
	"""Flatten a TensorFrame into a dense float feature matrix [N, D]."""

	xs = []
	for st in tensor_frame.stypes:
		feat = tensor_frame.feat_dict[st]
		if isinstance(feat, torch.Tensor):
			x = feat
		else:
			try:
				x = torch.as_tensor(feat)
			except Exception as e:
				raise TypeError(f"Unsupported feature tensor type for stype={st}") from e

		if x.dim() == 1:
			x = x.unsqueeze(-1)
		elif x.dim() > 2:
			x = x.reshape(x.shape[0], -1)

		xs.append(x.to(dtype=torch.float32))

	if not xs:
		raise ValueError("TensorFrame has no features (empty stypes list).")

	return torch.cat(xs, dim=1)


def row_gnn_after_materialize_fn(
	train_tensor_frame,
	val_tensor_frame,
	test_tensor_frame,
	config: dict,
	dataset_name: str,
	task_type: str,
):
	"""Offline row-embedding extraction after `materialize_fn`.

	Paper-aligned fairness rule: uses raw row feature vectors as node features.
	"""

	try:
		from torch_frame.data.loader import DataLoader
		from torch_frame.datasets.yandex import Yandex
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"Failed to import local torch_frame components required by TromPT row_gnn_after_materialize_fn."
		) from e

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError(
			"pandas is required for TromPT materialize-stage row-GNN injection. Install pandas to use this feature."
		) from _PANDAS_IMPORT_ERROR

	try:
		DGM_d_local = _import_dgm_d()
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"DGM_d (from DGM_pytorch/DGMlib) is required for row-level injection. "
			"Ensure the DGM_pytorch module is present and importable."
		) from e

	device = resolve_device(config)
	print(f"[MATERIALIZE-GNN-DGM] Using device: {device}")

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

	x_train = _tensor_frame_to_float_matrix(train_tensor_frame).to(device)
	x_val = _tensor_frame_to_float_matrix(val_tensor_frame).to(device)
	x_test = _tensor_frame_to_float_matrix(test_tensor_frame).to(device)

	y_train = train_tensor_frame.y
	y_val = val_tensor_frame.y

	if task_type in ["binclass", "multiclass"]:
		out_dim = int(torch.unique(torch.cat([y_train, y_val], dim=0)).numel())
		if task_type == "binclass" and out_dim != 2:
			out_dim = 2
	else:
		out_dim = 1

	n_train = int(x_train.shape[0])

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
			train_loss = F.cross_entropy(logits, y_train)
		else:
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
				val_loss = F.cross_entropy(logits_val, y_val)
			else:
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
			f"[MATERIALIZE-GNN-DGM] Epoch {epoch+1}/{gnn_epochs}, "
			f"Train Loss: {train_loss_val:.4f}, Val Loss: {val_loss_val:.4f}{suffix}"
		)

		if early_stop_counter >= patience:
			gnn_early_stop_epochs = epoch + 1
			print(
				f"[MATERIALIZE-GNN-DGM] Early stopping at epoch {epoch+1} (Val Loss: {val_loss_val:.4f})"
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

	train_df_emb = pd.DataFrame(emb_train.detach().cpu().numpy(), columns=row_emb_cols)
	val_df_emb = pd.DataFrame(emb_val.detach().cpu().numpy(), columns=row_emb_cols)
	test_df_emb = pd.DataFrame(emb_test.detach().cpu().numpy(), columns=row_emb_cols)

	train_df_emb["target"] = train_tensor_frame.y.detach().cpu().numpy()
	val_df_emb["target"] = val_tensor_frame.y.detach().cpu().numpy()
	test_df_emb["target"] = test_tensor_frame.y.detach().cpu().numpy()

	dataset = Yandex(train_df_emb, val_df_emb, test_df_emb, name=dataset_name, task_type=task_type)
	dataset.materialize(device=device)

	train_tensor_frame = dataset.tensor_frame[dataset.df["split_col"] == 0]
	val_tensor_frame = dataset.tensor_frame[dataset.df["split_col"] == 1]
	test_tensor_frame = dataset.tensor_frame[dataset.df["split_col"] == 2]

	batch_size = int(config.get("batch_size", 512))
	train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
	test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

	# After embedding injection, all features are numerical.
	col_stats = dataset.col_stats
	mutual_info_sort = None
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


def feature_gnn_after_materialize_fn(material_outputs: dict, config: dict, dataset_name: str, task_type: str) -> dict:
	"""Feature-level topology init after `materialize_fn` (inactive)."""

	train_tensor_frame = material_outputs["train_tensor_frame"]

	try:
		from torch_frame import stype
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"Failed to import torch_frame stype required by TromPT feature topology init."
		) from e

	feature_names = []
	for st in train_tensor_frame.stypes:
		feature_names.extend(list(train_tensor_frame.col_names_dict.get(st, [])))

	if not feature_names:
		# Fall back to numerical-only naming.
		feature_names = list(train_tensor_frame.col_names_dict.get(stype.numerical, []))

	init_mode = str(config.get("feature_topology_init", "chain")).lower()
	topology = _init_feature_topology_prior(feature_names, init_mode=init_mode)
	material_outputs["feature_topology"] = topology
	config["feature_topology_meta"] = _feature_topology_to_meta(topology)

	print(
		f"[MATERIALIZE] Feature-level topology initialized (inactive). "
		f"init_mode={topology['init_mode']}, num_features={topology['num_features']}"
	)
	return material_outputs


def trompt_core_fn_row_gnn(material_outputs, config, task_type, gnn_stage=None):
	"""TromPT core function (row-GNN capable)."""

	print("Executing trompt_core_fn_row_gnn")
	print(f"[CORE] gnn_stage={gnn_stage}")

	repo_root = Path(__file__).resolve().parents[2]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

	try:
		from torch.nn import LayerNorm, ModuleList, Parameter, ReLU, Sequential
		from torch.optim.lr_scheduler import ExponentialLR
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError("PyTorch is required to run TromPT.") from e

	try:
		from tqdm import tqdm
	except Exception:  # pragma: no cover
		def tqdm(x, **kwargs):
			return x

	try:
		from torch_frame import stype
		from torch_frame.data.stats import StatType
		from torch_frame.nn.conv import TromptConv
		from torch_frame.nn.decoder import TromptDecoder
		from torch_frame.nn.encoder.stype_encoder import EmbeddingEncoder, LinearEncoder, StypeEncoder
		from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
		from torch_frame.typing import NAStrategy
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"Failed to import local torch_frame components required by trompt_core_fn_row_gnn."
		) from e

	train_tensor_frame = material_outputs["train_tensor_frame"]
	train_loader = material_outputs["train_loader"]
	val_loader = material_outputs["val_loader"]
	test_loader = material_outputs["test_loader"]
	col_stats = material_outputs["col_stats"]
	device = material_outputs["device"]
	is_classification = bool(material_outputs["is_classification"])
	is_binary_class = bool(material_outputs["is_binary_class"])
	out_channels = int(material_outputs["out_channels"])
	metric_computer = material_outputs["metric_computer"]
	metric = material_outputs["metric"]

	channels = int(config.get("channels", material_outputs.get("channels", 128)))
	num_prompts = int(config.get("num_prompts", material_outputs.get("num_prompts", 128)))
	num_layers = int(config.get("num_layers", material_outputs.get("num_layers", 6)))
	patience = int(config.get("patience", 10))
	loss_threshold = float(config.get("loss_threshold", 1e-4))
	restore_best = bool(config.get("restore_best", True))

	col_names_dict = train_tensor_frame.col_names_dict
	num_cols = int(sum(len(v) for v in col_names_dict.values()))

	use_gnn = gnn_stage in ["encoding", "columnwise", "decoding"]
	gnn_hidden = int(config.get("gnn_hidden", 64))

	DGM_d_local = None
	if use_gnn:
		try:
			DGM_d_local = _import_dgm_d()
		except Exception as e:  # pragma: no cover
			raise ModuleNotFoundError(
				"DGM_d (from DGM_pytorch/DGMlib) is required for encoding/columnwise/decoding GNN stages."
			) from e

	# Learnable prompt tokens (shared across layers).
	x_prompt = Parameter(torch.empty(num_prompts, channels, device=device).normal_(std=0.01))
	torch.nn.init.normal_(x_prompt, std=0.01)

	# Online row-GNN components.
	gnn = None
	dgm_module = None
	self_attn = None
	attn_norm = None
	token_embed = None
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

		self_attn = torch.nn.MultiheadAttention(embed_dim=channels, num_heads=attn_heads, batch_first=True).to(device)
		attn_norm = torch.nn.LayerNorm(channels).to(device)

		# Single embedding buffer that can be sliced for both column tokens and prompt tokens.
		max_tokens = int(max(num_cols, num_prompts))
		token_embed = torch.nn.Parameter(torch.randn(max_tokens, channels, device=device))
		pool_query = torch.nn.Parameter(torch.randn(channels, device=device))

		if gnn_stage == "decoding":
			gnn = SimpleGCN(channels, gnn_hidden, out_channels, num_layers=2).to(device)
			print("[CORE] Decoding stage uses Self-Attention + DGM_d + GCN as the decoder.")
			print(f"[CORE]  - attention_heads={attn_heads}, dgm_k={dgm_k}, dgm_distance={dgm_distance}")
			print(f"[CORE]  - gnn_hidden={gnn_hidden}, out_channels={out_channels}")
		elif gnn_stage in ["encoding", "columnwise"]:
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

			label = "encoding" if gnn_stage == "encoding" else "columnwise"
			print(
				f"[CORE] {label} stage uses Self-Attention + DGM_d + GCN, then decodes back to tokens."
			)
			print(f"[CORE]  - attention_heads={attn_heads}, dgm_k={dgm_k}, dgm_distance={dgm_distance}")
			print(f"[CORE]  - gnn_hidden={gnn_hidden}, channels={channels}")

	# Per-layer feature encoders.
	encoders = ModuleList()
	for _ in range(num_layers):
		stype_encoder_dict_layer: dict[object, StypeEncoder]
		stype_encoder_dict_layer = {
			stype.categorical: EmbeddingEncoder(
				post_module=LayerNorm(channels),
				na_strategy=NAStrategy.MOST_FREQUENT,
			),
			stype.numerical: LinearEncoder(
				post_module=Sequential(ReLU(), LayerNorm(channels)),
				na_strategy=NAStrategy.MEAN,
			),
		}

		encoder = StypeWiseFeatureEncoder(
			out_channels=channels,
			col_stats=col_stats,
			col_names_dict=col_names_dict,
			stype_encoder_dict=stype_encoder_dict_layer,
		).to(device)
		encoders.append(encoder)

	trompt_convs = ModuleList([TromptConv(channels, num_cols, num_prompts).to(device) for _ in range(num_layers)])
	for conv in trompt_convs:
		conv.reset_parameters()

	trompt_decoder = TromptDecoder(channels, out_channels, num_prompts).to(device)
	trompt_decoder.reset_parameters()

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

	def _inject_tokens(tokens: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
		"""Inject row-GNN context into a token sequence [B,T,C]."""

		batch_size, t, c = tokens.shape
		embed_t = embed[:t].unsqueeze(0)

		tokens0 = tokens + embed_t
		tokens_norm = attn_norm(tokens0)
		attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
		tokens_attn = tokens0 + attn_out1
		ffn_out1 = ffn_pre(attn_norm(tokens_attn))
		tokens_attn = tokens_attn + ffn_out1

		pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(c)
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
		return tokens + fusion_alpha * tokens_out

	def _decode_with_gnn(tokens: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
		"""Decode row predictions with DGM+GCN from token sequence [B,T,C]."""

		_, t, c = tokens.shape
		embed_t = embed[:t].unsqueeze(0)

		tokens0 = tokens + embed_t
		tokens_norm = attn_norm(tokens0)
		attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
		tokens_attn = tokens0 + attn_out1

		pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(c)
		pool_weights = torch.softmax(pool_logits, dim=1)
		x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)

		x_dgm, edge_index_dgm, edge_weight_dgm = _dgm_forward(x_pooled)
		return gnn(x_dgm, edge_index_dgm, edge_weight=edge_weight_dgm)

	def forward_stacked(tf) -> torch.Tensor:
		batch_size = len(tf)
		x_prompt_batch = x_prompt.repeat(batch_size, 1, 1)

		prompts_outputs = []
		for i in range(num_layers):
			x, _ = encoders[i](tf)  # [B, F, C]
			if gnn_stage == "encoding" and use_gnn and i == 0:
				x = _inject_tokens(x, token_embed)

			if i == 0:
				prompt_in = x_prompt_batch
			else:
				prompt_in = prompts_outputs[-1]

			prompt_out = trompt_convs[i](x, prompt_in)  # [B, P, C]
			prompts_outputs.append(prompt_out)

		# Columnwise injection: inject once after all prompt updates.
		if gnn_stage == "columnwise" and use_gnn:
			prompts_outputs[-1] = _inject_tokens(prompts_outputs[-1], token_embed)

		outs = []
		for i in range(num_layers):
			out = trompt_decoder(prompts_outputs[i])
			out = out.view(batch_size, 1, out_channels)
			outs.append(out)
		return torch.cat(outs, dim=1)

	def forward(tf) -> torch.Tensor:
		if gnn_stage == "decoding" and use_gnn:
			# Build prompts as usual, then decode with row-GNN.
			batch_size = len(tf)
			x_prompt_batch = x_prompt.repeat(batch_size, 1, 1)

			prompt = x_prompt_batch
			for i in range(num_layers):
				x, _ = encoders[i](tf)
				prompt = trompt_convs[i](x, prompt)
			return _decode_with_gnn(prompt, token_embed)

		stacked = forward_stacked(tf)
		return stacked.mean(dim=1)

	lr = float(config.get("lr", 0.001))
	gamma = float(config.get("gamma", 0.95))

	params = [x_prompt]
	for encoder in encoders:
		params += list(encoder.parameters())
	for conv in trompt_convs:
		params += list(conv.parameters())
	params += list(trompt_decoder.parameters())

	if gnn is not None:
		params += list(gnn.parameters())
	if dgm_module is not None:
		params += list(dgm_module.parameters())
	if self_attn is not None:
		params += list(self_attn.parameters())
	if attn_norm is not None:
		params += list(attn_norm.parameters())
	if token_embed is not None:
		params += [token_embed]
	if pool_query is not None:
		params += [pool_query]
	if self_attn_out is not None:
		params += list(self_attn_out.parameters())
	if gcn_to_attn is not None:
		params += list(gcn_to_attn.parameters())
	if attn_out_norm is not None:
		params += list(attn_out_norm.parameters())
	if ffn_pre is not None:
		params += list(ffn_pre.parameters())
	if ffn_post is not None:
		params += list(ffn_post.parameters())
	if fusion_alpha_param is not None:
		params += [fusion_alpha_param]
	if dgm_gate_threshold_param is not None:
		params += [dgm_gate_threshold_param]

	optimizer = torch.optim.Adam(params, lr=lr)
	lr_scheduler = ExponentialLR(optimizer, gamma=gamma)

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
	test_metrics = []

	def _set_train_mode():
		for encoder in encoders:
			encoder.train()
		for conv in trompt_convs:
			conv.train()
		trompt_decoder.train()
		if dgm_module is not None:
			dgm_module.train()
		if gnn is not None:
			gnn.train()
		if self_attn is not None:
			self_attn.train()
		if self_attn_out is not None:
			self_attn_out.train()
		if attn_norm is not None:
			attn_norm.train()
		if attn_out_norm is not None:
			attn_out_norm.train()
		if gcn_to_attn is not None:
			gcn_to_attn.train()
		if ffn_pre is not None:
			ffn_pre.train()
		if ffn_post is not None:
			ffn_post.train()

	def _set_eval_mode():
		for encoder in encoders:
			encoder.eval()
		for conv in trompt_convs:
			conv.eval()
		trompt_decoder.eval()
		if dgm_module is not None:
			dgm_module.eval()
		if gnn is not None:
			gnn.eval()
		if self_attn is not None:
			self_attn.eval()
		if self_attn_out is not None:
			self_attn_out.eval()
		if attn_norm is not None:
			attn_norm.eval()
		if attn_out_norm is not None:
			attn_out_norm.eval()
		if gcn_to_attn is not None:
			gcn_to_attn.eval()
		if ffn_pre is not None:
			ffn_pre.eval()
		if ffn_post is not None:
			ffn_post.eval()

	def train_one_epoch(epoch: int) -> float:
		_set_train_mode()
		loss_accum = 0.0
		total_count = 0
		for tf in tqdm(train_loader, desc=f"Epoch: {epoch}"):
			tf = tf.to(device)
			pred = forward(tf)
			y = tf.y

			if is_classification:
				loss = F.cross_entropy(pred, y)
			else:
				loss = F.mse_loss(pred.view(-1), y.view(-1))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			loss_accum += float(loss) * len(y)
			total_count += len(y)

		return loss_accum / max(1, total_count)

	@torch.no_grad()
	def eval_loader(loader):
		_set_eval_mode()
		metric_computer.reset()
		loss_accum = 0.0
		total_count = 0
		for tf in loader:
			tf = tf.to(device)
			pred = forward(tf)
			if is_classification:
				loss = F.cross_entropy(pred, tf.y)
			else:
				loss = F.mse_loss(pred.view(-1), tf.y.view(-1))
			loss_accum += float(loss) * len(tf.y)
			total_count += len(tf.y)

			if is_binary_class:
				metric_computer.update(pred[:, 1], tf.y)
			elif is_classification:
				pred_class = pred.argmax(dim=-1)
				metric_computer.update(pred_class, tf.y)
			else:
				metric_computer.update(pred.view(-1), tf.y.view(-1))

		avg_loss = loss_accum / max(1, total_count)
		computed = metric_computer.compute().item()
		if not is_classification:
			computed = computed ** 0.5
		return computed, avg_loss

	epochs = int(config.get("epochs", 300))
	for epoch in range(1, epochs + 1):
		train_loss = train_one_epoch(epoch)
		train_metric, _ = eval_loader(train_loader)
		val_metric, val_loss = eval_loader(val_loader)
		test_metric, _ = eval_loader(test_loader)

		train_losses.append(train_loss)
		train_metrics.append(train_metric)
		val_metrics.append(val_metric)
		val_losses.append(val_loss)
		test_metrics.append(test_metric)

		improved = val_loss < best_val_loss - loss_threshold
		if improved:
			best_val_loss = val_loss
			best_val_metric = val_metric
			best_epoch = epoch
			early_stop_counter = 0

			if restore_best:
				best_states = {
					"x_prompt": x_prompt.detach().clone(),
					"encoders": [e.state_dict() for e in encoders],
					"convs": [c.state_dict() for c in trompt_convs],
					"decoder": trompt_decoder.state_dict(),
				}
				if gnn is not None:
					best_states["gnn"] = gnn.state_dict()
				if dgm_module is not None:
					best_states["dgm_module"] = dgm_module.state_dict()
				if self_attn is not None:
					best_states["self_attn"] = self_attn.state_dict()
				if attn_norm is not None:
					best_states["attn_norm"] = attn_norm.state_dict()
				if token_embed is not None:
					best_states["token_embed"] = token_embed.detach().clone()
				if pool_query is not None:
					best_states["pool_query"] = pool_query.detach().clone()
				if self_attn_out is not None:
					best_states["self_attn_out"] = self_attn_out.state_dict()
				if gcn_to_attn is not None:
					best_states["gcn_to_attn"] = gcn_to_attn.state_dict()
				if attn_out_norm is not None:
					best_states["attn_out_norm"] = attn_out_norm.state_dict()
				if ffn_pre is not None:
					best_states["ffn_pre"] = ffn_pre.state_dict()
				if ffn_post is not None:
					best_states["ffn_post"] = ffn_post.state_dict()
				if fusion_alpha_param is not None:
					best_states["fusion_alpha_param"] = fusion_alpha_param.detach().clone()
				if dgm_gate_threshold_param is not None:
					best_states["dgm_gate_threshold"] = dgm_gate_threshold_param.detach().clone()

			marker = "✓"
		else:
			early_stop_counter += 1
			marker = "✗"

		print(
			f"[CORE] Epoch {epoch:3d} {marker} | "
			f"Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
			f"Val {metric}: {val_metric:.4f}, Val Loss: {val_loss:.4f}, "
			f"Test {metric}: {test_metric:.4f} | "
			f"Early Stop: {early_stop_counter}/{patience}"
		)

		lr_scheduler.step()

		if early_stop_counter >= patience:
			early_stop_epochs = epoch
			print(
				f"[CORE] Early stopping at epoch {epoch} "
				f"(best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f})"
			)
			break

	if restore_best and best_states is not None:
		with torch.no_grad():
			x_prompt.copy_(best_states["x_prompt"])
		for i, e in enumerate(encoders):
			e.load_state_dict(best_states["encoders"][i])
		for i, c in enumerate(trompt_convs):
			c.load_state_dict(best_states["convs"][i])
		trompt_decoder.load_state_dict(best_states["decoder"])
		if gnn is not None and best_states.get("gnn") is not None:
			gnn.load_state_dict(best_states["gnn"])
		if dgm_module is not None and best_states.get("dgm_module") is not None:
			dgm_module.load_state_dict(best_states["dgm_module"])
		if self_attn is not None and best_states.get("self_attn") is not None:
			self_attn.load_state_dict(best_states["self_attn"])
		if attn_norm is not None and best_states.get("attn_norm") is not None:
			attn_norm.load_state_dict(best_states["attn_norm"])
		if token_embed is not None and best_states.get("token_embed") is not None:
			with torch.no_grad():
				token_embed.copy_(best_states["token_embed"])
		if pool_query is not None and best_states.get("pool_query") is not None:
			with torch.no_grad():
				pool_query.copy_(best_states["pool_query"])
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
		if fusion_alpha_param is not None and best_states.get("fusion_alpha_param") is not None:
			with torch.no_grad():
				fusion_alpha_param.copy_(best_states["fusion_alpha_param"])
		if dgm_gate_threshold_param is not None and best_states.get("dgm_gate_threshold") is not None:
			with torch.no_grad():
				dgm_gate_threshold_param.copy_(best_states["dgm_gate_threshold"])
		print(f"[CORE] Restored best weights from epoch {best_epoch}")

	test_metric, test_loss = eval_loader(test_loader)
	print(f"[CORE] Training complete. Best Val Loss: {best_val_loss:.4f}, Best Val {metric}: {best_val_metric:.4f}")
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
		"x_prompt": x_prompt,
		"encoders": encoders,
		"convs": trompt_convs,
		"decoder": trompt_decoder,
		"gnn": gnn,
		"train_loader": train_loader,
		"val_loader": val_loader,
		"test_loader": test_loader,
		"is_classification": is_classification,
		"is_binary_class": is_binary_class,
		"metric_computer": metric_computer,
		"metric": metric,
		"early_stop_epochs": early_stop_epochs,
		"gnn_stage": gnn_stage,
	}


def trompt_core_fn_feature_gnn(material_outputs, config, task_type, gnn_stage=None):
	"""TromPT core function (feature graphify).

	Paper-aligned behavior:
	- start/materialize: inactive (topology prior may be initialized, but no token mixing).
	- encoding/columnwise/decoding: topology-masked mixing becomes active once tokens exist.

	Note: TromPT's decoder is prompt-based; feature graphification is applied
	on feature tokens (columns) before prompt interaction.
	"""

	print(f"Executing trompt_core_fn_feature_gnn with gnn_stage={gnn_stage}")

	stage = "none" if gnn_stage is None else str(gnn_stage).lower()
	if stage in {"none", "start", "materialize"}:
		return trompt_core_fn_row_gnn(material_outputs, config, task_type, gnn_stage="none")

	if stage not in {"encoding", "columnwise", "decoding"}:
		raise ValueError(
			f"Unknown gnn_stage={stage!r} for graphify=feature. "
			"Expected one of: none/start/materialize/encoding/columnwise/decoding."
		)

	repo_root = Path(__file__).resolve().parents[2]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

	try:
		from torch.nn import LayerNorm, ModuleList, Parameter, ReLU, Sequential
		from torch.optim.lr_scheduler import ExponentialLR
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError("PyTorch is required to run TromPT.") from e

	try:
		from tqdm import tqdm
	except Exception:  # pragma: no cover
		def tqdm(x, **kwargs):
			return x

	try:
		from torch_frame import stype
		from torch_frame.nn.conv import TromptConv
		from torch_frame.nn.decoder import TromptDecoder
		from torch_frame.nn.encoder.stype_encoder import EmbeddingEncoder, LinearEncoder, StypeEncoder
		from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
		from torch_frame.typing import NAStrategy
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"Failed to import local torch_frame components required by trompt_core_fn_feature_gnn."
		) from e

	train_tensor_frame = material_outputs["train_tensor_frame"]
	train_loader = material_outputs["train_loader"]
	val_loader = material_outputs["val_loader"]
	test_loader = material_outputs["test_loader"]
	col_stats = material_outputs["col_stats"]
	device = material_outputs["device"]
	is_classification = bool(material_outputs["is_classification"])
	is_binary_class = bool(material_outputs["is_binary_class"])
	out_channels = int(material_outputs["out_channels"])
	metric_computer = material_outputs["metric_computer"]
	metric = material_outputs["metric"]

	channels = int(config.get("channels", material_outputs.get("channels", 128)))
	num_prompts = int(config.get("num_prompts", material_outputs.get("num_prompts", 128)))
	num_layers = int(config.get("num_layers", material_outputs.get("num_layers", 6)))
	patience = int(config.get("patience", 10))
	loss_threshold = float(config.get("loss_threshold", 1e-4))
	restore_best = bool(config.get("restore_best", True))

	col_names_dict = train_tensor_frame.col_names_dict
	num_cols = int(sum(len(v) for v in col_names_dict.values()))

	# Ensure a topology prior exists.
	topology = material_outputs.get("feature_topology")
	if topology is None:
		feature_names = []
		for st in train_tensor_frame.stypes:
			feature_names.extend(list(train_tensor_frame.col_names_dict.get(st, [])))
		init_mode = str(config.get("feature_topology_init", "chain")).lower()
		topology = _init_feature_topology_prior(feature_names, init_mode=init_mode)
		material_outputs["feature_topology"] = topology
		config["feature_topology_meta"] = _feature_topology_to_meta(topology)
		print(
			f"[CORE] Feature topology prior initialized in-core (inactive earlier). "
			f"init_mode={topology['init_mode']}, num_features={topology['num_features']}"
		)

	allowed_mask = topology["adjacency_mask"].to(device=device)
	topology_mixing = FeatureTopologyMaskedMixing(allowed_mask).to(device)
	fusion_alpha_param = torch.nn.Parameter(torch.tensor(-0.847, device=device))

	def _feature_inject(x_tokens: torch.Tensor) -> torch.Tensor:
		x_mp = topology_mixing(x_tokens)
		alpha = torch.sigmoid(fusion_alpha_param)
		return x_tokens + alpha * (x_mp - x_tokens)

	x_prompt = Parameter(torch.empty(num_prompts, channels, device=device).normal_(std=0.01))
	torch.nn.init.normal_(x_prompt, std=0.01)

	encoders = ModuleList()
	for _ in range(num_layers):
		stype_encoder_dict_layer: dict[object, StypeEncoder]
		stype_encoder_dict_layer = {
			stype.categorical: EmbeddingEncoder(
				post_module=LayerNorm(channels),
				na_strategy=NAStrategy.MOST_FREQUENT,
			),
			stype.numerical: LinearEncoder(
				post_module=Sequential(ReLU(), LayerNorm(channels)),
				na_strategy=NAStrategy.MEAN,
			),
		}
		encoders.append(
			StypeWiseFeatureEncoder(
				out_channels=channels,
				col_stats=col_stats,
				col_names_dict=col_names_dict,
				stype_encoder_dict=stype_encoder_dict_layer,
			).to(device)
		)

	trompt_convs = ModuleList([TromptConv(channels, num_cols, num_prompts).to(device) for _ in range(num_layers)])
	for conv in trompt_convs:
		conv.reset_parameters()

	trompt_decoder = TromptDecoder(channels, out_channels, num_prompts).to(device)
	trompt_decoder.reset_parameters()

	def forward(tf) -> torch.Tensor:
		batch_size = len(tf)
		prompt = x_prompt.repeat(batch_size, 1, 1)
		for i in range(num_layers):
			x, _ = encoders[i](tf)

			if stage == "encoding" and i == 0:
				x = _feature_inject(x)
			elif stage == "columnwise":
				x = _feature_inject(x)
			elif stage == "decoding" and i == (num_layers - 1):
				x = _feature_inject(x)

			prompt = trompt_convs[i](x, prompt)
		return trompt_decoder(prompt)

	lr = float(config.get("lr", 0.001))
	gamma = float(config.get("gamma", 0.95))

	params = [x_prompt] + [fusion_alpha_param] + list(topology_mixing.parameters())
	for encoder in encoders:
		params += list(encoder.parameters())
	for conv in trompt_convs:
		params += list(conv.parameters())
	params += list(trompt_decoder.parameters())

	optimizer = torch.optim.Adam(params, lr=lr)
	lr_scheduler = ExponentialLR(optimizer, gamma=gamma)

	def _set_train_mode():
		for encoder in encoders:
			encoder.train()
		for conv in trompt_convs:
			conv.train()
		trompt_decoder.train()
		topology_mixing.train()

	def _set_eval_mode():
		for encoder in encoders:
			encoder.eval()
		for conv in trompt_convs:
			conv.eval()
		trompt_decoder.eval()
		topology_mixing.eval()

	def train_one_epoch(epoch: int) -> float:
		_set_train_mode()
		loss_accum = 0.0
		total_count = 0
		for tf in tqdm(train_loader, desc=f"Epoch: {epoch}"):
			tf = tf.to(device)
			pred = forward(tf)
			if is_classification:
				loss = F.cross_entropy(pred, tf.y)
			else:
				loss = F.mse_loss(pred.view(-1), tf.y.view(-1))
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
		loss_accum = 0.0
		total_count = 0
		for tf in loader:
			tf = tf.to(device)
			pred = forward(tf)
			if is_classification:
				loss = F.cross_entropy(pred, tf.y)
			else:
				loss = F.mse_loss(pred.view(-1), tf.y.view(-1))
			loss_accum += float(loss) * len(tf.y)
			total_count += len(tf.y)

			if is_binary_class:
				metric_computer.update(pred[:, 1], tf.y)
			elif is_classification:
				metric_computer.update(pred.argmax(dim=-1), tf.y)
			else:
				metric_computer.update(pred.view(-1), tf.y.view(-1))

		avg_loss = loss_accum / max(1, total_count)
		computed = metric_computer.compute().item()
		if not is_classification:
			computed = computed ** 0.5
		return computed, avg_loss

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
	test_metrics = []

	epochs = int(config.get("epochs", 300))
	for epoch in range(1, epochs + 1):
		train_loss = train_one_epoch(epoch)
		train_metric, _ = eval_loader(train_loader)
		val_metric, val_loss = eval_loader(val_loader)
		test_metric, _ = eval_loader(test_loader)

		train_losses.append(train_loss)
		train_metrics.append(train_metric)
		val_metrics.append(val_metric)
		val_losses.append(val_loss)
		test_metrics.append(test_metric)

		improved = val_loss < best_val_loss - loss_threshold
		if improved:
			best_val_loss = val_loss
			best_val_metric = val_metric
			best_epoch = epoch
			early_stop_counter = 0
			if restore_best:
				best_states = {
					"x_prompt": x_prompt.detach().clone(),
					"encoders": [e.state_dict() for e in encoders],
					"convs": [c.state_dict() for c in trompt_convs],
					"decoder": trompt_decoder.state_dict(),
					"topology_mixing": topology_mixing.state_dict(),
					"fusion_alpha": fusion_alpha_param.detach().clone(),
				}
			marker = "✓"
		else:
			early_stop_counter += 1
			marker = "✗"

		print(
			f"[CORE] Epoch {epoch:3d} {marker} | "
			f"Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
			f"Val {metric}: {val_metric:.4f}, Val Loss: {val_loss:.4f}, "
			f"Test {metric}: {test_metric:.4f} | "
			f"Early Stop: {early_stop_counter}/{patience}"
		)

		lr_scheduler.step()

		if early_stop_counter >= patience:
			early_stop_epochs = epoch
			print(
				f"[CORE] Early stopping at epoch {epoch} "
				f"(best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f})"
			)
			break

	if restore_best and best_states is not None:
		with torch.no_grad():
			x_prompt.copy_(best_states["x_prompt"])
			fusion_alpha_param.copy_(best_states["fusion_alpha"])
		for i, e in enumerate(encoders):
			e.load_state_dict(best_states["encoders"][i])
		for i, c in enumerate(trompt_convs):
			c.load_state_dict(best_states["convs"][i])
		trompt_decoder.load_state_dict(best_states["decoder"])
		topology_mixing.load_state_dict(best_states["topology_mixing"])
		print(f"[CORE] Restored best weights from epoch {best_epoch}")

	test_metric, test_loss = eval_loader(test_loader)
	print(f"[CORE] Training complete. Best Val Loss: {best_val_loss:.4f}, Best Val {metric}: {best_val_metric:.4f}")
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
		"x_prompt": x_prompt,
		"encoders": encoders,
		"convs": trompt_convs,
		"decoder": trompt_decoder,
		"topology_mixing": topology_mixing,
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

	print("TromPT - staged execution")
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
					_mutual_info_sort,
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
						"dataset": dataset,
						"train_tensor_frame": train_tensor_frame,
						"val_tensor_frame": val_tensor_frame,
						"test_tensor_frame": test_tensor_frame,
						"gnn_early_stop_epochs": gnn_early_stop_epochs,
					}
				)

			results = trompt_core_fn_row_gnn(material_outputs, config, task_type, gnn_stage=stage)

		elif graphify == "feature":
			if stage == "start":
				train_df, val_df, test_df, gnn_early_stop_epochs = feature_gnn_after_start_fn(
					train_df, val_df, test_df, config, task_type
				)

			material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
			material_outputs["gnn_early_stop_epochs"] = gnn_early_stop_epochs

			if stage == "materialize":
				material_outputs = feature_gnn_after_materialize_fn(
					material_outputs,
					config,
					dataset_results["dataset"],
					task_type,
				)

			results = trompt_core_fn_feature_gnn(material_outputs, config, task_type, gnn_stage=stage)

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
#  python main.py --models trompt --dataset kaggle_Audit_Data --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  small+regression
#  python main.py --models trompt --dataset openml_The_Office_Dataset --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+binclass
#  python main.py --models trompt --dataset credit --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+multiclass
#  python main.py --models trompt --dataset eye --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  python main.py --models trompt --dataset helena --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+regression
#  python main.py --models trompt --dataset house --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0

