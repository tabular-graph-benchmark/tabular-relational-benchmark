from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
	import pandas as pd
except ModuleNotFoundError as e:  # pragma: no cover
	pd = None
	_PANDAS_IMPORT_ERROR = e

try:
	from sklearn.linear_model import LinearRegression, LogisticRegression
	from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
except ModuleNotFoundError as e:  # pragma: no cover
	LinearRegression = None
	LogisticRegression = None
	accuracy_score = None
	mean_squared_error = None
	roc_auc_score = None
	_SKLEARN_IMPORT_ERROR = e

try:
	from sklearn.neighbors import NearestNeighbors
except ModuleNotFoundError as e:  # pragma: no cover
	NearestNeighbors = None
	_SKLEARN_NN_IMPORT_ERROR = e

try:
	from torch_geometric.nn import GCNConv
except ModuleNotFoundError as e:  # pragma: no cover
	GCNConv = None
	_PYG_IMPORT_ERROR = e

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset


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
	- If `config['gpu'] < 0`, force CPU.
	- Otherwise fall back to default CUDA or CPU.
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
	"""A minimal GCN used by row-graphification hooks."""

	def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
		super().__init__()
		if GCNConv is None:  # pragma: no cover
			raise ModuleNotFoundError(
				"torch_geometric is required for SubTab row-graphification. Install torch-geometric to use this feature."
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


def knn_graph(x: torch.Tensor, k: int, directed: bool = False) -> torch.Tensor:
	"""Build a k-NN graph edge_index for node features `x`.

	Kept for parity with other backbones; DGM_d is the primary row graph learner.
	"""

	if NearestNeighbors is None:  # pragma: no cover
		raise ModuleNotFoundError(
			"scikit-learn is required for k-NN graph construction. Install scikit-learn to use this feature."
		) from _SKLEARN_NN_IMPORT_ERROR

	x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
	nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(x_np)
	_, indices = nbrs.kneighbors(x_np)

	edge_list = []
	n = x_np.shape[0]
	for i in range(n):
		for j in indices[i][1:]:
			edge_list.append([i, j])
			if not directed:
				edge_list.append([j, i])

	if not edge_list:
		edge_list = [[i, i] for i in range(n)]

	return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


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
	"""Convert DGM_d edges into an undirected self-looped graph, optionally weighted."""

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


def _init_feature_topology_prior(feature_names: Iterable[str], init_mode: str = "chain") -> dict:
	"""Initialize a schema-level feature topology prior."""

	feature_names = list(feature_names)
	n = int(len(feature_names))
	if n <= 0:
		raise ValueError("feature_names must be non-empty")

	init_mode = str(init_mode or "chain").lower()
	if init_mode not in {"chain", "fully_connected", "identity"}:
		raise ValueError(f"Unknown feature_topology_init={init_mode!r}.")

	adj = torch.zeros((n, n), dtype=torch.bool)
	if init_mode == "identity":
		adj.fill_(False)
	elif init_mode == "fully_connected":
		adj.fill_(True)
	else:  # chain
		adj.fill_(False)
		for i in range(n):
			adj[i, i] = True
			if i - 1 >= 0:
				adj[i, i - 1] = True
			if i + 1 < n:
				adj[i, i + 1] = True

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
	"""Topology-masked feature mixing shared across samples.

	tokens: [B, F, C] -> [B, F, C]
	"""

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


class TabularDataset(Dataset):
	"""A minimal DataFrame-backed dataset for SubTab."""

	def __init__(self, df: "pd.DataFrame", *, target_col: str = "target"):
		if pd is None:  # pragma: no cover
			raise ModuleNotFoundError("pandas is required for SubTab") from _PANDAS_IMPORT_ERROR
		self.df = df.reset_index(drop=True)
		self.target_col = target_col
		if target_col in self.df.columns:
			x_df = self.df.drop(columns=[target_col])
			self.y = self.df[target_col].values
		else:
			x_df = self.df
			self.y = np.zeros(len(self.df), dtype=np.float32)
		self.x = x_df.values.astype(np.float32)

	def __len__(self) -> int:
		return int(self.x.shape[0])

	def __getitem__(self, idx: int):
		return self.x[idx], self.y[idx]


def start_fn(train_df, val_df, test_df):
	return train_df, val_df, test_df


def materialize_fn(train_df, val_df, test_df, dataset_results=None, config=None) -> dict:
	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError("pandas is required for SubTab") from _PANDAS_IMPORT_ERROR

	config = config or {}
	batch_size = int(config.get("batch_size", 256))

	train_loader = TorchDataLoader(TabularDataset(train_df), batch_size=batch_size, shuffle=True)
	val_loader = TorchDataLoader(TabularDataset(val_df), batch_size=batch_size, shuffle=False)
	test_loader = TorchDataLoader(TabularDataset(test_df), batch_size=batch_size, shuffle=False)

	feature_names = [c for c in train_df.columns if c != "target"]

	return {
		"train_loader": train_loader,
		"val_loader": val_loader,
		"test_loader": test_loader,
		"feature_names": feature_names,
		"device": resolve_device(config),
	}


def row_gnn_after_start_fn(
	train_df: "pd.DataFrame",
	val_df: "pd.DataFrame",
	test_df: "pd.DataFrame",
	config: dict,
	task_type: str,
) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame", int]:
	"""Offline row-embedding extraction between start and materialize.

	Output schema:
	- Features are replaced by numerical embedding columns: N_feature_emb_1 .. N_feature_emb_{d}
	- Target is preserved in the `target` column.
	"""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError(
			"pandas is required for SubTab row-graphification. Install pandas to use this model."
		) from _PANDAS_IMPORT_ERROR

	device = resolve_device(config)
	x_train, x_val, x_test, y_train_np, y_val_np, y_test_np, feature_cols = _df_splits_to_tensors(
		train_df, val_df, test_df, device=device
	)

	emb_train, emb_val, emb_test, gnn_early_stop_epochs = _offline_row_embedding_extraction(
		x_train=x_train,
		x_val=x_val,
		x_test=x_test,
		y_train_np=y_train_np,
		y_val_np=y_val_np,
		y_test_np=y_test_np,
		task_type=task_type,
		config=config,
		num_cols=int(len(feature_cols)),
		stage_tag="START",
	)

	row_emb_cols = [f"N_feature_emb_{i + 1}" for i in range(int(emb_train.shape[1]))]
	train_df_emb = pd.DataFrame(emb_train, columns=row_emb_cols, index=train_df.index)
	val_df_emb = pd.DataFrame(emb_val, columns=row_emb_cols, index=val_df.index)
	test_df_emb = pd.DataFrame(emb_test, columns=row_emb_cols, index=test_df.index)
	train_df_emb["target"] = train_df["target"].values
	val_df_emb["target"] = val_df["target"].values
	test_df_emb["target"] = test_df["target"].values
	return train_df_emb, val_df_emb, test_df_emb, int(gnn_early_stop_epochs)


def row_gnn_after_materialize_fn(
	train_loader: TorchDataLoader,
	val_loader: TorchDataLoader,
	test_loader: TorchDataLoader,
	config: dict,
	task_type: str,
) -> tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader, int]:
	"""Offline row-embedding extraction after DataLoader materialization."""

	device = resolve_device(config)
	x_train, y_train_np = _collect_loader_xy(train_loader, device=device)
	x_val, y_val_np = _collect_loader_xy(val_loader, device=device)
	x_test, y_test_np = _collect_loader_xy(test_loader, device=device)

	num_cols = int(x_train.shape[1])
	emb_train, emb_val, emb_test, gnn_early_stop_epochs = _offline_row_embedding_extraction(
		x_train=x_train,
		x_val=x_val,
		x_test=x_test,
		y_train_np=y_train_np,
		y_val_np=y_val_np,
		y_test_np=y_test_np,
		task_type=task_type,
		config=config,
		num_cols=num_cols,
		stage_tag="MATERIALIZE",
	)

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError("pandas is required for SubTab") from _PANDAS_IMPORT_ERROR

	row_emb_cols = [f"N_feature_emb_{i + 1}" for i in range(int(emb_train.shape[1]))]
	train_df = pd.DataFrame(emb_train, columns=row_emb_cols)
	val_df = pd.DataFrame(emb_val, columns=row_emb_cols)
	test_df = pd.DataFrame(emb_test, columns=row_emb_cols)
	train_df["target"] = y_train_np
	val_df["target"] = y_val_np
	test_df["target"] = y_test_np

	batch_size = int(config.get("batch_size", 256))
	new_train_loader = TorchDataLoader(TabularDataset(train_df), batch_size=batch_size, shuffle=True)
	new_val_loader = TorchDataLoader(TabularDataset(val_df), batch_size=batch_size, shuffle=False)
	new_test_loader = TorchDataLoader(TabularDataset(test_df), batch_size=batch_size, shuffle=False)
	return new_train_loader, new_val_loader, new_test_loader, int(gnn_early_stop_epochs)


def feature_gnn_after_start_fn(
	train_df: "pd.DataFrame",
	val_df: "pd.DataFrame",
	test_df: "pd.DataFrame",
	config: dict,
	task_type: str,
) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame", int]:
	"""Feature topology init after start (inactive)."""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError("pandas is required for SubTab") from _PANDAS_IMPORT_ERROR

	feature_names = [c for c in train_df.columns if c != "target"]
	init_mode = str(config.get("feature_topology_init", "chain")).lower()
	topology = _init_feature_topology_prior(feature_names, init_mode=init_mode)
	config["feature_topology_meta"] = _feature_topology_to_meta(topology)
	print(
		f"[START-GNN] Feature topology initialized (inactive). init_mode={topology['init_mode']}, "
		f"num_features={topology['num_features']}"
	)
	return train_df, val_df, test_df, 0


def feature_gnn_after_materialize_fn(material_outputs: dict, config: dict, task_type: str) -> dict:
	"""Feature topology init after materialize (inactive)."""

	feature_names = material_outputs.get("feature_names")
	if not feature_names:
		return material_outputs
	init_mode = str(config.get("feature_topology_init", "chain")).lower()
	topology = _init_feature_topology_prior(feature_names, init_mode=init_mode)
	config["feature_topology_meta"] = _feature_topology_to_meta(topology)
	material_outputs["feature_topology"] = topology
	return material_outputs


@dataclass
class _SubTabSpec:
	n_subsets: int
	overlap: float
	subset_size: int
	actual_subset_size: int
	latent_dim: int


class _MLP(torch.nn.Module):
	def __init__(
		self,
		in_dim: int,
		hidden_dims: list[int],
		out_dim: int,
		*,
		dropout: float,
		use_batch_norm: bool,
	):
		super().__init__()
		dims = [int(in_dim)] + [int(d) for d in hidden_dims] + [int(out_dim)]
		layers: list[torch.nn.Module] = []
		for i in range(len(dims) - 1):
			layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
			if i < len(dims) - 2:
				if use_batch_norm:
					layers.append(torch.nn.BatchNorm1d(dims[i + 1]))
				layers.append(torch.nn.ReLU())
				if dropout > 0:
					layers.append(torch.nn.Dropout(dropout))
		self.net = torch.nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class SubTabAutoEncoder(torch.nn.Module):
	"""A lightweight SubTab-style subset autoencoder."""

	def __init__(
		self,
		subset_dim: int,
		*,
		latent_dim: int,
		hidden_dims: list[int],
		dropout: float,
		use_batch_norm: bool,
	):
		super().__init__()
		self.encoder = _MLP(
			subset_dim,
			hidden_dims=hidden_dims,
			out_dim=latent_dim,
			dropout=dropout,
			use_batch_norm=use_batch_norm,
		)
		self.decoder = _MLP(
			latent_dim,
			hidden_dims=list(reversed(hidden_dims)),
			out_dim=subset_dim,
			dropout=dropout,
			use_batch_norm=use_batch_norm,
		)
		self.proj = torch.nn.Sequential(
			torch.nn.Linear(latent_dim, latent_dim),
			torch.nn.ReLU(),
			torch.nn.Linear(latent_dim, latent_dim),
		)

	def encode(self, subset_x: torch.Tensor) -> torch.Tensor:
		return self.encoder(subset_x)

	def decode(self, latent: torch.Tensor) -> torch.Tensor:
		return self.decoder(latent)

	def project(self, latent: torch.Tensor) -> torch.Tensor:
		z = self.proj(latent)
		z = F.normalize(z, p=2, dim=-1)
		return z


def _make_subtab_spec(input_dim: int, config: dict) -> _SubTabSpec:
	n_subsets = int(config.get("n_subsets", 4))
	overlap = float(config.get("overlap", 0.75))
	subset_size = max(1, int(input_dim / max(1, n_subsets)))
	n_overlap = int(overlap * subset_size)
	actual_subset_size = int(min(input_dim, subset_size + 2 * n_overlap))
	latent_dim = int(config.get("subtab_latent_dim", 64))
	return _SubTabSpec(
		n_subsets=n_subsets,
		overlap=overlap,
		subset_size=subset_size,
		actual_subset_size=actual_subset_size,
		latent_dim=latent_dim,
	)


def _subset_generator(
	x_batch: torch.Tensor,
	*,
	spec: _SubTabSpec,
	add_noise: bool,
	noise_level: float,
) -> list[torch.Tensor]:
	n_features = int(x_batch.shape[1])
	subset_size = int(spec.subset_size)
	n_overlap = int(spec.overlap * subset_size)
	subsets: list[torch.Tensor] = []
	for i in range(int(spec.n_subsets)):
		start_idx = max(0, i * subset_size - n_overlap)
		end_idx = min(n_features, (i + 1) * subset_size + n_overlap)
		subset = x_batch[:, start_idx:end_idx]

		cur = int(subset.shape[1])
		if cur < int(spec.actual_subset_size):
			padding = torch.zeros(
				subset.shape[0],
				int(spec.actual_subset_size) - cur,
				device=subset.device,
				dtype=subset.dtype,
			)
			subset = torch.cat([subset, padding], dim=1)
		elif cur > int(spec.actual_subset_size):
			subset = subset[:, : int(spec.actual_subset_size)]

		if add_noise:
			subset = subset + torch.randn_like(subset) * float(noise_level)

		subsets.append(subset)
	return subsets


def _apply_feature_mixing_if_enabled(
	x_batch: torch.Tensor,
	*,
	mixer: FeatureTopologyMaskedMixing | None,
) -> torch.Tensor:
	if mixer is None:
		return x_batch
	tokens = x_batch.unsqueeze(-1)  # [B,F,1]
	mixed = mixer(tokens).squeeze(-1)
	return mixed


def _train_autoencoder(
	train_loader: TorchDataLoader,
	val_loader: TorchDataLoader,
	*,
	config: dict,
	device: torch.device,
	spec: _SubTabSpec,
	feature_mixer: FeatureTopologyMaskedMixing | None,
	row_latent_injector: "_RowLatentInjector" | None,
	epochs: int,
) -> tuple[SubTabAutoEncoder, dict, int]:
	lr = float(config.get("lr", 1e-3))
	weight_decay = float(config.get("weight_decay", 0.0))
	patience = int(config.get("patience", 10))
	loss_threshold = float(config.get("loss_threshold", 1e-4))
	dropout = float(config.get("subtab_dropout", 0.2))
	use_batch_norm = bool(int(config.get("subtab_batch_norm", 0)))
	hidden_dims = list(config.get("subtab_hidden_dims", [128, 128]))
	hidden_dims = [int(d) for d in hidden_dims]

	model = SubTabAutoEncoder(
		int(spec.actual_subset_size),
		latent_dim=int(spec.latent_dim),
		hidden_dims=hidden_dims,
		dropout=dropout,
		use_batch_norm=use_batch_norm,
	).to(device)

	params: list[torch.nn.Parameter] = list(model.parameters())
	if feature_mixer is not None:
		params += list(feature_mixer.parameters())
	if row_latent_injector is not None:
		params += list(row_latent_injector.parameters())

	optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

	add_noise = bool(int(config.get("add_noise", 1)))
	noise_level = float(config.get("noise_level", 0.1))

	best_val_loss = float("inf")
	early_stop_counter = 0
	early_stop_epochs = 0
	best_states: dict | None = None

	for epoch in range(int(epochs)):
		model.train()
		if feature_mixer is not None:
			feature_mixer.train()
		if row_latent_injector is not None:
			row_latent_injector.train()

		total_train = 0.0
		total_train_n = 0
		for xb, _ in train_loader:
			xb = xb.to(device)
			xb = _apply_feature_mixing_if_enabled(xb, mixer=feature_mixer)
			subsets = _subset_generator(
				xb,
				spec=spec,
				add_noise=add_noise,
				noise_level=noise_level,
			)

			latent_list = [model.encode(s) for s in subsets]
			latent_tokens = torch.stack(latent_list, dim=1)  # [B,S,D]
			if row_latent_injector is not None:
				latent_tokens = row_latent_injector(latent_tokens)

			recon_list = [model.decode(latent_tokens[:, i, :]) for i in range(latent_tokens.shape[1])]
			recon = torch.stack(recon_list, dim=1)  # [B,S,subset_dim]
			target = torch.stack(subsets, dim=1)
			loss = F.mse_loss(recon, target)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			b = int(xb.shape[0])
			total_train += float(loss.item()) * b
			total_train_n += b

		model.eval()
		if feature_mixer is not None:
			feature_mixer.eval()
		if row_latent_injector is not None:
			row_latent_injector.eval()

		total_val = 0.0
		total_val_n = 0
		with torch.no_grad():
			for xb, _ in val_loader:
				xb = xb.to(device)
				xb = _apply_feature_mixing_if_enabled(xb, mixer=feature_mixer)
				subsets = _subset_generator(
					xb,
					spec=spec,
					add_noise=False,
					noise_level=noise_level,
				)
				latent_list = [model.encode(s) for s in subsets]
				latent_tokens = torch.stack(latent_list, dim=1)
				if row_latent_injector is not None:
					latent_tokens = row_latent_injector(latent_tokens)
				recon_list = [model.decode(latent_tokens[:, i, :]) for i in range(latent_tokens.shape[1])]
				recon = torch.stack(recon_list, dim=1)
				target = torch.stack(subsets, dim=1)
				loss = F.mse_loss(recon, target)
				b = int(xb.shape[0])
				total_val += float(loss.item()) * b
				total_val_n += b

		train_loss = total_train / max(1, total_train_n)
		val_loss = total_val / max(1, total_val_n)
		improved = val_loss < best_val_loss - loss_threshold
		suffix = " (improved)" if improved else ""
		print(
			f"[SUBTAB][AE] Epoch {epoch+1}/{int(epochs)} TrainLoss={train_loss:.6f} "
			f"ValLoss={val_loss:.6f}{suffix}"
		)

		if improved:
			best_val_loss = val_loss
			early_stop_counter = 0
			best_states = {
				"model": model.state_dict(),
				"feature_mixer": feature_mixer.state_dict() if feature_mixer is not None else None,
				"row_latent_injector": row_latent_injector.state_dict() if row_latent_injector is not None else None,
			}
		else:
			early_stop_counter += 1

		if early_stop_counter >= patience:
			early_stop_epochs = epoch + 1
			print(f"[SUBTAB][AE] Early stopping at epoch {epoch+1}")
			break

	if best_states is not None:
		model.load_state_dict(best_states["model"])
		if feature_mixer is not None and best_states.get("feature_mixer") is not None:
			feature_mixer.load_state_dict(best_states["feature_mixer"])
		if row_latent_injector is not None and best_states.get("row_latent_injector") is not None:
			row_latent_injector.load_state_dict(best_states["row_latent_injector"])

	artifacts = {
		"feature_mixer": feature_mixer,
		"row_latent_injector": row_latent_injector,
	}
	return model, artifacts, int(early_stop_epochs)


def _encode_dataset(
	loader: TorchDataLoader,
	*,
	model: SubTabAutoEncoder,
	device: torch.device,
	spec: _SubTabSpec,
	feature_mixer: FeatureTopologyMaskedMixing | None,
	row_latent_injector: "_RowLatentInjector" | None,
) -> tuple[np.ndarray, np.ndarray]:
	model.eval()
	if feature_mixer is not None:
		feature_mixer.eval()
	if row_latent_injector is not None:
		row_latent_injector.eval()

	zs = []
	ys = []
	with torch.no_grad():
		for xb, yb in loader:
			xb = xb.to(device)
			xb = _apply_feature_mixing_if_enabled(xb, mixer=feature_mixer)
			subsets = _subset_generator(xb, spec=spec, add_noise=False, noise_level=0.0)
			latent_tokens = torch.stack([model.encode(s) for s in subsets], dim=1)
			if row_latent_injector is not None:
				latent_tokens = row_latent_injector(latent_tokens)
			pooled = latent_tokens.mean(dim=1)
			z = model.project(pooled)
			zs.append(z.detach().cpu().numpy())
			ys.append(np.asarray(yb))

	z_np = np.concatenate(zs, axis=0) if zs else np.zeros((0, int(spec.latent_dim)), dtype=np.float32)
	y_np = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.float32)
	return z_np, y_np


def _require_sklearn():
	if LogisticRegression is None or LinearRegression is None:
		raise ModuleNotFoundError(
			"scikit-learn is required for SubTab downstream evaluation. Install scikit-learn to use this model."
		) from _SKLEARN_IMPORT_ERROR


def _downstream_eval(z_train, y_train, z_val, y_val, z_test, y_test, task_type: str) -> tuple[float, float]:
	_require_sklearn()
	if accuracy_score is None or mean_squared_error is None or roc_auc_score is None:
		raise ModuleNotFoundError(
			"scikit-learn metrics are required for SubTab downstream evaluation."
		) from _SKLEARN_IMPORT_ERROR

	task_type = str(task_type).lower()
	if task_type == "regression":
		reg = LinearRegression()
		reg.fit(z_train, y_train)
		y_val_pred = reg.predict(z_val)
		y_test_pred = reg.predict(z_test)
		val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
		test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
		return val_rmse, test_rmse

	if task_type == "binclass":
		clf = LogisticRegression(max_iter=1200, C=1.0)
		clf.fit(z_train, y_train)
		proba_val = clf.predict_proba(z_val)[:, 1]
		proba_test = clf.predict_proba(z_test)[:, 1]
		return float(roc_auc_score(y_val, proba_val)), float(roc_auc_score(y_test, proba_test))

	if task_type == "multiclass":
		clf = LogisticRegression(max_iter=2000, multi_class="auto")
		clf.fit(z_train, y_train)
		y_val_pred = clf.predict(z_val)
		y_test_pred = clf.predict(z_test)
		return float(accuracy_score(y_val, y_val_pred)), float(accuracy_score(y_test, y_test_pred))

	raise ValueError(f"Unknown task_type={task_type!r}.")


class _RowLatentInjector(torch.nn.Module):
	"""Row graphification on latent tokens inside the SubTab core loop.

	Operates on minibatches: nodes are samples in the batch.
	"""

	def __init__(self, latent_dim: int, config: dict):
		super().__init__()
		try:
			DGM_d_local = _import_dgm_d()
		except Exception as e:  # pragma: no cover
			raise ModuleNotFoundError(
				"DGM_d (from DGM_pytorch/DGMlib) is required for SubTab row-graphification."
			) from e

		dgm_k = int(config.get("gnn_dgm_k", config.get("dgm_k", 10)))
		dgm_distance = str(config.get("gnn_dgm_distance", config.get("dgm_distance", "euclidean")))
		dgm_gate = str(config.get("gnn_dgm_gate", "none")).lower()
		dgm_gate_sharpness = float(config.get("gnn_dgm_gate_sharpness", 1.0))
		dgm_gate_threshold_init = float(config.get("gnn_dgm_gate_threshold_init", -10.0))
		dgm_eval_prune = bool(config.get("gnn_dgm_eval_prune", False))
		dgm_eval_prune_threshold = float(config.get("gnn_dgm_eval_prune_threshold", 0.5))

		class DGMEmbedWrapper(torch.nn.Module):
			def forward(self, x, A=None):
				return x

		self._dgm_gate = dgm_gate
		self._dgm_gate_sharpness = dgm_gate_sharpness
		self._dgm_eval_prune = dgm_eval_prune
		self._dgm_eval_prune_threshold = dgm_eval_prune_threshold

		self.dgm_gate_threshold_param = None
		if dgm_gate != "none":
			self.dgm_gate_threshold_param = torch.nn.Parameter(
				torch.tensor(dgm_gate_threshold_init, dtype=torch.float32)
			)

		self.dgm_module = DGM_d_local(DGMEmbedWrapper(), k=int(dgm_k), distance=dgm_distance)
		gnn_hidden = int(config.get("gnn_hidden", 64))
		self.gnn = SimpleGCN(int(latent_dim), int(gnn_hidden), int(latent_dim), num_layers=2)
		self.proj = torch.nn.Linear(int(latent_dim), int(latent_dim))
		self.fusion_alpha = torch.nn.Parameter(torch.tensor(-0.847, dtype=torch.float32))

	def forward(self, latent_tokens: torch.Tensor) -> torch.Tensor:
		if latent_tokens.dim() != 3:
			raise ValueError(f"Expected latent_tokens [B,S,D], got {tuple(latent_tokens.shape)}")
		b = int(latent_tokens.shape[0])
		if b <= 1:
			return latent_tokens

		pooled = latent_tokens.mean(dim=1)
		pooled_std = _standardize(pooled, dim=0)
		pooled_batched = pooled_std.unsqueeze(0)
		if hasattr(self.dgm_module, "k"):
			self.dgm_module.k = int(min(int(self.dgm_module.k), max(1, b - 1)))
		x_dgm, edge_index_dgm, logprobs_dgm = self.dgm_module(pooled_batched, A=None)
		x_dgm = x_dgm.squeeze(0)

		edge_index_dgm, edge_weight_dgm = _dgm_gate_and_prune_edges(
			edge_index_dgm,
			logprobs_dgm,
			b,
			gate=self._dgm_gate,
			gate_sharpness=self._dgm_gate_sharpness,
			gate_threshold_param=self.dgm_gate_threshold_param,
			is_training=bool(self.training),
			eval_prune=self._dgm_eval_prune,
			eval_prune_threshold=self._dgm_eval_prune_threshold,
		)

		gcn_out = self.gnn(x_dgm, edge_index_dgm, edge_weight=edge_weight_dgm)
		ctx = self.proj(gcn_out).unsqueeze(1)
		alpha = torch.sigmoid(self.fusion_alpha)
		return latent_tokens + alpha * ctx


def _train_supervised_decoding(
	train_loader: TorchDataLoader,
	val_loader: TorchDataLoader,
	test_loader: TorchDataLoader,
	*,
	config: dict,
	task_type: str,
	device: torch.device,
	spec: _SubTabSpec,
	feature_mixer: FeatureTopologyMaskedMixing | None,
	use_row_decoder: bool,
) -> dict:
	"""Train a supervised decoder/head for `gnn_stage='decoding'`."""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError("pandas is required for SubTab") from _PANDAS_IMPORT_ERROR

	# Model parts
	dropout = float(config.get("subtab_dropout", 0.2))
	use_batch_norm = bool(int(config.get("subtab_batch_norm", 0)))
	hidden_dims = [int(d) for d in list(config.get("subtab_hidden_dims", [128, 128]))]
	encoder = SubTabAutoEncoder(
		int(spec.actual_subset_size),
		latent_dim=int(spec.latent_dim),
		hidden_dims=hidden_dims,
		dropout=dropout,
		use_batch_norm=use_batch_norm,
	).to(device)

	task_type = str(task_type).lower()
	if task_type in {"binclass", "multiclass"}:
		# Infer num classes from train split.
		ys = []
		for _, yb in train_loader:
			ys.append(torch.as_tensor(yb).view(-1))
		y_all = torch.cat(ys, dim=0) if ys else torch.empty(0, dtype=torch.long)
		num_classes = int(torch.unique(y_all.long()).numel()) if y_all.numel() > 0 else 0
		if task_type == "binclass" and num_classes != 2:
			num_classes = 2
		out_dim = int(max(2, num_classes)) if task_type == "binclass" else int(max(1, num_classes))
	else:
		out_dim = 1

	if use_row_decoder:
		# Row-graph decoder: DGM + GCN + pred head.
		try:
			DGM_d_local = _import_dgm_d()
		except Exception as e:  # pragma: no cover
			raise ModuleNotFoundError(
				"DGM_d (from DGM_pytorch/DGMlib) is required for SubTab decoding row-graphification."
			) from e

		dgm_k = int(config.get("gnn_dgm_k", config.get("dgm_k", 10)))
		dgm_distance = str(config.get("gnn_dgm_distance", config.get("dgm_distance", "euclidean")))
		dgm_gate = str(config.get("gnn_dgm_gate", "none")).lower()
		dgm_gate_sharpness = float(config.get("gnn_dgm_gate_sharpness", 1.0))
		dgm_gate_threshold_init = float(config.get("gnn_dgm_gate_threshold_init", -10.0))
		dgm_eval_prune = bool(config.get("gnn_dgm_eval_prune", False))
		dgm_eval_prune_threshold = float(config.get("gnn_dgm_eval_prune_threshold", 0.5))

		class DGMEmbedWrapper(torch.nn.Module):
			def forward(self, x, A=None):
				return x

		dgm_module = DGM_d_local(DGMEmbedWrapper(), k=int(dgm_k), distance=dgm_distance).to(device)
		dgm_gate_threshold_param = None
		if dgm_gate != "none":
			dgm_gate_threshold_param = torch.nn.Parameter(
				torch.tensor(dgm_gate_threshold_init, device=device, dtype=torch.float32)
			)

		gnn_hidden = int(config.get("gnn_hidden", 64))
		gnn = SimpleGCN(int(spec.latent_dim), int(gnn_hidden), int(spec.latent_dim), num_layers=2).to(device)
		pred_head = torch.nn.Linear(int(spec.latent_dim), int(out_dim)).to(device)
	else:
		dgm_module = None
		dgm_gate_threshold_param = None
		gnn = None
		pred_head = torch.nn.Linear(int(spec.latent_dim), int(out_dim)).to(device)

	params = list(encoder.parameters()) + list(pred_head.parameters())
	if feature_mixer is not None:
		params += list(feature_mixer.parameters())
	if use_row_decoder and gnn is not None and dgm_module is not None:
		params += list(gnn.parameters()) + list(dgm_module.parameters())
		if dgm_gate_threshold_param is not None:
			params += [dgm_gate_threshold_param]

	lr = float(config.get("lr", 1e-3))
	weight_decay = float(config.get("weight_decay", 0.0))
	optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
	epochs = int(config.get("epochs", 200))
	patience = int(config.get("patience", 10))
	loss_threshold = float(config.get("loss_threshold", 1e-4))

	best_val_loss = float("inf")
	best_val_metric = float("nan")
	best_test_metric = float("nan")
	early_stop_counter = 0
	early_stop_epochs = 0
	best_states: dict | None = None

	def _forward_logits(xb: torch.Tensor) -> torch.Tensor:
		xb = _apply_feature_mixing_if_enabled(xb, mixer=feature_mixer)
		subsets = _subset_generator(xb, spec=spec, add_noise=False, noise_level=0.0)
		latent_tokens = torch.stack([encoder.encode(s) for s in subsets], dim=1)
		pooled = latent_tokens.mean(dim=1)

		if use_row_decoder:
			bsz = int(pooled.shape[0])
			if bsz <= 1:
				return pred_head(pooled)
			pooled_std = _standardize(pooled, dim=0)
			pooled_batched = pooled_std.unsqueeze(0)
			if hasattr(dgm_module, "k"):
				dgm_module.k = int(min(int(dgm_module.k), max(1, bsz - 1)))
			x_dgm, edge_index_dgm, logprobs_dgm = dgm_module(pooled_batched, A=None)
			x_dgm = x_dgm.squeeze(0)
			edge_index_dgm, edge_weight_dgm = _dgm_gate_and_prune_edges(
				edge_index_dgm,
				logprobs_dgm,
				bsz,
				gate=dgm_gate,
				gate_sharpness=dgm_gate_sharpness,
				gate_threshold_param=dgm_gate_threshold_param,
				is_training=bool(encoder.training),
				eval_prune=dgm_eval_prune,
				eval_prune_threshold=dgm_eval_prune_threshold,
			)
			gcn_out = gnn(x_dgm, edge_index_dgm, edge_weight=edge_weight_dgm)
			return pred_head(gcn_out)

		return pred_head(pooled)

	for epoch in range(epochs):
		encoder.train()
		pred_head.train()
		if feature_mixer is not None:
			feature_mixer.train()
		if use_row_decoder and gnn is not None and dgm_module is not None:
			gnn.train()
			dgm_module.train()

		total_train = 0.0
		total_train_n = 0
		for xb, yb in train_loader:
			xb = xb.to(device)
			yb_t = torch.as_tensor(yb).to(device)
			logits = _forward_logits(xb)
			if task_type in {"binclass", "multiclass"}:
				loss = F.cross_entropy(logits, yb_t.long())
			else:
				loss = F.mse_loss(logits.squeeze(), yb_t.float())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			b = int(xb.shape[0])
			total_train += float(loss.item()) * b
			total_train_n += b

		encoder.eval(); pred_head.eval()
		if feature_mixer is not None:
			feature_mixer.eval()
		if use_row_decoder and gnn is not None and dgm_module is not None:
			gnn.eval(); dgm_module.eval()

		val_loss, val_metric = _eval_supervised(val_loader, _forward_logits, task_type=task_type, device=device)
		test_loss, test_metric = _eval_supervised(test_loader, _forward_logits, task_type=task_type, device=device)

		improved = val_loss < best_val_loss - loss_threshold
		suffix = " (improved)" if improved else ""
		print(
			f"[SUBTAB][DECODING] Epoch {epoch+1}/{epochs} "
			f"TrainLoss={total_train/max(1,total_train_n):.6f} "
			f"ValLoss={val_loss:.6f} ValMetric={val_metric:.6f}{suffix}"
		)

		if improved:
			best_val_loss = float(val_loss)
			best_val_metric = float(val_metric)
			best_test_metric = float(test_metric)
			early_stop_counter = 0
			best_states = {
				"encoder": encoder.state_dict(),
				"pred_head": pred_head.state_dict(),
				"feature_mixer": feature_mixer.state_dict() if feature_mixer is not None else None,
				"gnn": gnn.state_dict() if gnn is not None else None,
				"dgm": dgm_module.state_dict() if dgm_module is not None else None,
			}
			if dgm_gate_threshold_param is not None:
				best_states["dgm_thr"] = dgm_gate_threshold_param.detach().clone()
		else:
			early_stop_counter += 1

		if early_stop_counter >= patience:
			early_stop_epochs = epoch + 1
			print(f"[SUBTAB][DECODING] Early stopping at epoch {epoch+1}")
			break

	if best_states is not None:
		encoder.load_state_dict(best_states["encoder"])
		pred_head.load_state_dict(best_states["pred_head"])
		if feature_mixer is not None and best_states.get("feature_mixer") is not None:
			feature_mixer.load_state_dict(best_states["feature_mixer"])
		if use_row_decoder and gnn is not None and best_states.get("gnn") is not None:
			gnn.load_state_dict(best_states["gnn"])
		if use_row_decoder and dgm_module is not None and best_states.get("dgm") is not None:
			dgm_module.load_state_dict(best_states["dgm"])
		if dgm_gate_threshold_param is not None and best_states.get("dgm_thr") is not None:
			with torch.no_grad():
				dgm_gate_threshold_param.copy_(best_states["dgm_thr"])

	return {
		"train_losses": [],
		"val_losses": [],
		"best_val_metric": float(best_val_metric),
		"best_test_metric": float(best_test_metric),
		"early_stop_epochs": int(early_stop_epochs),
		"gnn_early_stop_epochs": 0,
	}


def _eval_supervised(
	loader: TorchDataLoader,
	forward_logits,
	*,
	task_type: str,
	device: torch.device,
) -> tuple[float, float]:
	task_type = str(task_type).lower()
	y_true = []
	y_pred = []
	losses = []
	n_total = 0
	with torch.no_grad():
		for xb, yb in loader:
			xb = xb.to(device)
			yb_t = torch.as_tensor(yb).to(device)
			logits = forward_logits(xb)
			b = int(xb.shape[0])
			if task_type in {"binclass", "multiclass"}:
				loss = F.cross_entropy(logits, yb_t.long())
				probs = torch.softmax(logits, dim=-1)
				y_pred.append(probs.detach().cpu().numpy())
				y_true.append(yb_t.long().detach().cpu().numpy())
			else:
				loss = F.mse_loss(logits.squeeze(), yb_t.float())
				y_pred.append(logits.squeeze().detach().cpu().numpy())
				y_true.append(yb_t.float().detach().cpu().numpy())
			losses.append(float(loss.item()) * b)
			n_total += b

	avg_loss = float(sum(losses) / max(1, n_total))
	y_true_np = np.concatenate(y_true, axis=0) if y_true else np.array([])
	y_pred_np = np.concatenate(y_pred, axis=0) if y_pred else np.array([])
	metric = _compute_metric_from_predictions(task_type, y_true_np, y_pred_np)
	return avg_loss, float(metric)


def _compute_metric_from_predictions(task_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
	_require_sklearn()
	if y_true.size == 0:
		return float("nan")
	task_type = str(task_type).lower()
	if task_type == "regression":
		return float(np.sqrt(mean_squared_error(y_true, y_pred)))
	if task_type == "binclass":
		# y_pred: [N,2] probs
		proba = y_pred[:, 1] if y_pred.ndim == 2 and y_pred.shape[1] >= 2 else y_pred
		return float(roc_auc_score(y_true, proba))
	if task_type == "multiclass":
		pred_cls = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else y_pred
		return float(accuracy_score(y_true, pred_cls))
	raise ValueError(f"Unknown task_type={task_type!r}.")


def subtab_core_fn_row_gnn(material_outputs: dict, config: dict, task_type: str, *, gnn_stage: str):
	"""SubTab core with row graphification semantics."""

	stage = str(gnn_stage or "none").lower()
	if stage == "materialization":
		stage = "materialize"
	device = material_outputs.get("device") or resolve_device(config)

	train_loader = material_outputs["train_loader"]
	val_loader = material_outputs["val_loader"]
	test_loader = material_outputs["test_loader"]

	# Infer input dim.
	xb0, _ = next(iter(train_loader))
	input_dim = int(torch.as_tensor(xb0).shape[1])
	spec = _make_subtab_spec(input_dim, config)

	if stage == "decoding":
		return _train_supervised_decoding(
			train_loader,
			val_loader,
			test_loader,
			config=config,
			task_type=task_type,
			device=device,
			spec=spec,
			feature_mixer=None,
			use_row_decoder=True,
		)

	row_latent_injector = None
	if stage in {"encoding", "columnwise"}:
		row_latent_injector = _RowLatentInjector(int(spec.latent_dim), config).to(device)

	epochs = int(config.get("epochs", 200))
	model, artifacts, early_stop_epochs = _train_autoencoder(
		train_loader,
		val_loader,
		config=config,
		device=device,
		spec=spec,
		feature_mixer=None,
		row_latent_injector=row_latent_injector,
		epochs=epochs,
	)

	z_train, y_train = _encode_dataset(
		train_loader,
		model=model,
		device=device,
		spec=spec,
		feature_mixer=None,
		row_latent_injector=artifacts["row_latent_injector"],
	)
	z_val, y_val = _encode_dataset(
		val_loader,
		model=model,
		device=device,
		spec=spec,
		feature_mixer=None,
		row_latent_injector=artifacts["row_latent_injector"],
	)
	z_test, y_test = _encode_dataset(
		test_loader,
		model=model,
		device=device,
		spec=spec,
		feature_mixer=None,
		row_latent_injector=artifacts["row_latent_injector"],
	)

	best_val_metric, best_test_metric = _downstream_eval(
		z_train, y_train, z_val, y_val, z_test, y_test, task_type
	)
	return {
		"train_losses": [],
		"val_losses": [],
		"best_val_metric": float(best_val_metric),
		"best_test_metric": float(best_test_metric),
		"early_stop_epochs": int(early_stop_epochs),
		"gnn_early_stop_epochs": int(material_outputs.get("gnn_early_stop_epochs", 0)),
		"gnn_stage": stage,
	}


def subtab_core_fn_feature_gnn(material_outputs: dict, config: dict, task_type: str, *, gnn_stage: str):
	"""SubTab core with feature topology masked mixing."""

	stage = str(gnn_stage or "none").lower()
	if stage == "materialization":
		stage = "materialize"
	device = material_outputs.get("device") or resolve_device(config)

	train_loader = material_outputs["train_loader"]
	val_loader = material_outputs["val_loader"]
	test_loader = material_outputs["test_loader"]

	xb0, _ = next(iter(train_loader))
	input_dim = int(torch.as_tensor(xb0).shape[1])
	spec = _make_subtab_spec(input_dim, config)

	# Build mixer from topology prior.
	feature_names = material_outputs.get("feature_names") or [f"N_feature_{i + 1}" for i in range(input_dim)]
	init_mode = str(config.get("feature_topology_init", "chain")).lower()
	topology = _init_feature_topology_prior(feature_names, init_mode=init_mode)
	allowed = topology["adjacency_mask"].to(device)
	mixer = FeatureTopologyMaskedMixing(allowed).to(device)
	material_outputs["feature_topology"] = topology
	config["feature_topology_meta"] = _feature_topology_to_meta(topology)

	if stage == "decoding":
		return _train_supervised_decoding(
			train_loader,
			val_loader,
			test_loader,
			config=config,
			task_type=task_type,
			device=device,
			spec=spec,
			feature_mixer=mixer,
			use_row_decoder=False,
		)

	if stage in {"none", "start", "materialize"}:
		# Feature graphification is inactive at start/materialize. Keep behavior stable.
		mixer = None

	epochs = int(config.get("epochs", 200))
	model, artifacts, early_stop_epochs = _train_autoencoder(
		train_loader,
		val_loader,
		config=config,
		device=device,
		spec=spec,
		feature_mixer=mixer,
		row_latent_injector=None,
		epochs=epochs,
	)

	z_train, y_train = _encode_dataset(
		train_loader,
		model=model,
		device=device,
		spec=spec,
		feature_mixer=artifacts["feature_mixer"],
		row_latent_injector=None,
	)
	z_val, y_val = _encode_dataset(
		val_loader,
		model=model,
		device=device,
		spec=spec,
		feature_mixer=artifacts["feature_mixer"],
		row_latent_injector=None,
	)
	z_test, y_test = _encode_dataset(
		test_loader,
		model=model,
		device=device,
		spec=spec,
		feature_mixer=artifacts["feature_mixer"],
		row_latent_injector=None,
	)

	best_val_metric, best_test_metric = _downstream_eval(
		z_train, y_train, z_val, y_val, z_test, y_test, task_type
	)
	return {
		"train_losses": [],
		"val_losses": [],
		"best_val_metric": float(best_val_metric),
		"best_test_metric": float(best_test_metric),
		"early_stop_epochs": int(early_stop_epochs),
		"gnn_early_stop_epochs": int(material_outputs.get("gnn_early_stop_epochs", 0)),
		"gnn_stage": stage,
	}


def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
	"""SubTab entrypoint matching the backbone contract."""

	if pd is None:  # pragma: no cover
		raise ModuleNotFoundError("pandas is required for SubTab") from _PANDAS_IMPORT_ERROR

	stage = "none" if gnn_stage is None else str(gnn_stage).lower()
	if stage == "materialization":
		stage = "materialize"
	graphify = str(config.get("graphify", "row")).lower()
	task_type = str(dataset_results.get("info", {}).get("task_type", "binclass")).lower()

	train_df, val_df, test_df = start_fn(train_df, val_df, test_df)
	gnn_early_stop_epochs = 0

	try:
		if graphify == "row":
			if stage == "start":
				train_df, val_df, test_df, gnn_early_stop_epochs = row_gnn_after_start_fn(
					train_df, val_df, test_df, config, task_type
				)

			material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
			material_outputs["gnn_early_stop_epochs"] = int(gnn_early_stop_epochs)

			if stage == "materialize":
				tl, vl, tel, gnn_early_stop_epochs = row_gnn_after_materialize_fn(
					material_outputs["train_loader"],
					material_outputs["val_loader"],
					material_outputs["test_loader"],
					config,
					task_type,
				)
				material_outputs["train_loader"] = tl
				material_outputs["val_loader"] = vl
				material_outputs["test_loader"] = tel
				material_outputs["gnn_early_stop_epochs"] = int(gnn_early_stop_epochs)

			results = subtab_core_fn_row_gnn(material_outputs, config, task_type, gnn_stage=stage)
			return results

		if graphify == "feature":
			if stage == "start":
				train_df, val_df, test_df, _ = feature_gnn_after_start_fn(
					train_df, val_df, test_df, config, task_type
				)

			material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
			material_outputs["gnn_early_stop_epochs"] = 0
			material_outputs = feature_gnn_after_materialize_fn(material_outputs, config, task_type)

			results = subtab_core_fn_feature_gnn(material_outputs, config, task_type, gnn_stage=stage)
			return results

		raise ValueError(
			f"Unknown graphify mode: {graphify!r}. Expected 'row' or 'feature'."
		)
	except Exception as e:
		return {
			"train_losses": [],
			"val_losses": [],
			"best_val_metric": float("nan"),
			"best_test_metric": float("nan"),
			"early_stop_epochs": 0,
			"gnn_early_stop_epochs": int(gnn_early_stop_epochs),
			"error": str(e),
		}


def _df_splits_to_tensors(train_df, val_df, test_df, *, device: torch.device):
	all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
	feature_cols = [c for c in all_df.columns if c != "target"]
	x_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32, device=device)
	x_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32, device=device)
	x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32, device=device)
	y_train_np = train_df["target"].values
	y_val_np = val_df["target"].values
	y_test_np = test_df["target"].values
	return x_train, x_val, x_test, y_train_np, y_val_np, y_test_np, feature_cols


def _collect_loader_xy(loader: TorchDataLoader, *, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
	xs = []
	ys = []
	for xb, yb in loader:
		xs.append(torch.as_tensor(xb, dtype=torch.float32, device=device))
		ys.append(np.asarray(yb))
	x = torch.cat(xs, dim=0) if xs else torch.zeros((0, 0), dtype=torch.float32, device=device)
	y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.float32)
	return x, y


def _offline_row_embedding_extraction(
	*,
	x_train: torch.Tensor,
	x_val: torch.Tensor,
	x_test: torch.Tensor,
	y_train_np: np.ndarray,
	y_val_np: np.ndarray,
	y_test_np: np.ndarray,
	task_type: str,
	config: dict,
	num_cols: int,
	stage_tag: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
	"""Train DGM_d + GCN on train split, extract embeddings for all splits."""

	try:
		DGM_d_local = _import_dgm_d()
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"DGM_d (from DGM_pytorch/DGMlib) is required for SubTab row-graphification."
		) from e

	device = x_train.device
	task_type = str(task_type).lower()

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

	if task_type in {"binclass", "multiclass"}:
		y_all_np = np.concatenate([y_train_np, y_val_np, y_test_np])
		num_classes = int(len(np.unique(y_all_np)))
		if task_type == "binclass" and num_classes != 2:
			num_classes = 2
		out_dim = int(num_classes)
	else:
		out_dim = 1

	n_train = int(x_train.shape[0])

	class DGMEmbedWrapper(torch.nn.Module):
		def forward(self, x, A=None):
			return x

	dgm_k_train = int(min(dgm_k, max(1, n_train - 1)))
	dgm_module = DGM_d_local(DGMEmbedWrapper(), k=dgm_k_train, distance=dgm_distance).to(device)

	dgm_gate_threshold_param = None
	if dgm_gate != "none":
		dgm_gate_threshold_param = torch.nn.Parameter(
			torch.tensor(dgm_gate_threshold_init, device=device, dtype=torch.float32)
		)

	gnn = SimpleGCN(int(num_cols), int(gnn_hidden), int(gnn_out_dim), num_layers=2).to(device)
	pred_head = torch.nn.Linear(int(gnn_out_dim), int(out_dim)).to(device)

	params = list(gnn.parameters()) + list(pred_head.parameters()) + list(dgm_module.parameters())
	if dgm_gate_threshold_param is not None:
		params += [dgm_gate_threshold_param]
	optimizer = torch.optim.Adam(params, lr=lr)

	best_val_loss = float("inf")
	early_stop_counter = 0
	gnn_early_stop_epochs = 0
	best_states = None

	def forward_pass(x_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
		ns = int(x_tensor.shape[0])
		row_emb_std = _standardize(x_tensor, dim=0)
		row_emb_batched = row_emb_std.contiguous().unsqueeze(0)
		if hasattr(dgm_module, "k"):
			dgm_module.k = int(min(int(dgm_module.k), max(1, ns - 1)))
		row_emb_dgm, edge_index_dgm, logprobs_dgm = dgm_module(row_emb_batched, A=None)
		row_emb_dgm = row_emb_dgm.squeeze(0)

		edge_index_dgm, edge_weight_dgm = _dgm_gate_and_prune_edges(
			edge_index_dgm,
			logprobs_dgm,
			ns,
			gate=dgm_gate,
			gate_sharpness=dgm_gate_sharpness,
			gate_threshold_param=dgm_gate_threshold_param,
			is_training=bool(dgm_module.training),
			eval_prune=dgm_eval_prune,
			eval_prune_threshold=dgm_eval_prune_threshold,
		)

		gcn_out = gnn(row_emb_dgm, edge_index_dgm, edge_weight=edge_weight_dgm)
		logits = pred_head(gcn_out)
		return logits, gcn_out, logprobs_dgm

	for epoch in range(gnn_epochs):
		dgm_module.train(); gnn.train(); pred_head.train()
		optimizer.zero_grad()
		logits, _, logprobs = forward_pass(x_train)

		if task_type in {"binclass", "multiclass"}:
			y_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
			train_loss = F.cross_entropy(logits, y_train)
		else:
			y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)
			train_loss = F.mse_loss(logits.squeeze(), y_train)

		if logprobs is not None:
			train_loss = train_loss + (-logprobs.mean() * 0.01)

		train_loss.backward()
		optimizer.step()

		dgm_module.eval(); gnn.eval(); pred_head.eval()
		with torch.no_grad():
			logits_val, _, _ = forward_pass(x_val)
			if task_type in {"binclass", "multiclass"}:
				y_val = torch.tensor(y_val_np, dtype=torch.long, device=device)
				val_loss = F.cross_entropy(logits_val, y_val)
			else:
				y_val = torch.tensor(y_val_np, dtype=torch.float32, device=device)
				val_loss = F.mse_loss(logits_val.squeeze(), y_val)

		train_loss_val = float(train_loss.item())
		val_loss_val = float(val_loss.item())
		improved = val_loss_val < best_val_loss - loss_threshold
		suffix = " (improved)" if improved else ""
		print(
			f"[{stage_tag}-GNN-DGM] Epoch {epoch+1}/{gnn_epochs} "
			f"TrainLoss={train_loss_val:.4f} ValLoss={val_loss_val:.4f}{suffix}"
		)

		if improved:
			best_val_loss = val_loss_val
			early_stop_counter = 0
			best_states = {
				"gnn": gnn.state_dict(),
				"pred_head": pred_head.state_dict(),
				"dgm": dgm_module.state_dict(),
			}
			if dgm_gate_threshold_param is not None:
				best_states["thr"] = dgm_gate_threshold_param.detach().clone()
		else:
			early_stop_counter += 1

		if early_stop_counter >= patience:
			gnn_early_stop_epochs = epoch + 1
			print(f"[{stage_tag}-GNN-DGM] Early stopping at epoch {epoch+1}")
			break

	if best_states is not None:
		gnn.load_state_dict(best_states["gnn"])
		pred_head.load_state_dict(best_states["pred_head"])
		dgm_module.load_state_dict(best_states["dgm"])
		if dgm_gate_threshold_param is not None and best_states.get("thr") is not None:
			with torch.no_grad():
				dgm_gate_threshold_param.copy_(best_states["thr"])

	dgm_module.eval(); gnn.eval(); pred_head.eval()
	with torch.no_grad():
		_, emb_train, _ = forward_pass(x_train)
		_, emb_val, _ = forward_pass(x_val)
		_, emb_test, _ = forward_pass(x_test)

	return (
		emb_train.detach().cpu().numpy(),
		emb_val.detach().cpu().numpy(),
		emb_test.detach().cpu().numpy(),
		int(gnn_early_stop_epochs),
	)

#  small+binclass
#  python main.py --models subtab --dataset kaggle_Audit_Data --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  small+regression
#  python main.py --models subtab --dataset openml_The_Office_Dataset --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+binclass
#  python main.py --models subtab --dataset credit --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+multiclass
#  python main.py --models subtab --dataset eye --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  python main.py --models subtab --dataset helena --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
#  large+regression
#  python main.py --models subtab --dataset house --gnn_stages all --graphify all --gpu 1 --epochs 300 --restore_best 0
