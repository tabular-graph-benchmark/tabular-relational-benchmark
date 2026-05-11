from __future__ import annotations

if True:

	import math
	import sys
	
	from pathlib import Path
	from typing import Tuple

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
		from sklearn.preprocessing import StandardScaler
	except ModuleNotFoundError as e:  # pragma: no cover
		StandardScaler = None
		_SKLEARN_IMPORT_ERROR = e

	try:
		from sklearn.neighbors import NearestNeighbors
	except ModuleNotFoundError as e:  # pragma: no cover
		NearestNeighbors = None
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

		# Respect the CLI convention: gpu=-1 forces CPU.
		if gpu_id is not None:
			try:
				if int(gpu_id) < 0:
					print(f"[DEVICE] Using cpu (forced by config['gpu']={gpu_id})")
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
					pass
			print("[DEVICE] Using cuda")
			return torch.device("cuda")

		print("[DEVICE] Using cpu")
		return torch.device("cpu")


	def _import_dgm_d():
		"""Import DGM_d with a robust path fallback for this workspace."""

		try:
			from DGMlib.layers import DGM_d as _DGM_d

			return _DGM_d
		except Exception as first_exc:
			repo_root = Path(__file__).resolve().parents[3]
			dgm_path = repo_root / "DGM_pytorch"
			if dgm_path.exists() and str(dgm_path) not in sys.path:
				sys.path.insert(0, str(dgm_path))
			try:
				from DGMlib.layers import DGM_d as _DGM_d

				return _DGM_d
			except Exception as second_exc:
				raise second_exc from first_exc


	def _import_scarf_lib():
		"""Import scarf_lib from models/backbone/scarf_lib.

		The benchmark runner loads backbone files via importlib by path, so we do not
		rely on package imports through `models.*`.
		"""

		backbone_dir = Path(__file__).resolve().parent
		if str(backbone_dir) not in sys.path:
			sys.path.insert(0, str(backbone_dir))
		try:
			from scarf_lib.dataset import SCARFDataset
			from scarf_lib.loss import NTXent
			from scarf_lib.model import ScarfColumnwiseLayer, ScarfDecoder, ScarfEncoder

			return SCARFDataset, NTXent, ScarfEncoder, ScarfColumnwiseLayer, ScarfDecoder
		except Exception as e:
			raise ModuleNotFoundError(
				"Failed to import local scarf_lib. Expected it under models/backbone/scarf_lib."
			) from e


	def _standardize(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
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
		"""Apply gating and pruning to DGM_d edges using logprobs (edge probabilities).

		If gate == 'none', returns unweighted graph.
		If gate == 'sigmoid', returns weighted graph with edge weights in [0,1].
		Optionally prunes low-weight edges at eval time.
	"""
		if edge_index_dgm.numel() == 0:
			return _symmetrize_and_self_loop(edge_index_dgm, num_nodes), None

		gate = str(gate or "none").lower()
		if gate not in {"none", "sigmoid"}:
			raise ValueError(f"Unknown gate={gate!r}. Expected 'none' or 'sigmoid'.")

		edge_weight = None
		if gate != "none":
			if logprobs_dgm is None:
				raise ValueError("logprobs_dgm is required when gate is enabled.")

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
					f"logprobs_dgm numel ({lp.numel()}) does not match edges ({edge_index_dgm.shape[1]}). "
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
				else:
					for idx in range(inv.numel()):
						j = int(inv[idx].item())
						out[j] = torch.maximum(out[j], ew[idx])
						ew = out

		return ei, ew


	def _symmetrize_self_loop_and_coalesce(
		edge_index: torch.Tensor,
		num_nodes: int,
		edge_weight: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor | None]:
		if edge_index.numel() == 0:
			loops = torch.arange(num_nodes, device=edge_index.device)
			ei = torch.stack([loops, loops], dim=0)
			if edge_weight is None:
				return ei, None
			return ei, torch.ones(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype)

		rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
		loops = torch.arange(num_nodes, device=edge_index.device)
		self_edges = torch.stack([loops, loops], dim=0)
		ei = torch.cat([edge_index, rev, self_edges], dim=1)

		if edge_weight is not None:
			ew = torch.cat(
				[
					edge_weight,
					edge_weight,
					torch.ones(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype),
				],
				dim=0,
			)
		else:
			ew = None

		if coalesce is not None:
			try:
				# Newer PyG: coalesce(edge_index, edge_attr, m, n, reduce=...)
				ei, ew = coalesce(ei, ew, num_nodes, num_nodes, reduce="max")
			except TypeError:
				# Older PyG: coalesce(edge_index, edge_attr=None, num_nodes=None, reduce=...)
				ei, ew = coalesce(ei, ew, num_nodes=num_nodes, reduce="max")
			return ei, ew

		# Fallback: unique edges (no stable weights aggregation beyond max).
		edge_ids = ei[0] * num_nodes + ei[1]
		unique_ids, inv = torch.unique(edge_ids, sorted=False, return_inverse=True)
		ei0 = unique_ids // num_nodes
		ei1 = unique_ids % num_nodes
		ei = torch.stack([ei0, ei1], dim=0)
		if ew is None:
			return ei, None
		out = torch.zeros(unique_ids.shape[0], device=ew.device, dtype=ew.dtype)
		if hasattr(out, "scatter_reduce_"):
			out.scatter_reduce_(0, inv, ew, reduce="amax", include_self=False)
			return ei, out
		for idx in range(inv.numel()):  # pragma: no cover
			j = int(inv[idx].item())
			out[j] = torch.maximum(out[j], ew[idx])
		return ei, out


	class SimpleGCN(nn.Module):
		def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
			super().__init__()
			if GCNConv is None:  # pragma: no cover
				raise ModuleNotFoundError(
					"torch_geometric is required for SCARF GNN stages (GCNConv)."
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


	def _knn_graph(x: torch.Tensor, k: int, *, directed: bool = False) -> torch.Tensor:
		if NearestNeighbors is None:  # pragma: no cover
			raise ModuleNotFoundError(
				"scikit-learn is required for kNN graph fallback (NearestNeighbors)."
			) from _SKLEARN_IMPORT_ERROR
		x_np = x.detach().cpu().numpy()
		n = int(x_np.shape[0])
		if n <= 1:
			return torch.empty((2, 0), dtype=torch.long, device=x.device)
		k = int(min(max(1, k), n - 1))
		nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(x_np)
		_, idx = nbrs.kneighbors(x_np)
		edges = []
		for i in range(n):
			for j in idx[i][1:]:
				edges.append([i, int(j)])
				if not directed:
					edges.append([int(j), i])
		if not edges:
			return torch.empty((2, 0), dtype=torch.long, device=x.device)
		ei = torch.tensor(edges, dtype=torch.long, device=x.device).t().contiguous()
		return ei


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


	class _RowGnnCfg:
		def __init__(
			self,
			*,
			dgm_k: int,
			dgm_distance: str,
			dgm_reg: float,
			gnn_hidden: int,
			gnn_layers: int,
			gnn_out_dim: int,
			gnn_epochs: int,
			gnn_patience: int,
			gnn_loss_threshold: float,
			gnn_lr: float,
			dgm_gate: str = "sigmoid",
			dgm_gate_sharpness: float = 2.0,
			dgm_eval_prune: bool = False,
			dgm_eval_prune_threshold: float = 0.0,
		):
			self.dgm_k = int(dgm_k)
			self.dgm_distance = str(dgm_distance)
			self.dgm_reg = float(dgm_reg)
			self.gnn_hidden = int(gnn_hidden)
			self.gnn_layers = int(gnn_layers)
			self.gnn_out_dim = int(gnn_out_dim)
			self.gnn_epochs = int(gnn_epochs)
			self.gnn_patience = int(gnn_patience)
			self.gnn_loss_threshold = float(gnn_loss_threshold)
			self.gnn_lr = float(gnn_lr)
			self.dgm_gate = str(dgm_gate)
			self.dgm_gate_sharpness = float(dgm_gate_sharpness)
			self.dgm_eval_prune = bool(dgm_eval_prune)
			self.dgm_eval_prune_threshold = float(dgm_eval_prune_threshold)


	def _read_row_gnn_config(config: dict) -> _RowGnnCfg:
		return _RowGnnCfg(
			dgm_k=int(config.get("gnn_dgm_k", config.get("dgm_k", 10))),
			dgm_distance=str(config.get("gnn_dgm_distance", config.get("dgm_distance", "euclidean"))),
			dgm_reg=float(config.get("dgm_reg_weight", config.get("gnn_dgm_reg_weight", 0.01))),
			gnn_hidden=int(config.get("gnn_hidden", config.get("gnn_hidden_dim", 64))),
			gnn_layers=int(config.get("gnn_layers", 2)),
			gnn_out_dim=int(config.get("gnn_out_dim", config.get("gnn_hidden", 64))),
			gnn_epochs=int(config.get("gnn_epochs", config.get("epochs", 200))),
			gnn_patience=int(config.get("gnn_patience", config.get("patience", 10))),
			gnn_loss_threshold=float(config.get("gnn_loss_threshold", 1e-4)),
			gnn_lr=float(config.get("gnn_lr", config.get("lr", 1e-3))),
			dgm_gate=str(config.get("gnn_dgm_gate", config.get("dgm_gate", "sigmoid"))),
			dgm_gate_sharpness=float(config.get("gnn_dgm_gate_sharpness", config.get("dgm_gate_sharpness", 2.0))),
			dgm_eval_prune=bool(config.get("gnn_dgm_eval_prune", config.get("dgm_eval_prune", False))),
			dgm_eval_prune_threshold=float(config.get("gnn_dgm_eval_prune_threshold", config.get("dgm_eval_prune_threshold", 0.0))),
		)


	def row_gnn_after_start_fn(
		train_df: pd.DataFrame,
		val_df: pd.DataFrame,
		test_df: pd.DataFrame,
		config: dict,
		task_type: str,
	) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
		"""Offline row-embedding extraction between start and materialize.

		Paper-aligned behavior: use raw row feature vectors as node features.
		"""

		if pd is None:  # pragma: no cover
			raise ModuleNotFoundError("pandas is required for SCARF row graphification") from _PANDAS_IMPORT_ERROR

		device = resolve_device(config)
		use_dgm = bool(config.get("use_dgm", True))
		knn_k = int(config.get("gnn_knn", 5))
		DGM_d_local = None
		if use_dgm and device.type == "cuda":
			try:
				DGM_d_local = _import_dgm_d()
			except Exception:
				DGM_d_local = None
				print("[START-GNN] DGM_d unavailable; falling back to kNN graph.")

		cfg = _read_row_gnn_config(config)
		feature_cols = [c for c in train_df.columns if c != "target"]
		num_cols = int(len(feature_cols))
		if num_cols <= 0:
			raise ValueError("No feature columns found (expected columns other than 'target').")

		x_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32, device=device)
		x_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32, device=device)
		x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32, device=device)

		if task_type in {"binclass", "multiclass"}:
			y_all = np.concatenate([train_df["target"].values, val_df["target"].values, test_df["target"].values])
			out_dim = int(len(pd.unique(y_all)))
			if task_type == "binclass" and out_dim != 2:
				out_dim = 2
			y_train = torch.tensor(train_df["target"].values, dtype=torch.long, device=device)
			y_val = torch.tensor(val_df["target"].values, dtype=torch.long, device=device)
		else:
			out_dim = 1
			y_train = torch.tensor(train_df["target"].values, dtype=torch.float32, device=device)
			y_val = torch.tensor(val_df["target"].values, dtype=torch.float32, device=device)

		gnn = SimpleGCN(
			in_dim=num_cols,
			hidden_dim=int(cfg.gnn_hidden),
			out_dim=int(cfg.gnn_out_dim),
			num_layers=int(cfg.gnn_layers),
		).to(device)
		head = nn.Linear(int(cfg.gnn_out_dim), int(out_dim)).to(device)

		dgm = None
		params = list(gnn.parameters()) + list(head.parameters())
		if DGM_d_local is not None:
			class _Id(nn.Module):
				def forward(self, x, A=None):
					return x

			n_train = int(x_train.size(0))
			dgm = DGM_d_local(
				_Id(),
				k=int(min(cfg.dgm_k, max(1, n_train - 1))),
				distance=str(cfg.dgm_distance),
			).to(device)
			params = list(dgm.parameters()) + params

		opt = torch.optim.Adam(params, lr=float(cfg.gnn_lr))

		def _forward(x: torch.Tensor):
			b = int(x.size(0))
			x_std = _standardize(x, dim=0)
			if dgm is not None:
				x_dgm, edge_index_dgm, logprobs = dgm(x_std.unsqueeze(0), A=None)
				x_dgm = x_dgm.squeeze(0)
				edge_index, edge_weight = _dgm_gate_and_prune_edges(
					edge_index_dgm,
					logprobs,
					b,
					gate=str(cfg.dgm_gate),
					gate_sharpness=float(cfg.dgm_gate_sharpness),
					gate_threshold_param=None,
					is_training=dgm.training,
					eval_prune=bool(cfg.dgm_eval_prune),
					eval_prune_threshold=float(cfg.dgm_eval_prune_threshold),
				)
				emb = gnn(x_dgm, edge_index, edge_weight=edge_weight)
				logits = head(emb)
				return logits, emb, logprobs
			edge_index = _knn_graph(x_std, k=min(knn_k, max(1, b - 1))).to(device)
			edge_index, _ = _symmetrize_self_loop_and_coalesce(edge_index, b, edge_weight=None)
			emb = gnn(x_std, edge_index)
			logits = head(emb)
			return logits, emb, None

		epochs = int(cfg.gnn_epochs)
		patience = int(cfg.gnn_patience)
		loss_threshold = float(cfg.gnn_loss_threshold)
		best_val = float("inf")
		best_state = None
		pat = 0
		stop_epoch = 0

		for epoch in range(1, epochs + 1):
			if dgm is not None:
				dgm.train()
			gnn.train(); head.train()
			opt.zero_grad()
			logits, _, logprobs = _forward(x_train)
			if task_type in {"binclass", "multiclass"}:
				loss = F.cross_entropy(logits, y_train)
			else:
				loss = F.mse_loss(logits.squeeze(-1), y_train)
			if logprobs is not None:
				loss = loss + (-logprobs.mean()) * float(cfg.dgm_reg)
			loss.backward(); opt.step()

			if dgm is not None:
				dgm.eval()
			gnn.eval(); head.eval()
			with torch.no_grad():
				logits_v, _, _ = _forward(x_val)
				if task_type in {"binclass", "multiclass"}:
					val_loss = F.cross_entropy(logits_v, y_val)
				else:
					val_loss = F.mse_loss(logits_v.squeeze(-1), y_val)
			v = float(val_loss.item())
			if v < best_val - loss_threshold:
				best_val = v
				pat = 0
				best_state = {
					"gnn": gnn.state_dict(),
					"head": head.state_dict(),
				}
				if dgm is not None:
					best_state["dgm"] = dgm.state_dict()
			else:
				pat += 1
				if pat >= patience:
					stop_epoch = epoch
					break

		if stop_epoch == 0:
			stop_epoch = epochs
		if best_state is not None:
			if dgm is not None and best_state.get("dgm") is not None:
				dgm.load_state_dict(best_state["dgm"])
			gnn.load_state_dict(best_state["gnn"])
			head.load_state_dict(best_state["head"])

		if dgm is not None:
			dgm.eval()
		gnn.eval(); head.eval()
		with torch.no_grad():
			_, e_tr, _ = _forward(x_train)
			_, e_va, _ = _forward(x_val)
			_, e_te, _ = _forward(x_test)

		cols = [f"N_feature_emb_{i + 1}" for i in range(int(e_tr.size(1)))]
		train_out = pd.DataFrame(e_tr.detach().cpu().numpy(), columns=cols, index=train_df.index)
		val_out = pd.DataFrame(e_va.detach().cpu().numpy(), columns=cols, index=val_df.index)
		test_out = pd.DataFrame(e_te.detach().cpu().numpy(), columns=cols, index=test_df.index)
		train_out["target"] = train_df["target"].values
		val_out["target"] = val_df["target"].values
		test_out["target"] = test_df["target"].values
		return train_out, val_out, test_out, int(stop_epoch)


	def row_gnn_after_materialize_fn(material_outputs: dict, config: dict, dataset_name: str, task_type: str) -> dict:
		"""Row-level embedding extraction at materialize stage using DGM_d + GNN with edge gating.
		
		This uses the same edge gating mechanism as row_gnn_after_start_fn to ensure stable training.
		"""
		print("[MATERIALIZE-GNN] SCARF: row graphification with edge gating (DGM_d + GNN).")
		return material_outputs


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
			raise ModuleNotFoundError("pandas is required for SCARF feature topology init") from _PANDAS_IMPORT_ERROR

		feature_names = [c for c in train_df.columns if c != "target"]
		init_mode = str(config.get("feature_topology_init", "chain")).lower()
		topology = _init_feature_topology_prior(feature_names, init_mode=init_mode)
		config["feature_topology_meta"] = _feature_topology_to_meta(topology)
		print(
			"[START-GNN] Feature-level topology initialized (inactive). "
			f"init_mode={topology['init_mode']}, num_features={topology['num_features']}"
		)
		return train_df, val_df, test_df, 0


	def feature_gnn_after_materialize_fn(material_outputs: dict, config: dict, dataset_name: str, task_type: str) -> dict:
		print("[MATERIALIZE-GNN] SCARF: feature_gnn_after_materialize_fn is inactive (no-op).")
		return material_outputs


	def start_fn(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
		if pd is None:  # pragma: no cover
			raise ModuleNotFoundError("pandas is required to run SCARF") from _PANDAS_IMPORT_ERROR
		return train_df, val_df, test_df


	def materialize_fn(train_df, val_df, test_df, dataset_results, config):
		if pd is None:  # pragma: no cover
			raise ModuleNotFoundError("pandas is required to run SCARF") from _PANDAS_IMPORT_ERROR
		if StandardScaler is None:  # pragma: no cover
			raise ModuleNotFoundError(
				"scikit-learn is required for SCARF materialization (StandardScaler)."
			) from _SKLEARN_IMPORT_ERROR

		SCARFDataset, _NTXent, _ScarfEncoder, _ScarfColumnwiseLayer, _ScarfDecoder = _import_scarf_lib()

		batch_size = int(config.get("batch_size", 128))
		epochs = int(config.get("epochs", 200))
		device = resolve_device(config)

		if "target" not in train_df.columns:
			raise ValueError("SCARF expects a 'target' column in the input DataFrames.")

		train_x = train_df.drop(columns=["target"])
		train_y = train_df["target"]
		val_x = val_df.drop(columns=["target"])
		val_y = val_df["target"]
		test_x = test_df.drop(columns=["target"])
		test_y = test_df["target"]

		constant_cols = [c for c in train_x.columns if train_x[c].nunique(dropna=False) <= 1]
		if constant_cols:
			train_x = train_x.drop(columns=constant_cols)
			val_x = val_x.drop(columns=constant_cols)
			test_x = test_x.drop(columns=constant_cols)
			print(f"[MATERIALIZE] Dropped {len(constant_cols)} constant column(s).")

		scaler = StandardScaler()
		train_x = pd.DataFrame(scaler.fit_transform(train_x), columns=train_x.columns, index=train_df.index)
		val_x = pd.DataFrame(scaler.transform(val_x), columns=val_x.columns, index=val_df.index)
		test_x = pd.DataFrame(scaler.transform(test_x), columns=test_x.columns, index=test_df.index)

		train_ds = SCARFDataset(train_x.to_numpy(), train_y.to_numpy(), columns=train_x.columns)
		val_ds = SCARFDataset(val_x.to_numpy(), val_y.to_numpy(), columns=val_x.columns)
		test_ds = SCARFDataset(test_x.to_numpy(), test_y.to_numpy(), columns=test_x.columns)

		# Filter columns where low >= high (degenerate for uniform sampling).
		feat_low = train_ds.features_low
		feat_high = train_ds.features_high
		valid_cols = feat_low < feat_high
		if not np.all(valid_cols):
			keep = np.where(valid_cols)[0]
			print(f"[MATERIALIZE] Filtering out {int(np.sum(~valid_cols))} column(s) where low >= high.")
			train_x = train_x.iloc[:, keep]
			val_x = val_x.iloc[:, keep]
			test_x = test_x.iloc[:, keep]
			train_ds = SCARFDataset(train_x.to_numpy(), train_y.to_numpy(), columns=train_x.columns)
			val_ds = SCARFDataset(val_x.to_numpy(), val_y.to_numpy(), columns=val_x.columns)
			test_ds = SCARFDataset(test_x.to_numpy(), test_y.to_numpy(), columns=test_x.columns)

		from torch.utils.data import DataLoader

		train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
		val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
		test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

		task_type = dataset_results.get("info", {}).get("task_type", "binclass")
		is_classification = bool(task_type in {"binclass", "multiclass"})
		out_channels = int(len(np.unique(train_y))) if is_classification else 1
		is_binary_class = bool(is_classification and out_channels == 2)

		return {
			"batch_size": batch_size,
			"epochs": epochs,
			"device": device,
			"train_loader": train_loader,
			"val_loader": val_loader,
			"test_loader": test_loader,
			"train_ds": train_ds,
			"val_ds": val_ds,
			"test_ds": test_ds,
			"task_type": task_type,
			"is_classification": is_classification,
			"is_binary_class": is_binary_class,
			"out_channels": out_channels,
		}


	def scarf_core_fn(material_outputs: dict, config: dict, task_type: str, gnn_stage: str | None):
		"""Joint contrastive + supervised training with optional row-GNN injection."""

		SCARFDataset, NTXent, ScarfEncoder, ScarfColumnwiseLayer, ScarfDecoder = _import_scarf_lib()

		stage = "none" if gnn_stage is None else str(gnn_stage).lower()
		graphify = str(config.get("graphify", "row")).lower()
		if graphify == "feature" and stage in {"encoding", "columnwise", "decoding"}:
			print(
				f"[SCARF] graphify=feature keeps feature graphification inactive; "
				f"ignoring gnn_stage={stage!r} (treated as none)."
			)
			stage = "none"

		epochs = int(material_outputs["epochs"])
		device = material_outputs["device"]
		train_loader = material_outputs["train_loader"]
		val_loader = material_outputs["val_loader"]
		test_loader = material_outputs["test_loader"]
		train_ds = material_outputs["train_ds"]
		out_channels = int(material_outputs["out_channels"])

		dim_hidden_encoder = int(config.get("dim_hidden_encoder", 8))
		num_hidden_encoder = int(config.get("num_hidden_encoder", 3))
		dim_hidden_head = int(config.get("dim_hidden_head", 24))
		num_hidden_head = int(config.get("num_hidden_head", 2))
		dropout = float(config.get("dropout", 0.1))
		corruption_rate = float(config.get("corruption_rate", 0.6))
		sup_loss_weight = float(config.get("sup_loss_weight", 1.0))
		lr = float(config.get("lr", 1e-3))
		weight_decay = float(config.get("weight_decay", 1e-5))
		patience = int(config.get("patience", 10))
		loss_threshold = float(config.get("loss_threshold", 1e-4))
		restore_best = bool(int(config.get("restore_best", 1)))

		gnn_hidden = int(config.get("gnn_hidden", config.get("gnn_hidden_dim", 64)))
		gnn_layers = int(config.get("gnn_layers", 2))
		use_dgm = bool(config.get("use_dgm", True))
		dgm_k = int(config.get("gnn_dgm_k", config.get("dgm_k", 10)))
		dgm_distance = str(config.get("gnn_dgm_distance", config.get("dgm_distance", "euclidean")))
		dgm_reg_weight = float(config.get("dgm_reg_weight", 0.01))
		knn_k = int(config.get("gnn_knn", 5))

		if task_type == "multiclass":
			sup_out_dim = int(out_channels)
			metric_name = "Acc"
		elif task_type == "binclass":
			sup_out_dim = 1
			metric_name = "AUC"
		else:
			sup_out_dim = 1
			metric_name = "RMSE"

		encoder = ScarfEncoder(int(train_ds.shape[1]), dim_hidden_encoder, num_hidden_encoder, dropout).to(device)
		columnwise = ScarfColumnwiseLayer(dim_hidden_encoder).to(device)
		decoder = ScarfDecoder(dim_hidden_encoder, dim_hidden_head, num_hidden_head, dropout).to(device)

		gnn_encoding = (
			SimpleGCN(dim_hidden_encoder, gnn_hidden, dim_hidden_encoder, num_layers=max(2, gnn_layers)).to(device)
			if stage == "encoding"
			else None
		)
		gnn_columnwise = (
			SimpleGCN(dim_hidden_encoder, gnn_hidden, dim_hidden_encoder, num_layers=max(2, gnn_layers)).to(device)
			if stage == "columnwise"
			else None
		)
		gnn_decoder = (
			SimpleGCN(dim_hidden_encoder, gnn_hidden, sup_out_dim, num_layers=max(2, gnn_layers)).to(device)
			if stage == "decoding"
			else None
		)

		sup_head = nn.Linear(dim_hidden_encoder, sup_out_dim).to(device) if gnn_decoder is None else None

		# DGM module for online mini-batch graphs (optional).
		DGM_d_local = None
		if use_dgm and device.type == "cuda":
			try:
				DGM_d_local = _import_dgm_d()
			except Exception:
				DGM_d_local = None
				print("[SCARF] DGM_d unavailable; falling back to kNN graphs.")
		elif use_dgm and device.type != "cuda":
			print("[SCARF] Using CPU: disabling DGM_d and using kNN graphs instead.")

		dgm_module = None
		if DGM_d_local is not None and int(material_outputs.get("batch_size", 128)) > 1:
			class _DGMEmbedWrapper(nn.Module):
				def forward(self, x, A=None):
					return x

			dgm_k_train = int(min(dgm_k, max(1, int(material_outputs.get("batch_size", 128)) - 1)))
			dgm_module = DGM_d_local(_DGMEmbedWrapper(), k=dgm_k_train, distance=dgm_distance).to(device)

		feat_low = torch.tensor(train_ds.features_low, dtype=torch.float32, device=device)
		feat_high = torch.tensor(train_ds.features_high, dtype=torch.float32, device=device)
		feat_range = (feat_high - feat_low).clamp_min(1e-6)

		ntxent_temperature = float(config.get("temperature", 1.0))
		ntxent_loss = NTXent(temperature=ntxent_temperature).to(device)

		def _as_targets(t: torch.Tensor) -> torch.Tensor:
			t = torch.as_tensor(t, device=device)
			if task_type == "multiclass":
				return t.view(-1).long()
			if task_type == "binclass":
				return t.view(-1).float()
			return t.view(-1).float()

		def _build_graph(node_feats: torch.Tensor):
			"""Return (x_for_gnn, edge_index, dgm_reg_scalar)."""

			ns = int(node_feats.size(0))
			if ns <= 1:
				return (
					node_feats,
					torch.empty((2, 0), dtype=torch.long, device=device),
					torch.tensor(0.0, device=device),
				)

			x_std = _standardize(node_feats, dim=0)
			if dgm_module is not None:
				x_dgm, edge_index_dgm, logprobs = dgm_module(x_std.unsqueeze(0), A=None)
				x_dgm = x_dgm.squeeze(0)
				edge_index, _ = _symmetrize_self_loop_and_coalesce(edge_index_dgm, ns, edge_weight=None)
				dgm_reg = (-logprobs.mean()) if isinstance(logprobs, torch.Tensor) else torch.tensor(0.0, device=device)
				return x_dgm, edge_index, dgm_reg

			edge_index = _knn_graph(x_std, k=min(knn_k, max(1, ns - 1))).to(device)
			edge_index, _ = _symmetrize_self_loop_and_coalesce(edge_index, ns, edge_weight=None)
			return x_std, edge_index, torch.tensor(0.0, device=device)

		def _apply_stage_gnn(z: torch.Tensor, gnn: nn.Module | None):
			if gnn is None:
				return z, torch.tensor(0.0, device=device)
			x_for_gnn, edge_index, dgm_reg = _build_graph(z)
			z2 = gnn(x_for_gnn, edge_index)
			return z2, dgm_reg

		def _supervised_forward(z: torch.Tensor):
			if gnn_decoder is not None:
				z_for_gnn, edge_index, dgm_reg = _build_graph(z)
				logits = gnn_decoder(z_for_gnn, edge_index)
				return logits, dgm_reg
			assert sup_head is not None
			logits = sup_head(z)
			return logits, torch.tensor(0.0, device=device)

		def _supervised_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
			if task_type == "multiclass":
				return F.cross_entropy(logits, y)
			if task_type == "binclass":
				return F.binary_cross_entropy_with_logits(logits.view(-1), y)
			return F.mse_loss(logits.view(-1), y)

		def _metric_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
			# Keep dependencies minimal: use sklearn if available.
			logits_np = logits.detach().cpu().numpy()
			y_np = y.detach().cpu().numpy()
			if task_type == "multiclass":
				pred = np.argmax(logits_np, axis=1)
				return float((pred == y_np).mean())
			if task_type == "binclass":
				# roc_auc_score expects probabilities.
				try:
					from sklearn.metrics import roc_auc_score
				except Exception as e:  # pragma: no cover
					raise ModuleNotFoundError("scikit-learn is required for AUC metric.") from e
				prob = 1.0 / (1.0 + np.exp(-logits_np.reshape(-1)))
				return float(roc_auc_score(y_np.reshape(-1), prob))
			# RMSE
			diff = logits_np.reshape(-1) - y_np.reshape(-1)
			return float(np.sqrt(np.mean(diff * diff)))

		params: list[torch.nn.Parameter] = []
		modules = [encoder, columnwise, decoder, gnn_encoding, gnn_columnwise, gnn_decoder, sup_head, dgm_module]
		seen = set()
		for m in modules:
			if m is None:
				continue
			for p in m.parameters():
				if id(p) not in seen:
					seen.add(id(p))
					params.append(p)

		optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

		train_losses: list[float] = []
		val_losses: list[float] = []
		train_metrics: list[float] = []
		val_metrics: list[float] = []

		best_val_loss = float("inf")
		best_val_metric = float("-inf") if task_type in {"binclass", "multiclass"} else float("inf")
		best_states: dict = {}
		best_epoch = 0
		early_stop_epochs = 0
		stop_epoch = 0
		early_stop_counter = 0

		def _snapshot():
			return {
				"encoder": encoder.state_dict(),
				"columnwise": columnwise.state_dict(),
				"decoder": decoder.state_dict(),
				"gnn_encoding": gnn_encoding.state_dict() if gnn_encoding is not None else None,
				"gnn_columnwise": gnn_columnwise.state_dict() if gnn_columnwise is not None else None,
				"gnn_decoder": gnn_decoder.state_dict() if gnn_decoder is not None else None,
				"sup_head": sup_head.state_dict() if sup_head is not None else None,
				"dgm_module": dgm_module.state_dict() if dgm_module is not None else None,
			}

		def _restore(states: dict):
			encoder.load_state_dict(states["encoder"])
			columnwise.load_state_dict(states["columnwise"])
			decoder.load_state_dict(states["decoder"])
			if gnn_encoding is not None and states.get("gnn_encoding") is not None:
				gnn_encoding.load_state_dict(states["gnn_encoding"])
			if gnn_columnwise is not None and states.get("gnn_columnwise") is not None:
				gnn_columnwise.load_state_dict(states["gnn_columnwise"])
			if gnn_decoder is not None and states.get("gnn_decoder") is not None:
				gnn_decoder.load_state_dict(states["gnn_decoder"])
			if sup_head is not None and states.get("sup_head") is not None:
				sup_head.load_state_dict(states["sup_head"])
			if dgm_module is not None and states.get("dgm_module") is not None:
				dgm_module.load_state_dict(states["dgm_module"])

		for epoch in range(1, epochs + 1):
			encoder.train(); columnwise.train(); decoder.train()
			if gnn_encoding is not None:
				gnn_encoding.train()
			if gnn_columnwise is not None:
				gnn_columnwise.train()
			if gnn_decoder is not None:
				gnn_decoder.train()
			if sup_head is not None:
				sup_head.train()
			if dgm_module is not None:
				dgm_module.train()

			train_total = 0.0
			train_count = 0
			train_logits_all = []
			train_y_all = []

			for features, targets in train_loader:
				x = torch.as_tensor(features, device=device)
				y = _as_targets(targets)

				z = encoder(x)
				z, dgm_reg1 = _apply_stage_gnn(z, gnn_encoding)
				z = columnwise(z)
				z, dgm_reg2 = _apply_stage_gnn(z, gnn_columnwise)
				anchor = decoder(z)

				corruption_mask = torch.rand_like(x) > corruption_rate
				x_random = feat_low + torch.rand_like(x) * feat_range
				x_corrupt = torch.where(corruption_mask, x_random, x)

				z_c = encoder(x_corrupt)
				z_c, _ = _apply_stage_gnn(z_c, gnn_encoding)
				z_c = columnwise(z_c)
				z_c, _ = _apply_stage_gnn(z_c, gnn_columnwise)
				positive = decoder(z_c)

				contrastive = ntxent_loss(anchor, positive)
				logits, dgm_reg3 = _supervised_forward(z)
				sup_loss = _supervised_loss(logits, y)
				total_loss = contrastive + sup_loss_weight * sup_loss + dgm_reg_weight * (
					dgm_reg1 + dgm_reg2 + dgm_reg3
				)

				optimizer.zero_grad()
				total_loss.backward()
				torch.nn.utils.clip_grad_norm_(params, 1.0)
				optimizer.step()

				train_total += float(total_loss) * len(x)
				train_count += len(x)
				train_logits_all.append(logits.detach().cpu())
				train_y_all.append(y.detach().cpu())

			train_loss = train_total / max(1, train_count)
			train_losses.append(float(train_loss))
			if train_logits_all:
				train_metric = _metric_from_logits(
					torch.cat(train_logits_all, dim=0).to(device),
					torch.cat(train_y_all, dim=0).to(device),
				)
			else:
				train_metric = float("nan")
			train_metrics.append(float(train_metric))

			encoder.eval(); columnwise.eval(); decoder.eval()
			if gnn_encoding is not None:
				gnn_encoding.eval()
			if gnn_columnwise is not None:
				gnn_columnwise.eval()
			if gnn_decoder is not None:
				gnn_decoder.eval()
			if sup_head is not None:
				sup_head.eval()
			if dgm_module is not None:
				dgm_module.eval()

			val_loss_sum = 0.0
			val_count = 0
			val_logits_all = []
			val_y_all = []
			with torch.no_grad():
				for features, targets in val_loader:
					x = torch.as_tensor(features, device=device)
					y = _as_targets(targets)
					z = encoder(x)
					z, _ = _apply_stage_gnn(z, gnn_encoding)
					z = columnwise(z)
					z, _ = _apply_stage_gnn(z, gnn_columnwise)
					logits, _ = _supervised_forward(z)
					vloss = _supervised_loss(logits, y)
					val_loss_sum += float(vloss) * len(x)
					val_count += len(x)
					val_logits_all.append(logits.detach().cpu())
					val_y_all.append(y.detach().cpu())

			val_loss = val_loss_sum / max(1, val_count)
			val_losses.append(float(val_loss))
			if val_logits_all:
				val_metric = _metric_from_logits(
					torch.cat(val_logits_all, dim=0).to(device),
					torch.cat(val_y_all, dim=0).to(device),
				)
			else:
				val_metric = float("nan")
			val_metrics.append(float(val_metric))

			improved = val_loss < best_val_loss - loss_threshold
			if improved:
				best_val_loss = float(val_loss)
				best_val_metric = float(val_metric)
				best_states = _snapshot()
				best_epoch = epoch
				early_stop_counter = 0
			else:
				early_stop_counter += 1
				if early_stop_counter >= patience:
					stop_epoch = epoch
					break

		if stop_epoch == 0:
			stop_epoch = epochs
		early_stop_epochs = int(stop_epoch)

		if restore_best and best_states:
			_restore(best_states)
			print(f"[SCARF] Restored best weights from epoch {best_epoch}")

		# Test metric at the restored checkpoint.
		encoder.eval(); columnwise.eval(); decoder.eval()
		if gnn_encoding is not None:
			gnn_encoding.eval()
		if gnn_columnwise is not None:
			gnn_columnwise.eval()
		if gnn_decoder is not None:
			gnn_decoder.eval()
		if sup_head is not None:
			sup_head.eval()
		if dgm_module is not None:
			dgm_module.eval()

		test_logits_all = []
		test_y_all = []
		with torch.no_grad():
			for features, targets in test_loader:
				x = torch.as_tensor(features, device=device)
				y = _as_targets(targets)
				z = encoder(x)
				z, _ = _apply_stage_gnn(z, gnn_encoding)
				z = columnwise(z)
				z, _ = _apply_stage_gnn(z, gnn_columnwise)
				logits, _ = _supervised_forward(z)
				test_logits_all.append(logits.detach().cpu())
				test_y_all.append(y.detach().cpu())

		if test_logits_all:
			test_metric = _metric_from_logits(
				torch.cat(test_logits_all, dim=0).to(device),
				torch.cat(test_y_all, dim=0).to(device),
			)
		else:
			test_metric = float("nan")

		pre_stage_stop = int(material_outputs.get("gnn_early_stop_epochs", 0))
		reported_gnn_stop = pre_stage_stop if stage in {"start", "materialize"} else 0

		return {
			"gnn_early_stop_epochs": reported_gnn_stop,
			"train_losses": train_losses,
			"train_metrics": train_metrics,
			"val_losses": val_losses,
			"val_metrics": val_metrics,
			"best_val_loss": best_val_loss,
			"best_val_metric": best_val_metric,
			"final_metric": best_val_metric,
			"best_test_metric": float(test_metric),
			"metric": metric_name,
			"early_stop_epochs": int(early_stop_epochs),
			"gnn_stage": stage,
			"encoder": encoder,
			"columnwise": columnwise,
			"decoder": decoder,
			"gnn_encoding": gnn_encoding,
			"gnn_columnwise": gnn_columnwise,
			"gnn_decoder": gnn_decoder,
			"sup_head": sup_head,
			"dgm_module": dgm_module,
			"is_classification": bool(task_type in {"binclass", "multiclass"}),
			"is_binary_class": bool(task_type == "binclass"),
		}


	def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
		print("SCARF - staged execution")
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
					material_outputs = row_gnn_after_materialize_fn(
						material_outputs,
						config,
						dataset_results.get("dataset", "unknown"),
						task_type,
					)

				results = scarf_core_fn(material_outputs, config, task_type, gnn_stage=stage)

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
						dataset_results.get("dataset", "unknown"),
						task_type,
					)

				results = scarf_core_fn(material_outputs, config, task_type, gnn_stage=stage)
			else:
				raise ValueError(f"Unknown graphify mode: {graphify}. Expected 'row' or 'feature'.")

			results["graphify"] = graphify
			return results
		except Exception as e:
			return {
				"gnn_stage": stage,
				"graphify": str(config.get("graphify", "row")),
				"error": str(e),
			}

