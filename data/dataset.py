"""
HDF5-backed dataset for ACT imitation learning.

File layout (written by collect_data.py)
-----------------------------------------
demos.hdf5
  /episode_0000/
    observations  (T, obs_dim)   float32
    actions       (T, action_dim) float32
    success       ()              bool
  /episode_0001/
    ...

Training samples
----------------
Each index maps to a (timestep, episode) pair.
  obs     : flat_obs[t]                     shape (obs_dim,)
  actions : actions[t : t+K]  (padded)      shape (K, action_dim)

Padding: when t + K > T the last action is repeated to fill the chunk.
"""

from __future__ import annotations

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_stats(
    data_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-dimension mean and std for observations and actions.
    Used for normalising inputs at training / inference time.
    """
    all_obs, all_actions = [], []
    with h5py.File(data_path, "r") as f:
        for ep_key in f:
            all_obs.append(f[ep_key]["observations"][:])
            all_actions.append(f[ep_key]["actions"][:])

    obs_arr = np.concatenate(all_obs, axis=0)       # (N_total, obs_dim)
    act_arr = np.concatenate(all_actions, axis=0)   # (N_total, action_dim)

    obs_mean = obs_arr.mean(axis=0).astype(np.float32)
    obs_std = obs_arr.std(axis=0).astype(np.float32)
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std)   # avoid div-by-zero

    act_mean = act_arr.mean(axis=0).astype(np.float32)
    act_std = act_arr.std(axis=0).astype(np.float32)
    act_std = np.where(act_std < 1e-6, 1.0, act_std)

    return obs_mean, obs_std, act_mean, act_std


def load_demos(data_path: str) -> list[dict]:
    """Load all episodes from HDF5 into a list of dicts (in memory)."""
    demos = []
    with h5py.File(data_path, "r") as f:
        for ep_key in sorted(f.keys()):
            demos.append(
                {
                    "observations": f[ep_key]["observations"][:],
                    "actions": f[ep_key]["actions"][:],
                    "success": bool(f[ep_key].attrs.get("success", False)),
                }
            )
    return demos


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DemoDataset(Dataset):
    """
    PyTorch Dataset for ACT training.

    Each item is a (obs, action_chunk) pair where obs is normalised and
    action_chunk has shape (chunk_size, action_dim).
    """

    def __init__(
        self,
        demos: list[dict],
        chunk_size: int,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        indices: list[tuple[int, int]] | None = None,
    ):
        """
        Parameters
        ----------
        demos      : list of episode dicts with 'observations' and 'actions'
        chunk_size : K — number of future actions per sample
        obs_mean   : (obs_dim,)
        obs_std    : (obs_dim,)
        indices    : explicit list of (episode_idx, timestep_idx) pairs;
                     if None all valid timesteps are used.
        """
        self.demos = demos
        self.chunk_size = chunk_size
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = obs_std.astype(np.float32)

        if indices is not None:
            self._indices = indices
        else:
            self._indices = []
            for ep_idx, ep in enumerate(demos):
                T = len(ep["observations"])
                for t in range(T):
                    self._indices.append((ep_idx, t))

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ep_idx, t = self._indices[idx]
        ep = self.demos[ep_idx]

        obs = ep["observations"][t].astype(np.float32)
        obs = (obs - self.obs_mean) / self.obs_std

        T = len(ep["actions"])
        end = min(t + self.chunk_size, T)
        chunk = ep["actions"][t:end].astype(np.float32)      # (≤K, action_dim)

        # Pad with the last action if the episode is shorter than K
        if chunk.shape[0] < self.chunk_size:
            pad = np.tile(chunk[-1:], (self.chunk_size - chunk.shape[0], 1))
            chunk = np.concatenate([chunk, pad], axis=0)     # (K, action_dim)

        return (
            torch.from_numpy(obs),    # (obs_dim,)
            torch.from_numpy(chunk),  # (K, action_dim)
        )


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def make_train_val_datasets(
    data_path: str,
    chunk_size: int,
    train_ratio: float = 0.9,
) -> tuple[DemoDataset, DemoDataset, dict]:
    """
    Load demos, compute normalisation stats, split into train/val.

    Returns
    -------
    train_dataset, val_dataset, stats_dict
    """
    demos = load_demos(data_path)
    obs_mean, obs_std, act_mean, act_std = compute_stats(data_path)

    # Build all indices
    all_indices: list[tuple[int, int]] = []
    for ep_idx, ep in enumerate(demos):
        T = len(ep["observations"])
        for t in range(T):
            all_indices.append((ep_idx, t))

    rng = np.random.default_rng(seed=42)
    rng.shuffle(all_indices)
    split = int(len(all_indices) * train_ratio)
    train_idx = all_indices[:split]
    val_idx = all_indices[split:]

    train_ds = DemoDataset(demos, chunk_size, obs_mean, obs_std, train_idx)
    val_ds = DemoDataset(demos, chunk_size, obs_mean, obs_std, val_idx)

    stats = {
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "act_mean": act_mean,
        "act_std": act_std,
    }
    return train_ds, val_ds, stats
