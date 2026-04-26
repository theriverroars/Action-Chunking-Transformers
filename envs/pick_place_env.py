"""
Pick-and-place environment wrapper around panda-gym PandaPickAndPlace-v3.

Observation (flat, 22-dim):
  obs['observation'] (19)  +  obs['desired_goal'] (3)

Action (4-dim):  [dx, dy, dz, gripper]  — all in [-1, 1]
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

try:
    import panda_gym  # registers PandaPickAndPlace-v3
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "panda-gym is required.  Install it with:  pip install panda-gym"
    ) from exc


class PickPlaceEnv:
    """
    Thin wrapper around PandaPickAndPlace-v3 that:
      * returns a flat numpy observation (obs + desired_goal),
      * exposes episode metadata (success, step count),
      * optionally renders an RGB frame.
    """

    # Indices within the flat observation
    IDX_EE_POS = slice(0, 3)
    IDX_EE_VEL = slice(3, 6)
    IDX_FINGERS = 6
    IDX_OBJ_POS = slice(7, 10)
    IDX_OBJ_ROT = slice(10, 13)
    IDX_OBJ_VEL = slice(13, 16)
    IDX_OBJ_ANG_VEL = slice(16, 19)
    IDX_GOAL = slice(19, 22)

    def __init__(self, render: bool = False, seed: int | None = None):
        render_mode = "rgb_array"  # always rgb_array; rendering is opt-in
        self._env = gym.make(
            "PandaPickAndPlace-v3",
            render_mode=render_mode,
            max_episode_steps=200,
        )
        self._render = render
        self._seed = seed

        # Compute dimensions by doing a throw-away reset
        obs0, _ = self._env.reset(seed=seed)
        flat = self._flatten(obs0)
        self.obs_dim = flat.shape[0]            # 22
        self.action_dim = self._env.action_space.shape[0]  # 4

        self._step = 0
        self._max_steps = 200

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        obs_dict, info = self._env.reset(seed=seed if seed is not None else self._seed)
        self._step = 0
        return self._flatten(obs_dict), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        obs_dict, reward, terminated, truncated, info = self._env.step(action)
        self._step += 1
        done = terminated or truncated or (self._step >= self._max_steps)
        flat = self._flatten(obs_dict)
        return flat, float(reward), done, info

    def render(self) -> np.ndarray | None:
        """Return an RGB frame (H x W x 3) or None when rendering is off."""
        if not self._render:
            return None
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_ee_pos(self, flat_obs: np.ndarray) -> np.ndarray:
        return flat_obs[self.IDX_EE_POS].copy()

    def get_obj_pos(self, flat_obs: np.ndarray) -> np.ndarray:
        return flat_obs[self.IDX_OBJ_POS].copy()

    def get_goal_pos(self, flat_obs: np.ndarray) -> np.ndarray:
        return flat_obs[self.IDX_GOAL].copy()

    def get_fingers_width(self, flat_obs: np.ndarray) -> float:
        return float(flat_obs[self.IDX_FINGERS])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(obs_dict: dict) -> np.ndarray:
        """Concatenate observation and desired_goal into one vector."""
        return np.concatenate(
            [obs_dict["observation"], obs_dict["desired_goal"]]
        ).astype(np.float32)
