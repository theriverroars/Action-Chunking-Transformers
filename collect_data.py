"""
Collect pick-and-place demonstrations using the scripted expert.

Usage
-----
python collect_data.py [--num-demos 50] [--out data/demos.hdf5] [--seed 0]

The script runs the scripted expert for NUM_DEMOS episodes and saves the
observations and actions to an HDF5 file.  Only *successful* and *failed*
episodes are stored (all episodes are stored for data diversity; you can
filter to successes only via --success-only).

HDF5 layout
-----------
  /episode_0000/
    observations  (T, 22)   float32
    actions       (T, 4)    float32
    attrs: success (bool), episode_length (int)
  /episode_0001/
    ...
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
from tqdm import tqdm

from config import DEFAULT_CONFIG
from envs.pick_place_env import PickPlaceEnv
from expert import ScriptedExpert


# ---------------------------------------------------------------------------

def collect_episode(
    env: PickPlaceEnv,
    expert: ScriptedExpert,
    seed: int | None = None,
) -> dict:
    """Run one episode, return dict with observations, actions, success."""
    obs, _ = env.reset(seed=seed)
    expert.reset()

    observations, actions = [], []
    done = False

    while not done:
        action = expert.get_action(obs)
        observations.append(obs.copy())
        actions.append(action.copy())

        obs, _reward, done, info = env.step(action)
        if expert.is_done():
            # a few hold-still steps to let the object settle
            for _ in range(10):
                still_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                observations.append(obs.copy())
                actions.append(still_action.copy())
                obs, _reward, done, info = env.step(still_action)
                if done:
                    break
            break

    success = bool(info.get("is_success", False))
    return {
        "observations": np.array(observations, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "success": success,
    }


# ---------------------------------------------------------------------------

def collect_demos(
    num_demos: int,
    out_path: str,
    seed: int = 0,
    success_only: bool = False,
) -> None:
    """Collect *num_demos* episodes and save to HDF5."""
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    env = PickPlaceEnv(render=False)
    expert = ScriptedExpert()

    saved = 0
    attempts = 0
    successes = 0

    with h5py.File(out_path, "w") as hf:
        pbar = tqdm(total=num_demos, desc="Collecting demos")
        while saved < num_demos:
            ep_seed = seed + attempts
            ep = collect_episode(env, expert, seed=ep_seed)
            attempts += 1

            if ep["success"]:
                successes += 1

            if success_only and not ep["success"]:
                continue

            ep_key = f"episode_{saved:04d}"
            grp = hf.create_group(ep_key)
            grp.create_dataset("observations", data=ep["observations"])
            grp.create_dataset("actions", data=ep["actions"])
            grp.attrs["success"] = ep["success"]
            grp.attrs["episode_length"] = len(ep["observations"])

            saved += 1
            pbar.update(1)
            pbar.set_postfix(
                success_rate=f"{successes / attempts:.1%}",
                ep_len=len(ep["observations"]),
            )

        pbar.close()

    env.close()
    print(
        f"\nCollected {saved} demos  "
        f"({successes}/{attempts} successes = {successes/attempts:.1%})"
    )
    print(f"Saved to: {out_path}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect pick-and-place demos")
    parser.add_argument("--num-demos", type=int,
                        default=DEFAULT_CONFIG["num_demos"],
                        help="Number of episodes to collect (default: 50)")
    parser.add_argument("--out", type=str,
                        default=DEFAULT_CONFIG["data_path"],
                        help="Output HDF5 path (default: data/demos.hdf5)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for first episode (default: 0)")
    parser.add_argument("--success-only", action="store_true",
                        help="Only save successful episodes")
    args = parser.parse_args()

    t0 = time.time()
    collect_demos(
        num_demos=args.num_demos,
        out_path=args.out,
        seed=args.seed,
        success_only=args.success_only,
    )
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
