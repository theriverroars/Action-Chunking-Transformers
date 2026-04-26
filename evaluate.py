"""
Evaluate a trained ACT policy on the pick-and-place task.

Usage
-----
python evaluate.py [--ckpt checkpoints/best_model.pth]
                   [--num-episodes 50] [--video videos/eval_video.mp4]
                   [--success-only-video] [--seed 100]

Outputs
-------
  eval_results.txt     — success rate, mean episode length, std
  videos/eval_video.mp4 — video of (up to) 5 successful episodes
"""

from __future__ import annotations

import argparse
import os
import time

import imageio
import numpy as np
import torch
from tqdm import tqdm

from config import DEFAULT_CONFIG
from act.model import ACTPolicy
from envs.pick_place_env import PickPlaceEnv


# ---------------------------------------------------------------------------

def load_policy(ckpt_path: str, device: torch.device) -> tuple[ACTPolicy, dict, np.ndarray, np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    obs_mean = ckpt["obs_mean"]
    obs_std = ckpt["obs_std"]

    model = ACTPolicy(
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"],
        config=config,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(
        f"Loaded checkpoint: {ckpt_path}  "
        f"(epoch {ckpt.get('epoch', '?')}, "
        f"val_loss={ckpt.get('val_loss', float('nan')):.4f})"
    )
    return model, config, obs_mean, obs_std


# ---------------------------------------------------------------------------

def run_episode(
    env: PickPlaceEnv,
    model: ACTPolicy,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    device: torch.device,
    chunk_size: int,
    seed: int | None = None,
    record_frames: bool = False,
) -> dict:
    """
    Run one evaluation episode.

    The policy re-plans every step (i.e., one action per prediction call).
    Alternatively, the full K-action chunk is executed before re-planning —
    set exec_horizon = chunk_size for that behaviour.
    """
    obs, _ = env.reset(seed=seed)
    frames = []
    step = 0
    done = False

    while not done:
        if record_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        # Normalise observation
        obs_norm = (obs - obs_mean) / obs_std

        # Predict action chunk, execute first action
        actions = model.predict(obs_norm, device=device)  # (K, 4)
        action = actions[0]

        obs, _reward, done, info = env.step(action)
        step += 1

    success = bool(info.get("is_success", False))
    return {"success": success, "steps": step, "frames": frames}


# ---------------------------------------------------------------------------

def evaluate(
    ckpt_path: str,
    num_episodes: int = 50,
    video_path: str | None = "videos/eval_video.mp4",
    seed: int = 100,
    success_video_count: int = 5,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    model, config, obs_mean, obs_std = load_policy(ckpt_path, device)
    chunk_size = config["chunk_size"]

    env = PickPlaceEnv(render=True)

    successes = []
    episode_lengths = []
    video_frames: list[np.ndarray] = []
    success_episodes_recorded = 0

    with tqdm(total=num_episodes, desc="Evaluating") as pbar:
        for ep_idx in range(num_episodes):
            ep_seed = seed + ep_idx
            need_video = success_episodes_recorded < success_video_count

            result = run_episode(
                env=env,
                model=model,
                obs_mean=obs_mean,
                obs_std=obs_std,
                device=device,
                chunk_size=chunk_size,
                seed=ep_seed,
                record_frames=need_video,
            )

            successes.append(result["success"])
            episode_lengths.append(result["steps"])

            if result["success"] and need_video and result["frames"]:
                video_frames.extend(result["frames"])
                success_episodes_recorded += 1

            pbar.set_postfix(
                sr=f"{sum(successes) / len(successes):.1%}",
                ep_len=result["steps"],
            )
            pbar.update(1)

    env.close()

    # ---- Metrics --------------------------------------------------------
    sr = float(np.mean(successes))
    mean_len = float(np.mean(episode_lengths))
    std_len = float(np.std(episode_lengths))

    print(f"\n{'='*50}")
    print(f"Evaluation results over {num_episodes} episodes:")
    print(f"  Success rate  : {sr:.1%}  ({sum(successes)}/{num_episodes})")
    print(f"  Episode length: {mean_len:.1f} ± {std_len:.1f} steps")
    print(f"{'='*50}\n")

    # ---- Save results text file -----------------------------------------
    results_path = "eval_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Evaluation Results\n")
        f.write(f"==================\n")
        f.write(f"Checkpoint   : {ckpt_path}\n")
        f.write(f"Episodes     : {num_episodes}\n")
        f.write(f"Success rate : {sr:.1%} ({sum(successes)}/{num_episodes})\n")
        f.write(f"Mean ep. len : {mean_len:.1f} steps\n")
        f.write(f"Std ep. len  : {std_len:.1f} steps\n")
    print(f"Results saved to: {results_path}")

    # ---- Save video -----------------------------------------------------
    if video_path and video_frames:
        os.makedirs(os.path.dirname(video_path) if os.path.dirname(video_path) else ".", exist_ok=True)
        writer = imageio.get_writer(video_path, fps=30)
        for frame in video_frames:
            writer.append_data(frame)
        writer.close()
        print(f"Video saved to : {video_path}  ({len(video_frames)} frames)")

    return {
        "success_rate": sr,
        "mean_episode_length": mean_len,
        "std_episode_length": std_len,
        "successes": sum(successes),
        "total_episodes": num_episodes,
    }


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained ACT policy")
    parser.add_argument("--ckpt", type=str,
                        default=DEFAULT_CONFIG["checkpoint_path"])
    parser.add_argument("--num-episodes", type=int,
                        default=DEFAULT_CONFIG["num_eval_episodes"])
    parser.add_argument("--video", type=str,
                        default=DEFAULT_CONFIG["video_path"])
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--success-video-count", type=int,
                        default=DEFAULT_CONFIG["success_video_count"])
    args = parser.parse_args()

    t0 = time.time()
    evaluate(
        ckpt_path=args.ckpt,
        num_episodes=args.num_episodes,
        video_path=args.video,
        seed=args.seed,
        success_video_count=args.success_video_count,
    )
    print(f"Total evaluation time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
