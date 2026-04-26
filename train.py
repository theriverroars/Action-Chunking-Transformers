"""
Train the ACT policy on collected pick-and-place demonstrations.

Usage
-----
python train.py [--data data/demos.hdf5] [--epochs 300] [--batch-size 32]
                [--lr 1e-4] [--chunk-size 50] [--out checkpoints/]

Outputs
-------
  checkpoints/best_model.pth   — model with lowest validation loss
  checkpoints/last_model.pth   — model at final epoch
  checkpoints/stats.npz        — normalisation stats (obs_mean, obs_std)
  training_curves.png          — train/val loss plot
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEFAULT_CONFIG
from act.model import ACTPolicy
from data.dataset import make_train_val_datasets


# ---------------------------------------------------------------------------

def train(config: dict) -> None:
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ---- Data -----------------------------------------------------------
    print("Loading dataset …")
    train_ds, val_ds, stats = make_train_val_datasets(
        config["data_path"],
        chunk_size=config["chunk_size"],
        train_ratio=config["train_ratio"],
    )
    print(f"  train samples: {len(train_ds)}   val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # ---- Model ----------------------------------------------------------
    model = ACTPolicy(
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"],
        config=config,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=1e-6
    )

    # ---- Normalisation stats tensors ------------------------------------
    obs_mean_t = torch.tensor(stats["obs_mean"], dtype=torch.float32).to(device)
    obs_std_t = torch.tensor(stats["obs_std"], dtype=torch.float32).to(device)

    # ---- Output dirs ----------------------------------------------------
    out_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Training loop --------------------------------------------------
    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    train_kl_losses, train_recon_losses = [], []

    for epoch in range(1, config["num_epochs"] + 1):
        # -- Train --
        model.train()
        epoch_loss = epoch_recon = epoch_kl = 0.0
        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)   # (B, obs_dim)
            act_batch = act_batch.to(device)   # (B, K, action_dim)

            optimizer.zero_grad()
            total, recon, kl = model.compute_loss(obs_batch, act_batch)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()

        scheduler.step()

        n = len(train_loader)
        train_losses.append(epoch_loss / n)
        train_recon_losses.append(epoch_recon / n)
        train_kl_losses.append(epoch_kl / n)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                total, _, _ = model.compute_loss(obs_batch, act_batch)
                val_loss += total.item()

        val_loss /= max(len(val_loader), 1)
        val_losses.append(val_loss)

        # -- Logging (every 10 epochs) --
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{config['num_epochs']}  "
                f"train={train_losses[-1]:.4f}  "
                f"(recon={train_recon_losses[-1]:.4f}, kl={train_kl_losses[-1]:.4f})  "
                f"val={val_losses[-1]:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # -- Save best --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                    "obs_mean": stats["obs_mean"],
                    "obs_std": stats["obs_std"],
                },
                os.path.join(out_dir, "best_model.pth"),
            )

    # ---- Save last checkpoint ------------------------------------------
    torch.save(
        {
            "epoch": config["num_epochs"],
            "model_state_dict": model.state_dict(),
            "config": config,
            "obs_mean": stats["obs_mean"],
            "obs_std": stats["obs_std"],
        },
        os.path.join(out_dir, "last_model.pth"),
    )

    # ---- Save normalisation stats separately ---------------------------
    np.savez(
        os.path.join(out_dir, "stats.npz"),
        obs_mean=stats["obs_mean"],
        obs_std=stats["obs_std"],
    )

    # ---- Plot loss curves ----------------------------------------------
    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, label="train total")
    axes[0].plot(epochs, val_losses, label="val total")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss (recon + kl_weight × KL)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_recon_losses, label="train recon (L1)")
    axes[1].plot(epochs, train_kl_losses, label="train KL")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Reconstruction and KL Losses")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    curve_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()

    print(f"\nBest val loss: {best_val_loss:.4f}  (saved to {out_dir}/best_model.pth)")
    print(f"Training curves saved to: {curve_path}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ACT policy")
    parser.add_argument("--data", type=str,
                        default=DEFAULT_CONFIG["data_path"])
    parser.add_argument("--epochs", type=int,
                        default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--batch-size", type=int,
                        default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float,
                        default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--chunk-size", type=int,
                        default=DEFAULT_CONFIG["chunk_size"])
    parser.add_argument("--out", type=str, default="checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["data_path"] = args.data
    config["num_epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr
    config["chunk_size"] = args.chunk_size
    config["checkpoint_dir"] = args.out
    config["seed"] = args.seed

    t0 = time.time()
    train(config)
    print(f"Total training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
