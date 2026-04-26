# Action Chunking with Transformers — Pick-and-Place

Implementation of the **ACT** (Action Chunking with Transformers) imitation
learning framework applied to a **Franka Panda pick-and-place** task in
simulation.

> **Reference paper:** Zhao et al. (2023) *Learning Fine-Grained Bimanual
> Manipulation with Low-Cost Hardware.* [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)  
> **Reference code:** https://github.com/Shaka-Labs/ACT

---

## Overview

| Item | Detail |
|------|--------|
| Robot | Franka Panda 7-DoF (via panda-gym / PyBullet) |
| Task | Pick a randomly placed cube and place it at a random target |
| Success criterion | Cube within **5 cm** of goal at episode end |
| Policy | ACT — CVAE + Transformer encoder-decoder |
| Demonstrations | 50 scripted-expert episodes (HDF5) |
| Action space | Δ end-effector position (3) + gripper (1) |
| Observation | EEF pose, gripper width, cube pose, goal position (22-dim) |

---

## Architecture

```
Training
  ┌──────────────────────────────────────────────────┐
  │  CVAE Encoder (Transformer)                      │
  │    [CLS | obs_embed | a₀ … a_{K-1}]  →  μ, log σ│
  └──────────────────┬───────────────────────────────┘
                     │  z ~ N(μ, σ)   (reparameterise)
  ┌──────────────────▼───────────────────────────────┐
  │  Policy Decoder (Transformer)                    │
  │    encoder: [obs_embed | z_embed] → memory       │
  │    decoder: K query tokens × memory → K actions  │
  └──────────────────────────────────────────────────┘
  Loss = L1(pred_actions, gt_actions)
       + kl_weight × KL( N(μ,σ) ∥ N(0,I) )

Inference
  z = 0  (prior mean, no stochasticity)
  Policy decoder produces K-action chunk; first action is executed.
```

Key hyper-parameters (see `config.py`):

| Parameter | Value |
|-----------|-------|
| Chunk size *K* | 50 |
| Hidden dim | 256 |
| Attention heads | 8 |
| Encoder / decoder layers | 4 / 6 |
| Latent dim | 32 |
| KL weight | 10 |
| Learning rate | 1e-4 |
| Epochs | 300 |

---

## Repository Structure

```
Action-Chunking-Transformers/
├── config.py          # Hyperparameters & observation indices
├── requirements.txt   # Python dependencies
│
├── envs/
│   └── pick_place_env.py   # Panda-gym wrapper
│
├── act/
│   └── model.py            # ACTPolicy (CVAE + Transformer)
│
├── data/
│   └── dataset.py          # HDF5 dataset + normalisation
│
├── expert.py          # Scripted phase-based expert
├── collect_data.py    # Run expert → save HDF5 demos
├── train.py           # Train ACT policy
└── evaluate.py        # Evaluate + save metrics + record video
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/theriverroars/Action-Chunking-Transformers
cd Action-Chunking-Transformers

# 2. Install dependencies  (Python 3.10+ recommended)
pip install -r requirements.txt
```

---

## Step 1 — Collect Demonstrations

```bash
python collect_data.py --num-demos 50 --out data/demos.hdf5
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--num-demos` | 50 | Number of episodes to collect |
| `--out` | `data/demos.hdf5` | Output file |
| `--seed` | 0 | RNG seed for first episode |
| `--success-only` | off | Keep only successful episodes |

The scripted expert uses a six-phase proportional controller:
**pre-grasp → descend → close gripper → lift → transport → place → open gripper**

---

## Step 2 — Train

```bash
python train.py --data data/demos.hdf5 --epochs 300 --out checkpoints/
```

Outputs:

| File | Description |
|------|-------------|
| `checkpoints/best_model.pth` | Lowest-validation-loss checkpoint |
| `checkpoints/last_model.pth` | Final epoch checkpoint |
| `checkpoints/training_curves.png` | Loss vs. epoch plots |
| `checkpoints/stats.npz` | Observation normalisation statistics |

---

## Step 3 — Evaluate

```bash
python evaluate.py \
  --ckpt checkpoints/best_model.pth \
  --num-episodes 50 \
  --video videos/eval_video.mp4
```

Outputs:

| File | Description |
|------|-------------|
| `eval_results.txt` | Success rate, mean episode length, std |
| `videos/eval_video.mp4` | Video of successful pick-and-place runs |

---

## Deliverables

| Deliverable | Location |
|-------------|----------|
| Full working code | This repository |
| Demonstration data (50 eps) | `data/demos.hdf5` |
| Trained model checkpoint | `checkpoints/best_model.pth` |
| Training curves (PNG) | `checkpoints/training_curves.png` |
| Evaluation video (MP4) | `videos/eval_video.mp4` |
| Evaluation metrics | `eval_results.txt` |

---

## Citation

```bibtex
@inproceedings{zhao2023act,
  title   = {Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author  = {Zhao, Tony Z. and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  booktitle = {Robotics: Science and Systems (RSS)},
  year    = {2023}
}
```
