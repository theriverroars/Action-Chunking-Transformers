"""
Default configuration for the ACT pick-and-place project.

Observation layout for PandaPickAndPlace-v3 (flat = obs + desired_goal):
  [0:3]   ee_pos           end-effector position (m)
  [3:6]   ee_vel           end-effector velocity (m/s)
  [6]     fingers_width    gripper opening width (m)
  [7:10]  obj_pos          cube position (m)
  [10:13] obj_rot          cube orientation (euler rad)
  [13:16] obj_vel          cube linear velocity (m/s)
  [16:19] obj_ang_vel      cube angular velocity (rad/s)
  [19:22] desired_goal     target position (m)

Action layout:
  [0:3]   delta_ee_pos     end-effector displacement (normalised to [-1, 1])
  [3]     gripper          -1 = open, +1 = close
"""

# ---------------------------------------------------------------------------
# Indices used when parsing the flat observation
# ---------------------------------------------------------------------------
OBS_EE_POS = slice(0, 3)
OBS_EE_VEL = slice(3, 6)
OBS_FINGERS = 6
OBS_OBJ_POS = slice(7, 10)
OBS_OBJ_ROT = slice(10, 13)
OBS_OBJ_VEL = slice(13, 16)
OBS_OBJ_ANG_VEL = slice(16, 19)
OBS_GOAL = slice(19, 22)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # ---- environment -------------------------------------------------------
    "obs_dim": 22,          # len(obs['observation']) + len(desired_goal) = 19+3
    "action_dim": 4,        # [dx, dy, dz, gripper]
    "max_episode_steps": 200,

    # ---- ACT model ---------------------------------------------------------
    "chunk_size": 50,       # K — number of future actions to predict at once
    "hidden_dim": 256,      # Transformer hidden dimension d_model
    "nheads": 8,            # Multi-head attention heads
    "num_encoder_layers": 4,  # CVAE encoder Transformer layers
    "num_decoder_layers": 6,  # Policy decoder Transformer layers
    "dim_feedforward": 1024,  # FFN width inside Transformer
    "latent_dim": 32,         # CVAE latent space dimension
    "dropout": 0.1,
    "kl_weight": 10.0,      # Weight of KL term in ELBO loss

    # ---- training ----------------------------------------------------------
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "num_epochs": 300,
    "seed": 42,

    # ---- data collection ---------------------------------------------------
    "num_demos": 50,
    "train_ratio": 0.9,
    "data_path": "data/demos.hdf5",

    # ---- evaluation --------------------------------------------------------
    "num_eval_episodes": 50,
    "checkpoint_path": "checkpoints/best_model.pth",
    "video_path": "videos/eval_video.mp4",
    "success_threshold": 0.05,  # 5 cm from goal counts as success
    "success_video_count": 5,   # record at least this many successes in video
}
