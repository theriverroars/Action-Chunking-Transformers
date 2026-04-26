"""
ACT — Action Chunking with Transformers
========================================
Based on: Zhao et al. (2023) "Learning Fine-Grained Bimanual Manipulation
  with Low-Cost Hardware."  arXiv:2304.13705
Reference code: https://github.com/Shaka-Labs/ACT

Architecture summary
--------------------
Training
  1. CVAE encoder  : [CLS, obs_embed, action_embeds] → (mu, log_var)
  2. Reparameterise: z ~ N(mu, exp(0.5*log_var))
  3. Policy decoder: (obs_embed, z_embed) → K predicted actions
  Loss = L1(pred, target) + kl_weight * KL(q||p)

Inference
  z = 0  (prior mean)
  Policy decoder: (obs_embed, 0) → K predicted actions
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (batch_first)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape: (1, max_len, d_model)  — broadcast over batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# ACT Policy
# ---------------------------------------------------------------------------

class ACTPolicy(nn.Module):
    """
    Action Chunking with Transformers policy.

    Parameters
    ----------
    obs_dim   : dimension of the (flat) observation vector
    action_dim: dimension of one action step
    config    : dict with keys chunk_size, hidden_dim, nheads,
                num_encoder_layers, num_decoder_layers, dim_feedforward,
                latent_dim, dropout, kl_weight
    """

    def __init__(self, obs_dim: int, action_dim: int, config: dict):
        super().__init__()

        K = config["chunk_size"]
        H = config["hidden_dim"]
        Z = config["latent_dim"]
        nheads = config["nheads"]
        d_ff = config["dim_feedforward"]
        drop = config["dropout"]
        n_enc = config["num_encoder_layers"]
        n_dec = config["num_decoder_layers"]

        self.chunk_size = K
        self.latent_dim = Z
        self.kl_weight = config["kl_weight"]

        # ---- Shared observation projection ---------------------------------
        self.obs_proj = nn.Linear(obs_dim, H)

        # ---- CVAE encoder --------------------------------------------------
        # Processes [CLS, obs_embed, action_embeds]  →  (mu, log_var)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, H))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.action_proj = nn.Linear(action_dim, H)
        self.cvae_pos_enc = PositionalEncoding(H, max_len=K + 2, dropout=drop)

        cvae_enc_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nheads, dim_feedforward=d_ff,
            dropout=drop, batch_first=True, norm_first=True,
        )
        self.cvae_encoder = nn.TransformerEncoder(
            cvae_enc_layer, num_layers=n_enc, enable_nested_tensor=False
        )
        self.latent_mu = nn.Linear(H, Z)
        self.latent_logvar = nn.Linear(H, Z)

        # ---- Policy encoder (conditions the decoder) -----------------------
        # Processes [obs_embed, z_embed]  →  memory
        self.latent_proj = nn.Linear(Z, H)
        self.policy_pos_enc = PositionalEncoding(H, max_len=K + 2, dropout=drop)

        pol_enc_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nheads, dim_feedforward=d_ff,
            dropout=drop, batch_first=True, norm_first=True,
        )
        self.policy_encoder = nn.TransformerEncoder(
            pol_enc_layer, num_layers=n_enc, enable_nested_tensor=False
        )

        # ---- Policy decoder ------------------------------------------------
        # K learnable query tokens, cross-attend to memory  →  K actions
        self.query_embed = nn.Embedding(K, H)
        self.query_pos_enc = PositionalEncoding(H, max_len=K, dropout=drop)

        pol_dec_layer = nn.TransformerDecoderLayer(
            d_model=H, nhead=nheads, dim_feedforward=d_ff,
            dropout=drop, batch_first=True, norm_first=True,
        )
        self.policy_decoder = nn.TransformerDecoder(pol_dec_layer, num_layers=n_dec)

        # ---- Output head ---------------------------------------------------
        self.action_head = nn.Linear(H, action_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "cls_token" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # CVAE encode
    # ------------------------------------------------------------------

    def _cvae_encode(
        self,
        obs: torch.Tensor,                # (B, obs_dim)
        actions: torch.Tensor | None,     # (B, K, action_dim) or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, log_var), each (B, latent_dim)."""
        B = obs.shape[0]
        device = obs.device

        if actions is None:
            # Inference: use the prior  N(0, I)
            mu = torch.zeros(B, self.latent_dim, device=device)
            logvar = torch.zeros(B, self.latent_dim, device=device)
            return mu, logvar

        obs_emb = self.obs_proj(obs).unsqueeze(1)           # (B, 1, H)
        act_emb = self.action_proj(actions)                  # (B, K, H)
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, H)

        # [CLS | obs | a_0 … a_{K-1}]  →  (B, K+2, H)
        seq = torch.cat([cls, obs_emb, act_emb], dim=1)
        seq = self.cvae_pos_enc(seq)

        enc_out = self.cvae_encoder(seq)                     # (B, K+2, H)
        cls_out = enc_out[:, 0, :]                           # (B, H)

        mu = self.latent_mu(cls_out)                         # (B, Z)
        logvar = self.latent_logvar(cls_out)                 # (B, Z)
        return mu, logvar

    # ------------------------------------------------------------------
    # Reparameterisation trick
    # ------------------------------------------------------------------

    @staticmethod
    def _reparameterise(
        mu: torch.Tensor, logvar: torch.Tensor, training: bool
    ) -> torch.Tensor:
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu                # at inference use the mean

    # ------------------------------------------------------------------
    # Policy decode
    # ------------------------------------------------------------------

    def _policy_decode(
        self,
        obs: torch.Tensor,   # (B, obs_dim)
        z: torch.Tensor,     # (B, Z)
    ) -> torch.Tensor:       # (B, K, action_dim)
        B = obs.shape[0]

        obs_emb = self.obs_proj(obs).unsqueeze(1)            # (B, 1, H)
        z_emb = self.latent_proj(z).unsqueeze(1)             # (B, 1, H)

        enc_input = torch.cat([obs_emb, z_emb], dim=1)       # (B, 2, H)
        enc_input = self.policy_pos_enc(enc_input)
        memory = self.policy_encoder(enc_input)               # (B, 2, H)

        # Learnable decoder queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, K, H)
        queries = self.query_pos_enc(queries)

        dec_out = self.policy_decoder(queries, memory)        # (B, K, H)
        return self.action_head(dec_out)                      # (B, K, action_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: torch.Tensor,               # (B, obs_dim)
        actions: torch.Tensor | None = None,  # (B, K, action_dim) — training only
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        pred_actions : (B, K, action_dim)
        mu           : (B, latent_dim)
        logvar       : (B, latent_dim)
        """
        mu, logvar = self._cvae_encode(obs, actions)
        z = self._reparameterise(mu, logvar, self.training)
        pred_actions = self._policy_decode(obs, z)
        return pred_actions, mu, logvar

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        obs: torch.Tensor,      # (B, obs_dim)
        actions: torch.Tensor,  # (B, K, action_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total_loss : scalar
        recon_loss : scalar   (L1 reconstruction)
        kl_loss    : scalar   (KL divergence)
        """
        pred_actions, mu, logvar = self.forward(obs, actions)

        recon_loss = F.l1_loss(pred_actions, actions)

        # KL divergence  D_KL( N(mu, sigma) || N(0,I) )
        kl_loss = -0.5 * torch.mean(
            1.0 + logvar - mu.pow(2) - logvar.exp()
        )

        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, obs: np.ndarray, device: str | torch.device = "cpu") -> np.ndarray:
        """
        Predict a chunk of K actions from a single flat observation.

        Parameters
        ----------
        obs    : (obs_dim,) numpy array
        device : torch device string or object

        Returns
        -------
        actions : (K, action_dim) numpy array — clipped to [-1, 1]
        """
        self.eval()
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        pred, _, _ = self.forward(obs_t, actions=None)
        actions = pred.squeeze(0).cpu().numpy()
        return np.clip(actions, -1.0, 1.0)
