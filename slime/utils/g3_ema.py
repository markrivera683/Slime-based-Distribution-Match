from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from slime.utils.g2_core import _compute_cf_loss_terms, _get_fixed_cf_frequencies


class G3FeatureAdapter(nn.Module):
    """Residual feature adapter for G3 OPD-fused EMA experiments."""

    def __init__(self, feature_dim: int, rank: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        if int(rank) <= 0:
            raise ValueError("rank must be positive")
        if float(dropout) < 0.0 or float(dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.feature_dim = int(feature_dim)
        self.rank = int(rank)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.down = nn.Linear(self.feature_dim, self.rank)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(self.rank, self.feature_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.down.reset_parameters()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected last dim {self.feature_dim}, got {features.shape[-1]}")
        residual = features
        adapted = self.norm(features)
        adapted = self.down(adapted)
        adapted = self.activation(adapted)
        adapted = self.dropout(adapted)
        adapted = self.up(adapted)
        return residual + adapted


@torch.no_grad()
def copy_live_to_ema(live: nn.Module, ema: nn.Module) -> None:
    """Initialize EMA parameters and buffers from the live module."""
    ema.load_state_dict(live.state_dict())
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)


@torch.no_grad()
def update_ema_parameters(live: nn.Module, ema: nn.Module, beta: float) -> None:
    """Apply ema = beta * ema + (1 - beta) * live to floating-point EMA tensors."""
    if float(beta) < 0.0 or float(beta) > 1.0:
        raise ValueError("beta must be in [0, 1]")

    live_state = live.state_dict()
    ema_state = ema.state_dict()
    if live_state.keys() != ema_state.keys():
        raise ValueError("live and ema modules must have matching state_dict keys")

    beta = float(beta)
    one_minus_beta = 1.0 - beta
    for name, ema_tensor in ema_state.items():
        live_tensor = live_state[name].to(device=ema_tensor.device)
        if torch.is_floating_point(ema_tensor):
            ema_tensor.mul_(beta).add_(live_tensor.to(dtype=ema_tensor.dtype), alpha=one_minus_beta)
        else:
            ema_tensor.copy_(live_tensor)

    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    ema.eval()


def iter_trainable_adapter_parameters(adapter: nn.Module) -> Iterable[nn.Parameter]:
    """Return the trainable G3 adapter parameter set for optimizer construction."""
    return (parameter for parameter in adapter.parameters() if parameter.requires_grad)


def get_trainable_adapter_parameters(adapter: nn.Module) -> list[nn.Parameter]:
    """Materialize trainable G3 adapter parameters for optimizer construction/tests."""
    return list(iter_trainable_adapter_parameters(adapter))


def g3_feature_mse_loss(live_features: torch.Tensor, ema_features: torch.Tensor) -> torch.Tensor:
    """Differentiable live-vs-detached-EMA feature loss."""
    if live_features.shape != ema_features.shape:
        raise ValueError(f"live/ema feature shapes must match, got {live_features.shape} vs {ema_features.shape}")
    return torch.nn.functional.mse_loss(live_features, ema_features.detach())


def _prepare_opd_teacher_weights(
    teacher_scores: torch.Tensor,
    *,
    batch_size: int,
    num_groups: int,
    n_samples: int,
    num_blocks: int,
    device: torch.device,
    score_temperature: float,
) -> torch.Tensor:
    if teacher_scores.ndim not in {3, 4}:
        raise ValueError(
            "G3 OPD-CF feature loss expects teacher_scores shaped (B, G, N) or (B, G, N, K), "
            f"got {tuple(teacher_scores.shape)}"
        )
    if teacher_scores.shape[:3] != (batch_size, num_groups, n_samples):
        raise ValueError(
            "G3 OPD-CF feature loss teacher_scores must align with embeddings on B/G/N dims, "
            f"got scores={tuple(teacher_scores.shape)} embeddings={(batch_size, num_groups, n_samples, num_blocks)}"
        )
    if teacher_scores.ndim == 3:
        teacher_scores = teacher_scores.unsqueeze(-1).expand(-1, -1, -1, num_blocks)
    elif teacher_scores.shape[3] != num_blocks:
        raise ValueError(
            "G3 OPD-CF feature loss block-level teacher_scores must align with embedding K dim, "
            f"got scores={tuple(teacher_scores.shape)} K={num_blocks}"
        )

    temperature = float(score_temperature)
    if temperature <= 0.0:
        raise ValueError("G3 OPD-CF feature loss score_temperature must be positive.")

    scores = teacher_scores.detach().float().to(device=device) / temperature
    return torch.softmax(scores, dim=2).detach()


def g3_opd_cf_feature_loss(
    live_embedding: torch.Tensor,
    ema_target_embedding: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    cf_num_freqs: int = 128,
    cf_sigma: float = 1.0,
    cf_seed: int = 43,
    cf_alpha: float = 0.5,
    cf_beta: float = 0.5,
    score_temperature: float = 1.0,
) -> torch.Tensor:
    """Differentiable OPD-CF loss from live features to detached EMA target geometry.

    Shapes:
    - live_embedding / ema_target_embedding: (B, G, N, K, D)
    - teacher_scores: (B, G, N) or (B, G, N, K)

    The live side is the uniform student empirical distribution and remains
    differentiable. The target side uses the EMA feature support and detached
    teacher-score softmax weights, so gradients never flow into EMA target
    embeddings or teacher scores.
    """
    if live_embedding.ndim != 5 or ema_target_embedding.ndim != 5:
        raise ValueError(
            "G3 OPD-CF feature loss expects embeddings shaped (B, G, N, K, D), "
            f"got live={tuple(live_embedding.shape)} ema={tuple(ema_target_embedding.shape)}"
        )
    if live_embedding.shape != ema_target_embedding.shape:
        raise ValueError(
            "G3 OPD-CF feature loss live/EMA embedding shapes must match, "
            f"got {tuple(live_embedding.shape)} vs {tuple(ema_target_embedding.shape)}"
        )

    batch_size, num_groups, n_samples, num_blocks, feat_dim = live_embedding.shape
    if n_samples <= 1:
        raise ValueError("G3 OPD-CF feature loss requires N > 1 student rollouts per prompt group.")

    weights = _prepare_opd_teacher_weights(
        teacher_scores,
        batch_size=batch_size,
        num_groups=num_groups,
        n_samples=n_samples,
        num_blocks=num_blocks,
        device=live_embedding.device,
        score_temperature=score_temperature,
    )

    live_flat = live_embedding.permute(0, 1, 3, 2, 4).reshape(-1, n_samples, feat_dim).float()
    target_flat = ema_target_embedding.detach().permute(0, 1, 3, 2, 4).reshape(-1, n_samples, feat_dim).float()
    weight_flat = weights.permute(0, 1, 3, 2).reshape(-1, n_samples).to(device=live_flat.device)

    freqs = _get_fixed_cf_frequencies(
        input_dim=feat_dim,
        num_freqs=int(cf_num_freqs),
        sigma=float(cf_sigma),
        seed=int(cf_seed),
        device=live_flat.device,
    )

    live_proj = torch.einsum("fd,bnd->bfn", freqs, live_flat)
    live_real = torch.cos(live_proj).mean(dim=-1)
    live_imag = torch.sin(live_proj).mean(dim=-1)

    target_proj = torch.einsum("fd,bnd->bfn", freqs, target_flat)
    target_real_vals = torch.cos(target_proj)
    target_imag_vals = torch.sin(target_proj)
    target_real = (target_real_vals * weight_flat.unsqueeze(1)).sum(dim=-1)
    target_imag = (target_imag_vals * weight_flat.unsqueeze(1)).sum(dim=-1)

    return _compute_cf_loss_terms(
        target_real,
        target_imag,
        live_real,
        live_imag,
        cf_alpha,
        cf_beta,
    ).mean()


@dataclass
class G3EMAFeatureStepResult:
    """Detached metrics from one local G3 adapter/EMA feature step."""

    loss: torch.Tensor
    raw_feature_loss: torch.Tensor


class G3EMAAdapterController:
    """Local critic-side skeleton for live adapter, EMA adapter, optimizer, and EMA update."""

    def __init__(
        self,
        *,
        live_adapter: nn.Module,
        ema_adapter: nn.Module,
        optimizer: torch.optim.Optimizer,
        ema_beta: float,
        feature_loss_coef: float,
    ) -> None:
        if float(ema_beta) < 0.0 or float(ema_beta) > 1.0:
            raise ValueError("ema_beta must be in [0, 1]")
        if float(feature_loss_coef) < 0.0:
            raise ValueError("feature_loss_coef must be non-negative")
        self.live_adapter = live_adapter
        self.ema_adapter = ema_adapter
        self.optimizer = optimizer
        self.ema_beta = float(ema_beta)
        self.feature_loss_coef = float(feature_loss_coef)
        for parameter in self.ema_adapter.parameters():
            parameter.requires_grad_(False)
        self.ema_adapter.eval()

    @classmethod
    def create(
        cls,
        *,
        feature_dim: int,
        rank: int = 64,
        dropout: float = 0.0,
        lr: float = 5e-5,
        ema_beta: float = 0.99,
        feature_loss_coef: float = 0.1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "G3EMAAdapterController":
        if float(lr) <= 0.0:
            raise ValueError("lr must be positive")
        live_adapter = G3FeatureAdapter(feature_dim=feature_dim, rank=rank, dropout=dropout)
        ema_adapter = G3FeatureAdapter(feature_dim=feature_dim, rank=rank, dropout=dropout)
        if device is not None or dtype is not None:
            live_adapter = live_adapter.to(device=device, dtype=dtype)
            ema_adapter = ema_adapter.to(device=device, dtype=dtype)
        copy_live_to_ema(live_adapter, ema_adapter)
        optimizer = torch.optim.AdamW(get_trainable_adapter_parameters(live_adapter), lr=float(lr))
        return cls(
            live_adapter=live_adapter,
            ema_adapter=ema_adapter,
            optimizer=optimizer,
            ema_beta=ema_beta,
            feature_loss_coef=feature_loss_coef,
        )

    def train_feature_step(
        self,
        base_features: torch.Tensor,
        teacher_scores: torch.Tensor,
        *,
        cf_num_freqs: int = 128,
        cf_sigma: float = 1.0,
        cf_seed: int = 43,
        cf_alpha: float = 0.5,
        cf_beta: float = 0.5,
        score_temperature: float = 1.0,
    ) -> G3EMAFeatureStepResult:
        """Run one locally differentiable adapter step and then update EMA."""
        self.live_adapter.train()
        self.ema_adapter.eval()
        self.optimizer.zero_grad(set_to_none=True)

        frozen_base_features = base_features.detach()
        live_embedding = self.live_adapter(frozen_base_features)
        with torch.no_grad():
            ema_target_embedding = self.ema_adapter(frozen_base_features)

        raw_loss = g3_opd_cf_feature_loss(
            live_embedding,
            ema_target_embedding,
            teacher_scores,
            cf_num_freqs=cf_num_freqs,
            cf_sigma=cf_sigma,
            cf_seed=cf_seed,
            cf_alpha=cf_alpha,
            cf_beta=cf_beta,
            score_temperature=score_temperature,
        )
        loss = raw_loss * self.feature_loss_coef
        loss.backward()
        self.optimizer.step()
        update_ema_parameters(self.live_adapter, self.ema_adapter, self.ema_beta)
        return G3EMAFeatureStepResult(loss=loss.detach(), raw_feature_loss=raw_loss.detach())

    def state_dict(self) -> dict[str, Any]:
        return g3_adapter_checkpoint_state(self.live_adapter, self.ema_adapter, self.optimizer)

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        load_g3_adapter_checkpoint_state(
            self.live_adapter,
            self.ema_adapter,
            self.optimizer,
            state,
            strict=strict,
        )


def g3_adapter_checkpoint_state(
    live_adapter: nn.Module,
    ema_adapter: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Return the checkpoint payload for G3 adapter-only training state."""
    state: dict[str, Any] = {
        "live_adapter": live_adapter.state_dict(),
        "ema_adapter": ema_adapter.state_dict(),
    }
    if optimizer is not None:
        state["adapter_optimizer"] = optimizer.state_dict()
    return state


def load_g3_adapter_checkpoint_state(
    live_adapter: nn.Module,
    ema_adapter: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    state: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    """Load adapter-only checkpoint state, initializing missing EMA from live state."""
    if "live_adapter" not in state:
        raise KeyError("G3 adapter checkpoint is missing 'live_adapter'")
    live_adapter.load_state_dict(state["live_adapter"], strict=strict)
    if "ema_adapter" in state:
        ema_adapter.load_state_dict(state["ema_adapter"], strict=strict)
        ema_adapter.eval()
        for parameter in ema_adapter.parameters():
            parameter.requires_grad_(False)
    else:
        copy_live_to_ema(live_adapter, ema_adapter)

    if optimizer is not None and "adapter_optimizer" in state:
        optimizer.load_state_dict(state["adapter_optimizer"])


def select_g1_trainer_sync_source(args: Any) -> int | None:
    """Return the actor/critic broadcast source for trainer-side G1/G2/G3 rewards."""
    is_cf_l1oo = getattr(args, "distribution_reward_type", "pointwise") == "cf_l1oo"
    if (
        not is_cf_l1oo
        or getattr(args, "advantage_estimator", None) != "g1"
        or getattr(args, "g1_reward_location", None) != "trainer"
    ):
        return None

    if bool(getattr(args, "g3_enable", False)):
        return 1
    return 1 if getattr(args, "cf_target_mode", None) == "teacher" else 0


def is_g3_opd_fused_mode(args: Any) -> bool:
    """Return whether args request critic-owned G3 OPD-fused EMA mode."""
    return (
        bool(getattr(args, "g3_enable", False))
        and getattr(args, "distribution_reward_type", "pointwise") == "cf_l1oo"
        and getattr(args, "cf_target_mode", None) == "opd_onpolicy"
        and bool(getattr(args, "use_opd", False))
    )


def raise_if_g3_detached_reward_path(args: Any) -> None:
    if not bool(getattr(args, "g3_enable", False)):
        return
    # Keep G3 blocked on the detached trainer-side reward path until the critic
    # closure owns live adapter forward, EMA target forward, optimizer step, and
    # post-step EMA update in one differentiable path. TODO(g3): insert that
    # closure in MegatronTrainRayActor.train_critic after no-grad hidden/value
    # extraction and before sync_actor_critic_data, then broadcast critic-owned
    # token advantages/rewards after the adapter step has updated EMA.
    raise NotImplementedError(
        "G3 OPD-fused EMA requires a critic-side differentiable adapter/EMA training closure. "
        "The current trainer-side G1/G2 reward path consumes detached embeddings and cannot train the "
        "feature adapter safely yet. TODO(g3): add the critic closure in train_critic after hidden/value "
        "extraction and before actor/critic reward sync so it computes adapter loss, steps the adapter "
        "optimizer, updates EMA, and publishes token advantages/rewards from the critic."
    )
