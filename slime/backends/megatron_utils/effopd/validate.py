from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch


@dataclass
class EffOPDValidationScore:
    score: float
    combined_proxy: float
    cf_l1oo_reward_mean: float
    opd_reverse_kl_mean: float


def _mean_float(values: list[Any]) -> float | None:
    if not values:
        return None
    tensors = []
    for value in values:
        if isinstance(value, torch.Tensor):
            tensors.append(value.detach().float().reshape(-1).cpu())
        else:
            tensors.append(torch.tensor([float(value)], dtype=torch.float32))
    if not tensors:
        return None
    merged = torch.cat(tensors)
    if merged.numel() == 0:
        return None
    return float(merged.mean().item())


def select_dv_indices(
    *,
    num_samples: int,
    dv_size: int,
    seed: int,
    existing_indices: list[int] | None = None,
) -> list[int]:
    """Return stable D_v indices for the current rollout-sized population."""

    if num_samples <= 0:
        return []
    sample_size = min(max(int(dv_size), 0), int(num_samples))
    if (
        existing_indices
        and len(existing_indices) >= sample_size
        and all(0 <= int(index) < num_samples for index in existing_indices)
    ):
        return [int(index) for index in existing_indices[:sample_size]]
    indices = list(range(num_samples))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    return sorted(indices[:sample_size])


def slice_rollout_data_for_indices(rollout_data: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    """Slice per-sample rollout fields while leaving scalar/config fields untouched."""

    if not indices:
        return dict(rollout_data)
    num_samples = len(rollout_data.get("response_lengths") or rollout_data.get("tokens") or [])
    sliced: dict[str, Any] = {}
    for key, value in rollout_data.items():
        if isinstance(value, list) and len(value) == num_samples:
            sliced[key] = [value[index] for index in indices]
        else:
            sliced[key] = value
    return sliced


def score_from_terms(
    args,
    *,
    cf_rewards: list[Any] | None,
    teacher_log_probs: list[Any] | None,
    student_log_probs: list[Any] | None,
    mode: str | None = None,
) -> EffOPDValidationScore:
    cf_reward = _mean_float(cf_rewards or []) or 0.0
    reverse_kls = []
    if teacher_log_probs is not None and student_log_probs is not None:
        for student, teacher in zip(student_log_probs, teacher_log_probs, strict=True):
            student_tensor = student.detach().float()
            teacher_tensor = teacher.detach().float().to(device=student_tensor.device)
            if student_tensor.numel() != teacher_tensor.numel():
                raise ValueError(
                    "EffOPD D_v evaluator requires student/teacher logprob lengths to match; "
                    f"got {student_tensor.numel()} and {teacher_tensor.numel()}."
                )
            reverse_kls.append((student_tensor.reshape(-1) - teacher_tensor.reshape(-1)).cpu())
    opd_reverse_kl = _mean_float(reverse_kls) or 0.0
    combined = cf_reward - float(getattr(args, "opd_kl_coef", 1.0)) * opd_reverse_kl

    validation_mode = mode or getattr(args, "effopd_validation_mode", "opd_kl_shadow_cf")
    score = combined if validation_mode == "combined_gate" else -opd_reverse_kl
    return EffOPDValidationScore(
        score=float(score),
        combined_proxy=float(combined),
        cf_l1oo_reward_mean=float(cf_reward),
        opd_reverse_kl_mean=float(opd_reverse_kl),
    )


def score_from_rollout_data(args, rollout_data: dict[str, Any]) -> EffOPDValidationScore:
    """Score existing rollout terms for G2 cf_l1oo + SGLang OPD.

    G2 is logged as cf_l1oo reward; OPD is logged as reverse KL. The combined
    proxy mirrors how training adds the G2 reward signal and subtracts the OPD
    KL penalty, while keeping both terms separately visible in logs.
    """

    if rollout_data.get("teacher_log_probs") is not None and rollout_data.get("log_probs") is not None:
        return score_from_terms(
            args,
            cf_rewards=rollout_data.get("rewards"),
            teacher_log_probs=rollout_data.get("teacher_log_probs"),
            student_log_probs=rollout_data.get("log_probs"),
        )

    cf_reward = _mean_float(rollout_data.get("rewards") or []) or 0.0
    opd_reverse_kl = _mean_float(rollout_data.get("opd_reverse_kl") or []) or 0.0
    combined = cf_reward - float(getattr(args, "opd_kl_coef", 1.0)) * opd_reverse_kl
    mode = getattr(args, "effopd_validation_mode", "opd_kl_shadow_cf")
    score = combined if mode == "combined_gate" else -opd_reverse_kl
    return EffOPDValidationScore(
        score=float(score),
        combined_proxy=float(combined),
        cf_l1oo_reward_mean=float(cf_reward),
        opd_reverse_kl_mean=float(opd_reverse_kl),
    )
