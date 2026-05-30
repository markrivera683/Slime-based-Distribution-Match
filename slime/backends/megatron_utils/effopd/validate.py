from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch

from slime.utils.g2_core import log_probs_to_group_scores


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


def _per_sample_float_tensor(values: list[Any], *, device: torch.device) -> torch.Tensor | None:
    if not values:
        return None
    scalars = []
    for value in values:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().float().reshape(-1).to(device=device)
            if tensor.numel() == 0:
                return None
            scalars.append(tensor.mean())
        else:
            scalars.append(torch.tensor(float(value), dtype=torch.float32, device=device))
    if not scalars:
        return None
    return torch.stack(scalars, dim=0)


def align_dv_indices_to_prompt_groups(
    *,
    indices: list[int],
    num_samples: int,
    n_samples_per_prompt: int,
    max_size: int | None = None,
) -> list[int]:
    """Expand sample-level D_v indices so prompt groups remain intact."""

    if not indices:
        return []
    group_size = int(n_samples_per_prompt)
    if group_size <= 1:
        aligned = sorted({int(index) for index in indices if 0 <= int(index) < int(num_samples)})
        return aligned[: int(max_size)] if max_size is not None else aligned
    if max_size is not None and int(max_size) < group_size:
        raise ValueError(
            "EffOPD D_v size must be at least n_samples_per_prompt for opd_onpolicy combined_gate; "
            f"got effopd_dv_size={int(max_size)} and n_samples_per_prompt={group_size}."
        )
    if int(num_samples) <= 0:
        return []

    groups = sorted({int(index) // group_size for index in indices if 0 <= int(index) < int(num_samples)})
    aligned: list[int] = []
    for group_idx in groups:
        group_start = group_idx * group_size
        group_end = min(group_start + group_size, int(num_samples))
        if group_end - group_start == group_size:
            if max_size is not None and len(aligned) + group_size > int(max_size):
                break
            aligned.extend(range(group_start, group_end))
    return aligned


def select_dv_indices(
    *,
    num_samples: int,
    dv_size: int,
    seed: int,
    existing_indices: list[int] | None = None,
    n_samples_per_prompt: int = 1,
    require_complete_prompt_groups: bool = False,
) -> list[int]:
    """Return stable D_v indices for the current rollout-sized population."""

    if num_samples <= 0:
        return []
    sample_size = min(max(int(dv_size), 0), int(num_samples))
    group_size = int(n_samples_per_prompt)
    if require_complete_prompt_groups and group_size > 1:
        if int(dv_size) < group_size:
            raise ValueError(
                "EffOPD D_v size must be at least n_samples_per_prompt for opd_onpolicy combined_gate; "
                f"got effopd_dv_size={int(dv_size)} and n_samples_per_prompt={group_size}."
            )
        num_complete_groups = int(num_samples) // group_size
        if num_complete_groups <= 0:
            return []
        max_groups = min(num_complete_groups, int(dv_size) // group_size)
        if existing_indices:
            aligned = align_dv_indices_to_prompt_groups(
                indices=existing_indices,
                num_samples=num_samples,
                n_samples_per_prompt=group_size,
                max_size=int(dv_size),
            )
            existing_groups = sorted({index // group_size for index in aligned})
            if len(existing_groups) >= max_groups:
                return [
                    index
                    for group_idx in existing_groups[:max_groups]
                    for index in range(group_idx * group_size, group_idx * group_size + group_size)
                ]
        groups = list(range(num_complete_groups))
        rng = random.Random(int(seed))
        rng.shuffle(groups)
        selected_groups = sorted(groups[:max_groups])
        return [
            index
            for group_idx in selected_groups
            for index in range(group_idx * group_size, group_idx * group_size + group_size)
        ]
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


def _candidate_weighted_cf_reward_mean(
    args,
    *,
    cf_rewards: list[Any] | None,
    candidate_log_probs: list[torch.Tensor] | None,
    strict: bool = False,
) -> float | None:
    if not cf_rewards or candidate_log_probs is None:
        if strict and candidate_log_probs is not None:
            raise ValueError("EffOPD combined_gate opd_onpolicy requires cf_l1oo rewards for the weighted CF proxy.")
        return None

    n_samples_per_prompt = getattr(args, "n_samples_per_prompt", None)
    if n_samples_per_prompt is None:
        if strict:
            raise ValueError("EffOPD combined_gate opd_onpolicy requires n_samples_per_prompt.")
        return None
    n_samples_per_prompt = int(n_samples_per_prompt)
    if n_samples_per_prompt <= 0:
        if strict:
            raise ValueError(
                "EffOPD combined_gate opd_onpolicy requires n_samples_per_prompt to be positive; "
                f"got {n_samples_per_prompt}."
            )
        return None

    num_samples = len(candidate_log_probs)
    if num_samples == 0 or num_samples % n_samples_per_prompt != 0:
        if strict:
            raise ValueError(
                "EffOPD combined_gate opd_onpolicy requires candidate log_probs to contain complete prompt groups; "
                f"got {num_samples} samples with n_samples_per_prompt={n_samples_per_prompt}."
            )
        return None
    if len(cf_rewards) != num_samples:
        if strict:
            raise ValueError(
                "EffOPD combined_gate opd_onpolicy requires one cf_l1oo reward per candidate log_prob; "
                f"got {len(cf_rewards)} rewards and {num_samples} log_probs."
            )
        return None

    device = (
        candidate_log_probs[0].device
        if isinstance(candidate_log_probs[0], torch.Tensor)
        else torch.device("cpu")
    )
    rewards = _per_sample_float_tensor(cf_rewards, device=device)
    if rewards is None or rewards.numel() != num_samples:
        if strict:
            got = 0 if rewards is None else int(rewards.numel())
            raise ValueError(
                "EffOPD combined_gate opd_onpolicy requires scalar cf_l1oo rewards aligned with candidate log_probs; "
                f"got {got} rewards and {num_samples} log_probs."
            )
        return None

    normalization = str(getattr(args, "opd_cf_score_normalization", "mean"))
    scores = log_probs_to_group_scores(
        candidate_log_probs,
        num_groups=num_samples // n_samples_per_prompt,
        n_samples_per_prompt=n_samples_per_prompt,
        device=device,
        normalization=normalization,
        label="EffOPD candidate_log_probs",
    ).squeeze(0)
    weights = torch.softmax(scores, dim=1)
    grouped_rewards = rewards.view(-1, n_samples_per_prompt)
    return float((weights * grouped_rewards).sum(dim=1).mean().item())


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
    strict_weighted_cf_proxy: bool = False,
) -> EffOPDValidationScore:
    validation_mode = mode or getattr(args, "effopd_validation_mode", "combined_gate")
    candidate_weighted_cf_reward = None
    if validation_mode == "combined_gate":
        candidate_weighted_cf_reward = _candidate_weighted_cf_reward_mean(
            args,
            cf_rewards=cf_rewards,
            candidate_log_probs=student_log_probs,
            strict=bool(strict_weighted_cf_proxy),
        )
    cf_reward = (
        candidate_weighted_cf_reward
        if candidate_weighted_cf_reward is not None
        else _mean_float(cf_rewards or []) or 0.0
    )
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
    mode = getattr(args, "effopd_validation_mode", "combined_gate")
    score = combined if mode == "combined_gate" else -opd_reverse_kl
    return EffOPDValidationScore(
        score=float(score),
        combined_proxy=float(combined),
        cf_l1oo_reward_mean=float(cf_reward),
        opd_reverse_kl_mean=float(opd_reverse_kl),
    )
