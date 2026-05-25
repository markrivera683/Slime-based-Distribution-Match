from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch


NamedTensorIterable = Iterable[tuple[str, torch.Tensor]]


@torch.no_grad()
def snapshot_named_tensors(source: NamedTensorIterable) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for name, tensor in source:
        if not torch.is_floating_point(tensor):
            continue
        snapshot[name] = tensor.detach().cpu().clone()
    return snapshot


@torch.no_grad()
def restore_named_tensors(source: NamedTensorIterable, snapshot: Mapping[str, torch.Tensor]) -> None:
    for name, tensor in source:
        if name not in snapshot and not torch.is_floating_point(tensor):
            continue
        if name not in snapshot:
            raise KeyError(f"EffOPD restore missing tensor {name!r}")
        tensor.copy_(snapshot[name].to(device=tensor.device, dtype=tensor.dtype), non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def apply_extrapolation_from_snapshots(
    source: NamedTensorIterable,
    *,
    base: Mapping[str, torch.Tensor],
    previous: Mapping[str, torch.Tensor],
    scale: float,
) -> float:
    """Apply base + scale * (base - previous) to live tensors.

    Returns the local squared L2 norm of the displacement. The caller may all-reduce
    this value if a global norm is needed.
    """

    delta_norm_sq = 0.0
    for name, tensor in source:
        if not torch.is_floating_point(tensor):
            continue
        if name not in base:
            raise KeyError(f"EffOPD base snapshot missing tensor {name!r}")
        if name not in previous:
            raise KeyError(f"EffOPD previous snapshot missing tensor {name!r}")
        base_tensor = base[name].to(device=tensor.device, dtype=torch.float32)
        prev_tensor = previous[name].to(device=tensor.device, dtype=torch.float32)
        delta = base_tensor - prev_tensor
        delta_norm_sq += float(delta.float().pow(2).sum().detach().cpu().item())
        candidate = base_tensor + float(scale) * delta
        tensor.copy_(candidate.to(dtype=tensor.dtype), non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return delta_norm_sq


def should_include_parameter(name: str, *, include_patterns: list[str], exclude_patterns: list[str]) -> bool:
    import re

    if include_patterns and not any(re.search(pattern, name) for pattern in include_patterns):
        return False
    if exclude_patterns and any(re.search(pattern, name) for pattern in exclude_patterns):
        return False
    return True
