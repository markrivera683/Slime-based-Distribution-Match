#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import torch


OPENRLHF_EMBEDDING_UTILS = Path("/mnt/data/ebft-distribution-new/code/openrlhf/utils/embedding_utils.py")
DEFAULT_THRESHOLD = 1e-5

ALIASES = {
    "raw_student_gen": [
        "g2_raw_student_gen_tensor",
        "raw_student_gen_embeddings",
        "student_gen_embeddings_raw",
        "gen_embeddings_raw",
    ],
    "raw_student_gt": [
        "g2_raw_student_gt_tensor",
        "raw_student_gt_embeddings",
        "student_gt_embeddings_raw",
        "gt_embeddings_raw",
    ],
    "teacher_gen": [
        "g2_teacher_gen_tensor",
        "teacher_gen_embeddings",
        "g2_teacher_gen_embeddings_tensor",
    ],
    "whitened_student_gen": [
        "g2_whitened_student_gen_tensor",
        "whitened_student_gen_embeddings",
        "student_gen_embeddings_whitened",
        "gen_embeddings_whitened",
    ],
    "whitened_student_gt": [
        "g2_whitened_student_gt_tensor",
        "whitened_student_gt_embeddings",
        "student_gt_embeddings_whitened",
        "gt_embeddings_whitened",
    ],
    "cf_l1oo_rewards": [
        "g2_cf_l1oo_rewards",
        "cf_l1oo_rewards",
        "block_rewards",
    ],
    "token_advantages": [
        "token_advantages",
        "g1_token_advantages",
    ],
    "scalar_rewards": [
        "scalar_rewards",
        "rewards",
    ],
    "response_lengths": [
        "response_lengths",
    ],
}


def _install_optional_import_stubs() -> None:
    """Keep this parity script independent of OpenRLHF's distributed runtime deps."""
    sys.modules.setdefault("ray", types.ModuleType("ray"))
    pybase64 = sys.modules.setdefault("pybase64", types.ModuleType("pybase64"))
    if not hasattr(pybase64, "b64encode"):
        pybase64.b64encode = base64.b64encode
        pybase64.b64decode = base64.b64decode


def _load_openrlhf_embedding_utils(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"OpenRLHF embedding_utils.py not found: {path}")
    _install_optional_import_stubs()
    module_name = "_openrlhf_embedding_utils_for_g2_runtime_dump"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_dump(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dump to contain a dict, got {type(obj).__name__}: {path}")
    return obj


def _first_present(dump: dict[str, Any], canonical_key: str) -> tuple[str | None, Any | None]:
    for key in ALIASES.get(canonical_key, [canonical_key]):
        if key in dump:
            return key, dump[key]
    return None, None


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, torch.Tensor) for item in value):
        try:
            return torch.stack([item.detach().cpu() for item in value], dim=0)
        except RuntimeError:
            return torch.tensor([item.detach().cpu().flatten()[:8].tolist() for item in value], dtype=torch.float32)
    return torch.as_tensor(value)


def _as_float_tensor(value: Any) -> torch.Tensor:
    tensor = _as_tensor(value)
    if tensor.dtype == torch.bool:
        return tensor.to(dtype=torch.int32)
    if not torch.is_floating_point(tensor):
        return tensor.to(dtype=torch.float32)
    return tensor.float()


def _sample_values(tensor: torch.Tensor, limit: int) -> list[Any]:
    flat = tensor.detach().cpu().reshape(-1)
    return flat[:limit].tolist()


def _compare_tensor(
    *,
    section: str,
    key: str,
    expected: Any,
    actual: Any,
    threshold: float,
    sample_limit: int,
    expected_dump_key: str | None = None,
    actual_dump_key: str | None = None,
) -> dict[str, Any]:
    expected_tensor = _as_float_tensor(expected)
    actual_tensor = _as_float_tensor(actual)
    shape_match = tuple(expected_tensor.shape) == tuple(actual_tensor.shape)
    if shape_match:
        diff = (expected_tensor - actual_tensor).abs()
        max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
        mean_abs_diff = float(diff.mean().item()) if diff.numel() else 0.0
    else:
        max_abs_diff = None
        mean_abs_diff = None
    return {
        "section": section,
        "key": key,
        "passed": bool(shape_match and max_abs_diff is not None and max_abs_diff <= threshold),
        "threshold": threshold,
        "shape": {
            "expected": list(expected_tensor.shape),
            "actual": list(actual_tensor.shape),
            "match": shape_match,
        },
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "sample_values": {
            "expected": _sample_values(expected_tensor, sample_limit),
            "actual": _sample_values(actual_tensor, sample_limit),
        },
        "dump_keys": {
            "expected": expected_dump_key,
            "actual": actual_dump_key,
        },
    }


def _missing_result(section: str, key: str, *, side: str, aliases: list[str]) -> dict[str, Any]:
    return {
        "section": section,
        "key": key,
        "passed": False,
        "threshold": None,
        "shape": {"expected": [], "actual": [], "match": False},
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "sample_values": {"expected": [], "actual": []},
        "details": {"missing_side": side, "aliases_checked": aliases},
    }


def _shape_metadata(dump: dict[str, Any]) -> dict[str, Any]:
    meta = dump.get("g2_shape_metadata")
    return meta if isinstance(meta, dict) else {}


def _n_samples_per_prompt(dump: dict[str, Any]) -> int:
    meta = _shape_metadata(dump)
    value = meta.get("n_samples_per_prompt", dump.get("n_samples_per_prompt"))
    if value is None:
        raw_key, raw = _first_present(dump, "raw_student_gen")
        if raw_key is not None:
            raw_tensor = _as_tensor(raw)
            if raw_tensor.ndim == 5:
                return int(raw_tensor.shape[2])
        raise KeyError("Could not infer n_samples_per_prompt from dump metadata")
    return int(value)


def _raw_student_tensor_from_dump(dump: dict[str, Any], canonical_key: str) -> tuple[str | None, torch.Tensor | None]:
    key, value = _first_present(dump, canonical_key)
    if key is not None:
        tensor = _as_tensor(value)
        if tensor.ndim == 5:
            return key, tensor.float()

    list_key = "g2_raw_student_gen_embeddings" if canonical_key == "raw_student_gen" else "g2_raw_student_gt_embeddings"
    if list_key not in dump:
        return key, None
    flat = _as_tensor(dump[list_key]).float()
    if flat.ndim != 3:
        raise ValueError(f"{list_key} must stack to [samples, blocks, dim], got {tuple(flat.shape)}")
    n = _n_samples_per_prompt(dump)
    if flat.shape[0] % n != 0:
        raise ValueError(f"{list_key} sample count {flat.shape[0]} is not divisible by n_samples_per_prompt={n}")
    return list_key, flat.view(flat.shape[0] // n, n, flat.shape[1], flat.shape[2]).unsqueeze(0)


def _teacher_tensor_from_dump(dump: dict[str, Any]) -> tuple[str | None, torch.Tensor | None]:
    key, value = _first_present(dump, "teacher_gen")
    if key is not None:
        tensor = _as_tensor(value)
        if tensor.ndim == 5:
            return key, tensor.float()

    list_key = "g2_teacher_gen_embeddings"
    if list_key not in dump:
        return key, None
    flat = _as_tensor(dump[list_key]).float()
    if flat.ndim != 4:
        raise ValueError(f"{list_key} must stack to [samples, teacher_samples, blocks, dim], got {tuple(flat.shape)}")
    n = _n_samples_per_prompt(dump)
    if flat.shape[0] % n != 0:
        raise ValueError(f"{list_key} sample count {flat.shape[0]} is not divisible by n_samples_per_prompt={n}")
    groups = []
    for group_start in range(0, flat.shape[0], n):
        groups.append(flat[group_start])
    return list_key, torch.stack(groups, dim=0).unsqueeze(0)


def _cf_args(dump: dict[str, Any]) -> dict[str, Any]:
    args = dict(dump.get("g2_cf_args") or {})
    return {
        "cf_num_freqs": int(args.get("cf_num_freqs", dump.get("cf_num_freqs", 128))),
        "cf_sigma": float(args.get("cf_sigma", dump.get("cf_sigma", 1.0))),
        "cf_seed": int(args.get("cf_seed", dump.get("cf_seed", 43))),
        "cf_alpha": float(args.get("cf_alpha", dump.get("cf_alpha", 0.5))),
        "cf_beta": float(args.get("cf_beta", dump.get("cf_beta", 0.5))),
        "cf_reward_scale": float(args.get("cf_reward_scale", dump.get("cf_reward_scale", 1.0))),
        "cf_target_mode": str(args.get("cf_target_mode", dump.get("cf_target_mode", "teacher"))),
        "cf_target_num_refs": int(args.get("cf_target_num_refs", dump.get("cf_target_num_refs", 1))),
        "cf_target_std": float(args.get("cf_target_std", dump.get("cf_target_std", 0.05))),
        "cf_target_seed": int(args.get("cf_target_seed", args.get("cf_seed", dump.get("cf_seed", 43)))),
        "cf_teacher_lambda": float(args.get("cf_teacher_lambda", dump.get("cf_teacher_lambda", 0.0))),
    }


def _expected_token_advantages(block_rewards: torch.Tensor, response_lengths: list[int]) -> torch.Tensor:
    rewards = block_rewards.squeeze(0).reshape(-1, block_rewards.shape[-1]).float()
    rows = []
    for sample_idx, response_length in enumerate(response_lengths):
        num_blocks = int(rewards.shape[1])
        if int(response_length) % num_blocks != 0:
            raise ValueError(f"response_length {response_length} is not divisible by num_blocks={num_blocks}")
        rows.append(rewards[sample_idx].repeat(int(response_length) // num_blocks))
    return torch.stack(rows, dim=0)


def _make_tensor(shape: tuple[int, ...], *, offset: float, scale: float) -> torch.Tensor:
    total = 1
    for dim in shape:
        total *= int(dim)
    base = torch.arange(total, dtype=torch.float32).reshape(shape)
    return torch.sin(base * scale + offset) + 0.25 * torch.cos(base * scale * 0.7 - offset)


def _make_synthetic_slime_dump(openrlhf_utils: Any) -> dict[str, Any]:
    batch_size = 1
    num_groups = 2
    n_actor = 3
    n_teacher = 4
    num_blocks = 2
    feat_dim = 5
    generate_length = 2
    response_lengths = [num_blocks * generate_length] * (num_groups * n_actor)
    cf_args = {
        "cf_num_freqs": 17,
        "cf_sigma": 0.9,
        "cf_seed": 123,
        "cf_alpha": 0.35,
        "cf_beta": 0.65,
        "cf_reward_scale": 2.75,
        "cf_target_mode": "teacher",
        "cf_target_num_refs": 1,
        "cf_target_std": 0.05,
        "cf_target_seed": 123,
        "cf_teacher_lambda": 0.6,
    }

    raw_gen = _make_tensor((batch_size, num_groups, n_actor, num_blocks, feat_dim), offset=0.13, scale=0.071)
    raw_gt = _make_tensor((batch_size, num_groups, n_actor, num_blocks, feat_dim), offset=0.41, scale=0.053)
    teacher = _make_tensor((batch_size, num_groups, n_teacher, num_blocks, feat_dim), offset=-0.29, scale=0.067)
    white_gen, white_gt = openrlhf_utils.whiten_embeddings_batched(raw_gen, raw_gt, whiten_tol=1e-5, normalize=False)
    rewards = openrlhf_utils.get_cf_l1oo_rewards(
        white_gen,
        white_gt,
        cf_num_freqs=cf_args["cf_num_freqs"],
        cf_sigma=cf_args["cf_sigma"],
        cf_seed=cf_args["cf_seed"],
        cf_alpha=cf_args["cf_alpha"],
        cf_beta=cf_args["cf_beta"],
        cf_reward_scale=cf_args["cf_reward_scale"],
        cf_target_mode=cf_args["cf_target_mode"],
        cf_target_num_refs=cf_args["cf_target_num_refs"],
        cf_target_std=cf_args["cf_target_std"],
        cf_target_seed=cf_args["cf_target_seed"],
        teacher_embedding=teacher,
        cf_teacher_lambda=cf_args["cf_teacher_lambda"],
    )
    token_advantages = _expected_token_advantages(rewards, response_lengths)
    scalar_rewards = rewards.squeeze(0).reshape(-1, rewards.shape[-1]).mean(dim=-1)
    flat_gen = raw_gen.squeeze(0).reshape(-1, num_blocks, feat_dim)
    flat_gt = raw_gt.squeeze(0).reshape(-1, num_blocks, feat_dim)
    teacher_by_sample = []
    for group_idx in range(num_groups):
        for _ in range(n_actor):
            teacher_by_sample.append(teacher[0, group_idx])
    return {
        "source": "synthetic_slime_megatron_standard_g2_runtime",
        "g2_raw_student_gen_embeddings": [t.clone() for t in flat_gen],
        "g2_raw_student_gt_embeddings": [t.clone() for t in flat_gt],
        "g2_raw_student_gen_tensor": raw_gen,
        "g2_raw_student_gt_tensor": raw_gt,
        "g2_teacher_gen_embeddings": [t.clone() for t in teacher_by_sample],
        "g2_teacher_gen_tensor": teacher,
        "g2_whitened_student_gen_tensor": white_gen,
        "g2_whitened_student_gt_tensor": white_gt,
        "g2_cf_l1oo_rewards": rewards,
        "token_advantages": [row.clone() for row in token_advantages],
        "scalar_rewards": [float(x) for x in scalar_rewards],
        "response_lengths": response_lengths,
        "g2_cf_args": cf_args,
        "g2_shape_metadata": {
            "num_samples": num_groups * n_actor,
            "num_groups": num_groups,
            "n_samples_per_prompt": n_actor,
            "num_blocks": num_blocks,
            "embedding_dim": feat_dim,
            "teacher_samples_per_group": n_teacher,
        },
        "n_samples_per_prompt": n_actor,
        "whiten_tol": 1e-5,
    }


def _runtime_self_check(
    *,
    slime_dump: dict[str, Any],
    openrlhf_utils: Any,
    threshold: float,
    sample_limit: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    raw_gen_key, raw_gen = _raw_student_tensor_from_dump(slime_dump, "raw_student_gen")
    raw_gt_key, raw_gt = _raw_student_tensor_from_dump(slime_dump, "raw_student_gt")
    white_gen_key, white_gen_value = _first_present(slime_dump, "whitened_student_gen")
    white_gt_key, white_gt_value = _first_present(slime_dump, "whitened_student_gt")
    teacher_key, teacher = _teacher_tensor_from_dump(slime_dump)

    if white_gen_value is not None and white_gt_value is not None:
        white_gen = _as_tensor(white_gen_value).float()
        white_gt = _as_tensor(white_gt_value).float()
    elif raw_gen is not None and raw_gt is not None:
        white_gen, white_gt = openrlhf_utils.whiten_embeddings_batched(
            raw_gen,
            raw_gt,
            whiten_tol=float(slime_dump.get("whiten_tol", 1e-5)),
            normalize=False,
        )
        white_gen_key = raw_gen_key
        white_gt_key = raw_gt_key
    else:
        results.append(_missing_result("runtime self-check", "student embeddings", side="slime", aliases=ALIASES["raw_student_gen"]))
        return results

    if raw_gen is not None and raw_gt is not None:
        expected_white_gen, expected_white_gt = openrlhf_utils.whiten_embeddings_batched(
            raw_gen,
            raw_gt,
            whiten_tol=float(slime_dump.get("whiten_tol", 1e-5)),
            normalize=False,
        )
        results.append(
            _compare_tensor(
                section="runtime self-check",
                key="whitened_student_gen",
                expected=expected_white_gen,
                actual=white_gen,
                threshold=threshold,
                sample_limit=sample_limit,
                expected_dump_key=raw_gen_key,
                actual_dump_key=white_gen_key,
            )
        )
        results.append(
            _compare_tensor(
                section="runtime self-check",
                key="whitened_student_gt",
                expected=expected_white_gt,
                actual=white_gt,
                threshold=threshold,
                sample_limit=sample_limit,
                expected_dump_key=raw_gt_key,
                actual_dump_key=white_gt_key,
            )
        )

    if teacher is None:
        results.append(_missing_result("runtime self-check", "teacher_gen", side="slime", aliases=ALIASES["teacher_gen"]))
        return results

    cf_args = _cf_args(slime_dump)
    expected_rewards = openrlhf_utils.get_cf_l1oo_rewards(
        white_gen,
        white_gt,
        cf_num_freqs=cf_args["cf_num_freqs"],
        cf_sigma=cf_args["cf_sigma"],
        cf_seed=cf_args["cf_seed"],
        cf_alpha=cf_args["cf_alpha"],
        cf_beta=cf_args["cf_beta"],
        cf_reward_scale=cf_args["cf_reward_scale"],
        cf_target_mode=cf_args["cf_target_mode"],
        cf_target_num_refs=cf_args["cf_target_num_refs"],
        cf_target_std=cf_args["cf_target_std"],
        cf_target_seed=cf_args["cf_target_seed"],
        teacher_embedding=teacher,
        cf_teacher_lambda=cf_args["cf_teacher_lambda"],
    )

    rewards_key, actual_rewards = _first_present(slime_dump, "cf_l1oo_rewards")
    if actual_rewards is None:
        results.append(_missing_result("runtime self-check", "cf_l1oo_rewards", side="slime", aliases=ALIASES["cf_l1oo_rewards"]))
    else:
        results.append(
            _compare_tensor(
                section="runtime self-check",
                key="cf_l1oo_rewards",
                expected=expected_rewards,
                actual=actual_rewards,
                threshold=threshold,
                sample_limit=sample_limit,
                expected_dump_key=teacher_key,
                actual_dump_key=rewards_key,
            )
        )

    response_key, response_lengths = _first_present(slime_dump, "response_lengths")
    token_key, actual_tokens = _first_present(slime_dump, "token_advantages")
    if response_lengths is None:
        results.append(_missing_result("runtime self-check", "response_lengths", side="slime", aliases=ALIASES["response_lengths"]))
    elif actual_tokens is None:
        results.append(_missing_result("runtime self-check", "token_advantages", side="slime", aliases=ALIASES["token_advantages"]))
    else:
        expected_tokens = _expected_token_advantages(expected_rewards, [int(x) for x in response_lengths])
        results.append(
            _compare_tensor(
                section="runtime self-check",
                key="token_advantages",
                expected=expected_tokens,
                actual=actual_tokens,
                threshold=threshold,
                sample_limit=sample_limit,
                expected_dump_key=response_key,
                actual_dump_key=token_key,
            )
        )

    scalar_key, actual_scalars = _first_present(slime_dump, "scalar_rewards")
    if actual_scalars is not None:
        expected_scalars = expected_rewards.squeeze(0).reshape(-1, expected_rewards.shape[-1]).mean(dim=-1)
        results.append(
            _compare_tensor(
                section="runtime self-check",
                key="scalar_rewards",
                expected=expected_scalars,
                actual=actual_scalars,
                threshold=threshold,
                sample_limit=sample_limit,
                expected_dump_key="expected_cf_l1oo_rewards.mean",
                actual_dump_key=scalar_key,
            )
        )

    return results


def _openrlhf_dump_compare(
    *,
    slime_dump: dict[str, Any],
    openrlhf_dump: dict[str, Any],
    threshold: float,
    sample_limit: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for canonical_key in ALIASES:
        slime_key, slime_value = _first_present(slime_dump, canonical_key)
        openrlhf_key, openrlhf_value = _first_present(openrlhf_dump, canonical_key)
        if slime_value is None and openrlhf_value is None:
            continue
        if slime_value is None:
            results.append(_missing_result("openrlhf dump compare", canonical_key, side="slime", aliases=ALIASES[canonical_key]))
            continue
        if openrlhf_value is None:
            results.append(_missing_result("openrlhf dump compare", canonical_key, side="openrlhf", aliases=ALIASES[canonical_key]))
            continue
        results.append(
            _compare_tensor(
                section="openrlhf dump compare",
                key=canonical_key,
                expected=openrlhf_value,
                actual=slime_value,
                threshold=threshold,
                sample_limit=sample_limit,
                expected_dump_key=openrlhf_key,
                actual_dump_key=slime_key,
            )
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Slime/OpenRLHF standard G2 runtime dumps.")
    parser.add_argument("--slime-dump", type=Path)
    parser.add_argument("--openrlhf-dump", type=Path)
    parser.add_argument("--openrlhf-embedding-utils", type=Path, default=OPENRLHF_EMBEDDING_UTILS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--sample-values", type=int, default=8)
    parser.add_argument("--self-test", action="store_true", help="Run the compare logic against a synthetic in-memory dump.")
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    try:
        openrlhf_utils = _load_openrlhf_embedding_utils(args.openrlhf_embedding_utils)
        if args.self_test:
            slime_dump = _make_synthetic_slime_dump(openrlhf_utils)
        else:
            if args.slime_dump is None:
                raise ValueError("--slime-dump is required unless --self-test is set")
            slime_dump = _load_dump(args.slime_dump)
        results.extend(
            _runtime_self_check(
                slime_dump=slime_dump,
                openrlhf_utils=openrlhf_utils,
                threshold=args.threshold,
                sample_limit=args.sample_values,
            )
        )
        if args.openrlhf_dump is not None:
            results.extend(
                _openrlhf_dump_compare(
                    slime_dump=slime_dump,
                    openrlhf_dump=_load_dump(args.openrlhf_dump),
                    threshold=args.threshold,
                    sample_limit=args.sample_values,
                )
            )
    except Exception as exc:
        results.append(
            {
                "section": "script exception",
                "key": type(exc).__name__,
                "passed": False,
                "threshold": args.threshold,
                "shape": {"expected": [], "actual": [], "match": False},
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "sample_values": {"expected": [], "actual": []},
                "details": {"error": str(exc)},
            }
        )

    by_section: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        by_section.setdefault(item["section"], []).append(item)

    all_passed = all(item["passed"] for item in results)
    print(
        json.dumps(
            {
                "all_passed": all_passed,
                "slime_dump": "<synthetic>" if args.self_test else str(args.slime_dump),
                "openrlhf_dump": str(args.openrlhf_dump) if args.openrlhf_dump else None,
                "sections": by_section,
            },
            indent=2,
        )
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
