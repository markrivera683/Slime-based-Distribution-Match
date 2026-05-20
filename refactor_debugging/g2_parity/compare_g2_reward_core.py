#!/usr/bin/env python
from __future__ import annotations

import importlib.util
import json
import sys
import types
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENRLHF_EMBEDDING_UTILS = Path("/mnt/data/ebft-distribution-new/code/openrlhf/utils/embedding_utils.py")
THRESHOLD = 1e-6


@dataclass(frozen=True)
class LayoutCase:
    name: str
    prompt: str
    label: str
    teacher_completions: list[str]


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    _table = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "f": 6,
        "g": 7,
        "h": 8,
        "w": 21,
        "x": 22,
        "y": 23,
        "z": 24,
        "q": 25,
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._table[ch] for ch in str(text)]


def _install_optional_import_stubs() -> None:
    """Keep this parity script independent of Ray/SGLang runtime installs."""
    sys.modules.setdefault("ray", types.ModuleType("ray"))
    sys.modules.setdefault("sglang_router", types.ModuleType("sglang_router"))
    pybase64 = sys.modules.setdefault("pybase64", types.ModuleType("pybase64"))
    if not hasattr(pybase64, "b64encode"):
        import base64

        pybase64.b64encode = base64.b64encode
        pybase64.b64decode = base64.b64decode
    sglang_rollout = sys.modules.setdefault("slime.rollout.sglang_rollout", types.ModuleType("slime.rollout.sglang_rollout"))
    if not hasattr(sglang_rollout, "generate"):
        async def _unused_generate(*args, **kwargs):
            raise RuntimeError("sglang rollout is not used by this CPU parity script")

        sglang_rollout.generate = _unused_generate
    processing_utils = sys.modules.setdefault("slime.utils.processing_utils", types.ModuleType("slime.utils.processing_utils"))
    if not hasattr(processing_utils, "load_tokenizer"):
        def _unused_load_tokenizer(*args, **kwargs):
            raise RuntimeError("load_tokenizer is not used by this CPU parity script")

        processing_utils.load_tokenizer = _unused_load_tokenizer


def _load_openrlhf_embedding_utils() -> Any:
    if not OPENRLHF_EMBEDDING_UTILS.exists():
        raise FileNotFoundError(f"OpenRLHF embedding_utils.py not found: {OPENRLHF_EMBEDDING_UTILS}")

    _install_optional_import_stubs()
    module_name = "_openrlhf_embedding_utils_for_g2_parity"
    spec = importlib.util.spec_from_file_location(module_name, OPENRLHF_EMBEDDING_UTILS)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {OPENRLHF_EMBEDDING_UTILS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_slime_modules() -> tuple[Any, Any, Any, Any]:
    _install_optional_import_stubs()
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from slime.backends.megatron_utils import g1_fast
        from slime.rollout.g1_embedding import G1EmbeddingConfig, build_g1_teacher_full_sequence_inputs
        from slime.utils.types import Sample
    finally:
        try:
            sys.path.remove(str(REPO_ROOT))
        except ValueError:
            pass
    return g1_fast, G1EmbeddingConfig, build_g1_teacher_full_sequence_inputs, Sample


def _make_tensor(shape: tuple[int, ...], *, offset: float, scale: float) -> torch.Tensor:
    total = 1
    for dim in shape:
        total *= int(dim)
    base = torch.arange(total, dtype=torch.float32).reshape(shape)
    return torch.sin(base * scale + offset) + 0.25 * torch.cos(base * (scale * 0.7) - offset)


def _independent_openrlhf_prompt_pack(
    *,
    tokenizer: FakeTokenizer,
    prompt: str,
    label: str,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(label, add_special_tokens=False)
    packed = list(prompt_ids) + list(answer_ids)
    qa_mask = [0] * len(prompt_ids) + [1] * len(answer_ids)
    if len(packed) > prompt_length:
        raise ValueError(f"fixture prompt+label length {len(packed)} exceeds prompt_length={prompt_length}")
    pad_len = prompt_length - len(packed)
    packed.extend([int(tokenizer.pad_token_id)] * pad_len)
    qa_mask.extend([0] * pad_len)
    doc_ids = [0] * (prompt_length - pad_len) + [-1] * pad_len
    return (
        torch.tensor(packed, dtype=torch.long),
        torch.tensor(doc_ids, dtype=torch.long),
        torch.tensor(qa_mask, dtype=torch.long),
    )


def _independent_openrlhf_build_teacher_prompt(
    *,
    prompt_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    qa_masks: torch.Tensor,
    teacher_answers: dict[int, list[list[int]]],
    teacher_samples: int,
    pad_id: int,
) -> torch.Tensor:
    result = prompt_ids.unsqueeze(0).expand(teacher_samples, -1).clone()
    present_doc_ids = [int(doc_id) for doc_id in doc_ids.unique().tolist() if int(doc_id) >= 0]
    for doc_id in present_doc_ids:
        if doc_id not in teacher_answers:
            continue
        answer_positions = ((doc_ids == doc_id) & (qa_masks == 1)).nonzero(as_tuple=True)[0]
        if answer_positions.numel() == 0:
            continue
        answer_start = int(answer_positions[0].item())
        answer_len = int(answer_positions.numel())
        for sample_idx in range(teacher_samples):
            answer_ids = teacher_answers[doc_id][sample_idx]
            fill_len = min(answer_len, len(answer_ids))
            if fill_len > 0:
                result[sample_idx, answer_start : answer_start + fill_len] = torch.tensor(
                    answer_ids[:fill_len],
                    dtype=torch.long,
                )
            if len(answer_ids) < answer_len:
                result[sample_idx, answer_start + len(answer_ids) : answer_start + answer_len] = int(pad_id)
    return result


def _independent_openrlhf_teacher_full_sequences(
    *,
    tokenizer: FakeTokenizer,
    case: LayoutCase,
    prompt_length: int,
    context_length: int,
    generate_length: int,
    stride: int,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids, doc_ids, qa_masks = _independent_openrlhf_prompt_pack(
        tokenizer=tokenizer,
        prompt=case.prompt,
        label=case.label,
        prompt_length=prompt_length,
    )
    teacher_answers = {
        0: [tokenizer.encode(completion, add_special_tokens=False) for completion in case.teacher_completions]
    }
    teacher_prompt = _independent_openrlhf_build_teacher_prompt(
        prompt_ids=prompt_ids,
        doc_ids=doc_ids,
        qa_masks=qa_masks,
        teacher_answers=teacher_answers,
        teacher_samples=len(case.teacher_completions),
        pad_id=int(tokenizer.pad_token_id),
    )

    full_sequences = []
    for sample_idx in range(len(case.teacher_completions)):
        prompt_variant = teacher_prompt[sample_idx]
        block_tokens = []
        for block_idx in range(num_blocks):
            start = context_length + block_idx * stride
            end = start + generate_length
            if end <= prompt_variant.numel():
                block_tokens.append(prompt_variant[start:end])
            else:
                available = max(0, prompt_variant.numel() - start)
                if available > 0:
                    block_tokens.append(
                        torch.cat(
                            [
                                prompt_variant[start : start + available],
                                torch.full((generate_length - available,), int(tokenizer.pad_token_id), dtype=torch.long),
                            ]
                        )
                    )
                else:
                    block_tokens.append(torch.full((generate_length,), int(tokenizer.pad_token_id), dtype=torch.long))
        gen_region = torch.stack(block_tokens, dim=0).t().reshape(-1)
        full_sequences.append(torch.cat([prompt_ids, gen_region], dim=0))

    strided_qa = qa_masks[context_length:].unfold(0, generate_length, stride).t().reshape(-1)
    qa_full = torch.cat([qa_masks, strided_qa], dim=0).unsqueeze(0).expand(len(case.teacher_completions), -1)
    return torch.stack(full_sequences, dim=0), qa_full


def _compare_tensor(
    *,
    section: str,
    case: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    threshold: float = THRESHOLD,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shape_match = tuple(expected.shape) == tuple(actual.shape)
    if shape_match:
        diff = (expected.float() - actual.float()).abs()
        max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
        mean_abs_diff = float(diff.mean().item()) if diff.numel() else 0.0
    else:
        max_abs_diff = float("inf")
        mean_abs_diff = float("inf")
    payload_details = dict(details or {})
    payload_details["expected_values"] = expected.detach().cpu().tolist()
    payload_details["actual_values"] = actual.detach().cpu().tolist()
    return {
        "section": section,
        "case": case,
        "passed": shape_match and max_abs_diff <= threshold,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "threshold": threshold,
        "shape": {
            "expected": list(expected.shape),
            "actual": list(actual.shape),
            "match": shape_match,
        },
        "details": payload_details,
    }


def _run_layout_parity(
    *,
    G1EmbeddingConfig: Any,
    build_g1_teacher_full_sequence_inputs: Any,
    Sample: Any,
) -> list[dict[str, Any]]:
    tokenizer = FakeTokenizer()
    config = G1EmbeddingConfig(
        prompt_length=8,
        context_length=2,
        generate_length=2,
        stride=2,
        response_length=6,
        n_samples_per_prompt=3,
    )
    cases = [
        LayoutCase("teacher_answer_shorter_than_answer_span", "ab", "cdef", ["x"]),
        LayoutCase("teacher_answer_longer_than_answer_span", "ab", "cdef", ["xyzwq"]),
        LayoutCase("multiple_teacher_samples_m", "ab", "cdef", ["x", "xy", "zyx"]),
        LayoutCase("stride_block_reorder", "ab", "cdef", ["wxyz"]),
    ]

    results: list[dict[str, Any]] = []
    for case in cases:
        expected_full, expected_qa = _independent_openrlhf_teacher_full_sequences(
            tokenizer=tokenizer,
            case=case,
            prompt_length=config.prompt_length,
            context_length=config.context_length,
            generate_length=config.generate_length,
            stride=config.stride,
            num_blocks=config.num_blocks,
        )
        actual_full, actual_qa = build_g1_teacher_full_sequence_inputs(
            tokenizer=tokenizer,
            sample=Sample(prompt=case.prompt, label=case.label),
            teacher_completions=case.teacher_completions,
            config=config,
        )
        details = {
            "teacher_samples": len(case.teacher_completions),
            "num_blocks": config.num_blocks,
            "gen_region_expected": expected_full[:, config.prompt_length :].tolist(),
        }
        results.append(
            _compare_tensor(
                section="layout parity",
                case=f"{case.name}: full_sequences",
                expected=expected_full,
                actual=actual_full,
                details=details,
            )
        )
        results.append(
            _compare_tensor(
                section="layout parity",
                case=f"{case.name}: qa_masks",
                expected=expected_qa,
                actual=actual_qa,
                details=details,
            )
        )
    return results


def _make_reward_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = 1
    num_groups = 2
    n_actor = 3
    n_teacher = 5
    num_blocks = 2
    feat_dim = 4
    gen = _make_tensor((batch_size, num_groups, n_actor, num_blocks, feat_dim), offset=0.13, scale=0.071)
    gt = _make_tensor((batch_size, num_groups, n_actor, num_blocks, feat_dim), offset=0.41, scale=0.053)
    teacher = _make_tensor((batch_size, num_groups, n_teacher, num_blocks, feat_dim), offset=-0.29, scale=0.067)
    return gen, gt, teacher


def _flatten_actor_embeddings(tensor: torch.Tensor) -> list[torch.Tensor]:
    return [tensor[0, group_idx, sample_idx].clone() for group_idx in range(tensor.shape[1]) for sample_idx in range(tensor.shape[2])]


def _teacher_embeddings_by_actor_sample(teacher: torch.Tensor, *, n_actor: int) -> list[torch.Tensor]:
    return [teacher[0, group_idx].clone() for group_idx in range(teacher.shape[1]) for _ in range(n_actor)]


def _expected_token_advantages(rewards: torch.Tensor, *, generate_length: int) -> list[torch.Tensor]:
    flat = rewards.squeeze(0).reshape(-1, rewards.shape[-1])
    return [sample_rewards.repeat(generate_length) for sample_rewards in flat]


def _run_reward_path_parity(*, openrlhf_utils: Any, g1_fast: Any) -> list[dict[str, Any]]:
    gen_raw, gt_raw, teacher_raw = _make_reward_fixture()
    _, num_groups, n_actor, num_blocks, _ = gen_raw.shape
    n_teacher = teacher_raw.shape[2]
    generate_length = 2
    response_length = num_blocks * generate_length
    args = Namespace(
        n_samples_per_prompt=n_actor,
        distribution_reward_type="cf_l1oo",
        cf_target_mode="teacher",
        use_whitening=True,
        whiten_tol=1e-5,
        g1_response_length=response_length,
        g1_num_blocks=num_blocks,
        cf_num_freqs=17,
        cf_sigma=0.9,
        cf_seed=123,
        cf_alpha=0.35,
        cf_beta=0.65,
        cf_reward_scale=2.75,
        cf_teacher_lambda=0.6,
    )

    expected_gen, expected_gt = openrlhf_utils.whiten_embeddings_batched(
        gen_raw,
        gt_raw,
        whiten_tol=args.whiten_tol,
        normalize=False,
    )
    expected_rewards = openrlhf_utils.get_cf_l1oo_rewards(
        expected_gen,
        expected_gt,
        cf_num_freqs=args.cf_num_freqs,
        cf_sigma=args.cf_sigma,
        cf_seed=args.cf_seed,
        cf_alpha=args.cf_alpha,
        cf_beta=args.cf_beta,
        cf_reward_scale=args.cf_reward_scale,
        cf_target_mode="teacher",
        cf_target_num_refs=1,
        cf_target_std=0.05,
        cf_target_seed=args.cf_seed,
        teacher_embedding=teacher_raw,
        cf_teacher_lambda=args.cf_teacher_lambda,
    )
    expected_tokens = _expected_token_advantages(expected_rewards, generate_length=generate_length)
    expected_scalars = expected_rewards.squeeze(0).reshape(-1, num_blocks).mean(dim=-1)

    captured: dict[str, Any] = {}
    original_compute = g1_fast.compute_cf_l1oo_rewards

    def _capture_compute_cf_l1oo_rewards(gen: torch.Tensor, gt: torch.Tensor, *, teacher_embedding: torch.Tensor, **kwargs):
        captured["gen"] = gen.detach().clone()
        captured["gt"] = gt.detach().clone()
        captured["teacher"] = teacher_embedding.detach().clone()
        captured["kwargs"] = dict(kwargs)
        return original_compute(gen, gt, teacher_embedding=teacher_embedding, **kwargs)

    g1_fast.compute_cf_l1oo_rewards = _capture_compute_cf_l1oo_rewards
    try:
        token_advantages, scalar_rewards = g1_fast.compute_g1_token_advantages_from_embeddings(
            args,
            _flatten_actor_embeddings(gen_raw),
            _flatten_actor_embeddings(gt_raw),
            [response_length] * (num_groups * n_actor),
            teacher_gen_embeddings=_teacher_embeddings_by_actor_sample(teacher_raw, n_actor=n_actor),
        )
    finally:
        g1_fast.compute_cf_l1oo_rewards = original_compute

    details = {
        "actor_samples_n": n_actor,
        "teacher_samples_m": n_teacher,
        "m_differs_from_n": n_teacher != n_actor,
        "captured_cf_kwargs": captured.get("kwargs", {}),
    }
    results = [
        _compare_tensor(
            section="reward input parity",
            case="whitened gen passed to Slime trainer-side cf_l1oo",
            expected=expected_gen,
            actual=captured["gen"],
            threshold=1e-5,
            details=details,
        ),
        _compare_tensor(
            section="reward input parity",
            case="whitened gt passed to Slime trainer-side cf_l1oo",
            expected=expected_gt,
            actual=captured["gt"],
            threshold=1e-5,
            details=details,
        ),
        _compare_tensor(
            section="reward input parity",
            case="unwhitened teacher passed to Slime trainer-side cf_l1oo",
            expected=teacher_raw,
            actual=captured["teacher"],
            threshold=THRESHOLD,
            details=details,
        ),
    ]

    actual_tokens = torch.stack([tensor.float() for tensor in token_advantages], dim=0)
    expected_tokens_tensor = torch.stack(expected_tokens, dim=0)
    actual_scalars = torch.tensor(scalar_rewards, dtype=torch.float32)
    results.append(
        _compare_tensor(
            section="final reward/token advantage parity",
            case="expanded token advantages from trainer-side G2 rewards",
            expected=expected_tokens_tensor,
            actual=actual_tokens,
            threshold=1e-5,
            details=details,
        )
    )
    results.append(
        _compare_tensor(
            section="final reward/token advantage parity",
            case="scalar rewards from trainer-side G2 rewards",
            expected=expected_scalars.float(),
            actual=actual_scalars,
            threshold=1e-5,
            details=details,
        )
    )
    return results


def main() -> int:
    torch.set_num_threads(1)
    openrlhf_utils = _load_openrlhf_embedding_utils()
    g1_fast, G1EmbeddingConfig, build_g1_teacher_full_sequence_inputs, Sample = _load_slime_modules()

    results = []
    try:
        results.extend(
            _run_layout_parity(
                G1EmbeddingConfig=G1EmbeddingConfig,
                build_g1_teacher_full_sequence_inputs=build_g1_teacher_full_sequence_inputs,
                Sample=Sample,
            )
        )
        results.extend(_run_reward_path_parity(openrlhf_utils=openrlhf_utils, g1_fast=g1_fast))
    except Exception as exc:
        results.append(
            {
                "section": "script exception",
                "case": type(exc).__name__,
                "passed": False,
                "max_abs_diff": float("inf"),
                "mean_abs_diff": float("inf"),
                "threshold": THRESHOLD,
                "shape": {"expected": [], "actual": [], "match": False},
                "details": {"error": str(exc)},
            }
        )

    by_section: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        by_section.setdefault(item["section"], []).append(item)
    print(json.dumps({"all_passed": all(item["passed"] for item in results), "sections": by_section}, indent=2))

    failed = [item for item in results if not item["passed"]]
    if failed:
        for item in failed:
            print(
                "FAILED "
                f"[{item['section']}] {item['case']}: "
                f"shape={item['shape']} "
                f"max_abs_diff={item['max_abs_diff']:.9g} "
                f"mean_abs_diff={item['mean_abs_diff']:.9g} "
                f"threshold={item['threshold']:.9g} "
                f"details={item['details']}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
