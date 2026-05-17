#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EmbeddingThresholds:
    cosine_min: float = 0.998
    max_abs: float = 5e-2
    mean_abs: float = 5e-3
    rel_l2_max: float = 5e-2


@dataclass(frozen=True)
class RewardThresholds:
    max_abs: float = 5e-3
    mean_abs: float = 2e-3
    rel_l2_max: float = 5e-3


@dataclass(frozen=True)
class AdvantageThresholds:
    cosine_min: float = 0.995
    max_abs: float = 5e-2
    mean_abs: float = 1e-2
    rel_l2_max: float = 5e-2


STANDARD_ATTENTION_STATUS = "standard_megatron_attention"
BUILT_NOT_APPLIED_STATUS = "openrlhf_dense_mask_built_not_applied_to_megatron_te_thd"
APPLIED_THD_FALLBACK_STATUS = "openrlhf_dense_mask_applied_via_torch_thd_fallback"
NOT_DUMPED_STATUS = "not_dumped"
KNOWN_ATTENTION_MASK_STATUSES = {
    STANDARD_ATTENTION_STATUS,
    BUILT_NOT_APPLIED_STATUS,
    APPLIED_THD_FALLBACK_STATUS,
    NOT_DUMPED_STATUS,
}


@dataclass(frozen=True)
class MegatronDenseMaskStatus:
    raw_status: str
    ref_forward_mode: str
    openrlhf_exact: bool
    dense_mask_dumped: bool
    dense_mask_built: bool
    apply_dense_attention_mask: bool
    thd_fallback: bool
    status_label: str
    consistency_errors: tuple[str, ...]

    @property
    def consistency_ok(self) -> bool:
        return len(self.consistency_errors) == 0


def _flatten_embeddings(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([t.float() for t in values], dim=0)


def _sample_megatron_hidden(megatron_dump: dict) -> torch.Tensor:
    hidden = megatron_dump["hidden_states_post_sp_gather"].float()
    if hidden.ndim != 3 or hidden.shape[1] != 1:
        raise ValueError(f"Expected Megatron THD hidden [S, 1, H], got {hidden.shape}")
    total_lengths = [int(x) for x in megatron_dump["total_lengths"]]
    thd_seq = int(hidden.shape[0])
    packed = sum(total_lengths)
    if packed != thd_seq:
        raise ValueError(
            f"Megatron THD / total_lengths mismatch: sum(total_lengths)={packed} != "
            f"hidden_states_post_sp_gather.shape[0]={thd_seq}"
        )
    chunks: list[torch.Tensor] = []
    cursor = 0
    for length in total_lengths:
        chunks.append(hidden[cursor : cursor + length, 0])
        cursor += length
    if cursor != thd_seq:
        raise ValueError("Internal error: cursor did not consume full THD sequence after compacting by total_lengths")
    return torch.stack(chunks, dim=0)


def _require_same_shape(name: str, left: torch.Tensor, right: torch.Tensor) -> None:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"{name} shape mismatch: Megatron={tuple(left.shape)} OpenRLHF={tuple(right.shape)}")


def _cosine_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    _require_same_shape("cosine input", a, b)
    cos = F.cosine_similarity(a.reshape(-1, a.shape[-1]).float(), b.reshape(-1, b.shape[-1]).float(), dim=-1)
    return {
        "mean": float(cos.mean().item()),
        "median": float(cos.median().item()),
        "min": float(cos.min().item()),
        "max": float(cos.max().item()),
    }


def _close_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.float() - b.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Frobenius relative error ||a-b||_F / (||a||_F + eps)."""
    _require_same_shape("relative L2", a, b)
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    diff = a_f - b_f
    return float(torch.norm(diff) / (torch.norm(a_f) + 1e-12))


def _advantage_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    _require_same_shape("token advantages", a, b)
    sample_cos = F.cosine_similarity(a.float(), b.float(), dim=-1)
    close = _close_stats(a, b)
    return {
        "sample_cos_mean": float(sample_cos.mean().item()),
        "sample_cos_min": float(sample_cos.min().item()),
        "max_abs": close["max_abs"],
        "mean_abs": close["mean_abs"],
        "rel_l2": _relative_l2(a, b),
    }


def _optional_tensor_equal(left: torch.Tensor | None, right: torch.Tensor | None) -> bool | None:
    if left is None or right is None:
        return None
    return bool(torch.equal(left, right))


def _optional_attention_mask_semantic_equal(left: torch.Tensor | None, right: torch.Tensor | None) -> bool | None:
    if left is None or right is None:
        return None
    if tuple(left.shape) != tuple(right.shape):
        return False
    return bool(torch.equal(left == 0, right == 0))


def _require_same_count(name: str, left: list, right: list) -> None:
    if len(left) != len(right):
        raise ValueError(f"{name} sample count mismatch: Megatron={len(left)} OpenRLHF={len(right)}")


def _mask_summary(mask: torch.Tensor | None) -> dict[str, object] | None:
    if mask is None:
        return None
    allowed = mask == 0
    finite = torch.isfinite(mask)
    return {
        "shape": tuple(mask.shape),
        "allowed_count": int(allowed.sum().item()),
        "blocked_count": int((~allowed).sum().item()),
        "finite_count": int(finite.sum().item()),
    }


def _dump_has_dense_mask(megatron_dump: dict) -> bool:
    return any(
        megatron_dump.get(key) is not None
        for key in (
            "g1_attention_mask",
            "g1_attention_mask_shape",
            "g1_packed_attention_mask",
            "g1_packed_attention_mask_shape",
        )
    )


def _extract_megatron_dense_mask_status(megatron_dump: dict) -> MegatronDenseMaskStatus:
    raw_status = str(megatron_dump.get("g1_attention_mask_status", NOT_DUMPED_STATUS))
    ref_forward_mode = str(megatron_dump.get("g1_megatron_ref_forward_mode", "standard"))
    openrlhf_exact = ref_forward_mode == "openrlhf_exact"
    dense_mask_dumped = _dump_has_dense_mask(megatron_dump)
    status_says_dense_mask = raw_status in {BUILT_NOT_APPLIED_STATUS, APPLIED_THD_FALLBACK_STATUS}
    dense_mask_built = dense_mask_dumped or status_says_dense_mask
    applied_flag = bool(megatron_dump.get("g1_attention_mask_applied", False))
    thd_fallback = raw_status == APPLIED_THD_FALLBACK_STATUS
    apply_dense_attention_mask = applied_flag and thd_fallback

    if thd_fallback:
        status_label = "applied-via-torch-thd-fallback"
    elif dense_mask_built:
        status_label = "dense-mask-built-not-applied"
    elif openrlhf_exact:
        status_label = "openrlhf_exact-no-dense-mask"
    else:
        status_label = "standard-megatron-attention"

    errors: list[str] = []
    if raw_status not in KNOWN_ATTENTION_MASK_STATUSES:
        errors.append(f"unknown g1_attention_mask_status `{raw_status}`")
    if status_says_dense_mask and not openrlhf_exact:
        errors.append(
            f"g1_attention_mask_status `{raw_status}` requires "
            "`g1_megatron_ref_forward_mode=openrlhf_exact`"
        )
    if status_says_dense_mask and not dense_mask_dumped:
        errors.append(f"g1_attention_mask_status `{raw_status}` says a dense mask exists, but no dense mask was dumped")
    if raw_status == STANDARD_ATTENTION_STATUS and dense_mask_dumped:
        errors.append("standard attention status conflicts with dumped g1 dense attention mask metadata")
    if raw_status == BUILT_NOT_APPLIED_STATUS and applied_flag:
        errors.append("dense mask is marked built-not-applied, but g1_attention_mask_applied is True")
    if thd_fallback and not applied_flag:
        errors.append("THD fallback status requires g1_attention_mask_applied=True")
    if applied_flag and not thd_fallback:
        errors.append("g1_attention_mask_applied=True requires THD fallback status")

    return MegatronDenseMaskStatus(
        raw_status=raw_status,
        ref_forward_mode=ref_forward_mode,
        openrlhf_exact=openrlhf_exact,
        dense_mask_dumped=dense_mask_dumped,
        dense_mask_built=dense_mask_built,
        apply_dense_attention_mask=apply_dense_attention_mask,
        thd_fallback=thd_fallback,
        status_label=status_label,
        consistency_errors=tuple(errors),
    )


def _early_validate_tokens_and_masks(megatron: dict, openrlhf: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Fail before parity metrics if batch/shape contracts are broken."""
    if "tokens" not in megatron:
        raise ValueError("Megatron dump missing required key 'tokens'")
    if "sequences" not in openrlhf:
        raise ValueError("OpenRLHF dump missing required key 'sequences'")
    mt_list = megatron["tokens"]
    ot = openrlhf["sequences"].long()
    if len(mt_list) != ot.shape[0]:
        raise ValueError(f"tokens sample count mismatch: Megatron={len(mt_list)} OpenRLHF batch={ot.shape[0]}")
    megatron_tokens = torch.stack([t.long() for t in mt_list], dim=0)
    if megatron_tokens.shape != ot.shape:
        raise ValueError(
            f"Token tensor shape mismatch before hidden/embedding parity: "
            f"Megatron={tuple(megatron_tokens.shape)} OpenRLHF={tuple(ot.shape)}"
        )

    if "g1_qa_masks" not in megatron or "qa_masks" not in openrlhf:
        raise ValueError("Missing g1_qa_masks (Megatron) or qa_masks (OpenRLHF)")
    _require_same_count("g1_qa_masks", megatron["g1_qa_masks"], openrlhf["qa_masks"])
    mqa = torch.stack([t.long() for t in megatron["g1_qa_masks"]], dim=0)
    oqa = openrlhf["qa_masks"].long()
    if mqa.shape != oqa.shape:
        raise ValueError(
            f"QA mask tensor shape mismatch before parity metrics: Megatron={tuple(mqa.shape)} OpenRLHF={tuple(oqa.shape)}"
        )
    return megatron_tokens, mqa


def _check_embedding_family(
    name: str,
    cos: dict[str, float],
    close: dict[str, float],
    rel_l2: float,
    thr: EmbeddingThresholds,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if cos["min"] < thr.cosine_min:
        failures.append(f"`cosine_min` {cos['min']:.8f} < threshold {thr.cosine_min} ({name})")
    if close["max_abs"] > thr.max_abs:
        failures.append(f"`max_abs` {close['max_abs']:.8e} > threshold {thr.max_abs} ({name})")
    if close["mean_abs"] > thr.mean_abs:
        failures.append(f"`mean_abs` {close['mean_abs']:.8e} > threshold {thr.mean_abs} ({name})")
    if rel_l2 > thr.rel_l2_max:
        failures.append(f"`rel_l2` {rel_l2:.8e} > threshold {thr.rel_l2_max} ({name})")
    return len(failures) == 0, failures


def _check_reward_family(close: dict[str, float], rel_l2: float, thr: RewardThresholds) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if close["max_abs"] > thr.max_abs:
        failures.append(f"`max_abs` {close['max_abs']:.8e} > threshold {thr.max_abs} (scalar_rewards)")
    if close["mean_abs"] > thr.mean_abs:
        failures.append(f"`mean_abs` {close['mean_abs']:.8e} > threshold {thr.mean_abs} (scalar_rewards)")
    if rel_l2 > thr.rel_l2_max:
        failures.append(f"`rel_l2` {rel_l2:.8e} > threshold {thr.rel_l2_max} (scalar_rewards)")
    return len(failures) == 0, failures


def _check_advantage_family(stats: dict[str, float], thr: AdvantageThresholds) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if stats["sample_cos_min"] < thr.cosine_min:
        failures.append(
            f"`cosine_min` (per-token) {stats['sample_cos_min']:.8f} < threshold {thr.cosine_min} (g1_token_advantages)"
        )
    if stats["max_abs"] > thr.max_abs:
        failures.append(f"`max_abs` {stats['max_abs']:.8e} > threshold {thr.max_abs} (g1_token_advantages)")
    if stats["mean_abs"] > thr.mean_abs:
        failures.append(f"`mean_abs` {stats['mean_abs']:.8e} > threshold {thr.mean_abs} (g1_token_advantages)")
    if stats["rel_l2"] > thr.rel_l2_max:
        failures.append(f"`rel_l2` {stats['rel_l2']:.8e} > threshold {thr.rel_l2_max} (g1_token_advantages)")
    return len(failures) == 0, failures


def _family_gate_line(label: str, passed: bool | None, detail: str) -> str:
    """passed=None means the metric family was not present in both dumps (SKIP)."""
    if passed is None:
        return f"- **{label}**: **SKIP** — {detail}"
    status = "PASS" if passed else "FAIL"
    return f"- **{label}**: **{status}** — {detail}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare G1 Megatron vs OpenRLHF runtime dumps with strict parity metrics.")
    parser.add_argument("--megatron-dump", required=True)
    parser.add_argument("--openrlhf-dump", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--no-fail-on-strict",
        action="store_true",
        help="Do not exit with code 1 when the strict parity gate fails (default: exit 1 on failure).",
    )
    # Embedding family (hidden, gen, gt)
    parser.add_argument("--embedding-cosine-min", type=float, default=EmbeddingThresholds.cosine_min)
    parser.add_argument("--embedding-max-abs", type=float, default=EmbeddingThresholds.max_abs)
    parser.add_argument("--embedding-mean-abs", type=float, default=EmbeddingThresholds.mean_abs)
    parser.add_argument("--embedding-rel-l2-max", type=float, default=EmbeddingThresholds.rel_l2_max)
    # Rewards
    parser.add_argument("--reward-max-abs", type=float, default=RewardThresholds.max_abs)
    parser.add_argument("--reward-mean-abs", type=float, default=RewardThresholds.mean_abs)
    parser.add_argument("--reward-rel-l2-max", type=float, default=RewardThresholds.rel_l2_max)
    # Token advantages
    parser.add_argument("--advantage-cosine-min", type=float, default=AdvantageThresholds.cosine_min)
    parser.add_argument("--advantage-max-abs", type=float, default=AdvantageThresholds.max_abs)
    parser.add_argument("--advantage-mean-abs", type=float, default=AdvantageThresholds.mean_abs)
    parser.add_argument("--advantage-rel-l2-max", type=float, default=AdvantageThresholds.rel_l2_max)
    args = parser.parse_args()

    emb_thr = EmbeddingThresholds(
        cosine_min=args.embedding_cosine_min,
        max_abs=args.embedding_max_abs,
        mean_abs=args.embedding_mean_abs,
        rel_l2_max=args.embedding_rel_l2_max,
    )
    rew_thr = RewardThresholds(
        max_abs=args.reward_max_abs,
        mean_abs=args.reward_mean_abs,
        rel_l2_max=args.reward_rel_l2_max,
    )
    adv_thr = AdvantageThresholds(
        cosine_min=args.advantage_cosine_min,
        max_abs=args.advantage_max_abs,
        mean_abs=args.advantage_mean_abs,
        rel_l2_max=args.advantage_rel_l2_max,
    )

    megatron = torch.load(args.megatron_dump, map_location="cpu", weights_only=False)
    openrlhf = torch.load(args.openrlhf_dump, map_location="cpu", weights_only=False)

    megatron_tokens, _mqa = _early_validate_tokens_and_masks(megatron, openrlhf)
    openrlhf_tokens = openrlhf["sequences"].long()
    token_match = bool(torch.equal(megatron_tokens, openrlhf_tokens))
    qa_match = bool(torch.equal(torch.stack([t.long() for t in megatron["g1_qa_masks"]], dim=0), openrlhf["qa_masks"].long()))
    position_match = _optional_tensor_equal(megatron.get("g1_position_ids"), openrlhf.get("position_ids"))
    mask_match = _optional_attention_mask_semantic_equal(megatron.get("g1_attention_mask"), openrlhf.get("attention_mask"))
    megatron_mask_summary = _mask_summary(megatron.get("g1_attention_mask"))
    openrlhf_mask_summary = _mask_summary(openrlhf.get("attention_mask"))
    mask_status = _extract_megatron_dense_mask_status(megatron)

    megatron_hidden = _sample_megatron_hidden(megatron)
    openrlhf_hidden = openrlhf["hidden_states"].squeeze(-2).float()
    _require_same_shape("hidden states", megatron_hidden, openrlhf_hidden)
    hidden_cos = _cosine_stats(megatron_hidden, openrlhf_hidden)
    hidden_close = _close_stats(megatron_hidden, openrlhf_hidden)
    hidden_rel_l2 = _relative_l2(megatron_hidden, openrlhf_hidden)

    _require_same_count("gen embeddings", megatron["g1_gen_embedding"], openrlhf["g1_gen_embedding"])
    _require_same_count("gt embeddings", megatron["g1_gt_embedding"], openrlhf["g1_gt_embedding"])
    megatron_gen = _flatten_embeddings(megatron["g1_gen_embedding"])
    megatron_gt = _flatten_embeddings(megatron["g1_gt_embedding"])
    openrlhf_gen = _flatten_embeddings(openrlhf["g1_gen_embedding"])
    openrlhf_gt = _flatten_embeddings(openrlhf["g1_gt_embedding"])
    _require_same_shape("gen embeddings", megatron_gen, openrlhf_gen)
    _require_same_shape("gt embeddings", megatron_gt, openrlhf_gt)
    gen_cos = _cosine_stats(megatron_gen, openrlhf_gen)
    gt_cos = _cosine_stats(megatron_gt, openrlhf_gt)
    gen_close = _close_stats(megatron_gen, openrlhf_gen)
    gt_close = _close_stats(megatron_gt, openrlhf_gt)
    gen_rel_l2 = _relative_l2(megatron_gen, openrlhf_gen)
    gt_rel_l2 = _relative_l2(megatron_gt, openrlhf_gt)

    reward_stats = None
    reward_close = None
    reward_rel_l2 = None
    if "scalar_rewards" in megatron or "scalar_rewards" in openrlhf:
        if "scalar_rewards" not in megatron or "scalar_rewards" not in openrlhf:
            raise ValueError("scalar_rewards present in only one dump")
        _require_same_count("scalar_rewards", megatron["scalar_rewards"], openrlhf["scalar_rewards"])
        mrew = torch.as_tensor(megatron["scalar_rewards"], dtype=torch.float32)
        orew = torch.as_tensor(openrlhf["scalar_rewards"], dtype=torch.float32)
        _require_same_shape("scalar_rewards", mrew, orew)
        reward_close = _close_stats(mrew, orew)
        reward_rel_l2 = _relative_l2(mrew, orew)
        reward_stats = {**reward_close, "rel_l2": reward_rel_l2}

    advantage_stats = None
    if "g1_token_advantages" in megatron or "g1_token_advantages" in openrlhf:
        if "g1_token_advantages" not in megatron or "g1_token_advantages" not in openrlhf:
            raise ValueError("g1_token_advantages present in only one dump")
        _require_same_count(
            "g1_token_advantages",
            megatron["g1_token_advantages"],
            openrlhf["g1_token_advantages"],
        )
        ma = torch.stack([t.float() for t in megatron["g1_token_advantages"]], dim=0)
        oa = torch.stack([t.float() for t in openrlhf["g1_token_advantages"]], dim=0)
        _require_same_shape("g1_token_advantages", ma, oa)
        advantage_stats = _advantage_stats(ma, oa)

    ok_hidden, fail_hidden = _check_embedding_family("hidden states", hidden_cos, hidden_close, hidden_rel_l2, emb_thr)
    ok_gen, fail_gen = _check_embedding_family("gen embeddings", gen_cos, gen_close, gen_rel_l2, emb_thr)
    ok_gt, fail_gt = _check_embedding_family("GT embeddings", gt_cos, gt_close, gt_rel_l2, emb_thr)
    ok_rew = True
    fail_rew: list[str] = []
    if reward_stats is not None and reward_close is not None:
        ok_rew, fail_rew = _check_reward_family(reward_close, reward_rel_l2 or 0.0, rew_thr)
    ok_adv = True
    fail_adv: list[str] = []
    if advantage_stats is not None:
        ok_adv, fail_adv = _check_advantage_family(advantage_stats, adv_thr)

    ok_mask_status = mask_status.consistency_ok
    fail_mask_status = list(mask_status.consistency_errors)
    gate_ok = ok_hidden and ok_gen and ok_gt and ok_rew and ok_adv and ok_mask_status
    thd_raw_shape = tuple(megatron["hidden_states_post_sp_gather"].shape)
    total_lengths = [int(x) for x in megatron["total_lengths"]]
    thd_sum = sum(total_lengths)

    report = f"""# G1 Runtime Parity Report

## Inputs

- Megatron dump: `{args.megatron_dump}`
- OpenRLHF dump: `{args.openrlhf_dump}`

## Contract Checks

- Token IDs match: `{token_match}`
- QA masks match: `{qa_match}`
- Position ids match: `{position_match}`
- Attention mask tensors match: `{mask_match}`
- Megatron ref forward mode: `{mask_status.ref_forward_mode}`
- `openrlhf_exact` active: `{mask_status.openrlhf_exact}`
- Dense mask state: `{mask_status.status_label}`
- Dense mask dumped: `{mask_status.dense_mask_dumped}`
- Dense mask built: `{mask_status.dense_mask_built}`
- `apply_dense_attention_mask` effective: `{mask_status.apply_dense_attention_mask}`
- THD fallback active: `{mask_status.thd_fallback}`
- Megatron attention mask raw status: `{mask_status.raw_status}`
- Runtime mask status consistency: `{'PASS' if mask_status.consistency_ok else 'FAIL'}`

## Full-Group Hidden (THD compact vs OpenRLHF)

Megatron stores sequence-parallel gathered hidden in THD form `hidden_states_post_sp_gather` and **compacts** padding using `total_lengths` (sum of lengths must equal THD dim 0). Stacks per-sample slices into a batch tensor **[B, S, H]** using `torch.stack`, which requires **uniform sequence length S across the batch**. If your dump has variable `total_lengths`, align OpenRLHF hidden to the same per-sample slices (or pad both sides to a common max) before running this script.

| Field | Value |
|-------|------|
| Megatron THD raw shape `[S, T, H]` | `{thd_raw_shape}` |
| `total_lengths` (per-sample real tokens) | `{total_lengths}` |
| Sum(total_lengths) == THD S | `{thd_sum == thd_raw_shape[0]}` |
| Compact Megatron hidden `[batch, seq, hid]` | `{tuple(megatron_hidden.shape)}` |
| OpenRLHF hidden `[batch, seq, hid]` | `{tuple(openrlhf_hidden.shape)}` |
| Megatron gen embedding | `{tuple(megatron_gen.shape)}` |
| OpenRLHF gen embedding | `{tuple(openrlhf_gen.shape)}` |

## Mask Summary

- Megatron mask summary: `{megatron_mask_summary}`
- OpenRLHF mask summary: `{openrlhf_mask_summary}`
- Megatron mask status consistency detail:

```text
{chr(10).join(f"- {error}" for error in mask_status.consistency_errors) if not mask_status.consistency_ok else "(none)"}
```

## Strict Parity Gate (thresholds)

Embedding family thresholds: `cosine_min>={emb_thr.cosine_min}`, `max_abs<={emb_thr.max_abs}`, `mean_abs<={emb_thr.mean_abs}`, `rel_l2<={emb_thr.rel_l2_max}`.

Rewards: `max_abs<={rew_thr.max_abs}`, `mean_abs<={rew_thr.mean_abs}`, `rel_l2<={rew_thr.rel_l2_max}`.

Token advantages: `cosine_min>={adv_thr.cosine_min}`, `max_abs<={adv_thr.max_abs}`, `mean_abs<={adv_thr.mean_abs}`, `rel_l2<={adv_thr.rel_l2_max}`.

### Summary

{_family_gate_line("Hidden states", ok_hidden, "embedding thresholds")}
{_family_gate_line("Gen embeddings", ok_gen, "embedding thresholds")}
{_family_gate_line("GT embeddings", ok_gt, "embedding thresholds")}
{_family_gate_line("Scalar rewards", None if reward_stats is None else ok_rew, "not present in both dumps" if reward_stats is None else "reward thresholds")}
{_family_gate_line("Token advantages", None if advantage_stats is None else ok_adv, "not present in both dumps" if advantage_stats is None else "advantage thresholds")}
{_family_gate_line("Megatron dense-mask status", ok_mask_status, "runtime dump metadata consistency")}
- **Overall strict gate**: **{'PASS' if gate_ok else 'FAIL'}**

### Failure detail

```text
{chr(10).join(fail_hidden + fail_gen + fail_gt + fail_rew + fail_adv + fail_mask_status) if not gate_ok else "(none)"}
```

## Parity metrics

### Full Hidden States

Cosine (per position, last dim):

```text
mean   = {hidden_cos['mean']:.8f}
median = {hidden_cos['median']:.8f}
min    = {hidden_cos['min']:.8f}
max    = {hidden_cos['max']:.8f}
```

Abs / relative L2:

```text
max_abs   = {hidden_close['max_abs']:.8e}
mean_abs  = {hidden_close['mean_abs']:.8e}
rel_l2    = {hidden_rel_l2:.8e}
```

### Gen Block Embeddings

```text
mean   = {gen_cos['mean']:.8f}
median = {gen_cos['median']:.8f}
min    = {gen_cos['min']:.8f}
max    = {gen_cos['max']:.8f}
max_abs   = {gen_close['max_abs']:.8e}
mean_abs  = {gen_close['mean_abs']:.8e}
rel_l2    = {gen_rel_l2:.8e}
```

### GT Block Embeddings

```text
mean   = {gt_cos['mean']:.8f}
median = {gt_cos['median']:.8f}
min    = {gt_cos['min']:.8f}
max    = {gt_cos['max']:.8f}
max_abs   = {gt_close['max_abs']:.8e}
mean_abs  = {gt_close['mean_abs']:.8e}
rel_l2    = {gt_rel_l2:.8e}
```

### Rewards

```text
{f'''max_abs   = {reward_stats['max_abs']:.8e}
mean_abs  = {reward_stats['mean_abs']:.8e}
rel_l2    = {reward_stats['rel_l2']:.8e}''' if reward_stats is not None else "not available in both dumps"}
```

### Token Advantages

```text
{f'''per_token_cos_mean = {advantage_stats['sample_cos_mean']:.8f}
per_token_cos_min  = {advantage_stats['sample_cos_min']:.8f}
max_abs              = {advantage_stats['max_abs']:.8e}
mean_abs             = {advantage_stats['mean_abs']:.8e}
rel_l2               = {advantage_stats['rel_l2']:.8e}''' if advantage_stats is not None else "not available in both dumps"}
```

## Interpretation

This report compares the current Slime Megatron/ref fast path against OpenRLHF Critic on identical token IDs and QA masks.

Use the runtime mask status above to distinguish the staged contracts. `openrlhf_exact` means Megatron uses OpenRLHF-compatible position ids/RoPE inputs. `dense-mask-built-not-applied` means the OpenRLHF EBFT dense mask was dumped for comparison but did not affect Megatron attention. `applied-via-torch-thd-fallback` means `apply_dense_attention_mask` was effective through the slow diagnostic torch THD fallback, not the standard Megatron/TE `thd` fast path.
"""
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"[g1-runtime-parity] wrote {output_path}")
    print(report)

    if not args.no_fail_on_strict and not gate_ok:
        print("[g1-runtime-parity] STRICT GATE FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
