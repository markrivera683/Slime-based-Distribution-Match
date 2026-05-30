from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

from slime.rollout.g1_embedding import G1EmbeddingConfig, hidden_states_to_g1_embeddings
from slime.utils.g1_core import (
    compute_pointwise_rewards,
    compute_rloo_shaped_rewards,
    expand_block_rewards_to_token_advantages,
)
from slime.utils.g2_core import compute_cf_l1oo_rewards, compute_opd_cf_l1oo_rewards
from slime.utils.g3_ema import raise_if_g3_detached_reward_path


def _whiten_g2_standard_student_embeddings(
    gen_embedding: torch.Tensor,
    gt_embedding: torch.Tensor,
    *,
    whiten_tol: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply OpenRLHF-style sample-axis whitening to G2 student gen/gt tensors.

    Standard G2 builds the whitening matrix from actor samples on dim=2 and
    applies it only to student gen/gt embeddings. Teacher targets are produced by
    the teacher branch independently and must not participate in this whitening.
    """
    if gen_embedding.shape != gt_embedding.shape:
        raise ValueError(
            f"G2 gen/gt embedding shapes must match before whitening, got {gen_embedding.shape} vs {gt_embedding.shape}"
        )

    ndim = gen_embedding.ndim
    perm = [0, 1] + list(range(3, ndim - 1)) + [2, ndim - 1]
    inv_perm = [0] * ndim
    for idx, value in enumerate(perm):
        inv_perm[value] = idx

    gen_perm = gen_embedding.permute(*perm).contiguous()
    gt_perm = gt_embedding.permute(*perm).contiguous()

    *batch_dims, n_samples, embed_dim = gen_perm.shape
    batch_size = 1
    for dim in batch_dims:
        batch_size *= int(dim)

    gen_flat = gen_perm.reshape(batch_size, n_samples, embed_dim).float()
    gt_flat = gt_perm.reshape(batch_size, n_samples, embed_dim).float()

    try:
        u, singular_values, _ = torch.linalg.svd(gen_flat, full_matrices=False)
    except torch._C._LinAlgError:
        noise_scale = 1e-6 * gen_flat.abs().mean()
        gen_noisy = gen_flat + noise_scale * torch.randn_like(gen_flat)
        try:
            u, singular_values, _ = torch.linalg.svd(gen_noisy, full_matrices=False)
        except torch._C._LinAlgError:
            return gen_embedding, gt_embedding

    max_singular_value = singular_values.max(dim=-1, keepdim=True).values
    inv_s = torch.where(
        singular_values > float(whiten_tol) * max_singular_value,
        1.0 / (singular_values + 1e-12),
        torch.zeros_like(singular_values),
    )
    whitening = (u * inv_s.unsqueeze(-2)) @ u.transpose(-1, -2)

    gen_whitened = (whitening @ gen_flat).to(dtype=gen_embedding.dtype).reshape(*batch_dims, n_samples, embed_dim)
    gt_whitened = (whitening @ gt_flat).to(dtype=gt_embedding.dtype).reshape(*batch_dims, n_samples, embed_dim)

    return (
        gen_whitened.permute(*inv_perm).contiguous(),
        gt_whitened.permute(*inv_perm).contiguous(),
    )


def _teacher_log_probs_to_group_scores(
    *,
    teacher_log_probs: list[torch.Tensor] | None,
    num_groups: int,
    n_samples_per_prompt: int,
    device: torch.device,
    normalization: str,
) -> torch.Tensor:
    if teacher_log_probs is None:
        raise ValueError("OPD-CF-L1OO requires teacher_log_probs.")
    expected = int(num_groups) * int(n_samples_per_prompt)
    if len(teacher_log_probs) != expected:
        raise ValueError(
            "OPD-CF-L1OO teacher_log_probs must align with rollout samples; "
            f"got {len(teacher_log_probs)} for {expected} samples."
        )

    scores = []
    for sample_idx, log_probs in enumerate(teacher_log_probs):
        tensor = log_probs.detach().float().reshape(-1).to(device=device)
        if tensor.numel() == 0:
            raise ValueError(f"OPD-CF-L1OO teacher_log_probs[{sample_idx}] is empty.")
        if normalization == "mean":
            score = tensor.mean()
        elif normalization == "sum":
            score = tensor.sum()
        else:
            raise ValueError(
                "Unsupported --opd-cf-score-normalization "
                f"{normalization!r}; expected 'mean' or 'sum'."
            )
        scores.append(score)

    return torch.stack(scores, dim=0).view(1, int(num_groups), int(n_samples_per_prompt))


@torch.no_grad()
def _compute_opd_ebft_credit_rewards(
    gen_embedding: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    reward_scale: float,
) -> torch.Tensor:
    if gen_embedding.ndim != 5 or teacher_scores.ndim != 3:
        raise ValueError(
            "OPD-EBFT credit expects gen_embedding (B, G, N, K, D) and teacher_scores (B, G, N), "
            f"got gen={tuple(gen_embedding.shape)} scores={tuple(teacher_scores.shape)}"
        )
    if teacher_scores.shape != gen_embedding.shape[:3]:
        raise ValueError(
            "OPD-EBFT teacher_scores must align with gen_embedding on B/G/N dims, "
            f"got scores={tuple(teacher_scores.shape)} gen={tuple(gen_embedding.shape)}"
        )

    weights = torch.softmax(teacher_scores.float(), dim=2).to(device=gen_embedding.device, dtype=gen_embedding.dtype)
    target = (gen_embedding * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=2, keepdim=True)
    rewards = torch.nn.functional.cosine_similarity(gen_embedding, target.expand_as(gen_embedding), dim=-1)
    return rewards * float(reward_scale)


def g1_config_from_args(args: Namespace) -> G1EmbeddingConfig:
    """Build the shared strict G1 geometry used by the Megatron fast path."""
    return G1EmbeddingConfig(
        prompt_length=int(getattr(args, "g1_prompt_length", 384)),
        context_length=int(getattr(args, "g1_context_length", 8)),
        generate_length=int(getattr(args, "g1_generate_length", 8)),
        stride=int(getattr(args, "g1_stride", 8)),
        response_length=int(getattr(args, "g1_response_length", 376)),
        n_samples_per_prompt=int(getattr(args, "n_samples_per_prompt", 4)),
        hidden_state_method=str(getattr(args, "g1_hidden_state_method", "last_only")),
        qa_masking=bool(getattr(args, "g1_qa_masking", False)),
        document_masking=bool(getattr(args, "g1_document_masking", False)),
    )


def build_openrlhf_g1_attention_mask_and_position_ids(
    args: Namespace,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    doc_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror OpenRLHF's EBFT strided mask and position-id construction.

    This helper intentionally lives in the Megatron fast path so the ref forward,
    runtime dumps, and parity scripts all share one Slime-side definition of the
    OpenRLHF contract.
    """
    config = g1_config_from_args(args)
    if config.document_masking and doc_ids is None:
        raise ValueError("G1 document masking requires doc_ids")

    full_sequence_length = config.prompt_length + config.response_length
    num_blocks = config.num_blocks
    if config.response_length != config.generate_length * num_blocks:
        raise ValueError(
            "OpenRLHF G1 response layout requires "
            f"response_length == generate_length * num_blocks, got {config.response_length} and "
            f"{config.generate_length} * {num_blocks}"
        )

    if doc_ids is None:
        doc_ids = torch.zeros((batch_size, config.prompt_length), dtype=torch.long, device=device)
    else:
        doc_ids = doc_ids.to(device=device, dtype=torch.long)
        if doc_ids.shape != (batch_size, config.prompt_length):
            raise ValueError(f"Expected doc_ids shape {(batch_size, config.prompt_length)}, got {tuple(doc_ids.shape)}")

    min_value = torch.finfo(dtype).min
    attention_mask = torch.full(
        (batch_size, 1, full_sequence_length, full_sequence_length),
        min_value,
        dtype=dtype,
        device=device,
    )

    same_doc_mask = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
    same_doc_mask = same_doc_mask.unsqueeze(1)
    if not config.document_masking:
        same_doc_mask = same_doc_mask.fill_(True)

    causal_mask = torch.tril(torch.ones((config.prompt_length, config.prompt_length), dtype=torch.bool, device=device))
    causal_mask = causal_mask.view(1, 1, config.prompt_length, config.prompt_length)
    prompt_allowed = causal_mask & same_doc_mask[:, :, : config.prompt_length, : config.prompt_length]
    attention_mask[:, :, : config.prompt_length, : config.prompt_length].masked_fill_(prompt_allowed, 0.0)

    for gen_step in range(config.generate_length):
        for block_idx in range(num_blocks):
            generated_token_position = config.prompt_length + gen_step * num_blocks + block_idx
            context_window_end = min(
                block_idx * config.stride + config.context_length,
                config.prompt_length - config.generate_length,
            )
            cur_doc_ids = doc_ids[:, context_window_end + gen_step].unsqueeze(-1)
            context_doc_ids = doc_ids[:, :context_window_end]
            context_same_doc_idx = context_doc_ids == cur_doc_ids
            if not config.document_masking:
                context_same_doc_idx = context_same_doc_idx.fill_(True)

            row = attention_mask[:, 0, generated_token_position, :context_window_end]
            row[context_same_doc_idx] = 0.0
            attention_mask[:, 0, generated_token_position, generated_token_position] = 0.0

            if gen_step > 0:
                cur_anchor_idx = context_window_end + gen_step
                cur_doc = doc_ids[:, cur_anchor_idx]
                for prev_s in range(gen_step):
                    prev_anchor_idx = context_window_end + prev_s
                    prev_pos = config.prompt_length + prev_s * num_blocks + block_idx
                    same_doc_prev = doc_ids[:, prev_anchor_idx] == cur_doc
                    if not config.document_masking:
                        same_doc_prev = same_doc_prev.fill_(True)
                    attention_mask[same_doc_prev, 0, generated_token_position, prev_pos] = 0.0

    position_ids = torch.empty((batch_size, full_sequence_length), dtype=torch.long, device=device)
    if config.document_masking:
        boundaries = torch.zeros_like(doc_ids, dtype=torch.bool)
        boundaries[:, 0] = True
        boundaries[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
        global_pos = torch.arange(config.prompt_length, device=device).unsqueeze(0).expand(batch_size, -1)
        segment_start_pos = global_pos.masked_fill(~boundaries, 0)
        segment_start_pos, _ = torch.cummax(segment_start_pos, dim=1)
        prompt_positions = global_pos - segment_start_pos
        position_ids[:, : config.prompt_length] = prompt_positions
        block_starting_idx = torch.arange(0, num_blocks, device=device) * config.stride + config.context_length
        block_starting_positions = position_ids[:, block_starting_idx]
    else:
        position_ids[:, : config.prompt_length] = torch.arange(config.prompt_length, device=device)
        block_starting_positions = torch.arange(0, num_blocks, device=device) * config.stride + config.context_length
        block_starting_positions = block_starting_positions.view(1, num_blocks).expand(batch_size, -1)

    for gen_step in range(config.generate_length):
        step_start_idx = config.prompt_length + gen_step * num_blocks
        step_end_idx = step_start_idx + num_blocks
        position_ids[:, step_start_idx:step_end_idx] = block_starting_positions + gen_step

    return attention_mask, position_ids


def build_megatron_rotary_pos_emb_from_position_ids(rotary_pos_emb, position_ids: torch.Tensor) -> torch.Tensor:
    """Build Megatron RoPE frequencies for arbitrary per-token position ids."""
    if position_ids.ndim != 2 or position_ids.shape[0] != 1:
        raise ValueError(f"Expected packed THD position_ids with shape [1, T], got {tuple(position_ids.shape)}")

    inv_freq = rotary_pos_emb.inv_freq
    if inv_freq.device != position_ids.device:
        inv_freq = inv_freq.to(position_ids.device)
    positions = position_ids.reshape(-1).to(dtype=inv_freq.dtype)
    if getattr(rotary_pos_emb, "seq_len_interpolation_factor", None) is not None:
        positions = positions * (1.0 / rotary_pos_emb.seq_len_interpolation_factor)

    freqs = torch.outer(positions, inv_freq)
    if not rotary_pos_emb.rotary_interleaved:
        emb = torch.cat((freqs, freqs), dim=-1)
    else:
        emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(freqs.shape[0], -1)
    return emb[:, None, None, :]


def pack_openrlhf_g1_attention_mask(
    attention_mask: torch.Tensor,
    *,
    total_lengths: list[int],
    padded_total_length: int | None = None,
) -> torch.Tensor:
    """Pack per-sample OpenRLHF dense masks into Megatron THD token order."""
    if attention_mask.ndim != 4 or attention_mask.shape[1] != 1:
        raise ValueError(f"Expected attention mask [B, 1, S, S], got {tuple(attention_mask.shape)}")
    if len(total_lengths) != attention_mask.shape[0]:
        raise ValueError(
            f"total_lengths has {len(total_lengths)} entries but attention mask batch is {attention_mask.shape[0]}"
        )

    actual_total = sum(int(length) for length in total_lengths)
    packed_length = int(padded_total_length) if padded_total_length is not None else actual_total
    if packed_length < actual_total:
        raise ValueError(f"padded_total_length {packed_length} is shorter than packed token length {actual_total}")

    min_value = torch.finfo(attention_mask.dtype).min
    packed_mask = torch.full(
        (1, 1, packed_length, packed_length),
        min_value,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    cursor = 0
    for sample_idx, length in enumerate(total_lengths):
        length = int(length)
        sample_mask = attention_mask[sample_idx : sample_idx + 1, :, :length, :length]
        packed_mask[:, :, cursor : cursor + length, cursor : cursor + length] = sample_mask
        cursor += length

    if packed_length > actual_total:
        pad_idx = torch.arange(actual_total, packed_length, device=attention_mask.device)
        packed_mask[:, :, pad_idx, pad_idx] = 0.0

    return packed_mask


def openrlhf_dense_mask_thd_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    packed_attention_mask: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Small ref-only THD attention fallback that consumes an OpenRLHF dense mask."""
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError(
            "Expected THD query/key/value shapes [T, H, D], got "
            f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}"
        )
    if packed_attention_mask.ndim != 4 or packed_attention_mask.shape[:2] != (1, 1):
        raise ValueError(f"Expected packed mask [1, 1, T, T], got {tuple(packed_attention_mask.shape)}")

    query_length, query_heads, head_dim = query.shape
    key_length, key_heads, key_dim = key.shape
    value_length, value_heads, value_dim = value.shape
    if key_length != value_length or key_heads != value_heads or key_dim != head_dim:
        raise ValueError(
            "Incompatible THD attention shapes: "
            f"query={tuple(query.shape)} key={tuple(key.shape)} value={tuple(value.shape)}"
        )
    if packed_attention_mask.shape[-2] < query_length or packed_attention_mask.shape[-1] < key_length:
        raise ValueError(
            f"Packed mask {tuple(packed_attention_mask.shape)} is too small for query/key lengths "
            f"{query_length}/{key_length}"
        )
    if query_heads % key_heads != 0:
        raise ValueError(f"Query heads {query_heads} must be divisible by key/value heads {key_heads}")
    if query_heads != key_heads:
        repeat = query_heads // key_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)

    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    scores = torch.einsum("thd,shd->hts", query.float(), key.float()) * float(scale)
    mask = packed_attention_mask[:, :, :query_length, :key_length].to(device=scores.device, dtype=scores.dtype)
    scores = scores + mask[0]
    probs = torch.softmax(scores, dim=-1).to(dtype=value.dtype)
    output = torch.einsum("hts,shd->thd", probs, value)
    return output.to(dtype=query.dtype)


def _as_bsh(hidden_states: torch.Tensor, *, qkv_format: str) -> torch.Tensor:
    """Normalize Megatron decoder hidden layout to [B, S, H] for first fast path."""
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected Megatron hidden states with 3 dims, got {hidden_states.shape}")

    if qkv_format == "thd":
        # Megatron decoder returns [S, B, H], and slime uses B=1 for THD packed streams.
        if hidden_states.shape[1] != 1:
            raise ValueError(f"Expected THD hidden states [S, 1, H], got {hidden_states.shape}")
        return hidden_states.transpose(0, 1).contiguous()

    if qkv_format == "bshd":
        return hidden_states.contiguous()

    raise ValueError(f"Unsupported qkv_format for G1 Megatron embeddings: {qkv_format}")


def megatron_hidden_to_g1_embeddings(
    hidden_states: torch.Tensor,
    *,
    args: Namespace,
    total_lengths: list[int],
    g1_qa_masks: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Convert Megatron final hidden states into per-sample G1 gen/gt embeddings.

    The first Megatron fast path mirrors OpenRLHF's `last_only` hidden-state
    method: final hidden states are L2-normalized per token, expanded with a
    singleton feature axis, then passed through the shared block/groom helper.
    """
    if getattr(args, "g1_hidden_state_method", "last_only") != "last_only":
        raise ValueError("Megatron G1 fast path currently supports only g1_hidden_state_method='last_only'")
    if bool(getattr(args, "g1_document_masking", False)):
        raise ValueError("Megatron G1 fast path does not yet support g1_document_masking")

    from megatron.core import mpu

    if mpu.get_context_parallel_world_size() != 1:
        raise ValueError("Megatron G1 fast path currently requires context_parallel_world_size == 1")

    config = g1_config_from_args(args)
    hidden_bsh = _as_bsh(hidden_states, qkv_format=args.qkv_format).float()
    if hidden_bsh.shape[0] != 1:
        raise ValueError(f"Expected one packed THD batch, got hidden shape {hidden_bsh.shape}")
    hidden_flat = hidden_bsh[0]

    gen_embeddings: list[torch.Tensor] = []
    gt_embeddings: list[torch.Tensor] = []
    cursor = 0
    for sample_idx, (total_length, qa_mask) in enumerate(zip(total_lengths, g1_qa_masks, strict=True)):
        expected_length = config.prompt_length + config.response_length
        if int(total_length) != expected_length:
            raise ValueError(
                f"G1 Megatron sequence length {total_length} != expected {expected_length} at sample {sample_idx}"
            )
        sample_hidden = hidden_flat[cursor : cursor + total_length]
        cursor += total_length
        if sample_hidden.shape[0] != total_length:
            raise ValueError(f"Missing hidden states for G1 sample {sample_idx}: got {sample_hidden.shape[0]}")

        qa_mask = qa_mask.to(device=sample_hidden.device, dtype=torch.long)
        if qa_mask.numel() != total_length:
            raise ValueError(f"G1 qa mask length {qa_mask.numel()} != total_length {total_length} at sample {sample_idx}")

        sample_hidden = torch.nn.functional.normalize(sample_hidden, p=2, dim=-1)
        sample_hidden = sample_hidden.view(1, total_length, 1, sample_hidden.shape[-1])
        gen_embedding, gt_embedding = hidden_states_to_g1_embeddings(sample_hidden, qa_mask.view(1, -1), config)
        gen_embeddings.append(gen_embedding[0])
        gt_embeddings.append(gt_embedding[0])

    return gen_embeddings, gt_embeddings


def compute_g1_token_advantages_from_embeddings(
    args: Namespace,
    gen_embeddings: list[torch.Tensor],
    gt_embeddings: list[torch.Tensor],
    response_lengths: list[int],
    teacher_gen_embeddings: list[torch.Tensor] | None = None,
    *,
    teacher_log_probs: list[torch.Tensor] | None = None,
    g2_runtime_dump_path: str | None = None,
    g2_dump_writer_metadata: dict[str, Any] | None = None,
) -> tuple[list[torch.Tensor], list[float]]:
    """Compute trainer-side G1 token advantages from block embeddings.

    Embedding lists must follow the rollout batch dimension order grouped as
    `n_samples_per_prompt` contiguous samples per prompt (matching the rollout
    group layout). Caller code must preserve that order through the Megatron
    embedding forward (avoid micro-batch reshuffles).
    """
    raise_if_g3_detached_reward_path(args)
    if not gen_embeddings:
        return [], []
    if len(gen_embeddings) != len(gt_embeddings) or len(gen_embeddings) != len(response_lengths):
        raise ValueError("G1 embeddings and response_lengths must have the same number of samples")

    n_samples_per_prompt = int(getattr(args, "n_samples_per_prompt", len(gen_embeddings)))
    if n_samples_per_prompt <= 1:
        raise ValueError("G1 requires n_samples_per_prompt > 1")
    if len(gen_embeddings) % n_samples_per_prompt != 0:
        raise ValueError(
            f"G1 sample count {len(gen_embeddings)} is not divisible by n_samples_per_prompt={n_samples_per_prompt}"
        )

    gen = torch.stack([t.float() for t in gen_embeddings], dim=0)
    gt = torch.stack([t.float() for t in gt_embeddings], dim=0).to(device=gen.device)
    if gen.shape != gt.shape:
        raise ValueError(f"G1 gen/gt embedding shapes must match, got {gen.shape} vs {gt.shape}")

    num_samples, num_blocks, _ = gen.shape
    expected_num_blocks = getattr(args, "g1_num_blocks", None)
    if expected_num_blocks is not None and num_blocks != int(expected_num_blocks):
        raise ValueError(f"G1 num_blocks {num_blocks} != expected {int(expected_num_blocks)}")

    num_groups = num_samples // n_samples_per_prompt
    gen = gen.view(num_groups, n_samples_per_prompt, num_blocks, gen.shape[-1]).unsqueeze(0)
    gt = gt.view(num_groups, n_samples_per_prompt, num_blocks, gt.shape[-1]).unsqueeze(0)

    g2_runtime_payload: dict[str, Any] | None = None
    if getattr(args, "distribution_reward_type", "pointwise") == "cf_l1oo":
        cf_target_mode = getattr(args, "cf_target_mode", None)
        if cf_target_mode not in {"single", "teacher", "opd_onpolicy"}:
            raise ValueError(
                "G2 requires --cf-target-mode single, teacher, or opd_onpolicy with --distribution-reward-type cf_l1oo"
            )
        if cf_target_mode == "teacher" and teacher_gen_embeddings is None:
            raise NotImplementedError(
                "Standard G2 cf_l1oo teacher mode requires rollout_data['g2_teacher_gen_embeddings'] "
                "with one [M, num_blocks, hidden_dim] teacher embedding tensor per sample/group. "
                "Use the default SGLang rollout with --teacher-backend remote and compute trainer-side "
                "embeddings on the frozen critic, or provide this contract via a custom rollout/convert hook."
            )
        if cf_target_mode == "opd_onpolicy" and teacher_log_probs is None:
            raise ValueError("OPD-CF-L1OO requires rollout_data['teacher_log_probs'].")
        if teacher_gen_embeddings is not None and len(teacher_gen_embeddings) != len(gen_embeddings):
            raise ValueError(
                "G2 teacher_gen_embeddings must align with samples; "
                f"got {len(teacher_gen_embeddings)} teacher entries for {len(gen_embeddings)} samples"
            )
        teacher_by_group = []
        teacher = None
        if cf_target_mode == "teacher":
            for group_idx in range(num_groups):
                first_idx = group_idx * n_samples_per_prompt
                group_teacher = teacher_gen_embeddings[first_idx].float().to(device=gen.device)
                if group_teacher.ndim != 3:
                    raise ValueError(
                        "Each G2 teacher embedding entry must have shape [M, num_blocks, hidden_dim], "
                        f"got {tuple(group_teacher.shape)} at group {group_idx}"
                    )
                if group_teacher.shape[1:] != gen.shape[3:]:
                    raise ValueError(
                        "G2 teacher embedding block/feature dims must match actor embeddings, "
                        f"got teacher={tuple(group_teacher.shape)} actor={tuple(gen.shape[3:])}"
                    )
                for sample_offset in range(1, n_samples_per_prompt):
                    sample_idx = first_idx + sample_offset
                    candidate = teacher_gen_embeddings[sample_idx].float().to(device=gen.device)
                    if candidate.shape != group_teacher.shape:
                        raise ValueError(
                            "All samples in a G2 prompt group must carry the same group-level teacher embedding shape; "
                            f"group={group_idx} first={tuple(group_teacher.shape)} sample={sample_idx} got={tuple(candidate.shape)}"
                        )
                    if not torch.allclose(candidate, group_teacher, rtol=1e-5, atol=1e-6):
                        raise ValueError(
                            "All samples in a G2 prompt group must carry identical group-level teacher embeddings; "
                            f"mismatch at group={group_idx} sample={sample_idx}."
                        )
                teacher_by_group.append(group_teacher)
            teacher = torch.stack(teacher_by_group, dim=0).unsqueeze(0)
        if not bool(getattr(args, "use_whitening", False)):
            raise ValueError("Standard G2 requires --use-whitening to match OpenRLHF reward construction.")
        raw_gen = gen
        raw_gt = gt
        if cf_target_mode == "opd_onpolicy":
            gen, _ = _whiten_g2_standard_student_embeddings(
                gen,
                gen,
                whiten_tol=float(getattr(args, "whiten_tol", 1e-5)),
            )
        else:
            gen, gt = _whiten_g2_standard_student_embeddings(
                gen,
                gt,
                whiten_tol=float(getattr(args, "whiten_tol", 1e-5)),
            )
        cf_args = {
            "cf_num_freqs": int(getattr(args, "cf_num_freqs", 128)),
            "cf_sigma": float(getattr(args, "cf_sigma", 1.0)),
            "cf_seed": int(getattr(args, "cf_seed", 43)),
            "cf_alpha": float(getattr(args, "cf_alpha", 0.5)),
            "cf_beta": float(getattr(args, "cf_beta", 0.5)),
            "cf_reward_scale": float(getattr(args, "cf_reward_scale", 1.0)),
            "cf_teacher_lambda": float(getattr(args, "cf_teacher_lambda", 0.0)),
            "cf_target_mode": cf_target_mode,
            "cf_target_num_refs": int(getattr(args, "cf_target_num_refs", 1)),
            "cf_target_std": float(getattr(args, "cf_target_std", 0.05)),
            "cf_target_seed": int(getattr(args, "cf_target_seed", 43)),
            "opd_cf_score_temperature": float(getattr(args, "opd_cf_score_temperature", 1.0)),
            "opd_cf_score_normalization": str(getattr(args, "opd_cf_score_normalization", "mean")),
            "opd_credit_assignment": str(getattr(args, "opd_credit_assignment", "cf_l1oo")),
        }
        teacher_scores = None
        if cf_target_mode == "opd_onpolicy":
            teacher_scores = _teacher_log_probs_to_group_scores(
                teacher_log_probs=teacher_log_probs,
                num_groups=num_groups,
                n_samples_per_prompt=n_samples_per_prompt,
                device=gen.device,
                normalization=cf_args["opd_cf_score_normalization"],
            )
            teacher_scores = teacher_scores / cf_args["opd_cf_score_temperature"]
            if cf_args["opd_credit_assignment"] == "cf_l1oo":
                rewards = compute_opd_cf_l1oo_rewards(
                    gen,
                    teacher_scores,
                    cf_num_freqs=cf_args["cf_num_freqs"],
                    cf_sigma=cf_args["cf_sigma"],
                    cf_seed=cf_args["cf_seed"],
                    cf_alpha=cf_args["cf_alpha"],
                    cf_beta=cf_args["cf_beta"],
                    cf_reward_scale=cf_args["cf_reward_scale"],
                    score_temperature=1.0,
                )
            elif cf_args["opd_credit_assignment"] == "ebft":
                rewards = _compute_opd_ebft_credit_rewards(
                    gen,
                    teacher_scores,
                    reward_scale=cf_args["cf_reward_scale"],
                )
            else:
                raise ValueError(
                    "Unsupported --opd-credit-assignment "
                    f"{cf_args['opd_credit_assignment']!r}; expected 'cf_l1oo' or 'ebft'."
                )
        else:
            rewards = compute_cf_l1oo_rewards(
                gen,
                gt,
                teacher_embedding=teacher,
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
                cf_teacher_lambda=cf_args["cf_teacher_lambda"],
            )
        if g2_runtime_dump_path:
            g2_runtime_payload = {
                "source": "slime_megatron_standard_g2_runtime",
                "g2_dump_writer_metadata": dict(g2_dump_writer_metadata or {}),
                "g2_raw_student_gen_embeddings": [t.detach().cpu() for t in gen_embeddings],
                "g2_raw_student_gt_embeddings": [t.detach().cpu() for t in gt_embeddings],
                "g2_raw_student_gen_tensor": raw_gen.detach().cpu(),
                "g2_raw_student_gt_tensor": raw_gt.detach().cpu(),
                "g2_teacher_gen_embeddings": [t.detach().cpu() for t in teacher_gen_embeddings] if teacher_gen_embeddings is not None else None,
                "g2_teacher_gen_tensor": teacher.detach().cpu() if teacher is not None else None,
                "opd_cf_teacher_scores": teacher_scores.detach().cpu() if teacher_scores is not None else None,
                "g2_whitened_student_gen_tensor": gen.detach().cpu(),
                "g2_whitened_student_gt_tensor": gt.detach().cpu(),
                "g2_cf_l1oo_rewards": rewards.detach().cpu(),
                "g2_cf_args": cf_args,
                "g2_shape_metadata": {
                    "num_samples": int(num_samples),
                    "num_groups": int(num_groups),
                    "n_samples_per_prompt": int(n_samples_per_prompt),
                    "num_blocks": int(num_blocks),
                    "embedding_dim": int(gen.shape[-1]),
                    "teacher_samples_per_group": int(teacher.shape[2]) if teacher is not None else 0,
                    "raw_student_shape": list(raw_gen.shape),
                    "teacher_shape": list(teacher.shape) if teacher is not None else None,
                    "whitened_student_shape": list(gen.shape),
                    "cf_reward_shape": list(rewards.shape),
                },
                "g1_prompt_length": int(getattr(args, "g1_prompt_length", 384)),
                "g1_context_length": int(getattr(args, "g1_context_length", 8)),
                "g1_generate_length": int(getattr(args, "g1_generate_length", 8)),
                "g1_stride": int(getattr(args, "g1_stride", 8)),
                "g1_response_length": int(getattr(args, "g1_response_length", 376)),
                "n_samples_per_prompt": int(n_samples_per_prompt),
                "response_lengths": [int(x) for x in response_lengths],
                "use_whitening": bool(getattr(args, "use_whitening", False)),
                "whiten_tol": float(getattr(args, "whiten_tol", 1e-5)),
            }
        if cf_target_mode == "opd_onpolicy" and cf_args["opd_credit_assignment"] == "ebft":
            rloo_rewards = rewards.reshape(rewards.shape[0], num_groups * n_samples_per_prompt, num_blocks)
            shaped_rewards, _ = compute_rloo_shaped_rewards(
                rloo_rewards,
                torch.zeros_like(rloo_rewards),
                rloo_rewards,
                n_samples_per_prompt=n_samples_per_prompt,
                alignment_rew_coef=1.0,
                diversity_rew_coef=0.0,
            )
            shaped_rewards = shaped_rewards.reshape_as(rewards)
        else:
            shaped_rewards = rewards
    else:
        alignment_rew_coef = float(getattr(args, "alignment_rew_coef", 1.0))
        diversity_rew_coef = float(getattr(args, "diversity_rew_coef", 1.0))
        rewards, gt_rewards, diversity_rewards = compute_pointwise_rewards(
            gen,
            gt,
            alignment_rew_coef=alignment_rew_coef,
            diversity_rew_coef=diversity_rew_coef,
            use_whitening=bool(getattr(args, "use_whitening", True)),
        )
        shaped_rewards, _ = compute_rloo_shaped_rewards(
            rewards,
            diversity_rewards,
            gt_rewards,
            n_samples_per_prompt=n_samples_per_prompt,
            alignment_rew_coef=alignment_rew_coef,
            diversity_rew_coef=diversity_rew_coef,
        )

    rewards = rewards.squeeze(0).reshape(num_samples, num_blocks)
    shaped_rewards = shaped_rewards.squeeze(0).reshape(num_samples, num_blocks)

    token_advantages: list[torch.Tensor] = []
    scalar_rewards: list[float] = []
    for idx, response_length in enumerate(response_lengths):
        expected_response_length = getattr(args, "g1_response_length", None)
        if expected_response_length is not None and int(response_length) != int(expected_response_length):
            raise ValueError(
                f"G1 response_length {response_length} != expected {int(expected_response_length)} at sample {idx}"
            )
        if int(response_length) % num_blocks != 0:
            raise ValueError(f"response_length {response_length} is not divisible by G1 num_blocks {num_blocks}")
        generate_length = int(response_length) // num_blocks
        token_advantages.append(
            expand_block_rewards_to_token_advantages(
                shaped_rewards[idx],
                generate_length=generate_length,
                response_length=int(response_length),
            ).detach()
        )
        scalar_rewards.append(float(rewards[idx].mean().detach().cpu().item()))

    if g2_runtime_payload is not None:
        output_path = Path(g2_runtime_dump_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        g2_runtime_payload["token_advantages"] = [t.detach().cpu() for t in token_advantages]
        g2_runtime_payload["g1_token_advantages"] = [t.detach().cpu() for t in token_advantages]
        g2_runtime_payload["scalar_rewards"] = [float(x) for x in scalar_rewards]
        g2_runtime_payload["g2_block_rewards_flat"] = rewards.detach().cpu()
        g2_runtime_payload["g2_shaped_block_rewards_flat"] = shaped_rewards.detach().cpu()
        g2_runtime_payload["g2_shape_metadata"]["token_advantage_lengths"] = [
            int(t.numel()) for t in token_advantages
        ]
        torch.save(g2_runtime_payload, output_path)

    return token_advantages, scalar_rewards
