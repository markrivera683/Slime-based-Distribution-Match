from __future__ import annotations

from argparse import Namespace

import torch

from slime.rollout.g1_embedding import G1EmbeddingConfig, hidden_states_to_g1_embeddings
from slime.utils.g1_core import (
    compute_pointwise_rewards,
    compute_rloo_shaped_rewards,
    expand_block_rewards_to_token_advantages,
)


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
) -> tuple[list[torch.Tensor], list[float]]:
    """Compute trainer-side G1 token advantages from block embeddings.

    Embedding lists must follow the rollout batch dimension order grouped as
    `n_samples_per_prompt` contiguous samples per prompt (matching the rollout
    group layout). Caller code must preserve that order through the Megatron
    embedding forward (avoid micro-batch reshuffles).
    """
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

    return token_advantages, scalar_rewards
