"""G1 EBFT next-token layout helpers (no Megatron imports; safe for CPU unit tests)."""

from __future__ import annotations

from argparse import Namespace

import torch

from slime.utils.g1_ebft_loss import (
    build_ebft_g1_next_token_tensors,
    ebft_build_next_token_action_qa_advantages,
)


def attach_ebft_g1_next_token_contract_to_batch(batch: dict, args: Namespace) -> None:
    """Populate ``ebft_*`` per-sample tensors when ``--g1-use-ebft-loss`` is set."""
    if batch is None or not bool(getattr(args, "g1_use_ebft_loss", False)):
        return

    advs = batch.get("advantages")
    if advs is None:
        return

    rlens_raw = batch.get("response_lengths")
    t_lens_raw = batch.get("total_lengths")
    sequences = batch.get("g1_full_sequences")
    if sequences is None:
        raise ValueError(
            "EBFT loss requires `g1_full_sequences` on the batch (Megatron train `batch_keys` must "
            "request it; trainer-side G1 rollout supplies this field). "
            "Do not rely on `unconcat_tokens` alone for the EBFT contract."
        )

    rlens = rlens_raw
    if rlens is None:
        rlens = [int(adv.reshape(-1).numel()) for adv in advs]

    t_lens = t_lens_raw
    if t_lens is None:
        t_lens = [int(seq.reshape(-1).numel()) for seq in sequences]
    qa_masking = bool(getattr(args, "g1_qa_masking", False))
    qa_masks = batch.get("g1_qa_masks")
    if qa_masking and qa_masks is None:
        raise ValueError(
            "EBFT loss with `--g1-qa-masking` requires `g1_qa_masks` on the batch "
            "(train `batch_keys` must request it)."
        )
    strict_pl = int(getattr(args, "g1_prompt_length", 384))
    strict_rl = int(getattr(args, "g1_response_length", 376))

    actions: list[torch.Tensor] = []
    qas: list[torch.Tensor] = []
    ebft_advs: list[torch.Tensor] = []

    def _truncate(seq_tensor: torch.Tensor, length: int) -> torch.Tensor:
        flat = seq_tensor.reshape(-1)
        if int(flat.numel()) < length:
            raise ValueError(
                f"sequence tensor shorter than total_lengths ({flat.numel()} < {length}) for EBFT contract"
            )
        return flat[:length]

    for idx, (seq_tensor, RL, adv) in enumerate(zip(sequences, rlens, advs, strict=True)):
        total_len = int(t_lens[idx])
        seq_flat = _truncate(seq_tensor, total_len)
        RL = int(RL)

        qa_tensor = None
        if qa_masking:
            if idx >= len(qa_masks):
                raise ValueError(
                    "EBFT loss with `--g1-qa-masking` requires `g1_qa_masks` with one mask per sample "
                    f"(got {len(qa_masks)} masks for sample index {idx})."
                )
            qa_tensor = _truncate(qa_masks[idx], total_len)
        elif qa_masks is not None and idx < len(qa_masks):
            qa_tensor = _truncate(qa_masks[idx], total_len)

        if total_len == strict_pl + strict_rl and RL == strict_rl:
            a_mask, q_next, a_next = build_ebft_g1_next_token_tensors(
                g1_full_sequence=seq_flat,
                g1_qa_mask=qa_tensor,
                response_advantages=adv,
                g1_prompt_length=strict_pl,
                g1_response_length=strict_rl,
                qa_masking=qa_masking,
            )
        else:
            a_mask, q_next, a_next = ebft_build_next_token_action_qa_advantages(
                full_sequence_1d=seq_flat,
                response_advantages_1d=adv,
                qa_mask_full_1d=qa_tensor,
            )

        actions.append(a_mask)
        qas.append(q_next)
        ebft_advs.append(a_next)

    batch["ebft_action_mask_next"] = actions
    batch["ebft_qa_mask_next"] = qas
    batch["ebft_advantages_next"] = ebft_advs
    batch["ebft_seq_len_m1"] = [int(t.numel()) for t in actions]


__all__ = [
    "build_ebft_g1_next_token_tensors",
    "attach_ebft_g1_next_token_contract_to_batch",
]
