#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import torch


def _load_python_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _groom_last_token(block_hidden_states: torch.Tensor, block_qa_mask: torch.Tensor, *, qa_masking: bool) -> torch.Tensor:
    if not qa_masking:
        block_qa_mask = torch.ones_like(block_qa_mask)
    time_idx = torch.arange(block_hidden_states.shape[2], device=block_hidden_states.device)
    view_shape = [1] * block_hidden_states.ndim
    view_shape[2] = block_hidden_states.shape[2]
    time_idx = time_idx.view(*view_shape[:-1], 1)
    mask = block_qa_mask.bool()
    last_idx = time_idx.masked_fill(~mask, -1).amax(dim=2)
    safe_idx = last_idx.clamp_min(0).unsqueeze(2).expand(
        *block_hidden_states.shape[:2],
        1,
        block_hidden_states.shape[3],
        block_hidden_states.shape[4],
    )
    selected = block_hidden_states.gather(dim=2, index=safe_idx).squeeze(2)
    valid = (last_idx >= 0).to(dtype=selected.dtype)
    return selected * valid


def _attention_mask_summary(attention_mask: torch.Tensor) -> dict[str, object]:
    finite = torch.isfinite(attention_mask)
    allowed = attention_mask == 0
    return {
        "shape": tuple(attention_mask.shape),
        "dtype": str(attention_mask.dtype),
        "allowed_count": int(allowed.sum().item()),
        "blocked_count": int((~allowed).sum().item()),
        "finite_count": int(finite.sum().item()),
    }


def hidden_states_to_g1_embeddings(
    hidden_states: torch.Tensor,
    qa_masks: torch.Tensor,
    *,
    prompt_length: int,
    context_length: int,
    generate_length: int,
    stride: int,
    response_length: int,
    qa_masking: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = (prompt_length - generate_length - context_length) // stride + 1
    gt_hidden = hidden_states[:, context_length:prompt_length, :, :]
    gen_hidden = hidden_states[:, prompt_length:, :, :]
    gt_qa = qa_masks[:, context_length:prompt_length].view(gt_hidden.shape[0], gt_hidden.shape[1], 1, 1).repeat(
        1, 1, gt_hidden.shape[2], 1
    )
    gen_qa = qa_masks[:, prompt_length:].view(gen_hidden.shape[0], gen_hidden.shape[1], 1, 1).repeat(
        1, 1, gen_hidden.shape[2], 1
    )
    gt_blocks = gt_hidden.unfold(-3, generate_length, stride).permute(0, 1, 4, 2, 3)
    gt_qa_blocks = gt_qa.unfold(-3, generate_length, stride).permute(0, 1, 4, 2, 3)
    gen_blocks = gen_hidden.reshape(
        gen_hidden.shape[0],
        generate_length,
        num_blocks,
        gen_hidden.shape[-2],
        gen_hidden.shape[-1],
    ).transpose(-3, -4)
    gen_qa_blocks = gen_qa.reshape(
        gen_hidden.shape[0],
        generate_length,
        num_blocks,
        gen_hidden.shape[-2],
        1,
    ).transpose(-3, -4)
    gt_embedding = _groom_last_token(gt_blocks, gt_qa_blocks, qa_masking=qa_masking)
    gen_embedding = _groom_last_token(gen_blocks, gen_qa_blocks, qa_masking=qa_masking)
    return (
        gen_embedding.reshape(gen_embedding.shape[0], gen_embedding.shape[1], -1).float(),
        gt_embedding.reshape(gt_embedding.shape[0], gt_embedding.shape[1], -1).float(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-dump", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-path", default="/mnt/data/models/Qwen3.5-4B")
    parser.add_argument("--openrlhf-repo", default="/mnt/data/ebft-distribution-new/code")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()

    megatron_dump = torch.load(args.megatron_dump, map_location="cpu", weights_only=False)
    repo = Path(args.openrlhf_repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    critic_module = _load_python_module("g1_runtime_openrlhf_critic", repo / "openrlhf" / "models" / "critic.py")
    model_utils = _load_python_module("g1_runtime_openrlhf_model_utils", repo / "openrlhf" / "models" / "utils.py")
    Critic = critic_module.Critic

    prompt_length = int(megatron_dump["g1_prompt_length"])
    context_length = int(megatron_dump["g1_context_length"])
    generate_length = int(megatron_dump["g1_generate_length"])
    stride = int(megatron_dump["g1_stride"])
    response_length = int(megatron_dump["g1_response_length"])
    num_blocks = (prompt_length - generate_length - context_length) // stride + 1

    sequences = torch.stack([t.long() for t in megatron_dump["tokens"]], dim=0)
    qa_masks = torch.stack([t.long() for t in megatron_dump["g1_qa_masks"]], dim=0)
    doc_ids = torch.zeros((sequences.shape[0], prompt_length), dtype=torch.long)

    device = torch.device(args.device)
    bf16 = args.dtype == "bfloat16"
    critic = Critic(
        args.model_path,
        use_flash_attention_2=False,
        bf16=bf16,
        critic_sequence_level="last_token",
        gen_len=generate_length,
        hidden_state_method="last_only",
        feature_adapter_enable=False,
    ).to(device)
    critic.eval()

    attn_dtype = torch.bfloat16 if bf16 else torch.float32
    attention_mask, position_ids = model_utils.build_strided_attention_mask_and_positions(
        full_sequence_length=sequences.shape[1],
        prompt_length=prompt_length,
        context_length=context_length,
        generation_step=generate_length,
        max_generation_length=generate_length,
        stride=stride,
        num_blocks=num_blocks,
        device=device,
        doc_ids=doc_ids.to(device),
        document_masking=False,
        dtype=attn_dtype,
    )
    with torch.no_grad():
        hidden_states, _ = critic(
            sequences.to(device),
            attention_mask=attention_mask,
            pos_ids=position_ids,
            context_length=context_length,
            prompt_length=prompt_length,
            generate_max_len=generate_length,
            stride=stride,
            num_blocks=num_blocks,
            hidden_state_method="last_only",
            qa_masks=qa_masks.to(device),
            qa_masking=False,
            return_dtype=torch.float32,
        )
    hidden_cpu = hidden_states.detach().cpu()
    gen_embedding, gt_embedding = hidden_states_to_g1_embeddings(
        hidden_cpu,
        qa_masks,
        prompt_length=prompt_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
        response_length=response_length,
        qa_masking=False,
    )
    from slime.backends.megatron_utils.g1_fast import compute_g1_token_advantages_from_embeddings

    token_advantages, scalar_rewards = compute_g1_token_advantages_from_embeddings(
        Namespace(
            n_samples_per_prompt=int(megatron_dump["n_samples_per_prompt"]),
            alignment_rew_coef=1.0,
            diversity_rew_coef=1.0,
            use_whitening=True,
            g1_response_length=response_length,
        ),
        [t.detach().cpu() for t in gen_embedding],
        [t.detach().cpu() for t in gt_embedding],
        [response_length] * sequences.shape[0],
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source": "openrlhf_critic",
            "sequences": sequences,
            "qa_masks": qa_masks,
            "attention_mask_shape": tuple(attention_mask.shape),
            "attention_mask": attention_mask.detach().cpu(),
            "attention_mask_summary": _attention_mask_summary(attention_mask.detach().cpu()),
            "position_ids": position_ids.detach().cpu(),
            "hidden_states": hidden_cpu,
            "g1_gen_embedding": [t.detach().cpu() for t in gen_embedding],
            "g1_gt_embedding": [t.detach().cpu() for t in gt_embedding],
            "g1_token_advantages": [t.detach().cpu() for t in token_advantages],
            "scalar_rewards": [float(x) for x in scalar_rewards],
            "g1_prompt_length": prompt_length,
            "g1_context_length": context_length,
            "g1_generate_length": generate_length,
            "g1_stride": stride,
            "g1_response_length": response_length,
            "n_samples_per_prompt": int(megatron_dump["n_samples_per_prompt"]),
        },
        output_path,
    )
    print(f"[openrlhf-runtime-dump] wrote {output_path}")


if __name__ == "__main__":
    main()
