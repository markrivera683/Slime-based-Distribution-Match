"""Temporary G1 embedding path: after SGLang rollout, before group RM, write gen/gt
embeddings into ``Sample.metadata`` via a slow Hugging Face / OpenRLHF critic.

Aligned with g1-exact-replan (~L29–31): the first version does **not** change Megatron
internals; this module is a debug stash for strict closed-loop parity. A later phase
should replace it with a Megatron (or other fast) embedding producer inside the
training stack.
"""
from __future__ import annotations

import json
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from slime.rollout.sglang_rollout import generate as sglang_generate
from slime.utils.g1_core import get_num_strided_blocks
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample


@dataclass(frozen=True)
class G1EmbeddingConfig:
    prompt_length: int = 384
    context_length: int = 8
    generate_length: int = 8
    stride: int = 8
    response_length: int = 376
    n_samples_per_prompt: int = 4
    critic_model_path: str | None = None
    tokenizer_path: str | None = None
    openrlhf_repo: str = "/mnt/data/ebft-distribution-new/code"
    hidden_state_method: str = "last_only"
    embedding_device: str = "cuda"
    embedding_dtype: str = "bfloat16"
    qa_masking: bool = False
    document_masking: bool = False

    @property
    def num_blocks(self) -> int:
        return get_num_strided_blocks(
            prompt_length=self.prompt_length,
            context_length=self.context_length,
            generate_length=self.generate_length,
            stride=self.stride,
        )


def g1_embedding_config_from_args(args: Any) -> G1EmbeddingConfig:
    tokenizer_path = getattr(args, "g1_tokenizer_path", None) or getattr(args, "hf_checkpoint", None)
    return G1EmbeddingConfig(
        prompt_length=int(getattr(args, "g1_prompt_length", 384)),
        context_length=int(getattr(args, "g1_context_length", 8)),
        generate_length=int(getattr(args, "g1_generate_length", 8)),
        stride=int(getattr(args, "g1_stride", 8)),
        response_length=int(getattr(args, "g1_response_length", 376)),
        n_samples_per_prompt=int(getattr(args, "n_samples_per_prompt", 4)),
        critic_model_path=getattr(args, "g1_critic_model_path", None),
        tokenizer_path=tokenizer_path,
        openrlhf_repo=str(getattr(args, "g1_openrlhf_repo", "/mnt/data/ebft-distribution-new/code")),
        hidden_state_method=str(getattr(args, "g1_hidden_state_method", "last_only")),
        embedding_device=str(getattr(args, "g1_embedding_device", "cuda")),
        embedding_dtype=str(getattr(args, "g1_embedding_dtype", "bfloat16")),
        qa_masking=bool(getattr(args, "g1_qa_masking", False)),
        document_masking=bool(getattr(args, "g1_document_masking", False)),
    )


def _prompt_to_text(prompt: str | list[dict[str, Any]]) -> str:
    if isinstance(prompt, str):
        return prompt
    parts: list[str] = []
    for message in prompt:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return "\n".join(part for part in parts if part)


def build_g1_prompt_inputs(
    *,
    tokenizer: Any,
    sample: Sample,
    config: G1EmbeddingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack one slime sample into OpenRLHF-style prompt tokens, doc ids, and QA mask."""
    if sample.label is None:
        raise ValueError("G1 embedding requires sample.label to build gt_embedding")

    prompt_ids = tokenizer.encode(_prompt_to_text(sample.prompt), add_special_tokens=False)
    answer_ids = tokenizer.encode(str(sample.label), add_special_tokens=False)
    packed_ids = list(prompt_ids) + list(answer_ids)
    answer_mask = [0] * len(prompt_ids) + [1] * len(answer_ids)

    if len(packed_ids) > config.prompt_length:
        raise ValueError(
            f"G1 prompt+label length {len(packed_ids)} exceeds g1_prompt_length={config.prompt_length}"
        )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("G1 embedding tokenizer must define pad_token_id or eos_token_id")

    pad_len = config.prompt_length - len(packed_ids)
    packed_ids.extend([int(pad_id)] * pad_len)
    answer_mask.extend([0] * pad_len)
    doc_ids = [0] * (config.prompt_length - pad_len) + [-1] * pad_len

    return (
        torch.tensor(packed_ids, dtype=torch.long),
        torch.tensor(doc_ids, dtype=torch.long),
        torch.tensor(answer_mask, dtype=torch.long),
    )


def build_g1_full_sequence_inputs(
    *,
    tokenizer: Any,
    sample: Sample,
    config: G1EmbeddingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sample.response_length != config.response_length:
        raise ValueError(
            f"G1 first version requires response_length={config.response_length}, got {sample.response_length}"
        )
    if len(sample.tokens) < sample.response_length:
        raise ValueError(f"Sample has {len(sample.tokens)} tokens but response_length={sample.response_length}")

    prompt_ids, doc_ids_prompt, qa_prompt = build_g1_prompt_inputs(
        tokenizer=tokenizer,
        sample=sample,
        config=config,
    )
    response_ids = torch.tensor(sample.tokens[-sample.response_length :], dtype=torch.long)
    if response_ids.numel() != config.num_blocks * config.generate_length:
        raise ValueError(
            f"G1 response token count {response_ids.numel()} != num_blocks*generate_length "
            f"{config.num_blocks * config.generate_length}"
        )

    full_sequence = torch.cat([prompt_ids, response_ids], dim=0)
    gen_qa_mask = torch.ones(config.response_length, dtype=torch.long)
    qa_masks = torch.cat([qa_prompt, gen_qa_mask], dim=0)
    return full_sequence, doc_ids_prompt, qa_masks


def _groom_last_token(
    block_hidden_states: torch.Tensor,
    block_qa_mask: torch.Tensor,
    *,
    qa_masking: bool,
) -> torch.Tensor:
    """Select OpenRLHF last-token/groom embedding from [B, K, G, NF, H]."""
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


def hidden_states_to_g1_embeddings(
    hidden_states: torch.Tensor,
    qa_masks: torch.Tensor,
    config: G1EmbeddingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert OpenRLHF critic hidden states [B, S, NF, H] to [B, K, D] embeddings."""
    expected_len = config.prompt_length + config.response_length
    if hidden_states.ndim != 4:
        raise ValueError(f"Expected hidden states [B, S, NF, H], got {hidden_states.shape}")
    if hidden_states.shape[1] != expected_len:
        raise ValueError(f"Expected sequence length {expected_len}, got {hidden_states.shape[1]}")
    if qa_masks.shape != hidden_states.shape[:2]:
        raise ValueError(f"Expected qa_masks shape {hidden_states.shape[:2]}, got {qa_masks.shape}")

    gt_hidden = hidden_states[:, config.context_length : config.prompt_length, :, :]
    gen_hidden = hidden_states[:, config.prompt_length :, :, :]
    gt_qa = qa_masks[:, config.context_length : config.prompt_length].view(
        gt_hidden.shape[0], gt_hidden.shape[1], 1, 1
    ).repeat(1, 1, gt_hidden.shape[2], 1)
    gen_qa = qa_masks[:, config.prompt_length :].view(
        gen_hidden.shape[0], gen_hidden.shape[1], 1, 1
    ).repeat(1, 1, gen_hidden.shape[2], 1)

    gt_blocks = gt_hidden.unfold(-3, config.generate_length, config.stride).permute(0, 1, 4, 2, 3)
    gt_qa_blocks = gt_qa.unfold(-3, config.generate_length, config.stride).permute(0, 1, 4, 2, 3)

    gen_blocks = gen_hidden.reshape(
        gen_hidden.shape[0],
        config.generate_length,
        config.num_blocks,
        gen_hidden.shape[-2],
        gen_hidden.shape[-1],
    ).transpose(-3, -4)
    gen_qa_blocks = gen_qa.reshape(
        gen_hidden.shape[0],
        config.generate_length,
        config.num_blocks,
        gen_hidden.shape[-2],
        1,
    ).transpose(-3, -4)

    gt_embedding = _groom_last_token(gt_blocks, gt_qa_blocks, qa_masking=config.qa_masking)
    gen_embedding = _groom_last_token(gen_blocks, gen_qa_blocks, qa_masking=config.qa_masking)
    return (
        gen_embedding.reshape(gen_embedding.shape[0], gen_embedding.shape[1], -1).float(),
        gt_embedding.reshape(gt_embedding.shape[0], gt_embedding.shape[1], -1).float(),
    )


def _load_python_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _config_json_references_sglang(obj: object) -> bool:
    """True when config.json would steer Transformers toward SGLang config classes."""
    if isinstance(obj, dict):
        auto_map = obj.get("auto_map")
        if isinstance(auto_map, dict):
            for value in auto_map.values():
                if isinstance(value, str) and "sglang" in value.lower():
                    return True
        return any(_config_json_references_sglang(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_config_json_references_sglang(x) for x in obj)
    return False


def assert_g1_critic_checkpoint_is_transformers_hf(model_path: str) -> None:
    """Fail fast if the critic directory cannot be loaded by Transformers AutoModelForCausalLM.

    SGLang-served or SGLang-patched checkpoints often ship a ``config.json`` whose ``auto_map``
    points at ``sglang.*`` modules; OpenRLHF's ``Critic`` uses ``from_pretrained`` and will
    raise an obscure ``ValueError`` unless this is caught early.
    """
    root = Path(model_path).expanduser().resolve()
    cfg_path = root / "config.json"
    if not cfg_path.is_file():
        raise ValueError(
            f"--g1-critic-model-path must be a directory containing config.json; missing {cfg_path}"
        )
    try:
        data: object = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {cfg_path}: {exc}") from exc
    if _config_json_references_sglang(data):
        raise ValueError(
            "G1 slow embedding: --g1-critic-model-path points to an SGLang-patched checkpoint "
            "(config.json references sglang). OpenRLHF Critic loads via "
            "transformers.AutoModelForCausalLM, which does not accept that config class. "
            "Use a vanilla Hugging Face model directory for the critic only (e.g. original Hub "
            "snapshot), not the same tree as SGLang server weights unless its config.json is HF-native."
        )
    try:
        from transformers import AutoConfig

        auto_config = AutoConfig.from_pretrained(str(root), trust_remote_code=True)
    except Exception as exc:
        model_type = data.get("model_type") if isinstance(data, dict) else None
        raise ValueError(
            "G1 slow embedding: --g1-critic-model-path must be loadable by Transformers "
            "AutoConfig/AutoModelForCausalLM because OpenRLHF Critic calls from_pretrained. "
            f"Failed to load config from {root} (model_type={model_type!r}). "
            "Use a vanilla Hugging Face checkpoint supported by the current Transformers "
            "environment, or update the environment/model implementation before using this path."
        ) from exc
    if "sglang" in type(auto_config).__module__.lower():
        raise ValueError(
            "G1 slow embedding: --g1-critic-model-path resolved to an SGLang config class "
            f"({type(auto_config).__module__}.{type(auto_config).__name__}). OpenRLHF Critic "
            "requires a Transformers/HF-native model class for this temporary path."
        )


class SlowOpenRLHFG1EmbeddingProducer:
    def __init__(self, config: G1EmbeddingConfig) -> None:
        if config.critic_model_path is None:
            raise ValueError("--g1-critic-model-path is required for the slow G1 embedding producer")
        assert_g1_critic_checkpoint_is_transformers_hf(config.critic_model_path)
        self.config = config
        repo = Path(config.openrlhf_repo)
        if repo.exists() and str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        critic_module = _load_python_module(
            "slime_g1_openrlhf_critic",
            repo / "openrlhf" / "models" / "critic.py",
        )
        model_utils_module = _load_python_module(
            "slime_g1_openrlhf_model_utils",
            repo / "openrlhf" / "models" / "utils.py",
        )

        Critic = critic_module.Critic
        self._build_mask = model_utils_module.build_strided_attention_mask_and_positions
        self.device = torch.device(config.embedding_device)
        bf16 = config.embedding_dtype == "bfloat16"
        self.critic = Critic(
            config.critic_model_path,
            use_flash_attention_2=False,
            bf16=bf16,
            critic_sequence_level="last_token",
            gen_len=config.generate_length,
            hidden_state_method=config.hidden_state_method,
            feature_adapter_enable=False,
        ).to(self.device)
        self.critic.eval()

    @torch.no_grad()
    def embed(self, sequences: torch.Tensor, doc_ids_prompt: torch.Tensor, qa_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = sequences.to(self.device)
        doc_ids_prompt = doc_ids_prompt.to(self.device)
        qa_masks = qa_masks.to(self.device)
        attn_dtype = torch.bfloat16 if self.config.embedding_dtype == "bfloat16" else torch.float32
        attention_mask, position_ids = self._build_mask(
            full_sequence_length=sequences.shape[1],
            prompt_length=self.config.prompt_length,
            context_length=self.config.context_length,
            generation_step=self.config.generate_length,
            max_generation_length=self.config.generate_length,
            stride=self.config.stride,
            num_blocks=self.config.num_blocks,
            device=self.device,
            doc_ids=doc_ids_prompt,
            document_masking=self.config.document_masking,
            dtype=attn_dtype,
        )
        hidden_states, _ = self.critic(
            sequences,
            attention_mask=attention_mask,
            pos_ids=position_ids,
            context_length=self.config.context_length,
            prompt_length=self.config.prompt_length,
            generate_max_len=self.config.generate_length,
            stride=self.config.stride,
            num_blocks=self.config.num_blocks,
            hidden_state_method=self.config.hidden_state_method,
            qa_masks=qa_masks,
            qa_masking=self.config.qa_masking,
            return_dtype=torch.float32,
        )
        return hidden_states_to_g1_embeddings(hidden_states.cpu(), qa_masks.cpu(), self.config)


_PRODUCERS: dict[G1EmbeddingConfig, SlowOpenRLHFG1EmbeddingProducer] = {}


def _get_producer(config: G1EmbeddingConfig) -> SlowOpenRLHFG1EmbeddingProducer:
    if config not in _PRODUCERS:
        _PRODUCERS[config] = SlowOpenRLHFG1EmbeddingProducer(config)
    return _PRODUCERS[config]


def attach_g1_embeddings(args: Any, sample: Sample) -> Sample:
    config = g1_embedding_config_from_args(args)
    tokenizer = load_tokenizer(config.tokenizer_path, trust_remote_code=True)
    full_sequence, doc_ids_prompt, qa_masks = build_g1_full_sequence_inputs(
        tokenizer=tokenizer,
        sample=sample,
        config=config,
    )
    producer = _get_producer(config)
    gen_embedding, gt_embedding = producer.embed(
        full_sequence.unsqueeze(0),
        doc_ids_prompt.unsqueeze(0),
        qa_masks.unsqueeze(0),
    )
    sample.metadata["g1_gen_embedding"] = gen_embedding[0].tolist()
    sample.metadata["g1_gt_embedding"] = gt_embedding[0].tolist()
    sample.metadata["g1_num_blocks"] = config.num_blocks
    sample.metadata["g1_generate_length"] = config.generate_length
    return sample


async def generate_with_g1_embeddings(args: Any, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> Sample:
    if evaluation:
        raise ValueError("G1 group RM is not supported during eval rollout")
    config = g1_embedding_config_from_args(args)
    sampling_params = dict(sampling_params)
    sampling_params["max_new_tokens"] = config.response_length
    sampling_params["min_new_tokens"] = config.response_length
    sampling_params["stop"] = []
    sampling_params["stop_token_ids"] = []
    sampling_params["ignore_eos"] = True
    sample = await sglang_generate(args, sample, sampling_params)
    return attach_g1_embeddings(args, sample)


async def generate_fixed_length_for_g1(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    """Generate strict fixed-length G1 responses without rollout-side embeddings."""
    if evaluation:
        # Evaluation does not feed trainer-side G1 embeddings, so keep eval's
        # own max_response_len instead of forcing the training response length.
        return await sglang_generate(args, sample, sampling_params)
    config = g1_embedding_config_from_args(args)
    sampling_params = dict(sampling_params)
    sampling_params["max_new_tokens"] = config.response_length
    sampling_params["min_new_tokens"] = config.response_length
    sampling_params["stop"] = []
    sampling_params["stop_token_ids"] = []
    sampling_params["ignore_eos"] = True
    return await sglang_generate(args, sample, sampling_params)
