import logging
import os
import random
import socket
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from slime.ray.train_actor import TrainRayActor
from slime.utils import train_dump_utils
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group, init_process_group
from slime.utils.logging_utils import init_tracking
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.misc import Box
from slime.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from slime.utils.routing_replay import RoutingReplay
from slime.utils.timer import Timer, inverse_timer, timer, with_defer
from slime.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from .checkpoint import load_checkpoint
from .cp_utils import slice_log_prob_with_cp, slice_with_cp
from .data import DataIterator, get_data_iterator, log_perf_data, log_rollout_data, sync_actor_critic_data
from .g1_fast import compute_g1_token_advantages_from_embeddings
from .initialize import init, is_megatron_main_rank
from .loss import (
    collect_g1_runtime_dump_writer_metadata,
    compute_advantages_and_returns,
    get_g1_embeddings_from_hidden_states,
    get_log_probs_and_entropy,
    get_values,
    g1_runtime_dump_writer_only,
    g2_runtime_dump_writer_only,
)
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_distributed import UpdateWeightFromDistributed
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        if args.debug_rollout_only:
            self.args = args
            return 0

        monkey_patch_torch_dist()
        super().init(args, role, with_ref, with_opd_teacher)

        init(args)

        if is_megatron_main_rank():
            init_tracking(args, primary=False)

        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
        }
        dist.barrier(group=get_gloo_group())

        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        if role == "critic":
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters
            if getattr(self.args, "critic_lr_head", None) == 0:
                freeze_patterns = list(self.args.freeze_params_name_list or [])
                if "output_layer" not in freeze_patterns:
                    freeze_patterns.append("output_layer")
                self.args.freeze_params_name_list = freeze_patterns

        (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
            args, role
        )

        start_rollout_id = loaded_rollout_id + 1

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return start_rollout_id

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=args.megatron_to_hf_mode == "raw",
                translate_gpu_to_cpu=not self.args.enable_weights_backuper,
            ),
            single_tag=None if args.enable_weights_backuper else "actor",
        )
        self._active_model_tag: str | None = "actor"
        self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        # Load teacher model for Megatron-based on-policy distillation
        if with_opd_teacher:
            self.load_other_checkpoint("teacher", args.opd_teacher_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            # Prefer HF config vocab_size (which may include model-native padding)
            # over tokenizer vocab_size (which may be smaller, e.g. GPT-OSS).
            hf_vocab = getattr(self.hf_config, "vocab_size", None)
            self.args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size

        update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )

        # empty cache after initialization
        clear_memory()

        if self.args.offload_train:
            # recover to actor in the end.
            self._switch_model("actor")
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        destroy_process_groups()

        torch_memory_saver.pause()

        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        torch_memory_saver.resume()

        clear_memory()
        reload_process_groups()
        print_memory("after wake_up model")

    def _get_rollout_data(self, rollout_data_ref: Box) -> RolloutBatch:
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will receive the data.
        rollout_data = process_rollout_data(
            self.args,
            rollout_data_ref,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
        )
        # TODO: this is ugly, move to somewhere else?
        # move tokens to GPU in advance
        rollout_data["tokens"] = [
            torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device()) for t in rollout_data["loss_masks"]
        ]
        if "g1_token_advantages" in rollout_data:
            rollout_data["g1_token_advantages"] = [
                torch.tensor(t, dtype=torch.float32, device=torch.cuda.current_device())
                for t in rollout_data["g1_token_advantages"]
            ]
        if "g1_full_sequences" in rollout_data:
            rollout_data["g1_full_sequences"] = [
                torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in rollout_data["g1_full_sequences"]
            ]
        if "g1_qa_masks" in rollout_data:
            rollout_data["g1_qa_masks"] = [
                torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in rollout_data["g1_qa_masks"]
            ]
        if "g2_teacher_full_sequences" in rollout_data:
            rollout_data["g2_teacher_full_sequences"] = [
                torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device())
                for t in rollout_data["g2_teacher_full_sequences"]
            ]
        if "g2_teacher_qa_masks" in rollout_data:
            rollout_data["g2_teacher_qa_masks"] = [
                torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device())
                for t in rollout_data["g2_teacher_qa_masks"]
            ]
        if "g2_teacher_gen_embeddings" in rollout_data:
            rollout_data["g2_teacher_gen_embeddings"] = [
                torch.tensor(t, dtype=torch.float32, device=torch.cuda.current_device())
                for t in rollout_data["g2_teacher_gen_embeddings"]
            ]
        if "multimodal_train_inputs" in rollout_data:
            # Move multimodal training tensors to GPU in advance
            rollout_data["multimodal_train_inputs"] = [
                (
                    {
                        key: (
                            torch.from_numpy(v.copy()).to(device=torch.cuda.current_device())
                            if isinstance(v, np.ndarray)
                            else v.to(device=torch.cuda.current_device())
                        )
                        for key, v in mm_dict.items()
                    }
                    if mm_dict is not None
                    else None
                )
                for mm_dict in rollout_data["multimodal_train_inputs"]
            ]

        if self.args.qkv_format == "bshd":
            # TODO: micro-batch wise dynamic, possibly move to @data.py:get_data_iterator
            max_seq_len = max(rollout_data["total_lengths"])

            # pad to reduce memory fragmentation and maybe make the computation faster
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size

            rollout_data["max_seq_lens"] = [max_seq_len] * len(rollout_data["tokens"])

        for key in ["rollout_log_probs", "teacher_log_probs"]:
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                torch.tensor(
                    slice_log_prob_with_cp(
                        log_prob,
                        total_length,
                        response_length,
                        self.args.qkv_format,
                        rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                    ),
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
                for i, (log_prob, total_length, response_length) in enumerate(
                    zip(
                        rollout_data[key],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                )
            ]
        if "rollout_routed_experts" in rollout_data:
            rollout_data["rollout_routed_experts"] = [
                torch.from_numpy(r) for r in rollout_data["rollout_routed_experts"]
            ]
        return rollout_data

    def _is_standard_g2(self) -> bool:
        return (
            getattr(self.args, "distribution_reward_type", "pointwise") == "cf_l1oo"
            and getattr(self.args, "cf_target_mode", None) == "teacher"
        )

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
            )

        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        from slime.utils.routing_replay import RoutingReplay

        for iterator in data_iterator:
            iterator.reset()

        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        def pad_func(experts, pad):
            _, num_layers, topk = experts.shape
            pad = (
                torch.arange(
                    pad * num_layers * topk,
                    device=experts.device,
                    dtype=experts.dtype,
                ).reshape((pad, num_layers, topk))
                % self.args.num_experts
            )
            return torch.cat([experts, pad], dim=0)

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
            rollout_routed_experts = batch["rollout_routed_experts"]
            tokens = batch["tokens"]
            assert len(rollout_routed_experts) == len(tokens)
            for a, b in zip(rollout_routed_experts, tokens, strict=False):
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            # We need to pad the experts to the last token. We won't calculate loss on this token so this should be fine.
            # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
            rollout_routed_experts = [pad_func(r, 1) for r in rollout_routed_experts]
            # TODO: maybe extract a common process function for here and get_batch?
            rollout_routed_experts = [slice_with_cp(r, pad_func) for r in rollout_routed_experts]
            rollout_routed_experts = torch.cat(rollout_routed_experts, dim=0)
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            pad = (pad_size - rollout_routed_experts.size(0) % pad_size) % pad_size
            if pad != 0:
                rollout_routed_experts = pad_func(rollout_routed_experts, pad)

            if self.args.sequence_parallel:
                seqlen = rollout_routed_experts.size(0)
                assert seqlen % tp_size == 0
                start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                rollout_routed_experts = rollout_routed_experts[start:end]

            routing_replay_offset = 0
            for vp_stage, model in enumerate(self.model):
                config = model.module.config
                num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    # skip dense layer
                    if isinstance(config.moe_layer_freq, int):
                        if layer_id % config.moe_layer_freq != 0:
                            continue
                    elif isinstance(config.moe_layer_freq, list):
                        assert len(config.moe_layer_freq) == config.num_layers
                        if config.moe_layer_freq[layer_id] == 0:
                            continue
                    layer_routed_experts = rollout_routed_experts[:, layer_id]
                    RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
                    routing_replay_offset += 1
            assert routing_replay_offset == len(RoutingReplay.all_routing_replays)

        del rollout_data["rollout_routed_experts"]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def compute_g1_token_advantages(
        self,
        rollout_data: RolloutBatch,
    ) -> None:
        if self.args.g1_embedding_source != "megatron_ref" or self.args.g1_reward_location != "trainer":
            return
        standard_g2 = self._is_standard_g2()
        required = ["g1_full_sequences", "g1_qa_masks"]
        missing = [key for key in required if key not in rollout_data]
        if missing:
            raise ValueError(f"Trainer-side G1/G2 embedding path requires rollout_data keys {missing}")
        if not standard_g2 and "ref" not in self.weights_backuper.backup_tags:
            raise ValueError("Megatron G1 fast path requires a ref checkpoint/snapshot; set --ref-load")

        g1_rollout_data = dict(rollout_data)
        g1_rollout_data["tokens"] = rollout_data["g1_full_sequences"]
        g1_rollout_data["total_lengths"] = [int(t.numel()) for t in g1_rollout_data["tokens"]]
        g1_rollout_data["response_lengths"] = [int(self.args.g1_response_length)] * len(g1_rollout_data["tokens"])
        g1_rollout_data["loss_masks"] = [
            torch.ones(int(self.args.g1_response_length), dtype=torch.int, device=torch.cuda.current_device())
            for _ in g1_rollout_data["tokens"]
        ]
        if self.args.qkv_format == "bshd":
            max_seq_len = max(t.size(0) for t in g1_rollout_data["tokens"])
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size
            g1_rollout_data["max_seq_lens"] = [max_seq_len] * len(g1_rollout_data["tokens"])

        dump_path = os.getenv("G1_RUNTIME_DUMP_PATH")
        g2_dump_path = os.getenv("G2_RUNTIME_DUMP_PATH")
        g2_dump_enabled = bool(g2_dump_path and standard_g2 and g2_runtime_dump_writer_only())
        # Single DP rank clears the dump so later appends inside the ref forward cannot
        # merge unrelated runs that happen to reuse the path.
        if dump_path and g1_runtime_dump_writer_only():
            Path(dump_path).unlink(missing_ok=True)
        if g2_dump_enabled:
            Path(g2_dump_path).unlink(missing_ok=True)

        g1_data_iterator, g1_num_microbatches = get_data_iterator(self.args, self.model, g1_rollout_data)
        try:
            with timer("g1_megatron_embeddings"):
                embeddings = forward_only(
                    get_g1_embeddings_from_hidden_states,
                    self.args,
                    self.model,
                    g1_data_iterator,
                    g1_num_microbatches,
                    collect_hidden_states=True,
                    extra_batch_keys=["g1_qa_masks"],
                )
            # Non-last pipeline stages receive `{}` here; intermediate ranks must still
            # drop heavyweight G1 keys so rollout_data stays pipeline-safe.
            if not embeddings:
                return
            token_advantages, scalar_rewards = compute_g1_token_advantages_from_embeddings(
                self.args,
                embeddings["g1_gen_embedding"],
                embeddings["g1_gt_embedding"],
                rollout_data["response_lengths"],
                teacher_gen_embeddings=rollout_data.get("g2_teacher_gen_embeddings"),
                g2_runtime_dump_path=g2_dump_path if g2_dump_enabled else None,
                g2_dump_writer_metadata=collect_g1_runtime_dump_writer_metadata() if g2_dump_enabled else None,
            )
            if dump_path and g1_runtime_dump_writer_only():
                output_path = Path(dump_path)
                if output_path.exists():
                    dump = torch.load(output_path, map_location="cpu", weights_only=False)
                    dump["g1_token_advantages"] = [t.detach().cpu() for t in token_advantages]
                    dump["scalar_rewards"] = [float(x) for x in scalar_rewards]
                    torch.save(dump, output_path)
            rollout_data["g1_token_advantages"] = token_advantages
            rollout_data["rewards"] = scalar_rewards
        finally:
            # Strip large tensors after embedding pass unless EBFT loss plumbing needs masks/sequences.
            if not bool(getattr(self.args, "g1_use_ebft_loss", False)):
                rollout_data.pop("g1_full_sequences", None)
                rollout_data.pop("g1_qa_masks", None)
            rollout_data.pop("g2_teacher_full_sequences", None)
            rollout_data.pop("g2_teacher_qa_masks", None)

    def compute_g2_teacher_gen_embeddings(self, rollout_data: RolloutBatch) -> None:
        if getattr(self.args, "distribution_reward_type", "pointwise") != "cf_l1oo":
            return
        if getattr(self.args, "cf_target_mode", None) != "teacher":
            return
        if "g2_teacher_gen_embeddings" in rollout_data:
            return

        required = ["g2_teacher_full_sequences", "g2_teacher_qa_masks"]
        missing = [key for key in required if key not in rollout_data]
        if missing:
            raise ValueError(f"Standard G2 trainer-side teacher embedding requires rollout_data keys {missing}")

        n_samples_per_prompt = int(getattr(self.args, "n_samples_per_prompt", 1))
        n_teacher = int(getattr(self.args, "cf_teacher_n_samples", 0))
        if n_samples_per_prompt <= 0 or n_teacher <= 0:
            raise ValueError(
                f"Invalid G2 group sizes: n_samples_per_prompt={n_samples_per_prompt}, cf_teacher_n_samples={n_teacher}"
            )
        num_samples = len(rollout_data["tokens"])
        if num_samples % n_samples_per_prompt != 0:
            raise ValueError(
                f"G2 local sample count {num_samples} must be divisible by n_samples_per_prompt={n_samples_per_prompt}"
            )

        teacher_sequences = []
        teacher_qa_masks = []
        group_first_indices = []
        for group_start in range(0, num_samples, n_samples_per_prompt):
            first_sequences = rollout_data["g2_teacher_full_sequences"][group_start]
            first_masks = rollout_data["g2_teacher_qa_masks"][group_start]
            if first_sequences.ndim != 2 or first_sequences.shape[0] != n_teacher:
                raise ValueError(
                    "Each G2 teacher sequence entry must have shape [M, sequence_length], "
                    f"got {tuple(first_sequences.shape)} at group_start={group_start}"
                )
            if first_masks.shape != first_sequences.shape:
                raise ValueError(
                    f"G2 teacher qa mask shape {tuple(first_masks.shape)} must match sequences {tuple(first_sequences.shape)}"
                )
            for offset in range(1, n_samples_per_prompt):
                idx = group_start + offset
                if not torch.equal(rollout_data["g2_teacher_full_sequences"][idx], first_sequences):
                    raise ValueError(
                        "All samples in a G2 prompt group must share identical teacher full sequences; "
                        f"group_start={group_start} sample_offset={offset}"
                    )
                if not torch.equal(rollout_data["g2_teacher_qa_masks"][idx], first_masks):
                    raise ValueError(
                        "All samples in a G2 prompt group must share identical teacher qa masks; "
                        f"group_start={group_start} sample_offset={offset}"
                    )
            teacher_sequences.extend(first_sequences.unbind(dim=0))
            teacher_qa_masks.extend(first_masks.unbind(dim=0))
            group_first_indices.append(group_start)

        teacher_rollout_data = dict(rollout_data)
        teacher_rollout_data["tokens"] = teacher_sequences
        teacher_rollout_data["g1_qa_masks"] = teacher_qa_masks
        teacher_rollout_data["total_lengths"] = [int(t.numel()) for t in teacher_sequences]
        teacher_rollout_data["response_lengths"] = [int(self.args.g1_response_length)] * len(teacher_sequences)
        teacher_rollout_data["loss_masks"] = [
            torch.ones(int(self.args.g1_response_length), dtype=torch.int, device=torch.cuda.current_device())
            for _ in teacher_sequences
        ]
        if self.args.qkv_format == "bshd":
            max_seq_len = max(t.size(0) for t in teacher_sequences)
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size
            teacher_rollout_data["max_seq_lens"] = [max_seq_len] * len(teacher_sequences)

        teacher_data_iterator, teacher_num_microbatches = get_data_iterator(self.args, self.model, teacher_rollout_data)
        with timer("g2_teacher_megatron_embeddings"):
            embeddings = forward_only(
                get_g1_embeddings_from_hidden_states,
                self.args,
                self.model,
                teacher_data_iterator,
                teacher_num_microbatches,
                collect_hidden_states=True,
                extra_batch_keys=["g1_qa_masks"],
            )
        if not embeddings:
            return

        gen_embeddings = embeddings["g1_gen_embedding"]
        expected = len(group_first_indices) * n_teacher
        if len(gen_embeddings) != expected:
            raise ValueError(f"G2 teacher embedding count {len(gen_embeddings)} != expected {expected}")

        teacher_by_sample = [None] * num_samples
        cursor = 0
        for group_start in group_first_indices:
            group_teacher = torch.stack([gen_embeddings[cursor + i].float() for i in range(n_teacher)], dim=0)
            cursor += n_teacher
            for offset in range(n_samples_per_prompt):
                teacher_by_sample[group_start + offset] = group_teacher
        rollout_data["g2_teacher_gen_embeddings"] = teacher_by_sample

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        if self.args.debug_rollout_only:
            return

        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        if self.role == "critic":
            return self.train_critic(rollout_id, rollout_data)
        else:
            return self.train_actor(rollout_id, rollout_data)

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
            )
        )

        if self._is_standard_g2() and self.args.advantage_estimator == "g1" and self.args.g1_reward_location == "trainer":
            self.compute_g2_teacher_gen_embeddings(rollout_data)
            self.compute_g1_token_advantages(rollout_data)

        if rollout_id >= self.args.num_critic_only_steps and not self.args.critic_train_only:
            sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)

        compute_advantages_and_returns(self.args, rollout_data)

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
        )

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )

                # Forward teacher model to get teacher_log_probs for Megatron-based OPD
                if "teacher" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("teacher")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="teacher_",
                        )
                    )

                if self.args.advantage_estimator == "g1" and self.args.g1_reward_location == "trainer":
                    if not self._is_standard_g2():
                        if self.args.use_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                        self._switch_model("ref")
                        self.compute_g2_teacher_gen_embeddings(rollout_data)
                        self.compute_g1_token_advantages(rollout_data)

                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                        )
                    )
                    if self.args.use_rollout_routing_replay:
                        RoutingReplay.clear_all_forward()

                if self.args.use_critic:
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args, rollout_id, rollout_data)

            log_rollout_data(
                rollout_id,
                self.args,
                rollout_data,
            )

            # Train
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        # update the cpu actor weight to the latest model
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args)

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving.
        if self.args.offload_train:
            reload_process_groups()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from slime.backends.megatron_utils.model import save_hf_model

            save_hf_model(self.args, rollout_id, self.model)

        if self.args.offload_train:
            destroy_process_groups()

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_updatable_engines_and_lock.remote()
        )

        if self.args.offload_train:
            reload_process_groups()

        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0:
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if self.args.offload_train:
            destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        old_ckpt_step = None
        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step
        elif model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if old_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    def connect_actor_critic(
        self,
        actor_handle: ActorHandle | None = None,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        if self.role == "actor":
            master_address = ray.util.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            actor_handle.connect_actor_critic.remote(master_address=master_address, master_port=master_port)

        group_name = "actor_critic"
        world_size = 2
        self._actor_critic_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0 if self.role == "actor" else 1,
            group_name=group_name,
        )
