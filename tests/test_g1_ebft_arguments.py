import argparse
import importlib
import sys
import types
from argparse import Namespace

import pytest


def _install_arguments_import_stubs(monkeypatch):
    sglang_router = types.ModuleType("sglang_router")
    launch_router = types.ModuleType("sglang_router.launch_router")

    class _RouterArgs:
        @staticmethod
        def add_cli_args(*args, **kwargs):
            return None

    launch_router.RouterArgs = _RouterArgs
    monkeypatch.setitem(sys.modules, "sglang_router", sglang_router)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router)

    sglang_args = types.ModuleType("slime.backends.sglang_utils.arguments")
    sglang_args.sglang_parse_args = lambda: Namespace()
    sglang_args.validate_args = lambda args: None
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.arguments", sglang_args)

    logging_utils = types.ModuleType("slime.utils.logging_utils")
    logging_utils.configure_logger = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "slime.utils.logging_utils", logging_utils)


def _arguments_module(monkeypatch):
    _install_arguments_import_stubs(monkeypatch)
    return importlib.import_module("slime.utils.arguments")


def _base_slime_args(**overrides):
    values = dict(
        actor_num_gpus_per_node=1,
        actor_num_nodes=1,
        advantage_estimator="g1",
        cf_target_mode=None,
        ckpt_step=None,
        colocate=False,
        critic_load=None,
        critic_lr=None,
        critic_lr_head=None,
        critic_num_gpus_per_node=None,
        critic_num_nodes=None,
        critic_train_only=False,
        custom_config_path=None,
        custom_tis_function_path=None,
        debug_rollout_only=False,
        debug_train_only=False,
        distribution_reward_type="pointwise",
        dump_details=None,
        enable_mtp_training=False,
        entropy_coef=0.0,
        eps_clip=0.2,
        eps_clip_high=None,
        eval_config=None,
        eval_function_path=None,
        eval_interval=None,
        eval_max_context_len=None,
        eval_prompt_data=None,
        eval_reward_key=None,
        freeze_params_name_list=None,
        g1_context_length=8,
        g1_ebft_logprob_indexing="strict_block_source",
        g1_embedding_source="rollout",
        g1_generate_length=8,
        g1_prompt_length=24,
        g1_response_length=16,
        g1_reward_location="rollout",
        g1_stride=8,
        g1_use_ebft_loss=True,
        get_mismatch_metrics=False,
        global_batch_size=None,
        grpo_std_normalization=True,
        hf_checkpoint=None,
        kl_coef=0.0,
        kl_loss_coef=0.0,
        load=None,
        load_debug_rollout_data=None,
        log_probs_max_tokens_per_gpu=None,
        loss_type="policy_loss",
        lr=1e-6,
        max_tokens_per_gpu=None,
        megatron_to_hf_mode="raw",
        mtp_num_layers=None,
        n_samples_per_prompt=1,
        normalize_advantages=True,
        num_epoch=None,
        num_rollout=1,
        num_steps_per_rollout=None,
        offload=False,
        offload_rollout=None,
        offload_train=None,
        only_train_params_name_list=None,
        opd_teacher_load=None,
        opd_type=None,
        over_sampling_batch_size=None,
        qkv_format="thd",
        ref_ckpt_step=None,
        ref_load="/tmp/nonexistent-ref",
        reward_key=None,
        rollout_batch_size=1,
        rollout_function_path=None,
        rollout_global_dataset=False,
        rollout_max_context_len=None,
        rollout_max_prompt_len=None,
        rollout_num_gpus=1,
        save=None,
        save_interval=None,
        train_backend="megatron",
        train_memory_margin_bytes=0,
        use_dynamic_batch_size=False,
        use_effopd=False,
        use_kl_loss=False,
        use_opd=False,
        use_opsm=False,
        use_rollout_logprobs=False,
        use_rollout_routing_replay=False,
        use_routing_replay=False,
        use_slime_router=False,
        use_tis=False,
    )
    values.update(overrides)
    return Namespace(**values)


def test_g1_ebft_logprob_indexing_parser_default_and_choices(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    parser = argparse.ArgumentParser()
    arguments.get_slime_extra_args_provider()(parser)

    required_args = ["--rollout-batch-size", "1"]

    assert parser.parse_args(required_args).g1_ebft_logprob_indexing == "standard_next_token"
    assert (
        parser.parse_args([*required_args, "--g1-ebft-logprob-indexing", "strict_block_source"]).g1_ebft_logprob_indexing
        == "strict_block_source"
    )

    with pytest.raises(SystemExit):
        parser.parse_args([*required_args, "--g1-ebft-logprob-indexing", "bad_indexing"])


def test_slime_validate_args_accepts_strict_block_source_geometry(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args()

    arguments.slime_validate_args(args)

    assert args.g1_ebft_logprob_indexing == "strict_block_source"


def test_slime_validate_args_rejects_strict_block_source_non_strided_geometry(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args(g1_prompt_length=23)

    with pytest.raises(ValueError, match="valid G1 strided-block geometry"):
        arguments.slime_validate_args(args)


def test_slime_validate_args_rejects_strict_block_source_response_length_mismatch(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args(g1_response_length=24)

    with pytest.raises(ValueError, match="g1_response_length == g1_generate_length \\* num_blocks"):
        arguments.slime_validate_args(args)


def test_slime_validate_args_rejects_strict_block_source_without_ebft_loss(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args(g1_use_ebft_loss=False)

    with pytest.raises(ValueError, match="requires --g1-use-ebft-loss"):
        arguments.slime_validate_args(args)
