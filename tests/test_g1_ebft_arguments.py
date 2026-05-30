import argparse
import importlib
import sys
import types
from argparse import Namespace

import pytest


def _install_arguments_import_stubs(monkeypatch):
    yaml = types.ModuleType("yaml")
    yaml.safe_load = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "yaml", yaml)

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
        enable_weights_backuper=True,
        effopd_dv_size=50,
        effopd_lr_decay=0.5,
        effopd_max_k=5,
        effopd_validation_mode="combined_gate",
        freeze_params_name_list=None,
        g1_context_length=8,
        g1_ebft_logprob_indexing="strict_block_source",
        g1_ebft_rollout_mask_mode="none",
        g1_ebft_rollout_sampling_mode="standard",
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
        sglang_attention_backend=None,
        sglang_disable_overlap_schedule=False,
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


def _g2_opd_cf_l1oo_args(**overrides):
    values = dict(
        advantage_estimator="g1",
        cf_target_mode="opd_onpolicy",
        critic_lr=0.0,
        critic_lr_head=0.0,
        distribution_reward_type="cf_l1oo",
        g1_embedding_source="megatron_ref",
        g1_reward_location="trainer",
        n_samples_per_prompt=2,
        opd_type="sglang",
        use_effopd=True,
        use_opd=True,
        use_whitening=True,
        zero_stage=3,
    )
    values.update(overrides)
    return _base_slime_args(**values)


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


def test_effopd_validation_mode_parser_default_and_shadow_choices(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    parser = argparse.ArgumentParser()
    arguments.get_slime_extra_args_provider()(parser)

    required_args = ["--rollout-batch-size", "1"]

    assert parser.parse_args(required_args).effopd_validation_mode == "combined_gate"
    assert (
        parser.parse_args(
            [*required_args, "--effopd-validation-mode", "opd_kl_shadow_cf"]
        ).effopd_validation_mode
        == "opd_kl_shadow_cf"
    )
    assert (
        parser.parse_args(
            [*required_args, "--effopd-validation-mode", "combined_shadow"]
        ).effopd_validation_mode
        == "combined_shadow"
    )

    with pytest.raises(SystemExit):
        parser.parse_args([*required_args, "--effopd-validation-mode", "bad_mode"])


def test_g1_ebft_rollout_mask_mode_parser_default_and_choices(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    parser = argparse.ArgumentParser()
    arguments.get_slime_extra_args_provider()(parser)

    required_args = ["--rollout-batch-size", "1"]

    assert parser.parse_args(required_args).g1_ebft_rollout_mask_mode == "none"
    assert parser.parse_args([*required_args, "--g1-ebft-rollout-mask-mode", "dense4d"]).g1_ebft_rollout_mask_mode == "dense4d"
    assert parser.parse_args([*required_args, "--g1-ebft-rollout-mask-mode", "sparse_ir"]).g1_ebft_rollout_mask_mode == "sparse_ir"

    with pytest.raises(SystemExit):
        parser.parse_args([*required_args, "--g1-ebft-rollout-mask-mode", "bad_mode"])


def test_g1_ebft_rollout_sampling_mode_parser_default_and_choices(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    parser = argparse.ArgumentParser()
    arguments.get_slime_extra_args_provider()(parser)

    required_args = ["--rollout-batch-size", "1"]

    assert parser.parse_args(required_args).g1_ebft_rollout_sampling_mode == "standard"
    assert (
        parser.parse_args(
            [*required_args, "--g1-ebft-rollout-sampling-mode", "block_source"]
        ).g1_ebft_rollout_sampling_mode
        == "block_source"
    )

    with pytest.raises(SystemExit):
        parser.parse_args([*required_args, "--g1-ebft-rollout-sampling-mode", "bad_mode"])


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


def test_slime_validate_args_accepts_dense_block_source_rollout_sampling(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args(
        g1_context_length=8,
        g1_ebft_rollout_mask_mode="dense4d",
        g1_ebft_rollout_sampling_mode="block_source",
        g1_generate_length=1,
        g1_prompt_length=10,
        g1_response_length=2,
        g1_stride=1,
        sglang_attention_backend="torch_native",
        sglang_disable_overlap_schedule=True,
    )

    arguments.slime_validate_args(args)


def test_slime_validate_args_accepts_block_source_rollout_sampling_generate_length_gt_one(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args(
        g1_ebft_rollout_mask_mode="dense4d",
        g1_ebft_rollout_sampling_mode="block_source",
        sglang_attention_backend="torch_native",
        sglang_disable_overlap_schedule=True,
    )

    arguments.slime_validate_args(args)


def test_slime_validate_args_accepts_sparse_block_source_rollout_sampling(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args(
        g1_context_length=8,
        g1_ebft_rollout_mask_mode="sparse_ir",
        g1_ebft_rollout_sampling_mode="block_source",
        g1_generate_length=1,
        g1_prompt_length=10,
        g1_response_length=2,
        g1_stride=1,
        sglang_attention_backend="triton",
        sglang_disable_overlap_schedule=True,
    )

    arguments.slime_validate_args(args)


def test_slime_validate_args_rejects_block_source_rollout_sampling_without_mask(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args(
        g1_context_length=8,
        g1_ebft_rollout_sampling_mode="block_source",
        g1_generate_length=1,
        g1_prompt_length=10,
        g1_response_length=2,
        g1_stride=1,
        sglang_attention_backend="torch_native",
        sglang_disable_overlap_schedule=True,
    )

    with pytest.raises(ValueError, match="requires --g1-ebft-rollout-mask-mode dense4d or sparse_ir"):
        arguments.slime_validate_args(args)


def test_slime_validate_args_rejects_block_source_rollout_sampling_without_torch_native(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args(
        g1_context_length=8,
        g1_ebft_rollout_mask_mode="dense4d",
        g1_ebft_rollout_sampling_mode="block_source",
        g1_generate_length=1,
        g1_prompt_length=10,
        g1_response_length=2,
        g1_stride=1,
        sglang_attention_backend="triton",
        sglang_disable_overlap_schedule=True,
    )

    with pytest.raises(ValueError, match="dense4d requires --sglang-attention-backend torch_native"):
        arguments.slime_validate_args(args)


def test_slime_validate_args_rejects_sparse_block_source_rollout_sampling_without_triton(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args(
        g1_context_length=8,
        g1_ebft_rollout_mask_mode="sparse_ir",
        g1_ebft_rollout_sampling_mode="block_source",
        g1_generate_length=1,
        g1_prompt_length=10,
        g1_response_length=2,
        g1_stride=1,
        sglang_attention_backend="torch_native",
        sglang_disable_overlap_schedule=True,
    )

    with pytest.raises(ValueError, match="sparse_ir requires --sglang-attention-backend triton"):
        arguments.slime_validate_args(args)


def test_slime_validate_args_rejects_block_source_rollout_sampling_with_overlap_schedule(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args(
        g1_context_length=8,
        g1_ebft_rollout_mask_mode="dense4d",
        g1_ebft_rollout_sampling_mode="block_source",
        g1_generate_length=1,
        g1_prompt_length=10,
        g1_response_length=2,
        g1_stride=1,
        sglang_attention_backend="torch_native",
        sglang_disable_overlap_schedule=False,
    )

    with pytest.raises(ValueError, match="requires --sglang-disable-overlap-schedule"):
        arguments.slime_validate_args(args)


def test_slime_validate_args_rejects_mask_transport_in_standard_rollout_sampling(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args(g1_ebft_rollout_mask_mode="dense4d")

    with pytest.raises(ValueError, match="transport only"):
        arguments.slime_validate_args(args)


def test_slime_validate_args_rejects_rollout_mask_mode_without_strict_indexing(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args(g1_ebft_logprob_indexing="standard_next_token", g1_ebft_rollout_mask_mode="dense4d")

    with pytest.raises(ValueError, match="transport only"):
        arguments.slime_validate_args(args)


def test_slime_validate_args_rejects_rollout_mask_mode_without_ebft_loss(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args(g1_use_ebft_loss=False, g1_ebft_rollout_mask_mode="sparse_ir")

    with pytest.raises(ValueError, match="requires --g1-use-ebft-loss"):
        arguments.slime_validate_args(args)


@pytest.mark.parametrize(
    ("cf_target_mode", "overrides"),
    [
        ("opd_onpolicy", {}),
        (
            "teacher",
            {
                "cf_teacher_lambda": 0.6,
                "cf_teacher_n_samples": 4,
                "teacher_api_base": "http://127.0.0.1:30000/v1",
                "teacher_backend": "remote",
                "teacher_model_name": "teacher",
            },
        ),
    ],
)
def test_slime_validate_args_accepts_effopd_supported_cf_target_modes(monkeypatch, cf_target_mode, overrides):
    arguments = _arguments_module(monkeypatch)
    args = _g2_opd_cf_l1oo_args(cf_target_mode=cf_target_mode, **overrides)

    arguments.slime_validate_args(args)


def test_slime_validate_args_accepts_effopd_lr_decay_zero(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _g2_opd_cf_l1oo_args(effopd_lr_decay=0.0)

    arguments.slime_validate_args(args)


def test_slime_validate_args_rejects_opd_onpolicy_without_sglang_opd(monkeypatch, tmp_path):
    arguments = _arguments_module(monkeypatch)
    opd_teacher_load = tmp_path / "opd_teacher"
    opd_teacher_load.mkdir()
    (opd_teacher_load / "latest_checkpointed_iteration.txt").write_text("1\n", encoding="utf-8")
    args = _g2_opd_cf_l1oo_args(use_effopd=False, opd_type="megatron", opd_teacher_load=str(opd_teacher_load))

    with pytest.raises(ValueError, match="OPD-CF-L1OO requires --opd-type sglang"):
        arguments.slime_validate_args(args)


def test_slime_validate_args_rejects_effopd_without_sglang_opd(monkeypatch, tmp_path):
    arguments = _arguments_module(monkeypatch)
    opd_teacher_load = tmp_path / "opd_teacher"
    opd_teacher_load.mkdir()
    (opd_teacher_load / "latest_checkpointed_iteration.txt").write_text("1\n", encoding="utf-8")
    args = _g2_opd_cf_l1oo_args(opd_type="megatron", opd_teacher_load=str(opd_teacher_load))

    with pytest.raises(ValueError, match="EffOPD currently targets SGLang OPD"):
        arguments.slime_validate_args(args)


def test_slime_validate_args_rejects_effopd_without_cf_l1oo(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args(
        cf_target_mode="teacher",
        distribution_reward_type="pointwise",
        opd_type="sglang",
        use_effopd=True,
        use_opd=True,
    )

    with pytest.raises(ValueError, match="EffOPD currently targets G2 cf_l1oo reward"):
        arguments.slime_validate_args(args)


def test_slime_validate_args_rejects_effopd_colocate(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _g2_opd_cf_l1oo_args(colocate=True)

    with pytest.raises(ValueError, match="non-colocate"):
        arguments.slime_validate_args(args)


def test_slime_validate_args_rejects_effopd_without_weights_backuper(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _g2_opd_cf_l1oo_args(enable_weights_backuper=False)

    with pytest.raises(ValueError, match="weights backuper"):
        arguments.slime_validate_args(args)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"effopd_dv_size": 0}, "--effopd-dv-size must be positive"),
        ({"effopd_lr_decay": -0.1}, "--effopd-lr-decay must be non-negative"),
        ({"effopd_max_k": 0}, "--effopd-max-k must be >= 1"),
    ],
)
def test_slime_validate_args_rejects_invalid_effopd_numeric_settings(monkeypatch, override, match):
    arguments = _arguments_module(monkeypatch)
    args = _g2_opd_cf_l1oo_args(**override)

    with pytest.raises(ValueError, match=match):
        arguments.slime_validate_args(args)


def test_slime_validate_args_rejects_effopd_opd_onpolicy_dv_budget_smaller_than_group(monkeypatch):
    arguments = _arguments_module(monkeypatch)
    args = _g2_opd_cf_l1oo_args(effopd_dv_size=3, n_samples_per_prompt=4)

    with pytest.raises(ValueError, match="--effopd-dv-size must be >= --n-samples-per-prompt"):
        arguments.slime_validate_args(args)
