import argparse
from argparse import Namespace

import pytest
import torch

from slime.utils.g3_ema import (
    G3FeatureAdapter,
    G3EMAAdapterController,
    copy_live_to_ema,
    g3_feature_mse_loss,
    g3_opd_cf_feature_loss,
    get_trainable_adapter_parameters,
    is_g3_opd_fused_mode,
    load_g3_adapter_checkpoint_state,
    raise_if_g3_detached_reward_path,
    select_g1_trainer_sync_source,
    update_ema_parameters,
)
from slime.backends.megatron_utils.g3_critic import run_g3_opd_critic_closure_from_embeddings


def test_g3_feature_adapter_zero_init_is_identity_and_gradients_are_adapter_local():
    adapter = G3FeatureAdapter(feature_dim=5, rank=3, dropout=0.0)
    features = torch.randn(2, 4, 5, requires_grad=True)

    output = adapter(features)

    assert torch.allclose(output, features)
    output.sum().backward()
    assert features.grad is not None
    assert adapter.up.weight.grad is not None
    assert adapter.up.bias.grad is not None


def test_g3_feature_adapter_validates_shape_and_hyperparameters():
    with pytest.raises(ValueError, match="feature_dim"):
        G3FeatureAdapter(feature_dim=0)
    with pytest.raises(ValueError, match="rank"):
        G3FeatureAdapter(feature_dim=4, rank=0)
    with pytest.raises(ValueError, match="dropout"):
        G3FeatureAdapter(feature_dim=4, dropout=1.0)

    adapter = G3FeatureAdapter(feature_dim=4)
    with pytest.raises(ValueError, match="Expected last dim"):
        adapter(torch.randn(2, 3))


def test_g3_ema_update_uses_live_parameter_average_and_freezes_ema():
    live = G3FeatureAdapter(feature_dim=4, rank=2)
    ema = G3FeatureAdapter(feature_dim=4, rank=2)
    copy_live_to_ema(live, ema)

    before = {name: tensor.detach().clone() for name, tensor in ema.state_dict().items()}
    with torch.no_grad():
        for parameter in live.parameters():
            parameter.add_(2.0)

    update_ema_parameters(live, ema, beta=0.25)

    for name, ema_tensor in ema.state_dict().items():
        expected = 0.25 * before[name] + 0.75 * live.state_dict()[name]
        assert torch.allclose(ema_tensor, expected)
    assert all(not parameter.requires_grad for parameter in ema.parameters())
    assert ema.training is False


def test_g3_ema_copy_sets_eval_mode_for_dropout_determinism():
    live = G3FeatureAdapter(feature_dim=6, rank=4, dropout=0.75)
    ema = G3FeatureAdapter(feature_dim=6, rank=4, dropout=0.75)
    live.train()
    ema.train()
    with torch.no_grad():
        live.up.weight.fill_(0.1)
        live.up.bias.fill_(0.2)

    copy_live_to_ema(live, ema)

    features = torch.randn(8, 6)
    first = ema(features)
    second = ema(features)

    assert live.training is True
    assert ema.training is False
    assert all(not parameter.requires_grad for parameter in ema.parameters())
    torch.testing.assert_close(first, second)


def test_g3_feature_loss_detaches_ema_side_and_backprops_to_live_only():
    live = torch.randn(3, 5, requires_grad=True)
    ema = torch.randn(3, 5, requires_grad=True)

    loss = g3_feature_mse_loss(live, ema)
    loss.backward()

    assert live.grad is not None
    assert ema.grad is None


def test_g3_opd_cf_feature_loss_backprops_to_live_adapter_only():
    torch.manual_seed(7)
    adapter = G3FeatureAdapter(feature_dim=6, rank=4, dropout=0.0)
    base = torch.randn(1, 2, 3, 2, 6)
    ema_target = torch.randn(1, 2, 3, 2, 6, requires_grad=True)
    teacher_scores = torch.randn(1, 2, 3, requires_grad=True)

    live = adapter(base)
    loss = g3_opd_cf_feature_loss(
        live,
        ema_target,
        teacher_scores,
        cf_num_freqs=16,
        cf_sigma=1.3,
        cf_seed=11,
    )
    loss.backward()

    assert loss.ndim == 0
    assert adapter.up.weight.grad is not None
    assert adapter.up.bias.grad is not None
    assert adapter.up.weight.grad.abs().sum() > 0
    assert ema_target.grad is None
    assert teacher_scores.grad is None


def test_g3_opd_cf_feature_loss_validates_shapes_and_temperature():
    live = torch.randn(1, 1, 2, 1, 4)
    ema = torch.randn(1, 1, 2, 1, 4)
    scores = torch.randn(1, 1, 2)

    with pytest.raises(ValueError, match="N > 1"):
        g3_opd_cf_feature_loss(live[:, :, :1], ema[:, :, :1], scores[:, :, :1])
    with pytest.raises(ValueError, match="shapes must match"):
        g3_opd_cf_feature_loss(live, ema[..., :3], scores)
    with pytest.raises(ValueError, match="score_temperature"):
        g3_opd_cf_feature_loss(live, ema, scores, score_temperature=0.0)


def test_g3_adapter_controller_step_updates_live_and_ema_without_base_grad():
    torch.manual_seed(13)
    controller = G3EMAAdapterController.create(
        feature_dim=5,
        rank=3,
        dropout=0.0,
        lr=1e-2,
        ema_beta=0.5,
        feature_loss_coef=0.25,
    )
    base = torch.randn(1, 1, 3, 2, 5, requires_grad=True)
    teacher_scores = torch.tensor([[[0.0, 1.0, -1.0]]], requires_grad=True)
    live_before = {name: tensor.detach().clone() for name, tensor in controller.live_adapter.state_dict().items()}
    ema_before = {name: tensor.detach().clone() for name, tensor in controller.ema_adapter.state_dict().items()}

    result = controller.train_feature_step(
        base,
        teacher_scores,
        cf_num_freqs=12,
        cf_sigma=1.1,
        cf_seed=17,
    )

    assert result.loss.item() > 0.0
    assert result.raw_feature_loss.item() > 0.0
    assert base.grad is None
    assert teacher_scores.grad is None
    assert any(
        not torch.allclose(tensor, live_before[name])
        for name, tensor in controller.live_adapter.state_dict().items()
    )
    assert any(
        not torch.allclose(tensor, ema_before[name])
        for name, tensor in controller.ema_adapter.state_dict().items()
    )
    assert all(not parameter.requires_grad for parameter in controller.ema_adapter.parameters())
    assert controller.ema_adapter.training is False


def test_g3_critic_closure_steps_adapter_and_returns_sync_ready_advantages():
    torch.manual_seed(23)
    args = Namespace(
        g3_enable=True,
        distribution_reward_type="cf_l1oo",
        cf_target_mode="opd_onpolicy",
        use_opd=True,
        n_samples_per_prompt=3,
        g1_response_length=4,
        use_whitening=True,
        whiten_tol=1e-5,
        cf_num_freqs=12,
        cf_sigma=1.0,
        cf_seed=19,
        cf_alpha=0.5,
        cf_beta=0.5,
        cf_reward_scale=1.0,
        opd_cf_score_temperature=1.0,
        opd_cf_score_normalization="mean",
    )
    controller = G3EMAAdapterController.create(
        feature_dim=4,
        rank=3,
        dropout=0.0,
        lr=1e-2,
        ema_beta=0.5,
        feature_loss_coef=0.25,
    )
    live_before = {name: tensor.detach().clone() for name, tensor in controller.live_adapter.state_dict().items()}
    gen_embeddings = [
        torch.tensor([[1.0, 0.2, 0.0, 0.1], [0.5, 0.0, 0.4, 0.3]]),
        torch.tensor([[0.0, 1.0, 0.2, 0.1], [0.2, 0.7, 0.1, 0.5]]),
        torch.tensor([[0.1, 0.0, 1.0, 0.2], [0.1, 0.3, 0.8, 0.4]]),
    ]
    teacher_log_probs = [
        torch.tensor([-2.0, -2.2, -2.1, -2.3]),
        torch.tensor([-0.2, -0.3, -0.2, -0.4]),
        torch.tensor([-1.0, -1.1, -1.2, -1.0]),
    ]

    result = run_g3_opd_critic_closure_from_embeddings(
        args,
        controller,
        gen_embeddings,
        [4, 4, 4],
        teacher_log_probs,
    )

    assert result.feature_step.loss.item() > 0.0
    assert result.feature_step.raw_feature_loss.item() > 0.0
    assert tuple(result.teacher_scores.shape) == (1, 1, 3)
    assert tuple(result.block_rewards.shape) == (3, 2)
    assert len(result.token_advantages) == 3
    assert all(tuple(advantage.shape) == (4,) for advantage in result.token_advantages)
    assert all(torch.isfinite(advantage).all() for advantage in result.token_advantages)
    assert all(isinstance(value, float) for value in result.scalar_rewards)
    assert any(
        not torch.allclose(tensor, live_before[name])
        for name, tensor in controller.live_adapter.state_dict().items()
    )
    assert result.normalized_base_embedding.requires_grad is False
    assert result.reward_embedding.requires_grad is False


def test_g3_critic_closure_validates_sync_boundary_shapes():
    args = Namespace(
        g3_enable=True,
        distribution_reward_type="cf_l1oo",
        cf_target_mode="opd_onpolicy",
        use_opd=True,
        n_samples_per_prompt=2,
        g1_response_length=3,
        use_whitening=True,
        opd_cf_score_normalization="mean",
    )
    controller = G3EMAAdapterController.create(feature_dim=2, rank=1, lr=1e-2)
    embeddings = [torch.randn(2, 2), torch.randn(2, 2)]
    teacher_log_probs = [torch.randn(3), torch.randn(3)]

    with pytest.raises(ValueError, match="not divisible by num_blocks"):
        run_g3_opd_critic_closure_from_embeddings(
            args,
            controller,
            embeddings,
            [3, 3],
            teacher_log_probs,
        )


def test_g3_adapter_param_selection_and_checkpoint_missing_ema_init():
    live = G3FeatureAdapter(feature_dim=4, rank=2)
    ema = G3FeatureAdapter(feature_dim=4, rank=2)
    optimizer = torch.optim.AdamW(get_trainable_adapter_parameters(live), lr=1e-3)
    with torch.no_grad():
        live.up.bias.fill_(0.75)

    load_g3_adapter_checkpoint_state(
        live,
        ema,
        optimizer,
        {"live_adapter": live.state_dict()},
    )

    assert get_trainable_adapter_parameters(live) == list(live.parameters())
    for name, tensor in live.state_dict().items():
        torch.testing.assert_close(ema.state_dict()[name], tensor)
    assert all(not parameter.requires_grad for parameter in ema.parameters())


def test_g3_selects_critic_as_trainer_reward_sync_source():
    base = dict(
        distribution_reward_type="cf_l1oo",
        advantage_estimator="g1",
        g1_reward_location="trainer",
    )

    assert select_g1_trainer_sync_source(Namespace(**base, cf_target_mode="single", g3_enable=False)) == 0
    assert select_g1_trainer_sync_source(Namespace(**base, cf_target_mode="teacher", g3_enable=False)) == 1
    assert select_g1_trainer_sync_source(Namespace(**base, cf_target_mode="opd_onpolicy", g3_enable=True)) == 1
    pointwise = dict(base, distribution_reward_type="pointwise", cf_target_mode="single", g3_enable=True)
    assert select_g1_trainer_sync_source(Namespace(**pointwise)) is None


def test_g3_opd_fused_mode_predicate_is_explicit_to_actor_path():
    valid = Namespace(
        g3_enable=True,
        distribution_reward_type="cf_l1oo",
        cf_target_mode="opd_onpolicy",
        use_opd=True,
    )

    assert is_g3_opd_fused_mode(valid) is True
    assert is_g3_opd_fused_mode(Namespace(**{**vars(valid), "g3_enable": False})) is False
    assert is_g3_opd_fused_mode(Namespace(**{**vars(valid), "cf_target_mode": "teacher"})) is False
    assert is_g3_opd_fused_mode(Namespace(**{**vars(valid), "use_opd": False})) is False


def test_g3_parser_defaults_and_validation_blocks_minimal_contract(monkeypatch):
    from tests.test_g1_ebft_arguments import _arguments_module, _base_slime_args

    arguments = _arguments_module(monkeypatch)
    parser = argparse.ArgumentParser()
    arguments.get_slime_extra_args_provider()(parser)
    parsed = parser.parse_args(["--rollout-batch-size", "1"])

    assert parsed.g3_enable is False
    assert parsed.feature_adapter_enable is False
    assert parsed.feature_adapter_rank == 64
    assert parsed.feature_adapter_dropout == 0.0
    assert parsed.enable_ema is False
    assert parsed.ema_beta == 0.99
    assert parsed.g3_adapter_lr == 5e-5
    assert parsed.g3_feature_loss_coef == 0.1

    args = _base_slime_args(
        distribution_reward_type="cf_l1oo",
        cf_target_mode="opd_onpolicy",
        g1_embedding_source="megatron_ref",
        g1_reward_location="trainer",
        n_samples_per_prompt=2,
        opd_credit_assignment="cf_l1oo",
        use_opd=True,
        opd_type="sglang",
        use_whitening=True,
        critic_lr=0.0,
        critic_lr_head=0.0,
        zero_stage=3,
        g3_enable=True,
        feature_adapter_enable=True,
        enable_ema=True,
    )

    with pytest.raises(ValueError, match="critic-side differentiable adapter/EMA training closure"):
        arguments.slime_validate_args(args)


def test_g3_validation_rejects_rollout_side_before_detached_guard_can_be_bypassed(monkeypatch):
    from tests.test_g1_ebft_arguments import _arguments_module, _base_slime_args

    arguments = _arguments_module(monkeypatch)
    args = _base_slime_args(
        distribution_reward_type="cf_l1oo",
        cf_target_mode="opd_onpolicy",
        g1_embedding_source="rollout",
        g1_reward_location="rollout",
        n_samples_per_prompt=2,
        opd_credit_assignment="cf_l1oo",
        use_opd=True,
        opd_type="sglang",
        use_whitening=False,
        critic_lr=0.0,
        critic_lr_head=0.0,
        zero_stage=2,
        g3_enable=True,
        feature_adapter_enable=True,
        enable_ema=True,
    )

    with pytest.raises(ValueError, match="critic-side differentiable adapter/EMA training closure"):
        arguments.slime_validate_args(args)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"distribution_reward_type": "pointwise"}, "distribution-reward-type cf_l1oo"),
        ({"cf_target_mode": "teacher"}, "cf-target-mode opd_onpolicy"),
        ({"use_opd": False}, "requires --use-opd"),
        ({"opd_credit_assignment": "ebft"}, "opd-credit-assignment cf_l1oo"),
        ({"feature_adapter_enable": False}, "requires --feature-adapter-enable"),
        ({"enable_ema": False}, "requires --enable-ema"),
        ({"teacher_backend": "remote"}, "must not set --teacher-backend"),
    ],
)
def test_g3_validation_rejects_non_opd_fused_contract(monkeypatch, overrides, message):
    from tests.test_g1_ebft_arguments import _arguments_module, _base_slime_args

    arguments = _arguments_module(monkeypatch)
    values = dict(
        distribution_reward_type="cf_l1oo",
        cf_target_mode="opd_onpolicy",
        g1_embedding_source="megatron_ref",
        g1_reward_location="trainer",
        n_samples_per_prompt=2,
        opd_credit_assignment="cf_l1oo",
        use_opd=True,
        opd_type="sglang",
        use_whitening=True,
        critic_lr=0.0,
        critic_lr_head=0.0,
        zero_stage=3,
        g3_enable=True,
        feature_adapter_enable=True,
        enable_ema=True,
    )
    values.update(overrides)
    args = _base_slime_args(**values)

    with pytest.raises(ValueError, match=message):
        arguments.slime_validate_args(args)


def test_g3_runtime_guard_blocks_detached_trainer_side_reward_path():
    args = Namespace(g3_enable=True)

    with pytest.raises(NotImplementedError, match="critic-side differentiable adapter/EMA"):
        raise_if_g3_detached_reward_path(args)

    raise_if_g3_detached_reward_path(Namespace(g3_enable=False))
