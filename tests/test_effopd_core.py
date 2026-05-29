from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils.effopd.controller import EffOPDController
from slime.backends.megatron_utils.effopd.delta import apply_extrapolation_from_snapshots, restore_named_tensors
from slime.backends.megatron_utils.effopd.state import EffOPDState, load_effopd_state, save_effopd_state
from slime.backends.megatron_utils.effopd.validate import score_from_rollout_data, select_dv_indices


def test_effopd_power_of_two_trigger():
    positives = [1, 2, 4, 8, 16]
    negatives = [0, 3, 5, 6, 7, 9]
    for step in positives:
        assert EffOPDState(opd_update_step=step).should_trigger()
    for step in negatives:
        assert not EffOPDState(opd_update_step=step).should_trigger()


def test_effopd_extrapolation_uses_same_base_and_restores():
    param = torch.tensor([2.0, 4.0])
    base = {"w": torch.tensor([2.0, 4.0])}
    previous = {"w": torch.tensor([1.0, 1.0])}

    norm_sq = apply_extrapolation_from_snapshots([("w", param)], base=base, previous=previous, scale=2)
    assert torch.allclose(param, torch.tensor([4.0, 10.0]))
    assert norm_sq == 10.0

    restore_named_tensors([("w", param)], base)
    assert torch.allclose(param, base["w"])

    apply_extrapolation_from_snapshots([("w", param)], base=base, previous=previous, scale=4)
    # Candidate k=2 must be W_t + 4 * Delta, not previous candidate + 4 * Delta.
    assert torch.allclose(param, torch.tensor([6.0, 16.0]))


def test_effopd_combined_proxy_keeps_g2_and_opd_terms_separate():
    args = SimpleNamespace(opd_kl_coef=2.0, effopd_validation_mode="combined_gate")
    rollout_data = {
        "rewards": [1.0, 3.0],
        "opd_reverse_kl": [torch.tensor([0.25, 0.75]), torch.tensor([0.5])],
    }

    score = score_from_rollout_data(args, rollout_data)

    assert score.cf_l1oo_reward_mean == 2.0
    assert score.opd_reverse_kl_mean == 0.5
    assert score.combined_proxy == 1.0
    assert score.score == score.combined_proxy


def test_effopd_selects_stable_dv_indices():
    first = select_dv_indices(num_samples=10, dv_size=4, seed=7)
    second = select_dv_indices(num_samples=10, dv_size=4, seed=7)
    assert first == second
    assert len(first) == 4
    assert first == sorted(first)
    assert select_dv_indices(num_samples=3, dv_size=50, seed=7) == [0, 1, 2]


def test_effopd_state_persists_dv_indices(tmp_path):
    args = SimpleNamespace(effopd_state_dir=str(tmp_path))
    state = EffOPDState(opd_update_step=4, dv_seed=99, dv_indices=[1, 3], accepted_k=2)

    save_effopd_state(args, state, rollout_id=8, rank=0)
    restored = load_effopd_state(args, rank=0)

    assert restored is not None
    assert restored.opd_update_step == 4
    assert restored.dv_seed == 99
    assert restored.dv_indices == [1, 3]
    assert restored.accepted_k == 2


class _DummyBackuper:
    def __init__(self, source_getter):
        self.source_getter = source_getter
        self.backup_tags = set()
        self.snapshots = {}

    def backup(self, tag):
        self.backup_tags.add(tag)
        self.snapshots[tag] = {name: tensor.detach().cpu().clone() for name, tensor in self.source_getter()}

    def get(self, tag):
        return self.snapshots[tag]

    def copy(self, *, src_tag, dst_tag):
        self.backup_tags.add(dst_tag)
        self.snapshots[dst_tag] = {name: tensor.detach().clone() for name, tensor in self.snapshots[src_tag].items()}


def _effopd_args(tmp_path, *, mode="combined_gate"):
    return SimpleNamespace(
        use_effopd=True,
        effopd_state_dir=str(tmp_path),
        effopd_dv_seed=42,
        effopd_dv_size=2,
        effopd_max_triggers=-1,
        effopd_validation_mode=mode,
        effopd_max_k=3,
        effopd_lr_decay=0.5,
        effopd_force_weight_sync=True,
        opd_kl_coef=1.0,
        lr=1.0,
        load=None,
        save=str(tmp_path),
    )


def test_effopd_combined_gate_requires_evaluator(tmp_path):
    param = torch.tensor([2.0])
    source_getter = lambda: [("w", param)]
    controller = EffOPDController(
        args=_effopd_args(tmp_path),
        source_getter=source_getter,
        backuper=_DummyBackuper(source_getter),
        optimizer=SimpleNamespace(param_groups=[]),
        opt_param_scheduler=None,
    )

    with pytest.raises(RuntimeError, match="validation_evaluator"):
        controller.maybe_extrapolate(
            rollout_id=0,
            rollout_data={"response_lengths": [1], "rewards": [0.0], "opd_reverse_kl": [torch.tensor([0.0])]},
        )


def test_effopd_combined_gate_accepts_largest_passing_candidate(tmp_path):
    param = torch.tensor([1.0])
    source_getter = lambda: [("w", param)]
    backuper = _DummyBackuper(source_getter)

    def evaluator(_rollout_data, dv_indices):
        value = float(param.item())
        score = 0.0 if value <= 2.0 else 1.0 if value <= 4.0 else 0.5
        return SimpleNamespace(
            score=score,
            combined_proxy=score,
            cf_l1oo_reward_mean=score,
            opd_reverse_kl_mean=0.0,
        )

    controller = EffOPDController(
        args=_effopd_args(tmp_path),
        source_getter=source_getter,
        backuper=backuper,
        optimizer=SimpleNamespace(param_groups=[]),
        opt_param_scheduler=None,
        validation_evaluator=evaluator,
    )
    controller.initialise()
    param.fill_(2.0)

    result = controller.maybe_extrapolate(
        rollout_id=0,
        rollout_data={"response_lengths": [1, 1], "rewards": [0.0, 0.0], "opd_reverse_kl": [torch.tensor([0.0])]},
    )

    assert result.accepted
    assert result.accepted_k == 1
    assert torch.allclose(param, torch.tensor([4.0]))
