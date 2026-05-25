from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.distributed as dist

from slime.utils.distributed_utils import get_gloo_group
from slime.utils.tensor_backper import TensorBackuper

from .delta import apply_extrapolation_from_snapshots, restore_named_tensors, snapshot_named_tensors
from .state import (
    EFFOPD_TAG_PREV_POWER,
    EFFOPD_TAG_W0,
    EFFOPD_TERMINOLOGY,
    EffOPDResult,
    EffOPDState,
    effopd_tensor_state_path,
    load_effopd_state,
    save_effopd_state,
)
from .validate import score_from_rollout_data
from .validate import select_dv_indices

logger = logging.getLogger(__name__)


class EffOPDController:
    """EffOPD mechanism controller for the G2 cf_l1oo + SGLang OPD workflow.

    Shadow modes exercise trigger/state/delta plumbing without changing final
    weights. combined_gate requires a D_v evaluator that scores each live
    candidate before accepting an extrapolation.
    """

    def __init__(
        self,
        *,
        args,
        source_getter: Callable[[], Iterable[tuple[str, torch.Tensor]]],
        backuper: TensorBackuper,
        optimizer: Any,
        opt_param_scheduler: Any,
        validation_evaluator: Callable[[dict[str, Any], list[int]], Any] | None = None,
        decision_rank_getter: Callable[[], bool] | None = None,
    ) -> None:
        self.args = args
        self.source_getter = source_getter
        self.backuper = backuper
        self.optimizer = optimizer
        self.opt_param_scheduler = opt_param_scheduler
        self.validation_evaluator = validation_evaluator
        self.decision_rank_getter = decision_rank_getter or (lambda: self.rank == 0)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.state = load_effopd_state(args, rank=self.rank) or EffOPDState(
            dv_seed=int(getattr(args, "effopd_dv_seed", 42)),
        )
        self._loaded_prev_power_snapshot: dict[str, torch.Tensor] | None = None
        if self.state.prev_power_path:
            path = self.state.prev_power_path
            try:
                self._loaded_prev_power_snapshot = torch.load(path, map_location="cpu", weights_only=False)
                logger.info("Loaded EffOPD prev_power anchor from %s", path)
            except FileNotFoundError:
                logger.warning("EffOPD prev_power anchor not found at %s; next trigger may be skipped.", path)
        self._initialised = False

    def initialise(self) -> None:
        if self._initialised:
            return
        self.backuper.backup(EFFOPD_TAG_W0)
        if EFFOPD_TAG_PREV_POWER not in self.backuper.backup_tags:
            self.backuper.backup(EFFOPD_TAG_PREV_POWER)
        self._initialised = True
        if self.rank == 0:
            logger.info("EffOPD enabled for G2 cf_l1oo + SGLang OPD. %s", EFFOPD_TERMINOLOGY)

    def maybe_extrapolate(self, *, rollout_id: int, rollout_data: dict[str, Any]) -> EffOPDResult:
        if not bool(getattr(self.args, "use_effopd", False)):
            return EffOPDResult(enabled=False)

        self.initialise()
        self.state.opd_update_step += 1
        self.state.last_rollout_id = int(rollout_id)

        max_triggers = getattr(self.args, "effopd_max_triggers", None)
        if max_triggers is None:
            max_triggers = -1
        if not self.state.should_trigger(max_triggers=int(max_triggers)):
            save_effopd_state(self.args, self.state, rollout_id=rollout_id, rank=self.rank)
            return EffOPDResult(enabled=True, opd_update_step=self.state.opd_update_step)

        validation = score_from_rollout_data(self.args, rollout_data)
        base_score = validation.score
        best_score = base_score
        accepted_k = 0
        accepted = False

        num_samples = len(rollout_data.get("response_lengths") or rollout_data.get("tokens") or [])
        self.state.dv_indices = select_dv_indices(
            num_samples=num_samples,
            dv_size=int(getattr(self.args, "effopd_dv_size", 50)),
            seed=int(getattr(self.args, "effopd_dv_seed", self.state.dv_seed)),
            existing_indices=self.state.dv_indices,
        )

        base_snapshot = snapshot_named_tensors(self.source_getter())
        if self.state.opd_update_step == 1:
            previous_snapshot = self.backuper.get(EFFOPD_TAG_W0)
        elif self._loaded_prev_power_snapshot is not None:
            previous_snapshot = self._loaded_prev_power_snapshot
        else:
            previous_snapshot = self.backuper.get(EFFOPD_TAG_PREV_POWER)

        mode = getattr(self.args, "effopd_validation_mode", "opd_kl_shadow_cf")
        gate_enabled = mode == "combined_gate"
        max_k = int(getattr(self.args, "effopd_max_k", 5))
        delta_norm_sq = 0.0

        if gate_enabled:
            if self.validation_evaluator is None:
                raise RuntimeError(
                    "EffOPD combined_gate requires a real D_v validation_evaluator; "
                    "use a shadow mode for mechanism-only validation."
                )
            is_decision_rank = bool(self.decision_rank_getter())
            decision_src_rank = self._resolve_decision_src_rank(is_decision_rank)
            validation = self.validation_evaluator(rollout_data, self.state.dv_indices)
            if is_decision_rank:
                base_score = validation.score
                best_score = base_score
            for k in range(1, max_k + 1):
                candidate_delta_norm_sq = apply_extrapolation_from_snapshots(
                    self.source_getter(),
                    base=base_snapshot,
                    previous=previous_snapshot,
                    scale=2**k,
                )
                candidate_score = self.validation_evaluator(rollout_data, self.state.dv_indices)
                restore_named_tensors(self.source_getter(), base_snapshot)
                delta_norm_sq = candidate_delta_norm_sq
                candidate_passed = True
                if is_decision_rank:
                    candidate_passed = candidate_score.score >= best_score
                    if candidate_passed:
                        accepted_k = k
                        accepted = True
                        validation = candidate_score
                        best_score = candidate_score.score
                candidate_passed = self._broadcast_candidate_pass(
                    candidate_passed,
                    src_rank=decision_src_rank,
                )
                if not candidate_passed:
                    break

            accepted_k, best_score, force_weight_sync = self._broadcast_decision(
                accepted_k=accepted_k,
                best_score=best_score,
                force_weight_sync=accepted or bool(getattr(self.args, "effopd_force_weight_sync", True)),
                src_rank=decision_src_rank,
            )
            accepted = accepted_k > 0
            if accepted:
                delta_norm_sq = apply_extrapolation_from_snapshots(
                    self.source_getter(),
                    base=base_snapshot,
                    previous=previous_snapshot,
                    scale=2**accepted_k,
                )
        else:
            # Shadow mode: exercise delta math and immediately restore W_t.
            delta_norm_sq = apply_extrapolation_from_snapshots(
                self.source_getter(),
                base=base_snapshot,
                previous=previous_snapshot,
                scale=2.0,
            )
            restore_named_tensors(self.source_getter(), base_snapshot)
            force_weight_sync = bool(getattr(self.args, "effopd_force_weight_sync", True))

        if accepted:
            self._decay_learning_rate()
            self.state.num_accepts += 1
            self.state.accepted_k = accepted_k
            self.state.lr_scale *= float(getattr(self.args, "effopd_lr_decay", 0.5))
            self.backuper.backup("actor")
        else:
            restore_named_tensors(self.source_getter(), base_snapshot)
            self.backuper.backup("actor")
            self.state.accepted_k = 0

        # The next delta anchor is the accepted post-train checkpoint. This
        # follows the project plan's post-save semantics.
        self.backuper.copy(src_tag="actor", dst_tag=EFFOPD_TAG_PREV_POWER)
        prev_power_snapshot = snapshot_named_tensors(self.source_getter())
        prev_power_path = effopd_tensor_state_path(
            self.args,
            rollout_id=rollout_id,
            tag="prev_power",
            rank=self.rank,
        )
        prev_power_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(prev_power_snapshot, prev_power_path)
        self._loaded_prev_power_snapshot = prev_power_snapshot
        self.state.prev_power_step = self.state.opd_update_step
        self.state.prev_power_path = str(prev_power_path)
        self.state.num_triggers += 1
        self.state.last_score = float(best_score)
        self.state.last_combined_proxy = validation.combined_proxy
        save_effopd_state(self.args, self.state, rollout_id=rollout_id, rank=self.rank)

        result = EffOPDResult(
            enabled=True,
            triggered=True,
            accepted=accepted,
            accepted_k=accepted_k,
            force_weight_sync=force_weight_sync,
            opd_update_step=self.state.opd_update_step,
            base_score=base_score,
            best_score=best_score,
            combined_proxy=validation.combined_proxy,
            cf_l1oo_reward_mean=validation.cf_l1oo_reward_mean,
            opd_reverse_kl_mean=validation.opd_reverse_kl_mean,
            delta_norm=math.sqrt(max(delta_norm_sq, 0.0)),
            lr_scale=self.state.lr_scale,
            message=f"EffOPD {mode}: G2=cf_l1oo reward, OPD=SGLang distillation",
        )
        return result

    def _broadcast_decision(
        self,
        *,
        accepted_k: int,
        best_score: float,
        force_weight_sync: bool,
        src_rank: int | None,
    ) -> tuple[int, float, bool]:
        if not dist.is_initialized():
            return accepted_k, best_score, force_weight_sync
        if src_rank is None or src_rank < 0:
            raise RuntimeError("EffOPD combined_gate could not find a decision rank.")
        tensor = torch.tensor(
            [int(accepted_k), float(best_score), 1.0 if force_weight_sync else 0.0],
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        dist.broadcast(tensor, src=src_rank, group=get_gloo_group())
        return int(tensor[0].item()), float(tensor[1].item()), bool(int(tensor[2].item()))

    def _resolve_decision_src_rank(self, is_decision_rank: bool) -> int | None:
        if not dist.is_initialized():
            return None
        src_tensor = torch.tensor(
            [self.rank if is_decision_rank else -1],
            dtype=torch.int64,
            device=torch.device("cpu"),
        )
        dist.all_reduce(src_tensor, op=dist.ReduceOp.MAX, group=get_gloo_group())
        src_rank = int(src_tensor.item())
        if src_rank < 0:
            raise RuntimeError("EffOPD combined_gate could not find a decision rank.")
        return src_rank

    def _broadcast_candidate_pass(self, candidate_passed: bool, *, src_rank: int | None) -> bool:
        if not dist.is_initialized():
            return candidate_passed
        tensor = torch.tensor(
            [1 if candidate_passed else 0],
            dtype=torch.int32,
            device=torch.device("cpu"),
        )
        if src_rank is None or src_rank < 0:
            raise RuntimeError("EffOPD combined_gate could not find a decision rank.")
        dist.broadcast(tensor, src=src_rank, group=get_gloo_group())
        return bool(int(tensor.item()))

    def _decay_learning_rate(self) -> None:
        decay = float(getattr(self.args, "effopd_lr_decay", 0.5))
        if decay <= 0:
            return
        optimizers = [self.optimizer]
        inner = getattr(self.optimizer, "optimizer", None)
        if inner is not None and inner is not self.optimizer:
            optimizers.append(inner)
        for optimizer in optimizers:
            for group in getattr(optimizer, "param_groups", []) or []:
                group["lr"] = float(group.get("lr", getattr(self.args, "lr", 0.0))) * decay
        logger.info("EffOPD accepted extrapolation; decayed actor lr by factor %.6g", decay)
