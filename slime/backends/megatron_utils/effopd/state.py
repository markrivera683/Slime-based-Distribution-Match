from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


EFFOPD_TAG_W0 = "effopd_w0"
EFFOPD_TAG_PREV_POWER = "effopd_prev_power"
EFFOPD_TAG_TRIGGER_BASE = "effopd_trigger_base"
EFFOPD_TAG_ACCEPTED = "effopd_accepted"

EFFOPD_TERMINOLOGY = (
    "G2=cf_l1oo reward/distribution matching; OPD=SGLang teacher-logprob distillation; "
    "G2+OPD means cf_l1oo reward plus OPD KL penalty."
)


@dataclass
class EffOPDState:
    opd_update_step: int = 0
    prev_power_step: int = 0
    accepted_k: int = 0
    lr_scale: float = 1.0
    dv_seed: int = 42
    dv_indices: list[int] = field(default_factory=list)
    num_triggers: int = 0
    num_accepts: int = 0
    last_rollout_id: int = -1
    last_score: float | None = None
    last_combined_proxy: float | None = None
    prev_power_path: str | None = None

    @staticmethod
    def is_power_of_two(step: int) -> bool:
        return step > 0 and (step & (step - 1)) == 0

    def should_trigger(self, *, max_triggers: int | None = None) -> bool:
        if not self.is_power_of_two(self.opd_update_step):
            return False
        if max_triggers is not None and max_triggers >= 0 and self.num_triggers >= max_triggers:
            return False
        return True

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "EffOPDState":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in payload.items() if key in known})


@dataclass
class EffOPDResult:
    enabled: bool
    triggered: bool = False
    accepted: bool = False
    accepted_k: int = 0
    force_weight_sync: bool = False
    opd_update_step: int = 0
    base_score: float | None = None
    best_score: float | None = None
    combined_proxy: float | None = None
    cf_l1oo_reward_mean: float | None = None
    opd_reverse_kl_mean: float | None = None
    delta_norm: float | None = None
    lr_scale: float = 1.0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_effopd_state_dir(args) -> Path:
    state_dir = getattr(args, "effopd_state_dir", "auto")
    if state_dir and state_dir != "auto":
        return Path(state_dir).expanduser()
    save_dir = getattr(args, "save", None) or getattr(args, "load", None) or "."
    return Path(save_dir).expanduser() / "effopd"


def effopd_state_path(args, *, rollout_id: int | None = None, rank: int | None = None) -> Path:
    state_dir = resolve_effopd_state_dir(args)
    rank_prefix = "" if rank is None else f"rank{int(rank)}_"
    if rollout_id is None:
        return state_dir / f"{rank_prefix}effopd_state_latest.json"
    return state_dir / f"{rank_prefix}effopd_state_{int(rollout_id)}.json"


def save_effopd_state(args, state: EffOPDState, *, rollout_id: int | None = None, rank: int | None = None) -> Path:
    path = effopd_state_path(args, rollout_id=rollout_id, rank=rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.to_json_dict()
    payload["terminology"] = EFFOPD_TERMINOLOGY
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if rollout_id is not None:
        latest = effopd_state_path(args, rollout_id=None, rank=rank)
        latest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_effopd_state(args, *, rank: int | None = None) -> EffOPDState | None:
    path = effopd_state_path(args, rollout_id=None, rank=rank)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return EffOPDState.from_json_dict(payload)


def effopd_tensor_state_path(args, *, rollout_id: int, tag: str, rank: int | None = None) -> Path:
    state_dir = resolve_effopd_state_dir(args)
    rank_prefix = "" if rank is None else f"rank{int(rank)}_"
    return state_dir / f"{rank_prefix}{tag}_{int(rollout_id)}.pt"
