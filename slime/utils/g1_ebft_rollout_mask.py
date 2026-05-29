"""CPU-only strict G1 EBFT rollout mask contract helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from slime.utils.g1_core import get_num_strided_blocks
from slime.utils.g1_ebft_loss import G1_EBFT_LOGPROB_INDEXING_STRICT_BLOCK
from slime.utils.g1_ebft_loss import build_ebft_g1_logprob_pair_axis


PROMPT_SOURCE_KIND = "prompt_response_position"
GENERATED_SOURCE_KIND = "previous_generated"
PROMPT_KEY_KIND = "prompt"
GENERATED_KEY_KIND = "generated"
EBFT_SPAN_SPARSE_IR_VERSION = 1
EBFT_SPAN_SPARSE_IR_LAYOUT = "ebft_block_strided_v1"


@dataclass(frozen=True)
class G1EBFTRolloutSource:
    """One generated rollout token's sampling source logit row and target row."""

    step: int
    block: int
    response_position: int
    source_row: int
    source_kind: str
    target_position: int
    logprob_source_row: int


@dataclass(frozen=True)
class G1EBFTAllowedEdge:
    """Sparse attention-mask edge: query row may attend to key row."""

    query_position: int
    key_position: int
    query_kind: str
    key_kind: str


@dataclass(frozen=True)
class G1EBFTRolloutMaskContract:
    """Strict EBFT rollout layout plus sparse/dense mask projections."""

    prompt_length: int
    context_length: int
    generate_length: int
    stride: int
    num_blocks: int
    response_length: int
    full_sequence_length: int
    response_positions: tuple[int, ...]
    position_ids: tuple[int, ...]
    target_positions: tuple[int, ...]
    rollout_sources: tuple[G1EBFTRolloutSource, ...]
    logprob_source_rows: tuple[int, ...]
    logprob_target_positions: tuple[int, ...]
    allowed_edges: tuple[G1EBFTAllowedEdge, ...]
    qa_values: tuple[Any, ...] | None = None
    doc_ids: tuple[Any, ...] | None = None
    response_qa_values: tuple[Any, ...] | None = None
    response_doc_ids: tuple[Any, ...] | None = None
    document_masking: bool = False

    @property
    def rollout_source_rows(self) -> tuple[int, ...]:
        return tuple(source.source_row for source in self.rollout_sources)

    @property
    def rollout_anchor_positions(self) -> tuple[int, ...]:
        return self.response_positions

    @property
    def rollout_source_kinds(self) -> tuple[str, ...]:
        return tuple(source.source_kind for source in self.rollout_sources)

    def rollout_source_rows_by_step(self) -> tuple[tuple[int, ...], ...]:
        rows_by_step: list[tuple[int, ...]] = []
        for step in range(self.generate_length):
            rows_by_step.append(
                tuple(source.source_row for source in self.rollout_sources if source.step == step)
            )
        return tuple(rows_by_step)

    def to_sparse_allowed_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple((edge.query_position, edge.key_position) for edge in self.allowed_edges)

    def to_span_sparse_ir(self) -> dict[str, Any]:
        return build_ebft_span_sparse_ir_from_allowed_edges(
            allowed_edges=self.allowed_edges,
            seq_len=self.full_sequence_length,
            query_len=self.full_sequence_length,
            prefix_len=self.prompt_length,
            geometry_debug_fields={
                "prompt_length": self.prompt_length,
                "context_length": self.context_length,
                "generate_length": self.generate_length,
                "stride": self.stride,
                "num_blocks": self.num_blocks,
                "response_length": self.response_length,
                "full_sequence_length": self.full_sequence_length,
                "response_positions": self.response_positions,
                "rollout_anchor_positions": self.rollout_anchor_positions,
                "target_positions": self.target_positions,
                "rollout_source_rows": self.rollout_source_rows,
                "rollout_source_kinds": self.rollout_source_kinds,
                "logprob_source_rows": self.logprob_source_rows,
                "logprob_target_positions": self.logprob_target_positions,
                "document_masking": self.document_masking,
            },
        )

    def to_sparse_ir(self) -> dict[str, Any]:
        return self.to_span_sparse_ir()

    def to_mask_spec(self, *, mode: str) -> dict[str, Any]:
        return {
            "version": 1,
            "layout": "g1_strict_block_source",
            "mode": mode,
            "prompt_length": self.prompt_length,
            "context_length": self.context_length,
            "generate_length": self.generate_length,
            "stride": self.stride,
            "num_blocks": self.num_blocks,
            "response_length": self.response_length,
            "full_sequence_length": self.full_sequence_length,
            "response_positions": self.response_positions,
            "rollout_anchor_positions": self.rollout_anchor_positions,
            "target_positions": self.target_positions,
            "rollout_source_rows": self.rollout_source_rows,
            "rollout_source_kinds": self.rollout_source_kinds,
            "logprob_source_rows": self.logprob_source_rows,
            "logprob_target_positions": self.logprob_target_positions,
            "qa_values": self.qa_values,
            "doc_ids": self.doc_ids,
            "response_qa_values": self.response_qa_values,
            "response_doc_ids": self.response_doc_ids,
            "document_masking": self.document_masking,
        }

    def to_dense_allowed_matrix(self, *, batch_size: int = 1) -> torch.Tensor:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        allowed = torch.zeros(
            (1, 1, self.full_sequence_length, self.full_sequence_length),
            dtype=torch.bool,
            device=torch.device("cpu"),
        )
        for edge in self.allowed_edges:
            allowed[0, 0, edge.query_position, edge.key_position] = True
        return allowed.expand(batch_size, -1, -1, -1).clone()

    def to_dense_additive_mask(
        self,
        *,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        masked_value: float | None = None,
    ) -> torch.Tensor:
        allowed = self.to_dense_allowed_matrix(batch_size=batch_size)
        if masked_value is None:
            masked_value = torch.finfo(dtype).min
        mask = torch.full(allowed.shape, masked_value, dtype=dtype, device=torch.device("cpu"))
        mask.masked_fill_(allowed, 0.0)
        return mask


def build_ebft_span_sparse_ir_from_allowed_edges(
    *,
    allowed_edges: tuple[G1EBFTAllowedEdge, ...] | list[G1EBFTAllowedEdge],
    seq_len: int,
    query_len: int,
    prefix_len: int,
    geometry_debug_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical span sparse IR from logical allowed query/key edges."""

    seq_len = int(seq_len)
    query_len = int(query_len)
    prefix_len = int(prefix_len)
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if query_len < 1:
        raise ValueError(f"query_len must be >= 1, got {query_len}")
    if query_len > seq_len:
        raise ValueError(f"query_len must be <= seq_len ({query_len} > {seq_len})")
    if not (0 <= prefix_len <= seq_len):
        raise ValueError(f"prefix_len must be within [0, seq_len], got {prefix_len}")

    keys_by_query: list[list[int]] = [[] for _ in range(query_len)]
    seen: set[tuple[int, int]] = set()
    for edge in allowed_edges:
        query_position = int(edge.query_position)
        key_position = int(edge.key_position)
        if not (0 <= query_position < query_len):
            raise ValueError(f"allowed edge query_position out of sparse IR range: {edge}")
        if not (0 <= key_position < seq_len):
            raise ValueError(f"allowed edge key_position out of sparse IR range: {edge}")
        pair = (query_position, key_position)
        if pair in seen:
            raise ValueError(f"duplicate allowed edge: {pair}")
        seen.add(pair)
        keys_by_query[query_position].append(key_position)

    q_indptr: list[int] = [0]
    span_starts: list[int] = []
    span_ends: list[int] = []
    for keys in keys_by_query:
        if keys:
            sorted_keys = sorted(keys)
            start = sorted_keys[0]
            prev = sorted_keys[0]
            for key_position in sorted_keys[1:]:
                if key_position == prev + 1:
                    prev = key_position
                    continue
                span_starts.append(start)
                span_ends.append(prev + 1)
                start = key_position
                prev = key_position
            span_starts.append(start)
            span_ends.append(prev + 1)
        q_indptr.append(len(span_starts))

    sparse_ir: dict[str, Any] = {
        "version": EBFT_SPAN_SPARSE_IR_VERSION,
        "layout": EBFT_SPAN_SPARSE_IR_LAYOUT,
        "seq_len": seq_len,
        "query_len": query_len,
        "prefix_len": prefix_len,
        "q_indptr": tuple(q_indptr),
        "span_starts": tuple(span_starts),
        "span_ends": tuple(span_ends),
    }
    if geometry_debug_fields:
        sparse_ir.update(geometry_debug_fields)
    validate_ebft_span_sparse_ir(sparse_ir)
    return sparse_ir


def validate_ebft_span_sparse_ir(sparse_ir: dict[str, Any]) -> None:
    """Fail fast on malformed EBFT span sparse IR."""

    _normalize_ebft_span_sparse_ir(sparse_ir)


def ebft_span_sparse_ir_to_dense_allowed_matrix(
    sparse_ir: dict[str, Any],
    *,
    batch_size: int = 1,
) -> torch.Tensor:
    """Convert EBFT span sparse IR to a CPU bool allowed matrix oracle."""

    seq_len, query_len, _prefix_len, q_indptr, span_starts, span_ends = _normalize_ebft_span_sparse_ir(sparse_ir)
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    allowed = torch.zeros(
        (1, 1, query_len, seq_len),
        dtype=torch.bool,
        device=torch.device("cpu"),
    )
    for query_position in range(query_len):
        for span_idx in range(q_indptr[query_position], q_indptr[query_position + 1]):
            allowed[0, 0, query_position, span_starts[span_idx] : span_ends[span_idx]] = True
    return allowed.expand(batch_size, -1, -1, -1).clone()


def _normalize_ebft_span_sparse_ir(
    sparse_ir: dict[str, Any],
) -> tuple[int, int, int, list[int], list[int], list[int]]:
    if not isinstance(sparse_ir, dict):
        raise ValueError(f"sparse_ir must be a dict, got {type(sparse_ir).__name__}")
    version = sparse_ir.get("version")
    if version != EBFT_SPAN_SPARSE_IR_VERSION:
        raise ValueError(f"EBFT sparse IR version must be {EBFT_SPAN_SPARSE_IR_VERSION}, got {version!r}")
    layout = sparse_ir.get("layout")
    if layout != EBFT_SPAN_SPARSE_IR_LAYOUT:
        raise ValueError(f"EBFT sparse IR layout must be {EBFT_SPAN_SPARSE_IR_LAYOUT!r}, got {layout!r}")

    seq_len = _require_int_field(sparse_ir, "seq_len")
    query_len = _require_int_field(sparse_ir, "query_len")
    prefix_len = _require_int_field(sparse_ir, "prefix_len")
    if seq_len < 1:
        raise ValueError(f"sparse_ir seq_len must be >= 1, got {seq_len}")
    if query_len < 1:
        raise ValueError(f"sparse_ir query_len must be >= 1, got {query_len}")
    if query_len > seq_len:
        raise ValueError(f"sparse_ir query_len must be <= seq_len ({query_len} > {seq_len})")
    if not (0 <= prefix_len <= seq_len):
        raise ValueError(f"sparse_ir prefix_len must be within [0, seq_len], got {prefix_len}")

    q_indptr = _require_int_list_field(sparse_ir, "q_indptr")
    span_starts = _require_int_list_field(sparse_ir, "span_starts")
    span_ends = _require_int_list_field(sparse_ir, "span_ends")
    if len(q_indptr) != query_len + 1:
        raise ValueError(f"sparse_ir q_indptr length {len(q_indptr)} != query_len + 1 ({query_len + 1})")
    if not q_indptr or q_indptr[0] != 0:
        raise ValueError("sparse_ir q_indptr must start at 0")
    if any(left > right for left, right in zip(q_indptr, q_indptr[1:])):
        raise ValueError("sparse_ir q_indptr must be nondecreasing")
    if len(span_starts) != len(span_ends):
        raise ValueError(f"sparse_ir span_starts length {len(span_starts)} != span_ends length {len(span_ends)}")
    if q_indptr[-1] != len(span_starts):
        raise ValueError(f"sparse_ir q_indptr[-1] {q_indptr[-1]} != number of spans {len(span_starts)}")

    for query_position in range(query_len):
        prev_end: int | None = None
        for span_idx in range(q_indptr[query_position], q_indptr[query_position + 1]):
            start = span_starts[span_idx]
            end = span_ends[span_idx]
            if not (0 <= start < end <= seq_len):
                raise ValueError(
                    "sparse_ir span bounds must satisfy 0 <= start < end <= seq_len, "
                    f"got query={query_position} span=({start}, {end}) seq_len={seq_len}"
                )
            if prev_end is not None and start <= prev_end:
                raise ValueError(
                    "sparse_ir spans must be sorted, non-overlapping, and merged when adjacent; "
                    f"got query={query_position} previous_end={prev_end} next_start={start}"
                )
            prev_end = end

    return seq_len, query_len, prefix_len, q_indptr, span_starts, span_ends


def _require_int_field(sparse_ir: dict[str, Any], name: str) -> int:
    value = sparse_ir.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"sparse_ir {name} must be an int, got {value!r}")
    return value


def _require_int_list_field(sparse_ir: dict[str, Any], name: str) -> list[int]:
    value = sparse_ir.get(name)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"sparse_ir {name} must be a list/tuple of ints, got {type(value).__name__}")
    normalized = list(value)
    for item in normalized:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"sparse_ir {name} must contain only ints, got {item!r}")
    return normalized


def build_g1_ebft_rollout_mask_contract(
    *,
    prompt_length: int,
    context_length: int,
    generate_length: int,
    stride: int,
    qa_values: list[Any] | tuple[Any, ...] | None = None,
    doc_ids: list[Any] | tuple[Any, ...] | None = None,
    document_masking: bool = False,
) -> G1EBFTRolloutMaskContract:
    """Build the strict EBFT rollout layout and mask from CPU geometry.

    ``response_positions`` are prompt-side strided positions in time-major
    order. ``target_positions`` are the generated rows in the packed
    ``prompt + response`` sequence. ``rollout_anchor_positions`` keep the
    prompt-side token anchors, while ``rollout_source_rows`` are sampling logit
    rows consumed by SGLang block_source. ``logprob_source_rows`` are separately
    checked against the trainer's strict pair-axis helper.
    """

    prompt_length, context_length, generate_length, stride = _validate_geometry(
        prompt_length=prompt_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
    )
    num_blocks = get_num_strided_blocks(
        prompt_length=prompt_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
    )
    response_length = generate_length * num_blocks
    full_sequence_length = prompt_length + response_length

    qa_tuple = _normalize_optional_metadata("qa_values", qa_values, prompt_length)
    doc_tuple = _normalize_optional_metadata("doc_ids", doc_ids, prompt_length)
    if document_masking and doc_tuple is None:
        raise ValueError("document_masking requires doc_ids")

    response_positions: list[int] = []
    target_positions: list[int] = []
    rollout_sources: list[G1EBFTRolloutSource] = []
    prompt_position_ids = _build_prompt_position_ids(
        prompt_length=prompt_length,
        doc_ids=doc_tuple,
        document_masking=document_masking,
    )
    position_ids = list(prompt_position_ids)

    pair_source_rows, pair_target_positions, pair_action_mask = build_ebft_g1_logprob_pair_axis(
        prompt_length=prompt_length,
        response_length=response_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
        indexing=G1_EBFT_LOGPROB_INDEXING_STRICT_BLOCK,
        device=torch.device("cpu"),
    )
    action_pair_source_rows = tuple(int(row) for row in pair_source_rows[pair_action_mask].tolist())
    action_pair_target_positions = tuple(int(pos) for pos in pair_target_positions[pair_action_mask].tolist())

    for step in range(generate_length):
        for block in range(num_blocks):
            response_idx = step * num_blocks + block
            response_position = context_length + block * stride + step
            target_position = prompt_length + response_idx
            position_id = prompt_position_ids[context_length + block * stride] + step
            if step == 0:
                source_row = context_length + block * stride - 1
                source_kind = PROMPT_SOURCE_KIND
            else:
                source_row = prompt_length + (step - 1) * num_blocks + block
                source_kind = GENERATED_SOURCE_KIND

            response_positions.append(response_position)
            target_positions.append(target_position)
            position_ids.append(position_id)
            rollout_sources.append(
                G1EBFTRolloutSource(
                    step=step,
                    block=block,
                    response_position=response_position,
                    source_row=source_row,
                    source_kind=source_kind,
                    target_position=target_position,
                    logprob_source_row=action_pair_source_rows[response_idx],
                )
            )

    target_positions_tuple = tuple(target_positions)
    if target_positions_tuple != action_pair_target_positions:
        raise ValueError(
            "Strict EBFT rollout target positions diverged from trainer pair-axis: "
            f"{target_positions_tuple} != {action_pair_target_positions}"
        )
    rollout_source_rows_tuple = tuple(source.source_row for source in rollout_sources)
    if rollout_source_rows_tuple != action_pair_source_rows:
        raise ValueError(
            "Strict EBFT rollout source rows diverged from trainer pair-axis: "
            f"{rollout_source_rows_tuple} != {action_pair_source_rows}"
        )

    response_positions_tuple = tuple(response_positions)
    allowed_edges = _build_allowed_edges(
        prompt_length=prompt_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
        num_blocks=num_blocks,
        full_sequence_length=full_sequence_length,
        doc_ids=doc_tuple,
        document_masking=document_masking,
    )

    return G1EBFTRolloutMaskContract(
        prompt_length=prompt_length,
        context_length=context_length,
        generate_length=generate_length,
        stride=stride,
        num_blocks=num_blocks,
        response_length=response_length,
        full_sequence_length=full_sequence_length,
        response_positions=response_positions_tuple,
        position_ids=tuple(position_ids),
        target_positions=target_positions_tuple,
        rollout_sources=tuple(rollout_sources),
        logprob_source_rows=action_pair_source_rows,
        logprob_target_positions=action_pair_target_positions,
        allowed_edges=allowed_edges,
        qa_values=qa_tuple,
        doc_ids=doc_tuple,
        response_qa_values=_gather_optional_metadata(qa_tuple, response_positions_tuple),
        response_doc_ids=_gather_optional_metadata(doc_tuple, response_positions_tuple),
        document_masking=document_masking,
    )


def _validate_geometry(
    *,
    prompt_length: int,
    context_length: int,
    generate_length: int,
    stride: int,
) -> tuple[int, int, int, int]:
    prompt_length = int(prompt_length)
    context_length = int(context_length)
    generate_length = int(generate_length)
    stride = int(stride)
    if prompt_length < 1:
        raise ValueError(f"prompt_length must be >= 1, got {prompt_length}")
    if context_length < 1:
        raise ValueError(f"context_length must be >= 1, got {context_length}")
    if generate_length < 1:
        raise ValueError(f"generate_length must be >= 1, got {generate_length}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    return prompt_length, context_length, generate_length, stride


def _normalize_optional_metadata(
    name: str,
    values: list[Any] | tuple[Any, ...] | None,
    prompt_length: int,
) -> tuple[Any, ...] | None:
    if values is None:
        return None
    values_tuple = tuple(values)
    if len(values_tuple) != prompt_length:
        raise ValueError(f"{name} length {len(values_tuple)} != prompt_length {prompt_length}")
    return values_tuple


def _gather_optional_metadata(values: tuple[Any, ...] | None, positions: tuple[int, ...]) -> tuple[Any, ...] | None:
    if values is None:
        return None
    return tuple(values[pos] for pos in positions)


def _build_prompt_position_ids(
    *,
    prompt_length: int,
    doc_ids: tuple[Any, ...] | None,
    document_masking: bool,
) -> tuple[int, ...]:
    if not document_masking:
        return tuple(range(prompt_length))

    positions: list[int] = []
    segment_start = 0
    for idx in range(prompt_length):
        if idx == 0 or doc_ids[idx] != doc_ids[idx - 1]:
            segment_start = idx
        positions.append(idx - segment_start)
    return tuple(positions)


def _build_allowed_edges(
    *,
    prompt_length: int,
    context_length: int,
    generate_length: int,
    stride: int,
    num_blocks: int,
    full_sequence_length: int,
    doc_ids: tuple[Any, ...] | None,
    document_masking: bool,
) -> tuple[G1EBFTAllowedEdge, ...]:
    edges: list[G1EBFTAllowedEdge] = []

    def same_doc(left: int, right: int) -> bool:
        return not document_masking or doc_ids[left] == doc_ids[right]

    for query_position in range(prompt_length):
        for key_position in range(query_position + 1):
            if same_doc(query_position, key_position):
                edges.append(
                    G1EBFTAllowedEdge(
                        query_position=query_position,
                        key_position=key_position,
                        query_kind=PROMPT_KEY_KIND,
                        key_kind=PROMPT_KEY_KIND,
                    )
                )

    for step in range(generate_length):
        for block in range(num_blocks):
            query_position = prompt_length + step * num_blocks + block
            response_position = context_length + block * stride + step
            context_window_end = min(block * stride + context_length, prompt_length - generate_length)

            for key_position in range(context_window_end):
                if same_doc(response_position, key_position):
                    edges.append(
                        G1EBFTAllowedEdge(
                            query_position=query_position,
                            key_position=key_position,
                            query_kind=GENERATED_KEY_KIND,
                            key_kind=PROMPT_KEY_KIND,
                        )
                    )

            edges.append(
                G1EBFTAllowedEdge(
                    query_position=query_position,
                    key_position=query_position,
                    query_kind=GENERATED_KEY_KIND,
                    key_kind=GENERATED_KEY_KIND,
                )
            )

            for prev_step in range(step):
                prev_response_position = context_length + block * stride + prev_step
                if same_doc(response_position, prev_response_position):
                    edges.append(
                        G1EBFTAllowedEdge(
                            query_position=query_position,
                            key_position=prompt_length + prev_step * num_blocks + block,
                            query_kind=GENERATED_KEY_KIND,
                            key_kind=GENERATED_KEY_KIND,
                        )
                    )

    _validate_allowed_edges(edges, full_sequence_length)
    return tuple(edges)


def _validate_allowed_edges(edges: list[G1EBFTAllowedEdge], full_sequence_length: int) -> None:
    seen: set[tuple[int, int]] = set()
    for edge in edges:
        if not (0 <= edge.query_position < full_sequence_length):
            raise ValueError(f"allowed edge query_position out of range: {edge}")
        if not (0 <= edge.key_position < full_sequence_length):
            raise ValueError(f"allowed edge key_position out of range: {edge}")
        pair = (edge.query_position, edge.key_position)
        if pair in seen:
            raise ValueError(f"duplicate allowed edge: {pair}")
        seen.add(pair)
