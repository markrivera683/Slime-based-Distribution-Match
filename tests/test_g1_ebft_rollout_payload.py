import asyncio
import importlib.util
import sys
import types
from argparse import Namespace

import pytest


def _has_real_or_loaded_module(name):
    if name in sys.modules:
        return True
    return importlib.util.find_spec(name) is not None


def _install_stub_if_missing(name, module):
    if not _has_real_or_loaded_module(name):
        sys.modules[name] = module


_install_stub_if_missing("sglang_router", types.ModuleType("sglang_router"))

pybase64 = types.ModuleType("pybase64")
pybase64.b64decode = lambda value: value
_install_stub_if_missing("pybase64", pybase64)

tqdm_module = types.ModuleType("tqdm")
tqdm_module.tqdm = lambda iterable=None, *args, **kwargs: iterable if iterable is not None else []
_install_stub_if_missing("tqdm", tqdm_module)

if not _has_real_or_loaded_module("ray"):
    ray_module = types.ModuleType("ray")
    ray_module._private = types.SimpleNamespace(
        services=types.SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")
    )
    sys.modules["ray"] = ray_module

transformers_module = types.ModuleType("transformers")
transformers_module.AutoTokenizer = type(
    "AutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: object())}
)
transformers_module.AutoProcessor = type(
    "AutoProcessor",
    (),
    {"from_pretrained": staticmethod(lambda *args, **kwargs: None)},
)
transformers_module.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
transformers_module.ProcessorMixin = type("ProcessorMixin", (), {})
_install_stub_if_missing("transformers", transformers_module)

httpx_module = types.ModuleType("httpx")
httpx_module.AsyncClient = type("AsyncClient", (), {"__init__": lambda self, *args, **kwargs: None})
httpx_module.Limits = type("Limits", (), {"__init__": lambda self, *args, **kwargs: None})
httpx_module.Timeout = type("Timeout", (), {"__init__": lambda self, *args, **kwargs: None})
httpx_module.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
_install_stub_if_missing("httpx", httpx_module)

aiohttp_module = types.ModuleType("aiohttp")
aiohttp_module.ClientSession = type(
    "ClientSession", (), {"__init__": lambda self, *args, **kwargs: None, "closed": False}
)
aiohttp_module.TCPConnector = type("TCPConnector", (), {"__init__": lambda self, *args, **kwargs: None})
aiohttp_module.ClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, *args, **kwargs: None})
_install_stub_if_missing("aiohttp", aiohttp_module)

rm_hub_module = types.ModuleType("slime.rollout.rm_hub")


async def _unused_async_rm(*args, **kwargs):
    return 0.0


async def _unused_batched_async_rm(*args, **kwargs):
    return []


rm_hub_module.async_rm = _unused_async_rm
rm_hub_module.batched_async_rm = _unused_batched_async_rm
_install_stub_if_missing("slime.rollout.rm_hub", rm_hub_module)

from slime.rollout import sglang_rollout
from slime.utils.types import Sample


def _args(**overrides):
    values = dict(
        ci_test=False,
        g1_context_length=2,
        g1_document_masking=False,
        g1_ebft_logprob_indexing="strict_block_source",
        g1_ebft_rollout_mask_mode="none",
        g1_ebft_rollout_sampling_mode="standard",
        g1_generate_length=2,
        g1_prompt_length=6,
        g1_response_length=4,
        g1_stride=2,
        g1_use_ebft_loss=True,
        partial_rollout=False,
        router_policy=None,
        sglang_attention_backend="torch_native",
        sglang_disable_overlap_schedule=True,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        sglang_speculative_algorithm=None,
        use_rollout_routing_replay=False,
    )
    values.update(overrides)
    return Namespace(**values)


def _sample(**overrides):
    values = dict(
        prompt="abcdef",
        metadata={
            "g1_qa_values": [0, 0, 1, 1, 0, 0],
            "g1_doc_ids": [0, 0, 0, 0, 0, 0],
        },
    )
    values.update(overrides)
    return Sample(**values)


def _build_fields(args, sample=None):
    return sglang_rollout._build_g1_ebft_rollout_mask_payload_fields(
        args,
        sample or _sample(),
        input_ids=[1, 2, 3, 4, 5, 6],
        max_new_tokens=int(args.g1_response_length),
    )


def test_default_rollout_payload_has_no_ebft_fields():
    fields = _build_fields(_args(g1_ebft_rollout_mask_mode="none"))

    assert fields == {}


def test_strict_dense_rollout_payload_has_spec_position_ids_and_dense_mask():
    fields = _build_fields(
        _args(g1_ebft_rollout_mask_mode="dense4d", g1_ebft_rollout_sampling_mode="block_source")
    )

    assert set(fields) == {
        "ebft_rollout_sampling_mode",
        "ebft_mask_spec",
        "ebft_position_ids",
        "ebft_dense_attention_mask",
    }
    assert fields["ebft_rollout_sampling_mode"] == "block_source"
    assert fields["ebft_mask_spec"]["mode"] == "dense4d"
    assert fields["ebft_mask_spec"]["layout"] == "g1_strict_block_source"
    assert fields["ebft_mask_spec"]["rollout_anchor_positions"] == [2, 4, 3, 5]
    assert fields["ebft_mask_spec"]["rollout_source_rows"] == [1, 3, 6, 7]
    assert fields["ebft_mask_spec"]["logprob_source_rows"] == [1, 3, 6, 7]
    assert fields["ebft_position_ids"] == [0, 1, 2, 3, 4, 5, 2, 4, 3, 5]

    dense = fields["ebft_dense_attention_mask"]
    assert len(dense) == 1
    assert len(dense[0]) == 1
    assert len(dense[0][0]) == 10
    assert dense[0][0][6][0] == 0.0
    assert dense[0][0][6][2] < -1e20


def test_strict_sparse_rollout_payload_has_spec_position_ids_and_sparse_ir():
    fields = _build_fields(
        _args(
            g1_ebft_rollout_mask_mode="sparse_ir",
            g1_ebft_rollout_sampling_mode="block_source",
            sglang_attention_backend="triton",
        )
    )

    assert set(fields) == {
        "ebft_rollout_sampling_mode",
        "ebft_mask_spec",
        "ebft_position_ids",
        "ebft_sparse_ir",
    }
    assert fields["ebft_rollout_sampling_mode"] == "block_source"
    assert fields["ebft_mask_spec"]["mode"] == "sparse_ir"
    assert fields["ebft_mask_spec"]["rollout_source_rows"] == [1, 3, 6, 7]
    assert fields["ebft_sparse_ir"]["layout"] == "ebft_block_strided_v1"
    assert fields["ebft_sparse_ir"]["seq_len"] == 10
    assert fields["ebft_sparse_ir"]["query_len"] == 10
    assert fields["ebft_sparse_ir"]["prefix_len"] == 6
    assert "ebft_dense_attention_mask" not in fields


def test_rollout_payload_builder_accepts_block_source_generate_length_gt_one():
    args = _args(
        g1_ebft_rollout_mask_mode="dense4d",
        g1_ebft_rollout_sampling_mode="block_source",
        g1_generate_length=2,
        g1_response_length=4,
        g1_stride=2,
    )

    fields = _build_fields(args)

    assert fields["ebft_mask_spec"]["generate_length"] == 2
    assert fields["ebft_mask_spec"]["rollout_source_rows"] == [1, 3, 6, 7]


def test_rollout_payload_builder_rejects_missing_prompt_metadata():
    sample = _sample(metadata={"g1_qa_values": [0, 0, 1, 1, 0, 0]})

    with pytest.raises(ValueError, match="requires prompt metadata 'doc_ids'"):
        _build_fields(
            _args(g1_ebft_rollout_mask_mode="dense4d", g1_ebft_rollout_sampling_mode="block_source"),
            sample,
        )


def test_rollout_payload_builder_rejects_mask_transport_without_block_source_sampling():
    args = _args(g1_ebft_rollout_mask_mode="dense4d")

    with pytest.raises(ValueError, match="mask transport alone is not strict EBFT rollout"):
        _build_fields(args)


def test_rollout_payload_builder_rejects_invalid_strict_args():
    args = _args(
        g1_ebft_rollout_mask_mode="dense4d",
        g1_ebft_rollout_sampling_mode="block_source",
        g1_ebft_logprob_indexing="standard_next_token",
    )

    with pytest.raises(ValueError, match="requires --g1-ebft-logprob-indexing strict_block_source"):
        _build_fields(args)


def test_rollout_payload_builder_rejects_block_source_without_torch_native():
    args = _args(
        g1_ebft_rollout_mask_mode="dense4d",
        g1_ebft_rollout_sampling_mode="block_source",
        sglang_attention_backend="triton",
    )

    with pytest.raises(ValueError, match="dense4d requires --sglang-attention-backend torch_native"):
        _build_fields(args)


def test_rollout_payload_builder_rejects_sparse_block_source_without_triton():
    args = _args(
        g1_ebft_rollout_mask_mode="sparse_ir",
        g1_ebft_rollout_sampling_mode="block_source",
        sglang_attention_backend="torch_native",
    )

    with pytest.raises(ValueError, match="sparse_ir requires --sglang-attention-backend triton"):
        _build_fields(args)


def test_rollout_payload_builder_rejects_block_source_with_overlap_schedule():
    args = _args(
        g1_ebft_rollout_mask_mode="dense4d",
        g1_ebft_rollout_sampling_mode="block_source",
        sglang_disable_overlap_schedule=False,
    )

    with pytest.raises(ValueError, match="requires --sglang-disable-overlap-schedule"):
        _build_fields(args)


def test_rollout_payload_builder_rejects_non_matching_input_geometry():
    with pytest.raises(ValueError, match="input_ids length must match --g1-prompt-length"):
        sglang_rollout._build_g1_ebft_rollout_mask_payload_fields(
            _args(g1_ebft_rollout_mask_mode="dense4d", g1_ebft_rollout_sampling_mode="block_source"),
            _sample(),
            input_ids=[1, 2, 3],
            max_new_tokens=4,
        )


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3, 4, 5, 6]


class _FakeGenerateState:
    def __init__(self, args):
        self.args = args
        self.processor = None


def test_generate_sends_single_request_with_dense_ebft_payload(monkeypatch):
    calls = []

    async def fake_post(url, payload, headers=None):
        calls.append((url, payload, headers))
        return {
            "text": "xy",
            "meta_info": {
                "finish_reason": {"type": "length"},
                "output_token_logprobs": [(-0.1, 91), (-0.2, 92), (-0.3, 93), (-0.4, 94)],
            },
        }

    monkeypatch.setattr(sglang_rollout, "GenerateState", _FakeGenerateState)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    _FakeGenerateState.tokenizer = _FakeTokenizer()

    sample = asyncio.run(
        sglang_rollout.generate(
            _args(g1_ebft_rollout_mask_mode="dense4d", g1_ebft_rollout_sampling_mode="block_source"),
            _sample(),
            sampling_params={
                "max_new_tokens": 4,
                "stop": ["</s>"],
                "stop_token_ids": [2],
            },
        )
    )

    assert len(calls) == 1
    assert calls[0][1]["ebft_rollout_sampling_mode"] == "block_source"
    assert "ebft_mask_spec" in calls[0][1]
    assert "ebft_dense_attention_mask" in calls[0][1]
    assert calls[0][1]["input_ids"] == [1, 2, 3, 4, 5, 6]
    assert calls[0][1]["sampling_params"]["ignore_eos"] is True
    assert calls[0][1]["sampling_params"]["stop"] == []
    assert calls[0][1]["sampling_params"]["stop_token_ids"] == []
    assert sample.tokens == [1, 2, 3, 4, 5, 6, 91, 92, 93, 94]
