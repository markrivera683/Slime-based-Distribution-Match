import json
import sys
import types
from argparse import Namespace
from threading import Lock

import pytest
import torch

from slime.rollout.g2_teacher import (
    G2RemoteTeacherClient,
    G2RemoteTeacherConfig,
    attach_g2_teacher_completions,
)
from slime.utils.types import Sample


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _standard_g2_args(**overrides):
    args = dict(
        distribution_reward_type="cf_l1oo",
        cf_target_mode="teacher",
        advantage_estimator="g1",
        g1_embedding_source="megatron_ref",
        g1_reward_location="trainer",
        teacher_backend="remote",
        teacher_api_base="http://teacher:30000",
        teacher_model_name="teacher",
        cf_teacher_n_samples=2,
        cf_teacher_lambda=0.6,
        use_whitening=True,
        use_opd=False,
        opd_type=None,
        g1_use_ebft_loss=False,
        critic_lr=0.0,
        critic_lr_head=0.0,
        zero_stage=3,
    )
    args.update(overrides)
    return Namespace(**args)


def _opd_post_process_rewards(args, samples: list[Sample]):
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]
    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs
    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards


def _install_arguments_import_stubs(monkeypatch):
    yaml = types.ModuleType("yaml")
    yaml.safe_load = lambda stream: {}
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


def test_standard_g2_allows_sglang_opd_and_rejects_megatron_opd(monkeypatch):
    _install_arguments_import_stubs(monkeypatch)
    from slime.utils.arguments import assert_g2_standard_args

    assert_g2_standard_args(_standard_g2_args(use_opd=True, opd_type="sglang"))

    with pytest.raises(ValueError, match="SGLang OPD"):
        assert_g2_standard_args(_standard_g2_args(use_opd=True, opd_type="megatron"))


def test_g2_sglang_generate_requests_m_texts_in_one_call_and_records_stats(monkeypatch):
    requests = []
    lock = Lock()

    def _fake_urlopen(request, timeout):
        del timeout
        with lock:
            requests.append(request)
        payload = json.loads(request.data.decode("utf-8"))
        n_samples = int(payload["sampling_params"]["n"])
        return _Response({"text": [f"completion-{idx}" for idx in range(1, n_samples + 1)]})

    monkeypatch.setattr("slime.rollout.g2_teacher.urllib.request.urlopen", _fake_urlopen)
    client = G2RemoteTeacherClient(
        G2RemoteTeacherConfig(
            api_bases=("http://teacher:30000",),
            model_name="unused-for-sglang",
            api_style="sglang_generate",
            remote_batch_size=3,
            temperature=0.3,
            top_p=0.8,
            max_new_tokens=7,
        )
    )

    completions = client.sample_targets(["prompt"], 3)

    assert len(completions) == 1
    assert sorted(completions[0]) == ["completion-1", "completion-2", "completion-3"]
    assert len(requests) == 1
    assert len(client.last_stats) == 1
    stats = client.last_stats[0]
    assert stats["cache_hit"] is False
    assert stats["api_style"] == "sglang_generate"
    assert stats["api_base"] == "http://teacher:30000"
    assert stats["url"] == "http://teacher:30000/generate"
    assert stats["num_requests"] == 1
    assert stats["num_completions"] == 3
    assert stats["failed_attempts"] == 0
    assert stats["retries"] == 0
    assert stats["remote_batch_size"] == 3
    assert stats["sglang_multi_sample"] is True
    assert stats["latency_sec"] >= 0.0

    request = requests[0]
    assert request.full_url == "http://teacher:30000/generate"
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["text"] == "prompt"
    assert payload["sampling_params"]["max_new_tokens"] == 7
    assert payload["sampling_params"]["n"] == 3


def test_g2_sglang_generate_falls_back_to_concurrent_single_sample(monkeypatch):
    requests = []

    def _fake_urlopen(request, timeout):
        del timeout
        requests.append(request)
        payload = json.loads(request.data.decode("utf-8"))
        if payload["sampling_params"].get("n", 1) > 1:
            return _Response({"text": "single-completion"})
        return _Response({"text": f"fallback-{len(requests) - 1}"})

    monkeypatch.setattr("slime.rollout.g2_teacher.urllib.request.urlopen", _fake_urlopen)
    client = G2RemoteTeacherClient(
        G2RemoteTeacherConfig(
            api_bases=("http://teacher:30000",),
            model_name="unused-for-sglang",
            api_style="sglang_generate",
            remote_batch_size=3,
        )
    )

    completions = client.sample_targets(["prompt"], 3)

    assert completions == [["fallback-1", "fallback-2", "fallback-3"]]
    assert len(requests) == 4
    assert json.loads(requests[0].data.decode("utf-8"))["sampling_params"]["n"] == 3
    for request in requests[1:]:
        assert "n" not in json.loads(request.data.decode("utf-8"))["sampling_params"]
    assert client.last_stats[0]["num_requests"] == 4


def test_g2_attach_teacher_completions_writes_stats_metadata(monkeypatch):
    expected_stats = {
        "cache_hit": False,
        "latency_sec": 0.01,
        "num_completions": 2,
        "api_style": "sglang_generate",
        "api_base": "http://teacher:30000",
        "url": "http://teacher:30000/generate",
        "num_requests": 2,
        "failed_attempts": 0,
        "retries": 0,
        "remote_batch_size": 2,
        "sglang_multi_sample": True,
    }

    class _Client:
        last_stats = [expected_stats]

        def sample_targets(self, prompts, n_samples):
            assert prompts == ["ab"]
            assert n_samples == 2
            return [["xy", "yx"]]

    monkeypatch.setattr("slime.rollout.g2_teacher._get_client", lambda config: _Client())
    args = Namespace(
        distribution_reward_type="cf_l1oo",
        cf_target_mode="teacher",
        teacher_backend="remote",
        teacher_api_base="http://teacher:30000",
        teacher_model_name="teacher",
        teacher_api_key="EMPTY",
        teacher_api_style="sglang_generate",
        teacher_timeout=1,
        teacher_max_retries=1,
        teacher_remote_batch_size=2,
        teacher_temperature=0.7,
        teacher_top_p=0.95,
        teacher_max_new_tokens=16,
        teacher_system_prompt_text="",
        teacher_system_prompt_id="",
        teacher_cache_enable=False,
        cf_teacher_n_samples=2,
    )
    group = [Sample(index=0, prompt="ab"), Sample(index=1, prompt="ab")]

    attach_g2_teacher_completions(args, group)

    assert group[0].metadata["g2_teacher_stats"] == expected_stats
    assert group[1].metadata["g2_teacher_stats"] == expected_stats
    assert group[0].metadata["g2_teacher_stats"] is not expected_stats


def test_g2_attach_and_opd_postprocess_fields_coexist(monkeypatch):
    class _Client:
        last_stats = [{"num_requests": 1}]

        def sample_targets(self, prompts, n_samples):
            assert prompts == ["ab"]
            assert n_samples == 2
            return [["xy", "yx"]]

    monkeypatch.setattr("slime.rollout.g2_teacher._get_client", lambda config: _Client())
    args = Namespace(
        reward_key=None,
        distribution_reward_type="cf_l1oo",
        cf_target_mode="teacher",
        teacher_backend="remote",
        teacher_api_base="http://teacher:30000",
        teacher_model_name="teacher",
        teacher_api_key="EMPTY",
        teacher_api_style="sglang_generate",
        teacher_timeout=1,
        teacher_max_retries=1,
        teacher_remote_batch_size=2,
        teacher_temperature=0.7,
        teacher_top_p=0.95,
        teacher_max_new_tokens=16,
        teacher_system_prompt_text="",
        teacher_system_prompt_id="",
        teacher_cache_enable=False,
        cf_teacher_n_samples=2,
    )
    teacher_reward = {
        "meta_info": {
            "input_token_logprobs": [
                (0.0, 1),
                (-0.1, 2),
                (-0.2, 5),
                (-0.3, 6),
                (-0.4, 5),
                (-0.5, 6),
            ]
        }
    }
    group = [
        Sample(
            index=0,
            prompt="ab",
            label="cd",
            tokens=[1, 2, 5, 6, 5, 6],
            response_length=4,
            reward=teacher_reward,
        ),
        Sample(
            index=1,
            prompt="ab",
            label="cd",
            tokens=[1, 2, 6, 5, 6, 5],
            response_length=4,
            reward=teacher_reward,
        ),
    ]

    attach_g2_teacher_completions(args, group)
    _opd_post_process_rewards(args, group)

    assert group[0].metadata["g2_teacher_completions"] == ["xy", "yx"]
    assert group[1].metadata["g2_teacher_completions"] == ["xy", "yx"]
    torch.testing.assert_close(group[0].teacher_log_probs, torch.tensor([-0.2, -0.3, -0.4, -0.5]))
    torch.testing.assert_close(group[1].teacher_log_probs, torch.tensor([-0.2, -0.3, -0.4, -0.5]))
