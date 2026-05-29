import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
# Keep this distinct from whichever checkout is running the test so the
# negative TRAIN_ENTRY assertion exercises the inherited-root guard.
OLD_SLIME_ROOT = REPO_ROOT.with_name(f"{REPO_ROOT.name}_old")
MAIN_LAUNCHER = REPO_ROOT / "exper_scripts/main_test/run_g1_ebft_gt_qwen35_2b_main.sh"
STRICT_LAUNCHER = REPO_ROOT / "exper_scripts/main_test/run_g1_ebft_gt_qwen35_2b_strict_block_source.sh"


@pytest.fixture()
def launcher_env(tmp_path: Path) -> dict[str, str]:
    assert str(tmp_path).startswith("/tmp/")

    model_path = tmp_path / "model"
    ref_load = tmp_path / "ref_load"
    megatron_path = tmp_path / "megatron"
    data_dir = tmp_path / "data"
    bin_dir = tmp_path / "bin"
    output_root = tmp_path / "outputs"
    ray_tmpdir = tmp_path / "ray"

    for path in (model_path, ref_load, megatron_path, data_dir, bin_dir, output_root, ray_tmpdir):
        path.mkdir(parents=True, exist_ok=True)

    train_data = data_dir / "train.jsonl"
    train_data.write_text('{"prompt": "p", "label": "l"}\n', encoding="utf-8")

    ray_bin = bin_dir / "ray"
    ray_bin.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    ray_bin.chmod(0o755)

    slime_env_file = tmp_path / "empty_slime_env.sh"
    slime_env_file.write_text("# test env intentionally empty\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "ACTOR_NUM_GPUS_PER_NODE": "1",
            "ARTIFACT_DIR": str(tmp_path / "artifacts"),
            "COMPLETION_MAX_LENGTH": "16",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "ENABLE_ASYNC_TRAIN": "true",
            "ENABLE_SLIME_EVAL": "false",
            "G1_FILTER_TRAIN_DATA": "false",
            "G1_RESPONSE_LENGTH": "16",
            "GLOBAL_BATCH_SIZE": "4",
            "LOAD_PATH": str(output_root / "load"),
            "MEGATRON_PATH": str(megatron_path),
            "MODEL_PATH": str(model_path),
            "N_SAMPLES_PER_PROMPT": "1",
            "NUM_ROLLOUT": "1",
            "OUTPUT_ROOT": str(output_root),
            "PRINT_ONLY": "1",
            "PROMPT_MAX_LENGTH": "16",
            "PYTHON_BIN": sys.executable,
            "RAY_BIN": str(ray_bin),
            "RAY_TMPDIR": str(ray_tmpdir),
            "REF_LOAD": str(ref_load),
            "ROLLOUT_BATCH_SIZE": "4",
            "ROLLOUT_MAX_CONTEXT_LEN": "64",
            "ROLLOUT_NUM_GPUS_PER_ENGINE": "1",
            "RUN_NAME": "launcher_contract",
            "SAVE_PATH": str(output_root / "save"),
            "SLIME_ENV_FILE": str(slime_env_file),
            "SLIME_TRAIN_DATA": str(train_data),
            "TENSOR_MODEL_PARALLEL_SIZE": "1",
        }
    )
    env.pop("G1_EBFT_LOGPROB_INDEXING", None)
    return env


def _run_launcher(script: Path, env: dict[str, str]) -> str:
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout + proc.stderr


def _run_launcher_failure(script: Path, env: dict[str, str]) -> str:
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    return proc.stdout + proc.stderr


def test_main_launcher_print_only_uses_launcher_default_logprob_indexing(launcher_env: dict[str, str]) -> None:
    output = _run_launcher(MAIN_LAUNCHER, launcher_env)

    assert "--g1-ebft-logprob-indexing" not in output
    assert "--g1-ebft-rollout-sampling-mode" not in output
    assert "--g1-ebft-rollout-mask-mode" not in output
    assert "--sglang-disable-overlap-schedule" not in output
    assert "--sglang-grammar-backend" not in output
    assert "[preflight] G1_EBFT_LOGPROB_INDEXING=launcher-default" in output
    assert (
        "[preflight] G1_EBFT_ROLLOUT_SAMPLING_MODE=standard "
        "G1_EBFT_ROLLOUT_MASK_MODE=none SGLANG_GRAMMAR_BACKEND=launcher-default"
    ) in output


def test_strict_wrapper_print_only_adds_strict_block_source_indexing(launcher_env: dict[str, str]) -> None:
    output = _run_launcher(STRICT_LAUNCHER, launcher_env)

    assert "--g1-ebft-logprob-indexing strict_block_source" in output
    assert "--g1-ebft-rollout-sampling-mode" not in output
    assert "--g1-ebft-rollout-mask-mode" not in output
    assert "--sglang-grammar-backend none" in output
    assert "[preflight] G1_EBFT_LOGPROB_INDEXING=strict_block_source" in output
    assert (
        "[preflight] G1_EBFT_ROLLOUT_SAMPLING_MODE=standard "
        "G1_EBFT_ROLLOUT_MASK_MODE=none SGLANG_GRAMMAR_BACKEND=none"
    ) in output


def test_strict_wrapper_print_only_rejects_sparse_rollout_mask_without_block_source_sampling(
    launcher_env: dict[str, str],
) -> None:
    env = launcher_env.copy()
    env["G1_EBFT_ROLLOUT_MASK_MODE"] = "sparse_ir"
    env["SGLANG_ATTENTION_BACKEND"] = "triton"
    env["SGLANG_GRAMMAR_BACKEND"] = "xgrammar"

    output = _run_launcher_failure(STRICT_LAUNCHER, env)

    assert "transport only" in output


def test_strict_wrapper_print_only_rejects_block_source_sampling_without_dense4d(
    launcher_env: dict[str, str],
) -> None:
    env = launcher_env.copy()
    env["G1_EBFT_ROLLOUT_SAMPLING_MODE"] = "block_source"

    output = _run_launcher_failure(STRICT_LAUNCHER, env)

    assert "requires G1_EBFT_ROLLOUT_MASK_MODE=dense4d" in output


def test_main_launcher_print_only_adds_experimental_block_source_flags(
    launcher_env: dict[str, str],
) -> None:
    env = launcher_env.copy()
    env.update(
        {
            "G1_EBFT_LOGPROB_INDEXING": "strict_block_source",
            "G1_EBFT_ROLLOUT_MASK_MODE": "dense4d",
            "G1_EBFT_ROLLOUT_SAMPLING_MODE": "block_source",
            "SGLANG_ATTENTION_BACKEND": "torch_native",
            "SGLANG_DISABLE_OVERLAP_SCHEDULE": "true",
        }
    )

    output = _run_launcher(MAIN_LAUNCHER, env)

    assert "--g1-ebft-logprob-indexing strict_block_source" in output
    assert "--g1-generate-length 8" in output
    assert "--g1-ebft-rollout-sampling-mode block_source" in output
    assert "--g1-ebft-rollout-mask-mode dense4d" in output
    assert "--sglang-attention-backend torch_native" in output
    assert "--sglang-disable-overlap-schedule" in output
    assert (
        "[preflight] SGLANG_ATTENTION_BACKEND=torch_native "
        "SGLANG_DISABLE_OVERLAP_SCHEDULE=true"
    ) in output


def test_launcher_ignores_inherited_old_slime_root_for_train_entry(launcher_env: dict[str, str]) -> None:
    assert OLD_SLIME_ROOT != REPO_ROOT

    env = launcher_env.copy()
    env["SLIME_ROOT"] = str(OLD_SLIME_ROOT)

    output = _run_launcher(MAIN_LAUNCHER, env)

    assert f"[preflight] SLIME_ROOT={REPO_ROOT}" in output
    assert f"[preflight] TRAIN_ENTRY={REPO_ROOT}/train_async.py" in output
    assert "TRAIN_ENTRYPOINT=train_async.py" in output
    assert str(OLD_SLIME_ROOT / "train_async.py") not in output
