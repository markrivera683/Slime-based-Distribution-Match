import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TWO_NODE_LAUNCHER = REPO_ROOT / "exper_scripts/main_test/run_g2_opd_cf_l1oo_qwen35_2b_2node.sh"
LAUNCHERS = [
    pytest.param(
        REPO_ROOT / "exper_scripts/main_test/run_g2_opd_cf_l1oo_qwen35_2b_1node8.sh",
        id="1node8",
    ),
    pytest.param(
        TWO_NODE_LAUNCHER,
        id="2node",
    ),
]


@pytest.fixture()
def launcher_env(tmp_path: Path) -> dict[str, str]:
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
            "COLOCATE": "false",
            "COMPLETION_MAX_LENGTH": "16",
            "CRITIC_NUM_GPUS_PER_NODE": "1",
            "CUDA_VISIBLE_DEVICES": "0,1,2",
            "DEPLOY_MODE": "manual",
            "ENABLE_ASYNC_TRAIN": "false",
            "ENABLE_G2_POST_EVAL": "false",
            "ENABLE_SLIME_EVAL": "false",
            "EXPECTED_GPUS_PER_NODE": "0",
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
            "ROLLOUT_NUM_GPUS": "1",
            "ROLLOUT_NUM_GPUS_PER_ENGINE": "1",
            "RUN_NAME": "g2_opd_blockwise_launcher_contract",
            "SAVE_INTERVAL": "none",
            "SAVE_PATH": str(output_root / "save"),
            "SGLANG_503_DEBUG_MODE": "false",
            "SLIME_ENV_FILE": str(slime_env_file),
            "SLIME_ROOT": str(REPO_ROOT),
            "SLIME_TRAIN_DATA": str(train_data),
            "TEACHER_HOST": "127.0.0.1",
            "TENSOR_MODEL_PARALLEL_SIZE": "1",
            "TRAIN_ENTRYPOINT": "train.py",
        }
    )
    for key in (
        "G1_CONTEXT_LENGTH",
        "G1_EBFT_LOGPROB_INDEXING",
        "G1_EBFT_ROLLOUT_MASK_MODE",
        "G1_EBFT_ROLLOUT_SAMPLING_MODE",
        "G1_GENERATE_LENGTH",
        "G1_PROMPT_LENGTH",
        "G1_STRIDE",
        "G1_USE_EBFT_LOSS",
        "SGLANG_ATTENTION_BACKEND",
        "SGLANG_DISABLE_OVERLAP_SCHEDULE",
        "SGLANG_GRAMMAR_BACKEND",
        "TEACHER_EXTRA_SGLANG_ARGS",
    ):
        env.pop(key, None)
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


@pytest.mark.parametrize("script", LAUNCHERS)
def test_g2_opd_cf_l1oo_launchers_print_blockwise_flags(script: Path, launcher_env: dict[str, str]) -> None:
    output = _run_launcher(script, launcher_env)

    assert "--use-opd" in output
    assert "--distribution-reward-type cf_l1oo" in output
    assert "--opd-credit-assignment cf_l1oo" in output
    assert "--cf-target-mode opd_onpolicy" in output
    assert output.count("--g1-use-ebft-loss") == 1
    assert output.count("--g1-ebft-logprob-indexing strict_block_source") == 1
    assert output.count("--g1-ebft-rollout-sampling-mode block_source") == 1
    assert output.count("--g1-ebft-rollout-mask-mode sparse_ir") == 1
    assert output.count("--sglang-attention-backend triton") == 1
    assert output.count("--sglang-disable-overlap-schedule") == 1
    assert output.count("--sglang-grammar-backend none") == 1
    assert (
        "[preflight] SGLANG_ATTENTION_BACKEND=triton "
        "SGLANG_DISABLE_OVERLAP_SCHEDULE=true SGLANG_GRAMMAR_BACKEND=none"
    ) in output
    assert (
        "[preflight] G1_EBFT_LOGPROB_INDEXING=strict_block_source "
        "G1_EBFT_ROLLOUT_SAMPLING_MODE=block_source G1_EBFT_ROLLOUT_MASK_MODE=sparse_ir"
    ) in output
    if script.name.endswith("_1node8.sh"):
        assert "[layout] teacher CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 EXPECTED_TEACHER_GPUS=4" in output
        assert "[layout] student/Ray CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 EXPECTED_STUDENT_GPUS=4" in output
        assert "[layout] Ray/NUM_GPUS=4" in output
        assert "[preflight] RAY_NUM_GPUS=4 NUM_GPUS=4" in output
        assert "[submit] student CUDA_VISIBLE_DEVICES=4,5,6,7" in output

    run_context = Path(launcher_env["ARTIFACT_DIR"]) / "run_context.env"
    context = run_context.read_text(encoding="utf-8")
    context_lines = set(context.splitlines())
    assert "G1_EBFT_LOGPROB_INDEXING=strict_block_source" in context
    assert "G1_EBFT_ROLLOUT_SAMPLING_MODE=block_source" in context
    assert "G1_EBFT_ROLLOUT_MASK_MODE=sparse_ir" in context
    assert "SGLANG_ATTENTION_BACKEND=triton" in context
    assert "SGLANG_DISABLE_OVERLAP_SCHEDULE=true" in context
    assert "SGLANG_GRAMMAR_BACKEND=none" in context
    if script.name.endswith("_1node8.sh"):
        assert "TEACHER_CUDA_VISIBLE_DEVICES=0,1,2,3" in context_lines
        assert "TEACHER_NUM_GPUS=4" in context_lines
        assert "EXPECTED_TEACHER_GPUS=4" in context_lines
        assert "STUDENT_CUDA_VISIBLE_DEVICES=4,5,6,7" in context_lines
        assert "STUDENT_NUM_GPUS=4" in context_lines
        assert "EXPECTED_STUDENT_GPUS=4" in context_lines
        assert "CUDA_VISIBLE_DEVICES=4,5,6,7" in context_lines
        assert "NUM_GPUS=4" in context_lines
        assert "RAY_NUM_GPUS=4" in context_lines


def test_g2_2node_teacher_print_only_uses_safe_default_sglang_args(launcher_env: dict[str, str]) -> None:
    launcher_env.update(
        {
            "DEPLOY_ROLE": "teacher",
            "TEACHER_CONTEXT_LENGTH": "128",
            "TEACHER_CUDA_VISIBLE_DEVICES": "0,1,2",
            "TEACHER_DP_SIZE": "1",
            "TEACHER_MEM_FRACTION_STATIC": "0.1",
            "TEACHER_MODEL_PATH": launcher_env["MODEL_PATH"],
            "TEACHER_TP_SIZE": "1",
        }
    )

    output = _run_launcher(TWO_NODE_LAUNCHER, launcher_env)

    assert "[2node-opd-cf-l1oo-teacher] command:" in output
    assert "--disable-cuda-graph" in output
    assert "--attention-backend triton" in output
    assert "--sampling-backend pytorch" in output


def test_g2_2node_launcher_restores_repo_root_after_sourced_env(
    tmp_path: Path, launcher_env: dict[str, str]
) -> None:
    wrong_root = tmp_path / "wrong_slime_root"
    wrong_root.mkdir()
    wrong_build_script = wrong_root / "build_conda.sh"
    poisoned_slime_env = tmp_path / "poisoned_slime_env.sh"
    poisoned_slime_env.write_text(
        "\n".join(
            [
                f"export SLIME_ROOT={shlex.quote(str(wrong_root))}",
                f"export BUILD_SCRIPT={shlex.quote(str(wrong_build_script))}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    launcher_env.update(
        {
            "BUILD_SCRIPT": str(wrong_build_script),
            "SLIME_ENV_FILE": str(poisoned_slime_env),
            "SLIME_ROOT": str(wrong_root),
        }
    )

    output = _run_launcher(TWO_NODE_LAUNCHER, launcher_env)

    assert "[dlc] overriding SLIME_ROOT from sourced env:" in output
    assert f"-> {REPO_ROOT}" in output
    assert f"{REPO_ROOT}/train.py" in output
    assert f"{wrong_root}/train.py" not in output

    run_context = Path(launcher_env["ARTIFACT_DIR"]) / "run_context.env"
    context_lines = set(run_context.read_text(encoding="utf-8").splitlines())
    assert f"SLIME_ROOT={REPO_ROOT}" in context_lines
    assert "TRAIN_ENTRYPOINT=train.py" in context_lines
