from __future__ import annotations

import torch


def log_probs_to_sequence_scores(
    log_probs: list[torch.Tensor],
    *,
    device: torch.device,
    normalization: str,
    label: str = "log_probs",
) -> torch.Tensor:
    """Aggregate per-token logprobs into one score per sample."""

    scores = []
    for sample_idx, sample_log_probs in enumerate(log_probs):
        tensor = sample_log_probs.detach().float().reshape(-1).to(device=device)
        if tensor.numel() == 0:
            raise ValueError(f"{label}[{sample_idx}] is empty.")
        if normalization == "mean":
            score = tensor.mean()
        elif normalization == "sum":
            score = tensor.sum()
        else:
            raise ValueError(
                "Unsupported --opd-cf-score-normalization "
                f"{normalization!r}; expected 'mean' or 'sum'."
            )
        scores.append(score)

    return torch.stack(scores, dim=0)


def log_probs_to_group_scores(
    log_probs: list[torch.Tensor],
    *,
    num_groups: int,
    n_samples_per_prompt: int,
    device: torch.device,
    normalization: str,
    label: str = "log_probs",
) -> torch.Tensor:
    expected = int(num_groups) * int(n_samples_per_prompt)
    if len(log_probs) != expected:
        raise ValueError(
            f"{label} must align with grouped rollout samples; got {len(log_probs)} for {expected} samples."
        )
    return log_probs_to_sequence_scores(
        log_probs,
        device=device,
        normalization=normalization,
        label=label,
    ).view(1, int(num_groups), int(n_samples_per_prompt))


def _get_fixed_cf_frequencies(
    *,
    input_dim: int,
    num_freqs: int,
    sigma: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    freqs = torch.randn(int(num_freqs), int(input_dim), generator=generator, dtype=torch.float32)
    freqs = freqs / float(sigma)
    return freqs.to(device=device)


def _compute_cf_loss_terms(
    target_real: torch.Tensor,
    target_imag: torch.Tensor,
    gen_real: torch.Tensor,
    gen_imag: torch.Tensor,
    cf_alpha: float,
    cf_beta: float,
) -> torch.Tensor:
    target_norm = torch.sqrt(target_real * target_real + target_imag * target_imag)
    gen_norm = torch.sqrt(gen_real * gen_real + gen_imag * gen_imag)

    amp_diff = target_norm - gen_norm
    loss_amp = amp_diff * amp_diff
    loss_pha = 2 * (target_norm * gen_norm - gen_real * target_real - gen_imag * target_imag)
    loss_pha = loss_pha.clamp(min=1e-12)
    return torch.sqrt(float(cf_alpha) * loss_amp + float(cf_beta) * loss_pha)


def _build_cf_target_embedding(
    gt_embedding: torch.Tensor,
    *,
    cf_target_mode: str,
    cf_target_num_refs: int,
    cf_target_std: float,
    cf_target_seed: int,
    teacher_embedding: torch.Tensor | None,
    cf_teacher_lambda: float,
) -> torch.Tensor:
    target_embedding = gt_embedding[:, :, :1, :, :].float()
    if cf_target_mode == "single":
        return target_embedding
    if cf_target_mode != "teacher":
        raise ValueError(f"Unsupported G2 cf_target_mode: {cf_target_mode}")
    if teacher_embedding is None:
        raise ValueError("G2 teacher target mode requires teacher_embedding")
    if teacher_embedding.ndim != 5:
        raise ValueError(
            "G2 teacher target expects teacher_embedding with shape (B, G, M, K, D), "
            f"got {tuple(teacher_embedding.shape)}"
        )
    if gt_embedding.shape[:2] != teacher_embedding.shape[:2] or gt_embedding.shape[3:] != teacher_embedding.shape[3:]:
        raise ValueError(
            "G2 teacher_embedding must align with gt_embedding on batch/group/block/feature dims, "
            f"got gt={tuple(gt_embedding.shape)} teacher={tuple(teacher_embedding.shape)}"
        )

    lam = float(cf_teacher_lambda)
    if lam <= 0.0:
        return target_embedding

    teacher_float = teacher_embedding.float()
    m = teacher_float.shape[2]
    if m <= 0:
        raise ValueError("G2 teacher_embedding must contain at least one teacher sample")
    if lam >= 1.0:
        return teacher_float

    # Match OpenRLHF's integer GT repetition approximation:
    # r / (r + m) ~= 1 - lambda, so m / (r + m) ~= lambda.
    r = round(m * (1.0 - lam) / lam)
    r = max(min(r, m * 4), 1)
    gt_repeated = target_embedding.expand(-1, -1, r, -1, -1)
    return torch.cat([gt_repeated, teacher_float], dim=2)


@torch.no_grad()
def compute_cf_l1oo_rewards(
    gen_embedding: torch.Tensor,
    gt_embedding: torch.Tensor,
    *,
    teacher_embedding: torch.Tensor | None = None,
    cf_num_freqs: int = 128,
    cf_sigma: float = 1.0,
    cf_seed: int = 43,
    cf_alpha: float = 0.5,
    cf_beta: float = 0.5,
    cf_reward_scale: float = 1.0,
    cf_target_mode: str = "teacher",
    cf_target_num_refs: int = 1,
    cf_target_std: float = 0.05,
    cf_target_seed: int = 43,
    cf_teacher_lambda: float = 0.0,
) -> torch.Tensor:
    """OpenRLHF G2 cf_l1oo reward.

    Shapes:
    - gen_embedding / gt_embedding: (B, G, N, K, D)
    - teacher_embedding: (B, G, M, K, D), required only for teacher mode

    The returned reward has shape (B, G, N, K). Positive values mean removing
    the sample worsens the characteristic-function discrepancy.
    """
    if gen_embedding.ndim != 5 or gt_embedding.ndim != 5:
        raise ValueError(
            "G2 cf_l1oo expects non-token embeddings shaped (B, G, N, K, D), "
            f"got gen={tuple(gen_embedding.shape)} gt={tuple(gt_embedding.shape)}"
        )
    if gen_embedding.shape != gt_embedding.shape:
        raise ValueError(
            "G2 gen_embedding and gt_embedding must have identical shape, "
            f"got {tuple(gen_embedding.shape)} vs {tuple(gt_embedding.shape)}"
        )

    batch_size, num_groups, n_samples, num_blocks, feat_dim = gen_embedding.shape
    target_embedding = _build_cf_target_embedding(
        gt_embedding,
        cf_target_mode=cf_target_mode,
        cf_target_num_refs=cf_target_num_refs,
        cf_target_std=cf_target_std,
        cf_target_seed=cf_target_seed,
        teacher_embedding=teacher_embedding,
        cf_teacher_lambda=cf_teacher_lambda,
    )

    gen_flat = gen_embedding.permute(0, 1, 3, 2, 4).reshape(-1, n_samples, feat_dim).float()
    target_flat = target_embedding.permute(0, 1, 3, 2, 4).reshape(
        -1,
        target_embedding.shape[2],
        feat_dim,
    ).float()

    freqs = _get_fixed_cf_frequencies(
        input_dim=feat_dim,
        num_freqs=int(cf_num_freqs),
        sigma=float(cf_sigma),
        seed=int(cf_seed),
        device=gen_flat.device,
    )

    gen_proj = torch.einsum("fd,bnd->bfn", freqs, gen_flat)
    gen_real_vals = torch.cos(gen_proj)
    gen_imag_vals = torch.sin(gen_proj)
    gen_real = gen_real_vals.mean(dim=-1)
    gen_imag = gen_imag_vals.mean(dim=-1)

    target_proj = torch.einsum("fd,bnd->bfn", freqs, target_flat)
    target_real = torch.cos(target_proj).mean(dim=-1)
    target_imag = torch.sin(target_proj).mean(dim=-1)

    full_loss = _compute_cf_loss_terms(target_real, target_imag, gen_real, gen_imag, cf_alpha, cf_beta).mean(dim=-1)
    if n_samples == 1:
        rewards = -full_loss.unsqueeze(-1)
    else:
        loo_real = (gen_real_vals.sum(dim=-1, keepdim=True) - gen_real_vals) / float(n_samples - 1)
        loo_imag = (gen_imag_vals.sum(dim=-1, keepdim=True) - gen_imag_vals) / float(n_samples - 1)
        loo_loss = _compute_cf_loss_terms(
            target_real.unsqueeze(-1),
            target_imag.unsqueeze(-1),
            loo_real,
            loo_imag,
            cf_alpha,
            cf_beta,
        ).mean(dim=1)
        rewards = loo_loss - full_loss.unsqueeze(-1)

    rewards = rewards.reshape(batch_size, num_groups, num_blocks, n_samples).permute(0, 1, 3, 2).contiguous()
    return rewards.to(dtype=gen_embedding.dtype) * float(cf_reward_scale)


@torch.no_grad()
def compute_opd_cf_l1oo_rewards(
    gen_embedding: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    cf_num_freqs: int = 128,
    cf_sigma: float = 1.0,
    cf_seed: int = 43,
    cf_alpha: float = 0.5,
    cf_beta: float = 0.5,
    cf_reward_scale: float = 1.0,
    score_temperature: float = 1.0,
) -> torch.Tensor:
    """OPD-CF-L1OO reward over on-policy student rollout support.

    Shapes:
    - gen_embedding: (B, G, N, K, D)
    - teacher_scores: (B, G, N) or (B, G, N, K)

    Teacher scores are converted to weights with a softmax over the N student
    rollouts within each prompt group. The target distribution shares the same
    feature support as the student distribution, but is teacher-weighted.
    """
    if gen_embedding.ndim != 5:
        raise ValueError(
            "OPD-CF-L1OO expects gen_embedding shaped (B, G, N, K, D), "
            f"got {tuple(gen_embedding.shape)}"
        )
    if teacher_scores.ndim not in {3, 4}:
        raise ValueError(
            "OPD-CF-L1OO expects teacher_scores shaped (B, G, N) or (B, G, N, K), "
            f"got {tuple(teacher_scores.shape)}"
        )

    batch_size, num_groups, n_samples, num_blocks, feat_dim = gen_embedding.shape
    if n_samples <= 1:
        raise ValueError("OPD-CF-L1OO requires N > 1 student rollouts per prompt group.")
    if teacher_scores.shape[:3] != (batch_size, num_groups, n_samples):
        raise ValueError(
            "OPD-CF-L1OO teacher_scores must align with gen_embedding on B/G/N dims, "
            f"got scores={tuple(teacher_scores.shape)} gen={tuple(gen_embedding.shape)}"
        )
    if teacher_scores.ndim == 3:
        teacher_scores = teacher_scores.unsqueeze(-1).expand(-1, -1, -1, num_blocks)
    elif teacher_scores.shape[3] != num_blocks:
        raise ValueError(
            "OPD-CF-L1OO block-level teacher_scores must align with gen_embedding K dim, "
            f"got scores={tuple(teacher_scores.shape)} gen={tuple(gen_embedding.shape)}"
        )

    temperature = float(score_temperature)
    if temperature <= 0.0:
        raise ValueError("OPD-CF-L1OO score_temperature must be positive.")

    scores = teacher_scores.float() / temperature
    weights = torch.softmax(scores, dim=2)

    gen_flat = gen_embedding.permute(0, 1, 3, 2, 4).reshape(-1, n_samples, feat_dim).float()
    weight_flat = weights.permute(0, 1, 3, 2).reshape(-1, n_samples).to(device=gen_flat.device)

    freqs = _get_fixed_cf_frequencies(
        input_dim=feat_dim,
        num_freqs=int(cf_num_freqs),
        sigma=float(cf_sigma),
        seed=int(cf_seed),
        device=gen_flat.device,
    )

    gen_proj = torch.einsum("fd,bnd->bfn", freqs, gen_flat)
    gen_real_vals = torch.cos(gen_proj)
    gen_imag_vals = torch.sin(gen_proj)

    student_real = gen_real_vals.mean(dim=-1)
    student_imag = gen_imag_vals.mean(dim=-1)
    target_real = (gen_real_vals * weight_flat.unsqueeze(1)).sum(dim=-1)
    target_imag = (gen_imag_vals * weight_flat.unsqueeze(1)).sum(dim=-1)

    full_loss = _compute_cf_loss_terms(
        target_real,
        target_imag,
        student_real,
        student_imag,
        cf_alpha,
        cf_beta,
    ).mean(dim=-1)

    loo_real = (gen_real_vals.sum(dim=-1, keepdim=True) - gen_real_vals) / float(n_samples - 1)
    loo_imag = (gen_imag_vals.sum(dim=-1, keepdim=True) - gen_imag_vals) / float(n_samples - 1)
    loo_loss = _compute_cf_loss_terms(
        target_real.unsqueeze(-1),
        target_imag.unsqueeze(-1),
        loo_real,
        loo_imag,
        cf_alpha,
        cf_beta,
    ).mean(dim=1)
    rewards = loo_loss - full_loss.unsqueeze(-1)
    rewards = rewards.reshape(batch_size, num_groups, num_blocks, n_samples).permute(0, 1, 3, 2).contiguous()
    return rewards.to(dtype=gen_embedding.dtype) * float(cf_reward_scale)
