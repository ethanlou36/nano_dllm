from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch


def scheduler_mask_prob(t: torch.Tensor, schedule: str) -> torch.Tensor:
    if schedule == "linear":
        return t
    if schedule == "cosine":
        return 1.0 - torch.cos(0.5 * math.pi * t)
    raise ValueError(f"Unknown schedule '{schedule}'. Expected 'linear' or 'cosine'.")


def apply_mask_with_scheduler(
    clean_ids: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    special_token_ids: Sequence[int],
    schedule: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_prob = scheduler_mask_prob(t, schedule=schedule).unsqueeze(1).clamp(0.0, 1.0)
    random_scores = torch.rand(clean_ids.shape, device=clean_ids.device)
    loss_mask = random_scores < mask_prob

    is_special = torch.zeros_like(clean_ids, dtype=torch.bool)
    for token_id in special_token_ids:
        is_special |= clean_ids.eq(token_id)
    loss_mask &= ~is_special

    # Ensure at least one masked token per row (if there is any non-special token).
    candidate_positions = ~is_special
    no_mask_rows = loss_mask.sum(dim=1).eq(0).nonzero(as_tuple=False).flatten()
    for row_idx in no_mask_rows.tolist():
        candidates = candidate_positions[row_idx].nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        choice = candidates[torch.randint(0, candidates.numel(), (1,), device=clean_ids.device)]
        loss_mask[row_idx, choice] = True

    noisy_ids = clean_ids.clone()
    noisy_ids[loss_mask] = mask_token_id
    return noisy_ids, loss_mask
