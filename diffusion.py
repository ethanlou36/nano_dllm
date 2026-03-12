from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from data_loader import Vocab, build_vocab, load_token_ids, save_token_ids
from model import DIT


@dataclass
class MDLMTrainConfig:
    text_path: str = "data/shakespeare.txt"
    vocab_path: str = "data/vocab.json"
    bpe_path: str = "data/bpe_merges.json"
    token_ids_path: str = "data/token_ids.json"

    batch_size: int = 16
    seq_len: int = 256
    train_steps: int = 1000
    log_every: int = 10
    generate_tokens: int = 64
    generate_steps: int = 24
    generate_checkpoints: int = 4
    temperature: float = 1.0

    lr: float = 4e-4
    weight_decay: float = 0.01
    mask_schedule: str = "cosine"  # linear | cosine

    max_vocab_size: int = 20000
    num_merges: int = 5000

    hidden_size: int = 512
    cond_dim: int = 512
    n_heads: int = 8
    n_blocks: int = 8
    dropout: float = 0.1
    scale_by_sigma: bool = False

    seed: int = 42


def _build_model_config(cfg: MDLMTrainConfig) -> Dict[str, Dict[str, float]]:
    if cfg.hidden_size % cfg.n_heads != 0:
        raise ValueError("hidden_size must be divisible by n_heads")
    return {
        "model": {
            "hidden_size": cfg.hidden_size,
            "cond_dim": cfg.cond_dim,
            "n_heads": cfg.n_heads,
            "n_blocks": cfg.n_blocks,
            "dropout": cfg.dropout,
            "scale_by_sigma": cfg.scale_by_sigma,
        }
    }


def prepare_dataset(cfg: MDLMTrainConfig) -> Tuple[torch.Tensor, Vocab]:
    text_path = Path(cfg.text_path)
    vocab_path = Path(cfg.vocab_path)
    bpe_path = Path(cfg.bpe_path)
    token_ids_path = Path(cfg.token_ids_path)

    if token_ids_path.exists() and vocab_path.exists() and bpe_path.exists():
        vocab = Vocab.load(vocab_path)
        token_ids = json.loads(token_ids_path.read_text(encoding="utf-8"))
        return torch.tensor(token_ids, dtype=torch.long), vocab

    vocab, bpe = build_vocab(
        text_path=text_path,
        min_freq=1,
        max_vocab_size=cfg.max_vocab_size,
        num_merges=cfg.num_merges,
        min_pair_freq=2,
    )
    vocab.save(vocab_path)
    bpe.save(bpe_path)

    token_ids = load_token_ids(text_path=text_path, vocab=vocab, bpe=bpe)
    save_token_ids(token_ids, token_ids_path)
    return torch.tensor(token_ids, dtype=torch.long), vocab


def sample_batch_windows(
    token_ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if token_ids.numel() < seq_len:
        raise ValueError("Encoded corpus is shorter than seq_len")

    max_start = token_ids.numel() - seq_len
    starts = torch.randint(0, max_start + 1, (batch_size,), dtype=torch.long)
    batch = torch.stack([token_ids[s : s + seq_len] for s in starts.tolist()], dim=0)
    return batch.to(device=device, dtype=torch.long)


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


def mdlm_train_step(
    model: DIT,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
    mask_token_id: int,
    special_token_ids: Sequence[int],
    schedule: str,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    clean_batch = sample_batch_windows(
        token_ids=token_ids,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )
    t = torch.rand((batch_size,), device=device)
    noisy_batch, loss_mask = apply_mask_with_scheduler(
        clean_ids=clean_batch,
        t=t,
        mask_token_id=mask_token_id,
        special_token_ids=special_token_ids,
        schedule=schedule,
    )

    logits = model(noisy_batch, sigma=t)  # [B, L, V]
    per_token_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        clean_batch.reshape(-1),
        reduction="none",
    ).view_as(clean_batch)

    mask_float = loss_mask.float()
    denom = mask_float.sum().clamp_min(1.0)
    loss = (per_token_ce * mask_float).sum() / denom

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "mask_ratio": float(mask_float.mean().item()),
        "t_mean": float(t.mean().item()),
    }


def decode_token_pieces(tokens: Sequence[str]) -> str:
    if not tokens:
        return ""
    no_space_before = {".", ",", "!", "?", ";", ":", "'", "\"", ")", "]", "}"}
    no_space_after = {"(", "[", "{", "\""}

    out: list[str] = []
    for token in tokens:
        if not out:
            out.append(token)
            continue
        prev = out[-1]
        if token in no_space_before or prev in no_space_after:
            out.append(token)
        else:
            out.append(" " + token)
    return "".join(out)


def render_generation_state(vocab: Vocab, token_ids: Sequence[int]) -> str:
    visible_tokens: list[str] = []
    for token_id in token_ids:
        token = vocab.id_to_token[token_id]
        if token in ("[BOS]", "[EOS]", "[PAD]"):
            continue
        visible_tokens.append(token)
    return decode_token_pieces(visible_tokens)


@torch.no_grad()
def generate_short_passage(
    model: DIT,
    vocab: Vocab,
    device: torch.device,
    seq_len: int,
    steps: int,
    schedule: str,
    temperature: float,
    checkpoints: int,
) -> str:
    model.eval()
    seq_len = max(seq_len, 4)
    x = torch.full((1, seq_len), vocab.mask_id, dtype=torch.long, device=device)
    x[:, 0] = vocab.bos_id
    x[:, -1] = vocab.eos_id

    maskable = (x != vocab.bos_id) & (x != vocab.eos_id) & (x != vocab.pad_id)
    temp = max(temperature, 1e-5)
    checkpoint_every = max(1, steps // max(checkpoints, 1))

    print("\nGeneration Trace")
    print("initial(masked)")
    print(render_generation_state(vocab=vocab, token_ids=x[0].tolist()))

    for step in range(steps, 0, -1):
        t_value = step / float(steps)
        t = torch.full((1,), t_value, dtype=torch.float32, device=device)
        logits = model(x, sigma=t) / temp
        sampled = torch.distributions.Categorical(logits=logits).sample()

        masked_now = (x == vocab.mask_id) & maskable
        x[masked_now] = sampled[masked_now]

        if step > 1:
            next_t = torch.tensor([(step - 1) / float(steps)], device=device)
            next_mask_prob = scheduler_mask_prob(next_t, schedule=schedule).item()
            remask = (torch.rand_like(x, dtype=torch.float32) < next_mask_prob) & maskable
            x[remask] = vocab.mask_id

        should_log = step == steps or step == 1 or step % checkpoint_every == 0
        if should_log:
            masked_count = int(((x == vocab.mask_id) & maskable).sum().item())
            print(f"\ncheckpoint(step={step}/{steps}, masks_remaining={masked_count})")
            print(render_generation_state(vocab=vocab, token_ids=x[0].tolist()))

    token_ids = x[0].tolist()
    pieces = vocab.decode(token_ids, skip_special=True)
    return decode_token_pieces(pieces)


def train_mdlm(cfg: MDLMTrainConfig) -> None:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    token_ids, vocab = prepare_dataset(cfg)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = DIT(config=_build_model_config(cfg), vocab_size=len(vocab.id_to_token)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    special_ids = [vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.mask_id]
    loss_history: list[float] = []
    mask_history: list[float] = []
    t_history: list[float] = []
    start_time = time.time()

    for step in range(1, cfg.train_steps + 1):
        metrics = mdlm_train_step(
            model=model,
            optimizer=optimizer,
            token_ids=token_ids,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            mask_token_id=vocab.mask_id,
            special_token_ids=special_ids,
            schedule=cfg.mask_schedule,
            device=device,
        )
        loss_history.append(metrics["loss"])
        mask_history.append(metrics["mask_ratio"])
        t_history.append(metrics["t_mean"])

        if step == 1 or step % cfg.log_every == 0:
            print(
                f"step={step} "
                f"loss={metrics['loss']:.4f} "
                f"mask_ratio={metrics['mask_ratio']:.4f} "
                f"t_mean={metrics['t_mean']:.4f}"
            )

    elapsed = time.time() - start_time
    mean_loss = sum(loss_history) / len(loss_history)
    mean_mask = sum(mask_history) / len(mask_history)
    mean_t = sum(t_history) / len(t_history)
    print("\nTraining Summary")
    print(f"steps={cfg.train_steps} elapsed_s={elapsed:.2f} steps_per_s={cfg.train_steps / max(elapsed, 1e-8):.2f}")
    print(
        f"loss(last/mean/min/max)="
        f"{loss_history[-1]:.4f}/{mean_loss:.4f}/{min(loss_history):.4f}/{max(loss_history):.4f}"
    )
    print(f"mask_ratio_mean={mean_mask:.4f} t_mean={mean_t:.4f}")

    passage = generate_short_passage(
        model=model,
        vocab=vocab,
        device=device,
        seq_len=cfg.generate_tokens,
        steps=cfg.generate_steps,
        schedule=cfg.mask_schedule,
        temperature=cfg.temperature,
        checkpoints=cfg.generate_checkpoints,
    )
    print("\nGenerated Passage")
    print(passage)


if __name__ == "__main__":
    train_mdlm(MDLMTrainConfig())
