from __future__ import annotations

from typing import Sequence

import torch

from data_loader import Vocab
from model import DIT
from noise_scheduler import scheduler_mask_prob


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
    prefix_token_ids: Sequence[int] | None = None,
) -> str:
    model.eval()
    seq_len = max(seq_len, 4)
    x = torch.full((1, seq_len), vocab.mask_id, dtype=torch.long, device=device)
    fixed_positions = torch.zeros_like(x, dtype=torch.bool)

    x[:, 0] = vocab.bos_id
    fixed_positions[:, 0] = True

    if prefix_token_ids:
        max_prefix_len = max(0, seq_len - 2)
        prefix_ids = list(prefix_token_ids)[:max_prefix_len]
        if prefix_ids:
            prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
            x[:, 1 : 1 + len(prefix_ids)] = prefix_tensor
            fixed_positions[:, 1 : 1 + len(prefix_ids)] = True

    x[:, -1] = vocab.eos_id
    fixed_positions[:, -1] = True

    maskable = ~fixed_positions & (x != vocab.pad_id)
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
