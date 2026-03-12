from __future__ import annotations

import argparse

import torch

from data_loader import BPEEncoder, Vocab
from diffusion import MDLMTrainConfig, _build_model_config
from generation import generate_short_passage
from model import DIT


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def _load_model_and_config(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[DIT, Vocab, BPEEncoder, MDLMTrainConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = MDLMTrainConfig(**checkpoint["train_config"])
    vocab = Vocab.load(cfg.vocab_path)
    bpe = BPEEncoder.load(cfg.bpe_path)
    vocab_size = checkpoint.get("vocab_size", len(vocab.id_to_token))

    model = DIT(
        config=_build_model_config(cfg),
        vocab_size=vocab_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab, bpe, cfg


def _encode_prompt(prompt: str, vocab: Vocab, bpe: BPEEncoder) -> list[int]:
    if not prompt.strip():
        return []
    tokens = bpe.tokenize(prompt)
    return vocab.encode(tokens, add_bos_eos=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a trained checkpoint and run continuous generation.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to saved .pt checkpoint.")
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--generate-tokens", type=int, default=None)
    parser.add_argument("--generate-steps", type=int, default=None)
    parser.add_argument("--generate-checkpoints", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--mask-schedule", type=str, choices=("linear", "cosine"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    model, vocab, bpe, cfg = _load_model_and_config(args.checkpoint_path, device)

    generate_tokens = args.generate_tokens if args.generate_tokens is not None else cfg.generate_tokens
    generate_steps = args.generate_steps if args.generate_steps is not None else cfg.generate_steps
    generate_checkpoints = (
        args.generate_checkpoints if args.generate_checkpoints is not None else cfg.generate_checkpoints
    )
    temperature = args.temperature if args.temperature is not None else cfg.temperature
    mask_schedule = args.mask_schedule if args.mask_schedule is not None else cfg.mask_schedule

    print(f"Loaded checkpoint from: {args.checkpoint_path}")
    print(f"Device: {device}")
    print("Type a prompt and press Enter to generate, or type 'q' to quit.")

    while True:
        user_input = input("> ")
        stripped_input = user_input.strip()
        if stripped_input.lower() in {"q", "quit", "exit"}:
            break
        prefix_token_ids = _encode_prompt(stripped_input, vocab=vocab, bpe=bpe)
        passage = generate_short_passage(
            model=model,
            vocab=vocab,
            device=device,
            seq_len=generate_tokens,
            steps=generate_steps,
            schedule=mask_schedule,
            temperature=temperature,
            checkpoints=generate_checkpoints,
            prefix_token_ids=prefix_token_ids,
        )
        print("\nGenerated Passage")
        print(passage)


if __name__ == "__main__":
    main()
