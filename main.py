from __future__ import annotations

import argparse

from diffusion import MDLMTrainConfig, train_mdlm


def parse_args() -> tuple[MDLMTrainConfig, str | None]:
    parser = argparse.ArgumentParser(description="Train the masked diffusion language model.")
    parser.add_argument("--text-path", type=str, default=MDLMTrainConfig.text_path)
    parser.add_argument("--vocab-path", type=str, default=MDLMTrainConfig.vocab_path)
    parser.add_argument("--bpe-path", type=str, default=MDLMTrainConfig.bpe_path)
    parser.add_argument("--token-ids-path", type=str, default=MDLMTrainConfig.token_ids_path)

    parser.add_argument("--batch-size", type=int, default=MDLMTrainConfig.batch_size)
    parser.add_argument("--seq-len", type=int, default=MDLMTrainConfig.seq_len)
    parser.add_argument("--train-steps", type=int, default=MDLMTrainConfig.train_steps)
    parser.add_argument("--log-every", type=int, default=MDLMTrainConfig.log_every)
    parser.add_argument("--generate-tokens", type=int, default=MDLMTrainConfig.generate_tokens)
    parser.add_argument("--generate-steps", type=int, default=MDLMTrainConfig.generate_steps)
    parser.add_argument("--generate-checkpoints", type=int, default=MDLMTrainConfig.generate_checkpoints)
    parser.add_argument("--temperature", type=float, default=MDLMTrainConfig.temperature)

    parser.add_argument("--lr", type=float, default=MDLMTrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=MDLMTrainConfig.weight_decay)
    parser.add_argument("--mask-schedule", type=str, choices=("linear", "cosine"), default=MDLMTrainConfig.mask_schedule)

    parser.add_argument("--max-vocab-size", type=int, default=MDLMTrainConfig.max_vocab_size)
    parser.add_argument("--num-merges", type=int, default=MDLMTrainConfig.num_merges)

    parser.add_argument("--hidden-size", type=int, default=MDLMTrainConfig.hidden_size)
    parser.add_argument("--cond-dim", type=int, default=MDLMTrainConfig.cond_dim)
    parser.add_argument("--n-heads", type=int, default=MDLMTrainConfig.n_heads)
    parser.add_argument("--n-blocks", type=int, default=MDLMTrainConfig.n_blocks)
    parser.add_argument("--dropout", type=float, default=MDLMTrainConfig.dropout)
    parser.add_argument(
        "--scale-by-sigma",
        action=argparse.BooleanOptionalAction,
        default=MDLMTrainConfig.scale_by_sigma,
    )

    parser.add_argument("--seed", type=int, default=MDLMTrainConfig.seed)
    parser.add_argument(
        "--save-model-path",
        type=str,
        default=None,
        help="Optional checkpoint path (.pt) to save trained model + config.",
    )
    args = parser.parse_args()

    cfg = MDLMTrainConfig(
        text_path=args.text_path,
        vocab_path=args.vocab_path,
        bpe_path=args.bpe_path,
        token_ids_path=args.token_ids_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        train_steps=args.train_steps,
        log_every=args.log_every,
        generate_tokens=args.generate_tokens,
        generate_steps=args.generate_steps,
        generate_checkpoints=args.generate_checkpoints,
        temperature=args.temperature,
        lr=args.lr,
        weight_decay=args.weight_decay,
        mask_schedule=args.mask_schedule,
        max_vocab_size=args.max_vocab_size,
        num_merges=args.num_merges,
        hidden_size=args.hidden_size,
        cond_dim=args.cond_dim,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        scale_by_sigma=args.scale_by_sigma,
        seed=args.seed,
    )
    return cfg, args.save_model_path


if __name__ == "__main__":
    cfg, save_model_path = parse_args()
    train_mdlm(cfg, save_model_path=save_model_path)
