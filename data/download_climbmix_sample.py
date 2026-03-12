from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a text sample from karpathy/climbmix-400b-shuffle.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/climbmix_400b_sample.txt",
        help="Output path for sampled text file.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=20_000_000,
        help="Stop after writing this many characters.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help="Stop after processing this many rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chars = 0
    rows = 0
    stream = load_dataset("karpathy/climbmix-400b-shuffle", split="train", streaming=True)

    with output_path.open("w", encoding="utf-8") as f:
        for ex in stream:
            text = ex.get("text")
            if not text:
                continue
            f.write(text)
            f.write("\n")
            chars += len(text) + 1
            rows += 1
            if rows >= args.max_rows or chars >= args.max_chars:
                break

    print(f"saved={output_path} rows={rows} chars={chars}")


if __name__ == "__main__":
    main()
