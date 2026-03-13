from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SPECIAL_TOKENS = ("[PAD]", "[BOS]", "[EOS]", "[MASK]", "[UNK]")

SPACE_TOKEN = "<SP>"
NEWLINE_TOKEN = "<NL>"
TAB_TOKEN = "<TAB>"
CARRIAGE_RETURN_TOKEN = "<CR>"

_WHITESPACE_TO_TOKEN = {
    " ": SPACE_TOKEN,
    "\n": NEWLINE_TOKEN,
    "\t": TAB_TOKEN,
    "\r": CARRIAGE_RETURN_TOKEN,
}
_TOKEN_TO_WHITESPACE = {v: k for k, v in _WHITESPACE_TO_TOKEN.items()}
WHITESPACE_MARKER_TOKENS = set(_TOKEN_TO_WHITESPACE.keys())


PRETOKEN_PATTERN = re.compile(r"\s+|\w+|[^\w\s]", flags=re.UNICODE)


def pretokenize(text: str) -> List[str]:
    """Split into coarse tokens (including explicit whitespace markers) before BPE."""
    out: List[str] = []
    for token in PRETOKEN_PATTERN.findall(text):
        if token.isspace():
            for ch in token:
                out.append(_WHITESPACE_TO_TOKEN.get(ch, SPACE_TOKEN))
        else:
            out.append(token)
    return out


def _slugify_dataset_name(path: str | Path) -> str:
    stem = Path(path).stem
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return slug or "dataset"


@dataclass(frozen=True)
class DatasetArtifactPaths:
    vocab_path: Path
    bpe_path: Path
    token_ids_path: Path


def get_dataset_artifact_paths(
    text_path: str | Path,
    artifacts_dir: str | Path | None = None,
) -> DatasetArtifactPaths:
    # Keep artifacts scoped by source text file to avoid cross-dataset collisions.
    text_path = Path(text_path)
    base_dir = Path(artifacts_dir) if artifacts_dir is not None else text_path.parent
    slug = _slugify_dataset_name(text_path)
    return DatasetArtifactPaths(
        vocab_path=base_dir / f"{slug}_vocab.json",
        bpe_path=base_dir / f"{slug}_bpe_merges.json",
        token_ids_path=base_dir / f"{slug}_token_ids.json",
    )


def _count_pairs(word_freq: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
    pair_counts: Dict[Tuple[str, str], int] = {}
    for symbols, freq in word_freq.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
    return pair_counts


def _merge_pair_in_symbols(symbols: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
    merged: List[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


@dataclass
class BPEEncoder:
    merges: List[Tuple[str, str]]

    def __post_init__(self) -> None:
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.cache: Dict[str, List[str]] = {}

    @classmethod
    def train(
        cls,
        text: str,
        num_merges: int = 5000,
        min_pair_freq: int = 2,
    ) -> "BPEEncoder":
        token_freq: Dict[str, int] = {}
        for token in pretokenize(text):
            token_freq[token] = token_freq.get(token, 0) + 1

        word_freq: Dict[Tuple[str, ...], int] = {}
        for token, freq in token_freq.items():
            if not token:
                continue
            word_freq[tuple(token)] = freq

        merges: List[Tuple[str, str]] = []
        for _ in range(num_merges):
            pair_counts = _count_pairs(word_freq)
            if not pair_counts:
                break
            best_pair, best_count = max(pair_counts.items(), key=lambda x: x[1])
            if best_count < min_pair_freq:
                break

            next_word_freq: Dict[Tuple[str, ...], int] = {}
            for symbols, freq in word_freq.items():
                merged_symbols = _merge_pair_in_symbols(symbols, best_pair)
                next_word_freq[merged_symbols] = next_word_freq.get(merged_symbols, 0) + freq
            word_freq = next_word_freq
            merges.append(best_pair)

        return cls(merges=merges)

    def encode_token(self, token: str) -> List[str]:
        if token in WHITESPACE_MARKER_TOKENS:
            return [token]
        if token in self.cache:
            return list(self.cache[token])

        if len(token) <= 1:
            out = [token]
            self.cache[token] = out
            return list(out)

        symbols = list(token)
        while len(symbols) > 1:
            candidate_pairs = []
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.bpe_ranks.get(pair)
                if rank is not None:
                    candidate_pairs.append((rank, pair))
            if not candidate_pairs:
                break

            _, best_pair = min(candidate_pairs, key=lambda x: x[0])
            merged: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged

        self.cache[token] = list(symbols)
        return list(symbols)

    def tokenize(self, text: str) -> List[str]:
        pieces: List[str] = []
        for token in pretokenize(text):
            pieces.extend(self.encode_token(token))
        return pieces

    def save(self, path: str | Path) -> None:
        payload = {"merges": [list(pair) for pair in self.merges]}
        Path(path).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPEEncoder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(pair) for pair in payload["merges"]]
        return cls(merges=merges)


@dataclass
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["[PAD]"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["[BOS]"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["[EOS]"]

    @property
    def mask_id(self) -> int:
        return self.token_to_id["[MASK]"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["[UNK]"]

    def encode(self, tokens: Sequence[str], add_bos_eos: bool = True) -> List[int]:
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]
        if add_bos_eos:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> List[str]:
        out = []
        for idx in ids:
            token = self.id_to_token[idx]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            out.append(token)
        return out

    def save(self, path: str | Path) -> None:
        payload = {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(token_to_id=payload["token_to_id"], id_to_token=payload["id_to_token"])


def detokenize(tokens: Sequence[str]) -> str:
    out: List[str] = []
    for tok in tokens:
        out.append(_TOKEN_TO_WHITESPACE.get(tok, tok))
    return "".join(out)


def save_token_ids(token_ids: Sequence[int], path: str | Path) -> None:
    Path(path).write_text(json.dumps(list(token_ids), ensure_ascii=True), encoding="utf-8")


def build_vocab(
    text_path: str | Path,
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    num_merges: int = 5000,
    min_pair_freq: int = 2,
) -> tuple[Vocab, BPEEncoder]:
    text = Path(text_path).read_text(encoding="utf-8")
    bpe = BPEEncoder.train(text=text, num_merges=num_merges, min_pair_freq=min_pair_freq)
    tokens = bpe.tokenize(text)

    counts: Dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1

    kept = [tok for tok, freq in counts.items() if freq >= min_freq and tok not in SPECIAL_TOKENS]
    kept.sort(key=lambda t: (-counts[t], t))

    if max_vocab_size is not None:
        # max_vocab_size includes special tokens
        room = max(0, max_vocab_size - len(SPECIAL_TOKENS))
        kept = kept[:room]

    id_to_token = list(SPECIAL_TOKENS) + kept
    token_to_id = {tok: i for i, tok in enumerate(id_to_token)}
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token), bpe


def load_token_ids(text_path: str | Path, vocab: Vocab, bpe: BPEEncoder) -> List[int]:
    text = Path(text_path).read_text(encoding="utf-8")
    tokens = bpe.tokenize(text)
    return vocab.encode(tokens, add_bos_eos=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Build vocab/BPE/token_ids artifacts for a text dataset.")
    parser.add_argument("--text-path", type=str, default="data/shakespeare.txt")
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--vocab-path", type=str, default=None)
    parser.add_argument("--bpe-path", type=str, default=None)
    parser.add_argument("--token-ids-path", type=str, default=None)
    parser.add_argument("--max-vocab-size", type=int, default=20000)
    parser.add_argument("--num-merges", type=int, default=5000)
    parser.add_argument("--min-pair-freq", type=int, default=2)
    args = parser.parse_args()

    text_file = Path(args.text_path)
    auto_paths = get_dataset_artifact_paths(text_file, artifacts_dir=args.artifacts_dir)
    vocab_file = Path(args.vocab_path) if args.vocab_path else auto_paths.vocab_path
    bpe_file = Path(args.bpe_path) if args.bpe_path else auto_paths.bpe_path
    token_ids_file = Path(args.token_ids_path) if args.token_ids_path else auto_paths.token_ids_path
    vocab_file.parent.mkdir(parents=True, exist_ok=True)
    bpe_file.parent.mkdir(parents=True, exist_ok=True)
    token_ids_file.parent.mkdir(parents=True, exist_ok=True)

    vocab, bpe = build_vocab(
        text_file,
        min_freq=1,
        max_vocab_size=args.max_vocab_size,
        num_merges=args.num_merges,
        min_pair_freq=args.min_pair_freq,
    )
    vocab.save(vocab_file)
    bpe.save(bpe_file)

    token_ids = load_token_ids(text_file, vocab, bpe)
    save_token_ids(token_ids, token_ids_file)

    print("vocab_size:", len(vocab.id_to_token))
    print("bpe_merges:", len(bpe.merges))
    print("mask_token:", "[MASK]", "id=", vocab.mask_id)
    print("num_token_ids:", len(token_ids))
    print("text_path:", str(text_file))
    print("saved:", str(vocab_file), str(bpe_file), str(token_ids_file))
