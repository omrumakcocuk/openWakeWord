#!/usr/bin/env python3
"""Eğitim manifestini uygulamanın 80 ms akış yolunda değerlendir."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wake_word import CHUNK_SAMPLES, DEFAULT_MODEL, build_model, score_as_float, wav_chunks


MANIFEST = ROOT / "training" / "data" / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument(
        "--tail-ms",
        type=int,
        default=0,
        help="wake_word.py'nin 400 ms kuyruğuna eklenecek ek sessizlik",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = build_model(args.model.resolve())
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    silence_chunks = max(0, round(args.tail_ms / 80))
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)

    for item in manifest:
        model.reset()
        best_score = 0.0
        chunks = list(wav_chunks(ROOT / item["path"]))
        chunks.extend(np.zeros(CHUNK_SAMPLES, dtype=np.int16) for _ in range(silence_chunks))
        for samples in chunks:
            predictions = model.predict(samples)
            best_score = max(best_score, *(score_as_float(value) for value in predictions.values()))
        grouped[(int(item["label"]), str(item["text"]))].append(best_score)

    total_positive = total_negative = positive_hits = negative_hits = 0
    for (label, phrase), scores in sorted(grouped.items(), key=lambda item: (-item[0][0], item[0][1])):
        hits = sum(score >= args.threshold for score in scores)
        if label:
            total_positive += len(scores)
            positive_hits += hits
        else:
            total_negative += len(scores)
            negative_hits += hits
        print(
            f"{'POS' if label else 'NEG'} {phrase!r}: {hits}/{len(scores)} | "
            f"min={min(scores):.3f} ort={np.mean(scores):.3f} max={max(scores):.3f}"
        )

    print(f"\nPozitif yakalama: {positive_hits}/{total_positive}")
    print(f"Negatif yanlış tetikleme: {negative_hits}/{total_negative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
