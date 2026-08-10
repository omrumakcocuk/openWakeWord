#!/usr/bin/env python3
"""Manifest bölmesini uygulamanın 80 ms akış yolunda deterministik değerlendir."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wake_word import (
    CHUNK_DURATION_SECONDS,
    CHUNK_SAMPLES,
    DEFAULT_MODEL,
    MODEL_PRIME_CHUNKS,
    WAV_TAIL_DURATION_SECONDS,
    build_model,
    reset_and_prime_model,
    score_as_float,
    wav_chunks,
)

try:  # Hem ``python training/...`` hem modül olarak içe aktarma desteği.
    from .common import (
        MANIFEST_PATH,
        VALID_SPLITS,
        atomic_write_json,
        evaluation_set_sha256,
        load_manifest,
        sha256_file,
    )
except ImportError:
    from common import (
        MANIFEST_PATH,
        VALID_SPLITS,
        atomic_write_json,
        evaluation_set_sha256,
        load_manifest,
        sha256_file,
    )


CHUNK_MS = round(CHUNK_DURATION_SECONDS * 1000)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument(
        "--split",
        choices=(*VALID_SPLITS, "all"),
        default="test",
        help="Değerlendirilecek manifest bölmesi (varsayılan: ayrılmış test)",
    )
    parser.add_argument(
        "--tail-ms",
        type=int,
        default=0,
        help="wake_word.py'nin 400 ms kuyruğuna eklenecek ek sessizlik",
    )
    parser.add_argument("--json-output", type=Path, help="Makinece okunabilir JSON özet yolu")
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold 0 ile 1 arasında olmalı")
    if args.tail_ms < 0:
        parser.error("--tail-ms negatif olamaz")
    return args


def best_prediction_score(predictions: dict[str, object]) -> float:
    if not predictions:
        raise RuntimeError("openWakeWord modelinden skor gelmedi")
    return max(score_as_float(value) for value in predictions.values())


def evaluate_item(
    model: object,
    wav_path: Path,
    *,
    extra_tail_chunks: int,
) -> float:
    # Canlı uygulamayla aynı 26 sessizlik karesi kullanılır. Böylece
    # openWakeWord reset'indeki rastgele tampon kararı etkilemez.
    reset_and_prime_model(model)  # type: ignore[arg-type]
    best_score = 0.0
    for samples in wav_chunks(wav_path):
        best_score = max(
            best_score,
            best_prediction_score(model.predict(samples)),  # type: ignore[attr-defined]
        )
    silence = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
    for _ in range(extra_tail_chunks):
        best_score = max(
            best_score,
            best_prediction_score(model.predict(silence)),  # type: ignore[attr-defined]
        )
    return best_score


def build_summary(
    grouped: dict[tuple[int, str], list[float]],
    *,
    threshold: float,
    split: str,
    model_path: Path,
    manifest_path: Path,
    model_sha256: str,
    manifest_sha256: str,
    evaluation_set_hash: str,
    warmup_ms: int,
    extra_tail_ms: int,
) -> dict[str, object]:
    total_positive = total_negative = positive_hits = negative_hits = 0
    per_phrase: list[dict[str, object]] = []
    for (label, phrase), scores in sorted(grouped.items(), key=lambda item: (-item[0][0], item[0][1])):
        hits = sum(score >= threshold for score in scores)
        if label:
            total_positive += len(scores)
            positive_hits += hits
        else:
            total_negative += len(scores)
            negative_hits += hits
        per_phrase.append(
            {
                "label": label,
                "text": phrase,
                "examples": len(scores),
                "threshold_hits": hits,
                "minimum_score": float(min(scores)),
                "mean_score": float(np.mean(scores)),
                "median_score": float(np.median(scores)),
                "maximum_score": float(max(scores)),
            }
        )

    tp, fn = positive_hits, total_positive - positive_hits
    fp, tn = negative_hits, total_negative - negative_hits
    recall = tp / total_positive if total_positive else None
    false_positive_rate = fp / total_negative if total_negative else None
    precision = tp / (tp + fp) if tp + fp else None
    accuracy = (tp + tn) / (total_positive + total_negative)
    return {
        "schema_version": 2,
        "evaluation_protocol": "openwakeword-streaming-v1",
        "model": str(model_path),
        "model_sha256": model_sha256,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "evaluation_set_sha256": evaluation_set_hash,
        "split": split,
        "threshold": threshold,
        "warmup_ms": warmup_ms,
        "built_in_wav_tail_ms": round(WAV_TAIL_DURATION_SECONDS * 1000),
        "additional_tail_ms": extra_tail_ms,
        "positive_examples": total_positive,
        "negative_examples": total_negative,
        "positive_hits": positive_hits,
        "false_positive_hits": negative_hits,
        "recall": recall,
        "precision": precision,
        "false_positive_rate": false_positive_rate,
        "accuracy": accuracy,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "per_phrase": per_phrase,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = args.model.resolve()
    manifest_path = args.manifest.resolve()
    model_hash = sha256_file(model_path)
    manifest_hash = sha256_file(manifest_path)
    model = build_model(model_path)
    manifest = load_manifest(manifest_path)
    selected = [item for item in manifest if args.split == "all" or item.split == args.split]
    if not selected:
        raise ValueError(f"Manifestte {args.split!r} bölmesi için kayıt yok")
    evaluation_hash = evaluation_set_sha256(selected)

    extra_tail_chunks = math.ceil(args.tail_ms / CHUNK_MS)
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for index, item in enumerate(selected, start=1):
        score = evaluate_item(
            model,
            item.absolute_path,
            extra_tail_chunks=extra_tail_chunks,
        )
        grouped[(item.label, item.text)].append(score)
        if index % 25 == 0 or index == len(selected):
            print(f"Akış testi: {index}/{len(selected)}", flush=True)

    summary = build_summary(
        grouped,
        threshold=args.threshold,
        split=args.split,
        model_path=model_path,
        manifest_path=manifest_path,
        model_sha256=model_hash,
        manifest_sha256=manifest_hash,
        evaluation_set_hash=evaluation_hash,
        warmup_ms=MODEL_PRIME_CHUNKS * CHUNK_MS,
        extra_tail_ms=extra_tail_chunks * CHUNK_MS,
    )
    if sha256_file(model_path) != model_hash:
        raise RuntimeError("Model dosyası değerlendirme sırasında değişti; rapor yazılmadı")
    if sha256_file(manifest_path) != manifest_hash:
        raise RuntimeError("Manifest değerlendirme sırasında değişti; rapor yazılmadı")
    if evaluation_set_sha256(selected) != evaluation_hash:
        raise RuntimeError("WAV veri seti değerlendirme sırasında değişti; rapor yazılmadı")
    for phrase in summary["per_phrase"]:  # type: ignore[index]
        prefix = "POS" if phrase["label"] else "NEG"
        print(
            f"{prefix} {phrase['text']!r}: {phrase['threshold_hits']}/{phrase['examples']} | "
            f"min={phrase['minimum_score']:.3f} ort={phrase['mean_score']:.3f} "
            f"max={phrase['maximum_score']:.3f}"
        )
    print(
        f"\nPozitif yakalama: {summary['positive_hits']}/{summary['positive_examples']}\n"
        f"Negatif yanlış tetikleme: {summary['false_positive_hits']}/{summary['negative_examples']}"
    )
    if args.json_output:
        atomic_write_json(args.json_output.resolve(), summary)
        print(f"JSON özeti: {args.json_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
