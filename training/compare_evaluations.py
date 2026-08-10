#!/usr/bin/env python3
"""Aday akış raporunu recall-korumalı kabul kurallarıyla karşılaştır."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:  # Hem ``python training/...`` hem modül olarak içe aktarma desteği.
    from .common import evaluation_set_sha256, load_manifest, sha256_file
except ImportError:
    from common import evaluation_set_sha256, load_manifest, sha256_file


EXPECTED_SCHEMA_VERSION = 2
EXPECTED_EVALUATION_PROTOCOL = "openwakeword-streaming-v1"


def load_report(path: Path) -> dict[str, object]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Geçersiz JSON raporu: {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"Rapor bir JSON nesnesi olmalı: {path}")
    return report


def phrase_hits(report: dict[str, object], label: int) -> dict[str, tuple[int, int]]:
    rows = report.get("per_phrase")
    if not isinstance(rows, list):
        raise ValueError("Raporda 'per_phrase' listesi bulunamadı")
    result: dict[str, tuple[int, int]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("label") != label:
            continue
        text = row.get("text")
        hits = row.get("threshold_hits")
        examples = row.get("examples")
        if not isinstance(text, str) or not isinstance(hits, int) or not isinstance(examples, int):
            raise ValueError("Geçersiz ifade metriği")
        if examples < 0 or hits < 0 or hits > examples:
            raise ValueError(f"Geçersiz ifade sayaçları: {text!r} ({hits}/{examples})")
        if text in result:
            raise ValueError(f"Raporda yinelenen ifade metriği var: etiket={label}, {text!r}")
        result[text] = (hits, examples)
    return result


def report_consistency_failures(report: dict[str, object], name: str) -> list[str]:
    """Özet sayaçların ifade satırlarıyla iç tutarlılığını denetle."""

    failures: list[str] = []
    for label, examples_key, hits_key in (
        (1, "positive_examples", "positive_hits"),
        (0, "negative_examples", "false_positive_hits"),
    ):
        try:
            rows = phrase_hits(report, label)
        except ValueError as exc:
            failures.append(f"{name} raporu: {exc}")
            continue
        row_examples = sum(examples for _, examples in rows.values())
        row_hits = sum(hits for hits, _ in rows.values())
        if report.get(examples_key) != row_examples:
            failures.append(
                f"{name} raporunda {examples_key} ifade satırlarıyla tutarsız"
            )
        if report.get(hits_key) != row_hits:
            failures.append(f"{name} raporunda {hits_key} ifade satırlarıyla tutarsız")
    matrix = report.get("confusion_matrix")
    positive_examples = report.get("positive_examples")
    negative_examples = report.get("negative_examples")
    positive_hits = report.get("positive_hits")
    false_positive_hits = report.get("false_positive_hits")
    if not isinstance(matrix, dict):
        failures.append(f"{name} raporunda confusion_matrix yok")
    elif all(
        isinstance(value, int)
        for value in (
            positive_examples,
            negative_examples,
            positive_hits,
            false_positive_hits,
        )
    ):
        expected_matrix = {
            "tn": negative_examples - false_positive_hits,
            "fp": false_positive_hits,
            "fn": positive_examples - positive_hits,
            "tp": positive_hits,
        }
        if matrix != expected_matrix:
            failures.append(f"{name} raporunda confusion_matrix sayaçlarla tutarsız")
    return failures


def compare_reports(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> list[str]:
    """Boş liste adayın kabul edilebilir olduğunu gösterir."""

    failures = report_consistency_failures(baseline, "Baseline")
    failures.extend(report_consistency_failures(candidate, "Aday"))
    for name, report in (("Baseline", baseline), ("Aday", candidate)):
        if report.get("schema_version") != EXPECTED_SCHEMA_VERSION:
            failures.append(
                f"{name} schema_version={report.get('schema_version')!r}; "
                f"beklenen {EXPECTED_SCHEMA_VERSION}"
            )
        if report.get("evaluation_protocol") != EXPECTED_EVALUATION_PROTOCOL:
            failures.append(
                f"{name} evaluation_protocol={report.get('evaluation_protocol')!r}; "
                f"beklenen {EXPECTED_EVALUATION_PROTOCOL!r}"
            )
    if baseline.get("split") != "test" or candidate.get("split") != "test":
        failures.append("Kabul karşılaştırması yalnız 'test' bölmesiyle yapılabilir")
    for key in (
        "schema_version",
        "evaluation_protocol",
        "split",
        "threshold",
        "warmup_ms",
        "built_in_wav_tail_ms",
        "additional_tail_ms",
        "manifest_sha256",
        "evaluation_set_sha256",
        "positive_examples",
        "negative_examples",
    ):
        if key not in baseline or key not in candidate:
            failures.append(f"Karşılaştırma alanı eksik: {key!r}")
            continue
        if baseline.get(key) != candidate.get(key):
            failures.append(
                f"Karşılaştırılamayan rapor alanı {key!r}: "
                f"{baseline.get(key)!r} != {candidate.get(key)!r}"
            )

    baseline_positive = baseline.get("positive_hits")
    candidate_positive = candidate.get("positive_hits")
    if not isinstance(baseline_positive, int) or not isinstance(candidate_positive, int):
        failures.append("Toplam pozitif yakalama sayıları eksik")
    elif candidate_positive < baseline_positive:
        failures.append(
            f"Toplam pozitif yakalama geriledi: {baseline_positive} -> {candidate_positive}"
        )
    candidate_positive_examples = candidate.get("positive_examples")
    if (
        isinstance(candidate_positive, int)
        and isinstance(candidate_positive_examples, int)
        and candidate_positive != candidate_positive_examples
    ):
        failures.append(
            "Aday test setindeki bütün hedef örneklerini yakalamıyor: "
            f"{candidate_positive}/{candidate_positive_examples}"
        )

    baseline_fp = baseline.get("false_positive_hits")
    candidate_fp = candidate.get("false_positive_hits")
    if not isinstance(baseline_fp, int) or not isinstance(candidate_fp, int):
        failures.append("Yanlış pozitif sayıları eksik")
    elif candidate_fp > baseline_fp:
        failures.append(f"Yanlış pozitif arttı: {baseline_fp} -> {candidate_fp}")

    try:
        baseline_phrases = phrase_hits(baseline, label=1)
        candidate_phrases = phrase_hits(candidate, label=1)
    except ValueError as exc:
        failures.append(f"Pozitif ifade metrikleri karşılaştırılamadı: {exc}")
        return failures
    if baseline_phrases.keys() != candidate_phrases.keys():
        failures.append("Pozitif ifade listeleri aynı değil")
    else:
        for phrase, (baseline_hits, baseline_examples) in baseline_phrases.items():
            candidate_hits, candidate_examples = candidate_phrases[phrase]
            if candidate_examples != baseline_examples:
                failures.append(
                    f"{phrase!r} örnek sayısı değişti: "
                    f"{baseline_examples} -> {candidate_examples}"
                )
            elif candidate_hits < baseline_hits:
                failures.append(
                    f"{phrase!r} yakalaması geriledi: {baseline_hits} -> {candidate_hits}"
                )

    try:
        baseline_negatives = phrase_hits(baseline, label=0)
        candidate_negatives = phrase_hits(candidate, label=0)
    except ValueError as exc:
        failures.append(f"Negatif ifade metrikleri karşılaştırılamadı: {exc}")
        return failures
    if baseline_negatives.keys() != candidate_negatives.keys():
        failures.append("Negatif ifade listeleri aynı değil")
    else:
        for phrase, (baseline_hits, baseline_examples) in baseline_negatives.items():
            candidate_hits, candidate_examples = candidate_negatives[phrase]
            if candidate_examples != baseline_examples:
                failures.append(
                    f"Negatif {phrase!r} örnek sayısı değişti: "
                    f"{baseline_examples} -> {candidate_examples}"
                )
            elif candidate_hits > baseline_hits:
                failures.append(
                    f"Negatif {phrase!r} yanlış tetiklemesi arttı: "
                    f"{baseline_hits} -> {candidate_hits}"
                )
    return failures


def verify_report_artifacts(
    report: dict[str, object],
    *,
    expected_model: Path,
) -> list[str]:
    """Raporun hâlâ diskteki aynı model, manifest ve WAV'lara ait olduğunu doğrula."""

    failures: list[str] = []
    expected_model = expected_model.resolve()
    model_hash = report.get("model_sha256")
    if not isinstance(model_hash, str):
        failures.append("Raporda geçerli model_sha256 yok")
    elif not expected_model.is_file():
        failures.append(f"Model dosyası bulunamadı: {expected_model}")
    elif sha256_file(expected_model) != model_hash:
        failures.append(f"Model hash'i raporla eşleşmiyor: {expected_model}")

    manifest_value = report.get("manifest")
    manifest_hash = report.get("manifest_sha256")
    if not isinstance(manifest_value, str) or not manifest_value:
        failures.append("Raporda geçerli manifest yolu yok")
        return failures
    manifest_path = Path(manifest_value).resolve()
    if not manifest_path.is_file():
        failures.append(f"Manifest bulunamadı: {manifest_path}")
        return failures
    if not isinstance(manifest_hash, str) or sha256_file(manifest_path) != manifest_hash:
        failures.append(f"Manifest hash'i raporla eşleşmiyor: {manifest_path}")
        return failures

    try:
        manifest = load_manifest(manifest_path)
        split = report.get("split")
        selected = [item for item in manifest if split == "all" or item.split == split]
        actual_set_hash = evaluation_set_sha256(selected)
    except (FileNotFoundError, OSError, ValueError) as exc:
        failures.append(f"Değerlendirme seti doğrulanamadı: {exc}")
        return failures
    if actual_set_hash != report.get("evaluation_set_sha256"):
        failures.append("WAV içerikleri/metadata rapordaki değerlendirme setiyle eşleşmiyor")
    return failures


def verify_training_metrics(
    metrics_path: Path,
    candidate_model: Path,
    candidate_report: dict[str, object],
) -> list[str]:
    """Eğitim metriklerini aynı aday ONNX, manifest ve test setine bağla."""

    try:
        metrics = load_report(metrics_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return [f"Aday eğitim metrikleri okunamadı: {exc}"]
    expected_hash = metrics.get("model_sha256")
    if not isinstance(expected_hash, str):
        return ["Aday eğitim metriklerinde model_sha256 yok"]
    if not candidate_model.is_file() or sha256_file(candidate_model) != expected_hash:
        return ["Aday eğitim metrikleri verilen ONNX modeliyle eşleşmiyor"]
    if metrics.get("evaluation_split") != "test":
        return ["Aday eğitim metriklerinin evaluation_split alanı 'test' olmalı"]
    train_splits = metrics.get("train_splits")
    if not isinstance(train_splits, list) or "test" in train_splits:
        return ["Aday eğitim metriklerinde test bölmesi eğitimden ayrı değil"]
    if metrics.get("manifest_sha256") != candidate_report.get("manifest_sha256"):
        return ["Aday eğitim metrikleri akış raporuyla aynı manifeste ait değil"]
    if metrics.get("evaluation_set_sha256") != candidate_report.get("evaluation_set_sha256"):
        return ["Aday eğitim metrikleri akış raporuyla aynı test WAV'larına ait değil"]
    manifest_value = candidate_report.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        return ["Aday akış raporunda manifest yolu yok"]
    try:
        manifest = load_manifest(Path(manifest_value).resolve())
        current_training_items = [item for item in manifest if item.split in train_splits]
        current_test_items = [item for item in manifest if item.split == "test"]
        current_training_hash = evaluation_set_sha256(current_training_items)
        current_test_hash = evaluation_set_sha256(current_test_items)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return [f"Aday eğitim/test seti yeniden doğrulanamadı: {exc}"]
    if metrics.get("training_set_sha256") != current_training_hash:
        return ["Aday eğitim metrikleri mevcut eğitim WAV setiyle eşleşmiyor"]
    if metrics.get("evaluation_set_sha256") != current_test_hash:
        return ["Aday eğitim metrikleri mevcut test WAV setiyle eşleşmiyor"]
    return []


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--candidate-training-metrics", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    baseline = load_report(args.baseline)
    candidate = load_report(args.candidate)
    failures = compare_reports(baseline, candidate)
    failures.extend(verify_report_artifacts(baseline, expected_model=args.baseline_model))
    failures.extend(verify_report_artifacts(candidate, expected_model=args.candidate_model))
    failures.extend(
        verify_training_metrics(
            args.candidate_training_metrics,
            args.candidate_model.resolve(),
            candidate,
        )
    )
    if failures:
        print("ADAY REDDEDİLDİ:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(
        "ADAY KABUL: Model/veri kimliği doğrulandı; hedef yakalama gerilemedi "
        "ve hiçbir negatif ifade kötüleşmedi."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
