from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import wake_word


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hey_orbit.onnx"
METRICS_PATH = ROOT / "models" / "hey_orbit_metrics.json"
TEST_REPORT_PATH = ROOT / "models" / "evaluation" / "hey_orbit_test.json"
ALL_REPORT_PATH = ROOT / "models" / "evaluation" / "hey_orbit_all.json"


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON nesnesi bekleniyordu: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RepositoryMetadataTests(unittest.TestCase):
    def test_tracked_metrics_and_reports_belong_to_current_model(self) -> None:
        expected_hash = sha256_file(MODEL_PATH)
        metrics = load_json(METRICS_PATH)
        test_report = load_json(TEST_REPORT_PATH)
        all_report = load_json(ALL_REPORT_PATH)

        self.assertEqual(metrics["model_sha256"], expected_hash)
        self.assertEqual(test_report["model_sha256"], expected_hash)
        self.assertEqual(all_report["model_sha256"], expected_hash)

    def test_streaming_summaries_match_detailed_reports(self) -> None:
        metrics = load_json(METRICS_PATH)["deterministic_streaming_evaluation"]
        self.assertIsInstance(metrics, dict)
        for name, path in (("test", TEST_REPORT_PATH), ("all", ALL_REPORT_PATH)):
            report = load_json(path)
            summary = metrics[name]  # type: ignore[index]
            self.assertEqual(summary["positive_hits"], report["positive_hits"])
            self.assertEqual(summary["positive_examples"], report["positive_examples"])
            self.assertEqual(summary["false_positive_hits"], report["false_positive_hits"])
            self.assertEqual(summary["negative_examples"], report["negative_examples"])
            self.assertEqual(summary["evaluation_set_sha256"], report["evaluation_set_sha256"])

    def test_evaluation_protocol_matches_runtime_defaults(self) -> None:
        for path in (TEST_REPORT_PATH, ALL_REPORT_PATH):
            report = load_json(path)
            self.assertEqual(report["schema_version"], 2)
            self.assertEqual(report["evaluation_protocol"], "openwakeword-streaming-v1")
            self.assertEqual(report["threshold"], wake_word.DEFAULT_THRESHOLD)
            self.assertEqual(
                report["warmup_ms"],
                round(
                    wake_word.MODEL_PRIME_CHUNKS
                    * wake_word.CHUNK_DURATION_SECONDS
                    * 1000
                ),
            )
            self.assertEqual(
                report["built_in_wav_tail_ms"],
                round(wake_word.WAV_TAIL_DURATION_SECONDS * 1000),
            )

    def test_accepted_model_report_is_not_mislabeled_as_independent(self) -> None:
        report = load_json(TEST_REPORT_PATH)
        self.assertEqual(
            report["evaluation_role"],
            "post_training_synthetic_regression_reference",
        )
        self.assertIs(report["independent_holdout"], False)
        self.assertIs(report["dataset_generated_after_model"], True)


if __name__ == "__main__":
    unittest.main()
