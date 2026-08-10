from __future__ import annotations

import tempfile
import unittest
import warnings
import contextlib
import io
import json
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

from training.common import (
    ManifestItem,
    assign_splits,
    evaluation_set_sha256,
    sha256_file,
    validate_manifest_records,
)
from training.compare_evaluations import compare_reports, verify_training_metrics
from training.generate_turkish_samples import parse_args as parse_generator_args
from training.train_hey_orbit import (
    WINDOW,
    export_onnx,
    fit_audio_window,
    parse_args as parse_training_args,
    validate_onnx_parity,
)


def item(group: str, *, label: int = 0, text: str = "benzer ifade") -> ManifestItem:
    relative = f"training/data/negative/{group}.wav"
    return ManifestItem(
        path=relative,
        absolute_path=Path("/") / relative,
        label=label,
        text=text,
        split="",
        group=group,
        voice=None,
    )


class ManifestSplitTests(unittest.TestCase):
    def test_three_groups_cover_every_split_deterministically(self) -> None:
        source = [item(f"speaker-{index}") for index in range(3)]
        first = assign_splits(source, seed=42)
        second = assign_splits(list(reversed(source)), seed=42)
        self.assertEqual({entry.split for entry in first}, {"train", "dev", "test"})
        self.assertEqual(
            {entry.group: entry.split for entry in first},
            {entry.group: entry.split for entry in second},
        )

    def test_group_members_never_cross_splits(self) -> None:
        source = []
        for group_index in range(6):
            source.extend([item(f"group-{group_index}"), item(f"group-{group_index}")])
        # Manifest yolları normalde benzersizdir; assign_splits yalnızca kaynak
        # grup davranışını test ettiği için burada aynı kaydın iki türevi var.
        assigned = assign_splits(source, seed=7)
        for group_index in range(6):
            splits = {entry.split for entry in assigned if entry.group == f"group-{group_index}"}
            self.assertEqual(len(splits), 1)

    def test_manifest_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "training" / "data"
            data_dir.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "güvenli"):
                validate_manifest_records(
                    [
                        {
                            "path": "training/data/../secret.wav",
                            "label": 0,
                            "text": "hayır",
                            "split": "train",
                            "group": "bad",
                        }
                    ],
                    project_root=root,
                    data_dir=data_dir,
                    require_files=False,
                    allow_legacy_split=False,
                )

    def test_manifest_rejects_negative_split_gap_when_avoidable(self) -> None:
        records = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "training" / "data"
            data_dir.mkdir(parents=True)
            for index in range(3):
                records.append(
                    {
                        "path": f"training/data/negative/{index}.wav",
                        "label": 0,
                        "text": "hey robot",
                        "split": "train" if index < 2 else "dev",
                        "group": f"negative-{index}",
                    }
                )

    def test_manifest_rejects_positive_phrase_split_gap_when_avoidable(self) -> None:
        records = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "training" / "data"
            data_dir.mkdir(parents=True)
            for index in range(3):
                records.append(
                    {
                        "path": f"training/data/positive/{index}.wav",
                        "label": 1,
                        "text": "heyorbit",
                        "split": "train" if index < 2 else "dev",
                        "group": f"positive-{index}",
                    }
                )
            with self.assertRaisesRegex(ValueError, "kapsamı eksik"):
                validate_manifest_records(
                    records,
                    project_root=root,
                    data_dir=data_dir,
                    require_files=False,
                    allow_legacy_split=False,
                )
            with self.assertRaisesRegex(ValueError, "kapsamı eksik"):
                validate_manifest_records(
                    records,
                    project_root=root,
                    data_dir=data_dir,
                    require_files=False,
                    allow_legacy_split=False,
                )


class EvaluationComparisonTests(unittest.TestCase):
    @staticmethod
    def report(
        *,
        positive_hits: int,
        false_positive_hits: int,
        phrase_hits: int,
    ) -> dict[str, object]:
        return {
            "schema_version": 2,
            "evaluation_protocol": "openwakeword-streaming-v1",
            "split": "test",
            "threshold": 0.70,
            "warmup_ms": 2080,
            "built_in_wav_tail_ms": 400,
            "additional_tail_ms": 0,
            "manifest_sha256": "same-manifest",
            "evaluation_set_sha256": "same-wavs",
            "positive_examples": 3,
            "negative_examples": 2,
            "positive_hits": positive_hits,
            "false_positive_hits": false_positive_hits,
            "confusion_matrix": {
                "tn": 2 - false_positive_hits,
                "fp": false_positive_hits,
                "fn": 3 - positive_hits,
                "tp": positive_hits,
            },
            "per_phrase": [
                {
                    "label": 1,
                    "text": "heyorbit",
                    "examples": 3,
                    "threshold_hits": phrase_hits,
                },
                {
                    "label": 0,
                    "text": "hey robot",
                    "examples": 2,
                    "threshold_hits": false_positive_hits,
                },
            ],
        }

    def test_accepts_non_regressing_candidate(self) -> None:
        baseline = self.report(positive_hits=3, false_positive_hits=1, phrase_hits=3)
        candidate = self.report(positive_hits=3, false_positive_hits=0, phrase_hits=3)
        self.assertEqual(compare_reports(baseline, candidate), [])

    def test_rejects_recall_regression_even_when_false_positives_improve(self) -> None:
        baseline = self.report(positive_hits=3, false_positive_hits=1, phrase_hits=3)
        candidate = self.report(positive_hits=2, false_positive_hits=0, phrase_hits=2)
        failures = compare_reports(baseline, candidate)
        self.assertTrue(any("pozitif" in failure.lower() for failure in failures))
        self.assertTrue(any("heyorbit" in failure for failure in failures))

    def test_rejects_false_positive_regression(self) -> None:
        baseline = self.report(positive_hits=3, false_positive_hits=0, phrase_hits=3)
        candidate = self.report(positive_hits=3, false_positive_hits=1, phrase_hits=3)
        failures = compare_reports(baseline, candidate)
        self.assertTrue(any("Yanlış pozitif arttı" in failure for failure in failures))

    def test_rejects_negative_phrase_shift_when_total_is_unchanged(self) -> None:
        baseline = self.report(positive_hits=3, false_positive_hits=1, phrase_hits=3)
        candidate = self.report(positive_hits=3, false_positive_hits=1, phrase_hits=3)
        baseline["per_phrase"].append(  # type: ignore[union-attr]
            {"label": 0, "text": "hey orbi", "examples": 1, "threshold_hits": 0}
        )
        baseline["negative_examples"] = 3
        baseline["confusion_matrix"]["tn"] = 2  # type: ignore[index]
        candidate["per_phrase"][1]["threshold_hits"] = 0  # type: ignore[index]
        candidate["per_phrase"].append(  # type: ignore[union-attr]
            {"label": 0, "text": "hey orbi", "examples": 1, "threshold_hits": 1}
        )
        candidate["negative_examples"] = 3
        candidate["confusion_matrix"]["tn"] = 2  # type: ignore[index]

        failures = compare_reports(baseline, candidate)
        self.assertTrue(any("hey orbi" in failure for failure in failures))

    def test_rejects_non_test_reports_for_acceptance(self) -> None:
        baseline = self.report(positive_hits=3, false_positive_hits=0, phrase_hits=3)
        candidate = self.report(positive_hits=3, false_positive_hits=0, phrase_hits=3)
        baseline["split"] = candidate["split"] = "all"
        failures = compare_reports(baseline, candidate)
        self.assertTrue(any("yalnız 'test'" in failure for failure in failures))


class TrainingArgumentTests(unittest.TestCase):
    def parse_error(self, *arguments: str) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_training_args(arguments)
        self.assertEqual(raised.exception.code, 2)

    def test_cannot_overwrite_accepted_model_or_metrics(self) -> None:
        self.parse_error("--output-model", "models/hey_orbit.onnx")
        self.parse_error("--metrics-output", "models/hey_orbit_metrics.json")

    def test_candidate_defaults_preserve_runtime_threshold(self) -> None:
        args = parse_training_args(())
        self.assertEqual(args.threshold, 0.70)

    def test_generator_help_is_side_effect_free_argument_parsing(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_generator_args(("--help",))
        self.assertEqual(raised.exception.code, 0)


class TrainingMetricsVerificationTests(unittest.TestCase):
    def test_metrics_are_bound_to_candidate_and_test_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model_path = directory / "candidate.onnx"
            metrics_path = directory / "metrics.json"
            train_path = directory / "train.wav"
            test_path = directory / "test.wav"
            model_path.write_bytes(b"candidate-model")
            train_path.write_bytes(b"train-audio")
            test_path.write_bytes(b"test-audio")
            items = [
                ManifestItem(
                    path="training/data/positive/train.wav",
                    absolute_path=train_path,
                    label=1,
                    text="heyorbit",
                    split="train",
                    group="train",
                ),
                ManifestItem(
                    path="training/data/positive/test.wav",
                    absolute_path=test_path,
                    label=1,
                    text="heyorbit",
                    split="test",
                    group="test",
                ),
            ]
            training_hash = evaluation_set_sha256([items[0]])
            test_hash = evaluation_set_sha256([items[1]])
            report = {
                "manifest": "training/data/manifest.json",
                "manifest_sha256": "manifest-hash",
                "evaluation_set_sha256": test_hash,
            }
            metrics_path.write_text(
                json.dumps(
                    {
                        "model_sha256": sha256_file(model_path),
                        "evaluation_split": "test",
                        "train_splits": ["train"],
                        "manifest_sha256": "manifest-hash",
                        "training_set_sha256": training_hash,
                        "evaluation_set_sha256": test_hash,
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "training.compare_evaluations.load_manifest",
                return_value=items,
            ):
                self.assertEqual(
                    verify_training_metrics(metrics_path, model_path, report),
                    [],
                )
                model_path.write_bytes(b"different-model")
                self.assertTrue(verify_training_metrics(metrics_path, model_path, report))


class EvaluationSetHashTests(unittest.TestCase):
    def test_hash_includes_actual_audio_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wav_path = Path(temporary) / "sample.wav"
            wav_path.write_bytes(b"first")
            sample = ManifestItem(
                path="training/data/positive/sample.wav",
                absolute_path=wav_path,
                label=1,
                text="heyorbit",
                split="test",
                group="sample",
            )
            first_hash = evaluation_set_sha256([sample])
            wav_path.write_bytes(b"second")
            self.assertNotEqual(first_hash, evaluation_set_sha256([sample]))

class AugmentationTests(unittest.TestCase):
    def test_long_positive_is_fitted_without_cropping_target(self) -> None:
        rng = np.random.default_rng(123)
        audio = np.ones(WINDOW + 12_000, dtype=np.float32) * 0.25
        fitted = fit_audio_window(audio, rng, positive=True)
        self.assertEqual(fitted.shape, (WINDOW,))
        active = np.flatnonzero(fitted)
        self.assertGreaterEqual(int(active[0]), 1)
        self.assertLessEqual(int(active[-1]), WINDOW - 2)
        # Tüm uzun klip yeniden örneklendiğinden iki uç da aktiftir; rastgele
        # 2 saniyelik crop ile hedefin bir hecesi atılmaz.
        self.assertGreater(len(active), 30_000)

    def test_nearly_full_positive_never_requests_negative_padding(self) -> None:
        audio = np.ones(WINDOW - 100, dtype=np.float32) * 0.1
        for seed in range(20):
            fitted = fit_audio_window(audio, np.random.default_rng(seed), positive=True)
            self.assertEqual(fitted.shape, (WINDOW,))


class OnnxParityTests(unittest.TestCase):
    def test_export_matches_sklearn_scores(self) -> None:
        rng = np.random.default_rng(9)
        features = rng.normal(size=(24, 16, 96)).astype(np.float32)
        labels = np.asarray([0, 1] * 12, dtype=np.int8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            classifier = MLPClassifier(
                hidden_layer_sizes=(6,),
                max_iter=8,
                random_state=3,
            ).fit(features.reshape(len(features), -1), labels)
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "candidate.onnx"
            export_onnx(classifier, model_path)
            parity = validate_onnx_parity(classifier, model_path, features)
        self.assertEqual(parity["checked_examples"], len(features))
        self.assertLess(float(parity["max_absolute_error"]), 1e-5)


if __name__ == "__main__":
    unittest.main()
