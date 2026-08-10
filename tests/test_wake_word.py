from __future__ import annotations

import argparse
import contextlib
import io
import subprocess
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import wake_word


class ParseArgsTests(unittest.TestCase):
    def parse_error(self, *arguments: str) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                wake_word.parse_args(arguments)
        self.assertEqual(raised.exception.code, 2)

    def test_defaults_keep_existing_detection_sensitivity(self) -> None:
        args = wake_word.parse_args(())

        self.assertEqual(args.threshold, 0.70)
        self.assertEqual(args.confirmation_frames, 1)
        self.assertEqual(args.cooldown, 2.0)
        self.assertEqual(args.vad_threshold, 0.10)
        self.assertEqual(args.release_threshold, 0.30)
        self.assertEqual(args.rearm_frames, 5)

    def test_wav_and_device_are_mutually_exclusive(self) -> None:
        self.parse_error("--wav", "sample.wav", "--device", "hw:1,0")

    def test_rejects_non_finite_numeric_arguments(self) -> None:
        for option, value in (
            ("--threshold", "nan"),
            ("--threshold", "inf"),
            ("--cooldown", "nan"),
            ("--cooldown", "inf"),
            ("--release-threshold", "nan"),
            ("--release-threshold", "inf"),
            ("--vad-threshold", "nan"),
            ("--vad-threshold", "inf"),
        ):
            with self.subTest(option=option, value=value):
                self.parse_error(option, value)

    def test_rejects_invalid_confirmation_frame_count(self) -> None:
        self.parse_error("--confirmation-frames", "0")

    def test_rejects_invalid_rearm_settings(self) -> None:
        self.parse_error("--release-threshold", "0.70")
        self.parse_error("--release-threshold", "-0.01")
        self.parse_error("--rearm-frames", "0")
        self.parse_error("--vad-threshold", "-0.01")
        self.parse_error("--vad-threshold", "1.01")


class DetectionGateTests(unittest.TestCase):
    def test_default_gate_triggers_on_one_threshold_frame(self) -> None:
        gate = wake_word.DetectionGate()

        self.assertTrue(gate.observe("hey_orbit", 0.70, 0.08))

    def test_optional_confirmation_requires_consecutive_frames(self) -> None:
        gate = wake_word.DetectionGate(
            threshold=0.70,
            cooldown_seconds=0,
            confirmation_frames=2,
        )

        self.assertFalse(gate.observe("hey_orbit", 0.80, 0.08))
        self.assertFalse(gate.observe("hey_orbit", 0.20, 0.16))
        self.assertFalse(gate.observe("hey_orbit", 0.80, 0.24))
        self.assertTrue(gate.observe("hey_orbit", 0.75, 0.32))

    def test_confirmation_does_not_mix_model_names(self) -> None:
        gate = wake_word.DetectionGate(
            threshold=0.70,
            cooldown_seconds=0,
            confirmation_frames=2,
        )

        self.assertFalse(gate.observe("first", 0.90, 0.08))
        self.assertFalse(gate.observe("second", 0.90, 0.16))
        self.assertTrue(gate.observe("second", 0.90, 0.24))

    def test_cooldown_does_not_accumulate_candidate_frames(self) -> None:
        gate = wake_word.DetectionGate(
            threshold=0.70,
            cooldown_seconds=2.0,
            confirmation_frames=2,
        )

        self.assertFalse(gate.observe("hey_orbit", 0.90, 0.08))
        self.assertTrue(gate.observe("hey_orbit", 0.90, 0.16))
        for timestamp in (0.24, 0.32, 0.40, 0.48, 0.56):
            self.assertFalse(gate.observe("hey_orbit", 0.10, timestamp))
        self.assertFalse(gate.observe("hey_orbit", 0.90, 1.00))
        self.assertFalse(gate.observe("hey_orbit", 0.90, 2.16))
        self.assertTrue(gate.observe("hey_orbit", 0.90, 2.24))

    def test_exact_cooldown_boundary_is_not_lost_to_float_rounding(self) -> None:
        gate = wake_word.DetectionGate(cooldown_seconds=2.0)

        self.assertTrue(gate.observe("hey_orbit", 0.90, 0.32))
        for timestamp in (0.40, 0.48, 0.56, 0.64, 0.72):
            self.assertFalse(gate.observe("hey_orbit", 0.10, timestamp))
        # 2.32 - 0.32 ikili kayan noktada 2.0'ın çok az altında kalabilir.
        self.assertTrue(gate.observe("hey_orbit", 0.90, 2.32))

    def test_continuous_high_score_cannot_retrigger_after_cooldown(self) -> None:
        gate = wake_word.DetectionGate(cooldown_seconds=2.0)

        self.assertTrue(gate.observe("hey_orbit", 0.90, 0.08))
        for timestamp in (0.16, 1.00, 2.08, 3.00, 10.00):
            self.assertFalse(gate.observe("hey_orbit", 0.90, timestamp))

    def test_rearm_requires_consecutive_low_scores(self) -> None:
        gate = wake_word.DetectionGate(cooldown_seconds=0, rearm_frames=3)

        self.assertTrue(gate.observe("hey_orbit", 0.90, 0.08))
        self.assertFalse(gate.observe("hey_orbit", 0.10, 0.16))
        self.assertFalse(gate.observe("hey_orbit", 0.10, 0.24))
        self.assertFalse(gate.observe("hey_orbit", 0.40, 0.32))
        self.assertFalse(gate.observe("hey_orbit", 0.10, 0.40))
        self.assertFalse(gate.observe("hey_orbit", 0.10, 0.48))
        self.assertFalse(gate.observe("hey_orbit", 0.10, 0.56))
        self.assertTrue(gate.observe("hey_orbit", 0.90, 0.64))

    def test_continuous_speech_prevents_rearm(self) -> None:
        gate = wake_word.DetectionGate(
            cooldown_seconds=0,
            rearm_frames=3,
            rearm_vad_threshold=0.10,
        )

        self.assertTrue(gate.observe("hey_orbit", 0.90, 0.08, speech_score=0.80))
        for timestamp in (0.16, 0.24, 0.32, 0.40, 0.48):
            self.assertFalse(
                gate.observe("hey_orbit", 0.0, timestamp, speech_score=0.80)
            )
        for timestamp in (2.08, 4.08, 6.08, 8.08):
            self.assertFalse(
                gate.observe("hey_orbit", 0.90, timestamp, speech_score=0.80)
            )
        for timestamp in (8.16, 8.24, 8.32):
            self.assertFalse(
                gate.observe("hey_orbit", 0.0, timestamp, speech_score=0.01)
            )
        self.assertTrue(gate.observe("hey_orbit", 0.90, 8.40, speech_score=0.80))

    def test_rejects_non_finite_observations(self) -> None:
        gate = wake_word.DetectionGate()

        with self.assertRaises(ValueError):
            gate.observe("hey_orbit", float("nan"), 0.08)
        with self.assertRaises(ValueError):
            gate.observe("hey_orbit", 0.90, float("inf"))
        with self.assertRaises(ValueError):
            gate.observe("hey_orbit", 0.90, 0.08, speech_score=float("nan"))


class ModelPrimingTests(unittest.TestCase):
    def test_reset_is_followed_by_full_silent_context(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.reset_calls = 0
                self.frames: list[np.ndarray] = []

            def reset(self) -> None:
                self.reset_calls += 1

            def predict(self, samples: np.ndarray) -> dict[str, float]:
                self.frames.append(samples.copy())
                return {"hey_orbit": 0.0}

        model = FakeModel()

        wake_word.reset_and_prime_model(model)  # type: ignore[arg-type]

        self.assertEqual(model.reset_calls, 1)
        self.assertEqual(len(model.frames), wake_word.MODEL_PRIME_CHUNKS)
        for samples in model.frames:
            self.assertEqual(samples.dtype, np.int16)
            self.assertEqual(samples.shape, (wake_word.CHUNK_SAMPLES,))
            self.assertFalse(np.any(samples))

    def test_reset_also_clears_vad_state_and_history(self) -> None:
        class FakeBuffer(list[float]):
            def clear(self) -> None:
                super().clear()

        class FakeVad:
            def __init__(self) -> None:
                self.reset_calls = 0
                self.prediction_buffer = FakeBuffer([0.8, 0.9])

            def reset_states(self) -> None:
                self.reset_calls += 1

        class FakeModel:
            def __init__(self) -> None:
                self.vad = FakeVad()

            def reset(self) -> None:
                pass

            def predict(self, _samples: np.ndarray) -> dict[str, float]:
                return {"hey_orbit": 0.0}

        model = FakeModel()

        wake_word.reset_and_prime_model(model)  # type: ignore[arg-type]

        self.assertEqual(model.vad.reset_calls, 1)
        self.assertEqual(model.vad.prediction_buffer, [])

    def test_post_detection_prime_preserves_live_vad_context(self) -> None:
        class FakeVad:
            def __init__(self) -> None:
                self.reset_calls = 0
                self.prediction_buffer = [0.72, 0.81, 0.76]

            def reset_states(self) -> None:
                self.reset_calls += 1

        class FakeModel:
            def __init__(self) -> None:
                self.vad = FakeVad()
                self.vad_threshold = 0.10
                self.predict_calls = 0

            def reset(self) -> None:
                pass

            def predict(self, _samples: np.ndarray) -> dict[str, float]:
                self.predict_calls += 1
                if self.vad_threshold > 0:
                    self.vad.prediction_buffer.append(0.0)
                return {"hey_orbit": 0.0}

        model = FakeModel()

        wake_word.reset_and_prime_model(  # type: ignore[arg-type]
            model,
            preserve_vad_context=True,
        )

        self.assertEqual(model.vad.reset_calls, 0)
        self.assertEqual(model.vad.prediction_buffer, [0.72, 0.81, 0.76])
        self.assertEqual(model.vad_threshold, 0.10)
        self.assertEqual(model.predict_calls, wake_word.MODEL_PRIME_CHUNKS)


class ModelConstructionTests(unittest.TestCase):
    def test_build_model_attaches_repository_vad_when_enabled(self) -> None:
        fake_model = mock.Mock()
        fake_vad = mock.Mock()

        with (
            mock.patch.object(wake_word, "Model", return_value=fake_model),
            mock.patch.object(wake_word, "VAD", return_value=fake_vad) as vad_class,
        ):
            result = wake_word.build_model(
                wake_word.DEFAULT_MODEL,
                vad_threshold=0.10,
            )

        self.assertIs(result, fake_model)
        self.assertEqual(fake_model.vad_threshold, 0.10)
        self.assertIs(fake_model.vad, fake_vad)
        vad_class.assert_called_once_with(
            model_path=str(wake_word.MODELS_DIR / "silero_vad.onnx")
        )


class AudioTimelineTests(unittest.TestCase):
    def test_wav_frame_time_uses_audio_duration(self) -> None:
        self.assertAlmostEqual(wake_word.wav_frame_end_time(0), 0.08)
        self.assertAlmostEqual(wake_word.wav_frame_end_time(25), 2.08)

    def test_wav_main_applies_cooldown_on_audio_timeline(self) -> None:
        args = argparse.Namespace(
            model=Path("models/hey_orbit.onnx"),
            threshold=0.70,
            cooldown=2.0,
            device=None,
            wav=Path("sample.wav"),
            confirmation_frames=1,
            release_threshold=0.30,
            rearm_frames=5,
            vad_threshold=0.0,
            list_devices=False,
            quiet=True,
        )

        class FakeModel:
            def __init__(self) -> None:
                self.frame_index = 0

            def predict(self, _samples: np.ndarray) -> dict[str, float]:
                score = 0.90 if self.frame_index in (0, 26) else 0.0
                self.frame_index += 1
                return {"hey_orbit": score}

        frames = [
            np.zeros(wake_word.CHUNK_SAMPLES, dtype=np.int16)
            for _ in range(27)
        ]
        output = io.StringIO()
        with (
            mock.patch.object(wake_word, "parse_args", return_value=args),
            mock.patch.object(wake_word, "build_model", return_value=FakeModel()),
            mock.patch.object(wake_word, "wav_chunks", return_value=iter(frames)),
            mock.patch.object(wake_word, "reset_and_prime_model"),
            mock.patch.object(wake_word.time, "monotonic", return_value=0.0),
            contextlib.redirect_stdout(output),
        ):
            return_code = wake_word.main()

        self.assertEqual(return_code, 0)
        self.assertEqual(output.getvalue().count("UYANDIRMA SÖZCÜĞÜ ALGILANDI"), 2)


class MicrophoneCleanupTests(unittest.TestCase):
    def test_killed_arecord_process_is_waited_for(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO(b"\0" * wake_word.CHUNK_BYTES)
                self.returncode = None
                self.calls: list[str] = []
                self.wait_count = 0

            def poll(self) -> None:
                return None

            def terminate(self) -> None:
                self.calls.append("terminate")

            def wait(self, timeout: float | None = None) -> int:
                self.calls.append("wait")
                self.wait_count += 1
                if self.wait_count == 1:
                    raise subprocess.TimeoutExpired("arecord", timeout)
                return -9

            def kill(self) -> None:
                self.calls.append("kill")

        process = FakeProcess()
        with (
            mock.patch.object(wake_word.shutil, "which", return_value="/usr/bin/arecord"),
            mock.patch.object(wake_word.subprocess, "Popen", return_value=process),
        ):
            chunks = wake_word.microphone_chunks(None)
            samples = next(chunks)
            chunks.close()

        self.assertEqual(samples.shape, (wake_word.CHUNK_SAMPLES,))
        self.assertEqual(process.calls, ["terminate", "wait", "kill", "wait"])


class PredictionTests(unittest.TestCase):
    def test_best_prediction_rejects_empty_or_non_finite_results(self) -> None:
        with self.assertRaises(RuntimeError):
            wake_word.best_prediction({})
        with self.assertRaises(ValueError):
            wake_word.best_prediction({"hey_orbit": float("nan")})

    def test_aligned_vad_score_matches_openwakeword_window(self) -> None:
        model = mock.Mock()
        model.vad.prediction_buffer = list(range(10))

        self.assertEqual(wake_word.aligned_vad_score(model), 5.0)


if __name__ == "__main__":
    unittest.main()
