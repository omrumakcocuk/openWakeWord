#!/usr/bin/env python3
"""openWakeWord ile mikrofon veya WAV dosyasından uyandırma sözcüğü algıla."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
import types
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterator, Mapping, Sequence

import numpy as np

# openWakeWord paketinin __init__ dosyası, kullanılmasa bile özel konuşmacı
# doğrulayıcısını ve onun ağır scikit-learn/SciPy bağımlılıklarını içe aktarır.
# Bu uygulama doğrulayıcı kullanmadığı için hafif bir uyumluluk modülü sağlıyoruz.
_verifier_stub = types.ModuleType("openwakeword.custom_verifier_model")


def _verifier_not_enabled(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("Bu uygulamada özel konuşmacı doğrulayıcısı etkin değil")


_verifier_stub.train_custom_verifier = _verifier_not_enabled  # type: ignore[attr-defined]
sys.modules.setdefault("openwakeword.custom_verifier_model", _verifier_stub)

from openwakeword.model import Model
from openwakeword.vad import VAD


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DEFAULT_MODEL = MODELS_DIR / "hey_orbit.onnx"
SAMPLE_RATE_HZ = 16_000
SAMPLE_CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
CHUNK_DURATION_SECONDS = 0.080
CHUNK_SAMPLES = round(SAMPLE_RATE_HZ * CHUNK_DURATION_SECONDS)
CHUNK_BYTES = CHUNK_SAMPLES * SAMPLE_WIDTH_BYTES
WAV_TAIL_DURATION_SECONDS = 0.400
WAV_TAIL_CHUNKS = round(WAV_TAIL_DURATION_SECONDS / CHUNK_DURATION_SECONDS)
MELSPECTROGRAM_CONTEXT_CHUNKS = 10  # 76 mel karesi için yaklaşık 800 ms
MODEL_CONTEXT_CHUNKS = 16  # hey_orbit.onnx modelinin yaklaşık 1,28 sn bağlamı
MODEL_PRIME_CHUNKS = MELSPECTROGRAM_CONTEXT_CHUNKS + MODEL_CONTEXT_CHUNKS
DEFAULT_THRESHOLD = 0.70
DEFAULT_COOLDOWN_SECONDS = 2.0
DEFAULT_CONFIRMATION_FRAMES = 1
DEFAULT_VAD_THRESHOLD = 0.10
DEFAULT_RELEASE_THRESHOLD = 0.30
DEFAULT_REARM_FRAMES = 5
TIME_COMPARISON_ABS_TOLERANCE = 1e-12


@dataclass
class DetectionGate:
    """Skorları tetikleme kararına dönüştüren durum makinesi.

    Varsayılan ``confirmation_frames=1`` mevcut tek-kare davranışını korur.
    Tetiklemeden sonraki histerezis yalnız tekrar algılamaları sınırlar. Daha
    katı ardışık-kare doğrulaması yalnızca CLI'da açıkça istenirse etkinleşir.
    """

    threshold: float = DEFAULT_THRESHOLD
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    confirmation_frames: int = DEFAULT_CONFIRMATION_FRAMES
    release_threshold: float = DEFAULT_RELEASE_THRESHOLD
    rearm_frames: int = DEFAULT_REARM_FRAMES
    rearm_vad_threshold: float | None = None
    _last_detection: float = field(default=-float("inf"), init=False)
    _candidate_name: str | None = field(default=None, init=False)
    _candidate_frames: int = field(default=0, init=False)
    _armed: bool = field(default=True, init=False)
    _release_frames: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        validate_detection_settings(
            self.threshold,
            self.cooldown_seconds,
            self.confirmation_frames,
            self.release_threshold,
            self.rearm_frames,
        )
        if self.rearm_vad_threshold is not None:
            validate_vad_threshold(self.rearm_vad_threshold)

    def observe(
        self,
        name: str,
        score: float,
        timestamp: float,
        speech_score: float | None = None,
    ) -> bool:
        """Bir model skorunu işle ve bu kare tetikliyorsa ``True`` döndür."""

        if not math.isfinite(score):
            raise ValueError("Model skoru sonlu bir sayı olmalı")
        if not math.isfinite(timestamp):
            raise ValueError("Algılama zamanı sonlu bir sayı olmalı")
        if speech_score is not None and not math.isfinite(speech_score):
            raise ValueError("Konuşma etkinliği skoru sonlu bir sayı olmalı")

        # Algılamadan sonra skor belirgin biçimde düşmeden kapıyı tekrar kurma.
        # Böylece sürekli arka plan sesi yüksek skor üretirse cooldown aralıklarıyla
        # art arda yanlış tetikleme oluşmaz. İlk algılama bundan etkilenmez.
        if not self._armed:
            self._clear_candidate()
            speech_is_quiet = self.rearm_vad_threshold is None or (
                speech_score is not None and speech_score < self.rearm_vad_threshold
            )
            if score <= self.release_threshold and speech_is_quiet:
                self._release_frames += 1
                if self._release_frames >= self.rearm_frames:
                    self._armed = True
                    self._release_frames = 0
            else:
                self._release_frames = 0
            return False

        # Cooldown sırasındaki yüksek kareler yeni bir aday biriktirmesin.
        elapsed = timestamp - self._last_detection
        if elapsed < self.cooldown_seconds and not math.isclose(
            elapsed,
            self.cooldown_seconds,
            rel_tol=0.0,
            abs_tol=TIME_COMPARISON_ABS_TOLERANCE,
        ):
            self._clear_candidate()
            return False

        if score < self.threshold:
            self._clear_candidate()
            return False

        if name == self._candidate_name:
            self._candidate_frames += 1
        else:
            self._candidate_name = name
            self._candidate_frames = 1

        if self._candidate_frames < self.confirmation_frames:
            return False

        self._last_detection = timestamp
        self._armed = False
        self._release_frames = 0
        self._clear_candidate()
        return True

    def _clear_candidate(self) -> None:
        self._candidate_name = None
        self._candidate_frames = 0


def validate_detection_settings(
    threshold: float,
    cooldown_seconds: float,
    confirmation_frames: int,
    release_threshold: float = DEFAULT_RELEASE_THRESHOLD,
    rearm_frames: int = DEFAULT_REARM_FRAMES,
) -> None:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("--threshold 0 ile 1 arasında sonlu bir sayı olmalı")
    if not math.isfinite(cooldown_seconds) or cooldown_seconds < 0:
        raise ValueError("--cooldown sonlu ve negatif olmayan bir sayı olmalı")
    if confirmation_frames < 1:
        raise ValueError("--confirmation-frames en az 1 olmalı")
    if not math.isfinite(release_threshold) or not 0 <= release_threshold < threshold:
        raise ValueError(
            "--release-threshold 0 ile --threshold arasında sonlu bir sayı olmalı"
        )
    if rearm_frames < 1:
        raise ValueError("--rearm-frames en az 1 olmalı")


def validate_vad_threshold(vad_threshold: float) -> None:
    if not math.isfinite(vad_threshold) or not 0 <= vad_threshold <= 1:
        raise ValueError("--vad-threshold 0 ile 1 arasında sonlu bir sayı olmalı")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mikrofonda Türkçe 'Hey Orbit' uyandırma ifadesini dinler."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Kullanılacak ONNX modeli (varsayılan: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Algılama eşiği, 0-1 (varsayılan: {DEFAULT_THRESHOLD:.2f})",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        help=(
            "İki algılama arasındaki en kısa süre, saniye "
            f"(varsayılan: {DEFAULT_COOLDOWN_SECONDS:g})"
        ),
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--device",
        help="ALSA kayıt aygıtı; örneğin hw:1,0 (varsayılan sistem aygıtı)",
    )
    source_group.add_argument(
        "--wav",
        type=Path,
        help=f"Mikrofon yerine {SAMPLE_RATE_HZ // 1000} kHz mono WAV dosyasını tara",
    )
    parser.add_argument(
        "--confirmation-frames",
        type=int,
        default=DEFAULT_CONFIRMATION_FRAMES,
        help=(
            "Tetikleme için gereken art arda eşik-üstü kare sayısı; "
            f"1 mevcut hassasiyeti korur (varsayılan: {DEFAULT_CONFIRMATION_FRAMES})"
        ),
    )
    parser.add_argument(
        "--release-threshold",
        type=float,
        default=DEFAULT_RELEASE_THRESHOLD,
        help=(
            "Algılamadan sonra yeniden kurulmak için skorun altına inmesi gereken "
            f"eşik (varsayılan: {DEFAULT_RELEASE_THRESHOLD:.2f})"
        ),
    )
    parser.add_argument(
        "--rearm-frames",
        type=int,
        default=DEFAULT_REARM_FRAMES,
        help=(
            "Yeniden kurulmadan önce gereken art arda düşük skorlu kare sayısı "
            f"(varsayılan: {DEFAULT_REARM_FRAMES})"
        ),
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=DEFAULT_VAD_THRESHOLD,
        help=(
            "Konuşma etkinliği eşiği; 0 filtreyi kapatır "
            f"(varsayılan: {DEFAULT_VAD_THRESHOLD:.2f})"
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="ALSA kayıt aygıtlarını göster ve çık",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Canlı skorları gizle, yalnızca algılamaları göster",
    )
    args = parser.parse_args(argv)
    try:
        validate_detection_settings(
            args.threshold,
            args.cooldown,
            args.confirmation_frames,
            args.release_threshold,
            args.rearm_frames,
        )
        validate_vad_threshold(args.vad_threshold)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def list_devices() -> int:
    if not shutil.which("arecord"):
        print("Hata: 'arecord' bulunamadı. 'alsa-utils' paketini kurun.", file=sys.stderr)
        return 1
    return subprocess.run(["arecord", "-L"], check=False).returncode


def read_exact(stream: BinaryIO, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        part = stream.read(size - len(data))
        if not part:
            break
        data.extend(part)
    return bytes(data)


def microphone_chunks(device: str | None) -> Iterator[np.ndarray]:
    if not shutil.which("arecord"):
        raise RuntimeError("'arecord' bulunamadı; 'alsa-utils' paketini kurun")

    command = [
        "arecord",
        "-q",
        "-t",
        "raw",
        "-f",
        "S16_LE",
        "-r",
        str(SAMPLE_RATE_HZ),
        "-c",
        str(SAMPLE_CHANNELS),
    ]
    if device:
        command.extend(["-D", device])

    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert process.stdout is not None
    try:
        while True:
            raw = read_exact(process.stdout, CHUNK_BYTES)
            if len(raw) != CHUNK_BYTES:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Ses kaydı durdu (arecord çıkış kodu: {process.returncode})"
                    )
                continue
            yield np.frombuffer(raw, dtype=np.int16)
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                # Zombi süreç bırakmamak için kill sonrası mutlaka reap et.
                process.wait()
        else:
            process.wait()


def wav_chunks(path: Path) -> Iterator[np.ndarray]:
    with wave.open(str(path), "rb") as wav_file:
        audio_format = (
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getframerate(),
        )
        if audio_format != (SAMPLE_CHANNELS, SAMPLE_WIDTH_BYTES, SAMPLE_RATE_HZ):
            raise ValueError(
                f"WAV dosyası {SAMPLE_RATE_HZ // 1000} kHz, mono ve 16-bit PCM olmalı "
                f"(mevcut: kanal={audio_format[0]}, bit={audio_format[1] * 8}, "
                f"hız={audio_format[2]} Hz)"
            )
        while raw := wav_file.readframes(CHUNK_SAMPLES):
            samples = np.frombuffer(raw, dtype="<i2")
            if len(samples) < CHUNK_SAMPLES:
                samples = np.pad(samples, (0, CHUNK_SAMPLES - len(samples)))
            yield samples.astype(np.int16, copy=False)
        # Son hecelerin modelin kayan bağlamında skorlanabilmesi için akışı
        # kısa bir sessizlikle tamamla. Mikrofonda bu devam doğal olarak vardır.
        for _ in range(WAV_TAIL_CHUNKS):
            yield np.zeros(CHUNK_SAMPLES, dtype=np.int16)


def build_model(
    model_path: Path,
    vad_threshold: float = DEFAULT_VAD_THRESHOLD,
) -> Model:
    validate_vad_threshold(vad_threshold)
    required = [
        model_path,
        MODELS_DIR / "melspectrogram.onnx",
        MODELS_DIR / "embedding_model.onnx",
    ]
    vad_model_path = MODELS_DIR / "silero_vad.onnx"
    if vad_threshold > 0:
        required.append(vad_model_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Gerekli model dosyaları bulunamadı:\n- "
            + "\n- ".join(missing)
            + "\n'Hey Orbit' modelini models/hey_orbit.onnx adıyla yerleştirin. "
            + "Ayrıntılar için MODEL_EGITIMI.md dosyasına bakın."
        )

    model = Model(
        wakeword_models=[str(model_path)],
        inference_framework="onnx",
        melspec_model_path=str(MODELS_DIR / "melspectrogram.onnx"),
        embedding_model_path=str(MODELS_DIR / "embedding_model.onnx"),
    )
    if vad_threshold > 0:
        # openWakeWord paketindeki varsayılan VAD yolu kurulum biçimine bağlıdır.
        # Repoda hash'i sabitlenmiş yerel modeli açıkça kullan.
        model.vad_threshold = vad_threshold
        model.vad = VAD(model_path=str(vad_model_path))
    return model


def score_as_float(value: object) -> float:
    values = np.asarray(value).reshape(-1)
    if values.size == 0:
        raise ValueError("Model boş bir skor üretti")
    score = float(values[0])
    if not math.isfinite(score):
        raise ValueError("Model sonlu olmayan bir skor üretti")
    return score


def best_prediction(predictions: Mapping[str, object]) -> tuple[str, float]:
    if not predictions:
        raise RuntimeError("Model herhangi bir tahmin üretmedi")
    return max(
        ((name, score_as_float(score)) for name, score in predictions.items()),
        key=lambda item: item[1],
    )


def aligned_vad_score(model: Model) -> float | None:
    """openWakeWord'ün mevcut wake skoru için kullandığı hizalı VAD skorunu döndür."""

    vad = getattr(model, "vad", None)
    if vad is None:
        return None
    frames = list(vad.prediction_buffer)[-7:-4]
    if not frames:
        return 0.0
    return max(float(value) for value in frames)


def reset_and_prime_model(
    model: Model,
    *,
    preserve_vad_context: bool = False,
) -> None:
    """Wake bağlamını sessizlikle prime et; gerekirse gerçek VAD geçmişini koru."""

    model.reset()
    vad = getattr(model, "vad", None)
    if vad is not None and not preserve_vad_context:
        vad.reset_states()
        vad.prediction_buffer.clear()
    # Wake modelini prime eden yapay sessizlik, canlı konuşmanın VAD geçmişini
    # ezmemeli. Aksi halde tetikleme sonrası ortam yanlışlıkla sessiz görünür ve
    # kapı konuşma sürerken yeniden kurulabilir.
    original_vad_threshold = getattr(model, "vad_threshold", 0.0)
    if vad is not None and preserve_vad_context:
        model.vad_threshold = 0.0
    silence = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
    try:
        for _ in range(MODEL_PRIME_CHUNKS):
            model.predict(silence)
    finally:
        if vad is not None and preserve_vad_context:
            model.vad_threshold = original_vad_threshold


def wav_frame_end_time(frame_index: int) -> float:
    """Bir WAV karesinin dosyanın ses zaman çizelgesindeki bitişini döndür."""

    if frame_index < 0:
        raise ValueError("Kare sırası negatif olamaz")
    return (frame_index + 1) * CHUNK_DURATION_SECONDS


def main() -> int:
    args = parse_args()
    try:
        if args.list_devices:
            return list_devices()

        model = build_model(args.model.resolve(), vad_threshold=args.vad_threshold)
        reset_and_prime_model(model)
        chunks = wav_chunks(args.wav.resolve()) if args.wav else microphone_chunks(args.device)
        source = str(args.wav) if args.wav else (args.device or "varsayılan mikrofon")
        print(f"Dinleniyor: {source}")
        print(f"Model: {args.model.name} | eşik: {args.threshold:.2f}")
        if args.vad_threshold > 0:
            print(f"Konuşma etkinliği filtresi: {args.vad_threshold:.2f} (muhafazakâr)")
        else:
            print("Konuşma etkinliği filtresi kapalı")
        if args.confirmation_frames == 1:
            print("Ardışık kare doğrulaması yok (tek kare)")
        else:
            print(f"Ardışık kare doğrulaması: {args.confirmation_frames} kare")
        rearm_condition = f"skor <= {args.release_threshold:.2f}"
        if args.vad_threshold > 0:
            rearm_condition += f" ve VAD < {args.vad_threshold:.2f}"
        print(f"Yeniden kurma: {rearm_condition} için {args.rearm_frames} kare")
        print("Durdurmak için Ctrl+C tuşlarına basın.\n")

        detection_gate = DetectionGate(
            threshold=args.threshold,
            cooldown_seconds=args.cooldown,
            confirmation_frames=args.confirmation_frames,
            release_threshold=args.release_threshold,
            rearm_frames=args.rearm_frames,
            rearm_vad_threshold=(args.vad_threshold if args.vad_threshold > 0 else None),
        )

        for frame_index, samples in enumerate(chunks):
            predictions = model.predict(samples)
            event_time = (
                wav_frame_end_time(frame_index)
                if args.wav
                else time.monotonic()
            )
            best_name, best_score = best_prediction(predictions)
            vad_score = aligned_vad_score(model)

            if detection_gate.observe(
                best_name,
                best_score,
                event_time,
                speech_score=vad_score,
            ):
                vad_text = "" if vad_score is None else f", VAD {vad_score:.3f}"
                print(
                    f"\r[{time.strftime('%H:%M:%S')}] UYANDIRMA SÖZCÜĞÜ ALGILANDI: "
                    f"{best_name} ({best_score:.3f}{vad_text}){' ' * 12}"
                )
                # Algılanan ifadenin ses/model bağlamı sonraki kararı etkilemesin.
                reset_and_prime_model(model, preserve_vad_context=True)
            elif not args.quiet:
                print(
                    f"\r{best_name}: {best_score:.3f}{' ' * 12}",
                    end="",
                    flush=True,
                )
        if args.wav:
            print("\nDosya taraması tamamlandı.")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError, wave.Error) as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nDinleme durduruldu.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
