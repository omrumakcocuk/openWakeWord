#!/usr/bin/env python3
"""openWakeWord ile mikrofon veya WAV dosyasından uyandırma sözcüğü algıla."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import types
import wave
from pathlib import Path
from typing import BinaryIO, Iterator

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


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DEFAULT_MODEL = MODELS_DIR / "hey_orbit.onnx"
CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz
CHUNK_BYTES = CHUNK_SAMPLES * 2  # signed 16-bit mono
WAV_TAIL_CHUNKS = 5  # Dosya sonunda canlı akışı taklit eden 400 ms sessizlik


def parse_args() -> argparse.Namespace:
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
        default=0.70,
        help="Algılama eşiği, 0-1 (varsayılan: 0.70)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="İki algılama arasındaki en kısa süre, saniye (varsayılan: 2)",
    )
    parser.add_argument(
        "--device",
        help="ALSA kayıt aygıtı; örneğin hw:1,0 (varsayılan sistem aygıtı)",
    )
    parser.add_argument("--wav", type=Path, help="Mikrofon yerine 16 kHz mono WAV dosyasını tara")
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
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold 0 ile 1 arasında olmalı")
    if args.cooldown < 0:
        parser.error("--cooldown negatif olamaz")
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
        "16000",
        "-c",
        "1",
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
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


def wav_chunks(path: Path) -> Iterator[np.ndarray]:
    with wave.open(str(path), "rb") as wav_file:
        audio_format = (
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getframerate(),
        )
        if audio_format != (1, 2, 16000):
            raise ValueError(
                "WAV dosyası 16 kHz, mono ve 16-bit PCM olmalı "
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


def build_model(model_path: Path) -> Model:
    required = [
        model_path,
        MODELS_DIR / "melspectrogram.onnx",
        MODELS_DIR / "embedding_model.onnx",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Gerekli model dosyaları bulunamadı:\n- "
            + "\n- ".join(missing)
            + "\n'Hey Orbit' modelini models/hey_orbit.onnx adıyla yerleştirin. "
            + "Ayrıntılar için MODEL_EGITIMI.md dosyasına bakın."
        )

    return Model(
        wakeword_models=[str(model_path)],
        inference_framework="onnx",
        melspec_model_path=str(MODELS_DIR / "melspectrogram.onnx"),
        embedding_model_path=str(MODELS_DIR / "embedding_model.onnx"),
    )


def score_as_float(value: object) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def main() -> int:
    args = parse_args()
    if args.list_devices:
        return list_devices()

    try:
        model = build_model(args.model.resolve())
        chunks = wav_chunks(args.wav.resolve()) if args.wav else microphone_chunks(args.device)
        source = str(args.wav) if args.wav else (args.device or "varsayılan mikrofon")
        print(f"Dinleniyor: {source}")
        print(f"Model: {args.model.name} | eşik: {args.threshold:.2f}")
        print("Filtre ve ardışık kare doğrulaması yok")
        print("Durdurmak için Ctrl+C tuşlarına basın.\n")

        last_detection = -float("inf")

        for samples in chunks:
            predictions = model.predict(samples)
            now = time.monotonic()
            best_name, best_score = max(
                ((name, score_as_float(score)) for name, score in predictions.items()),
                key=lambda item: item[1],
            )
            frame_hit = best_score >= args.threshold

            if frame_hit and now - last_detection >= args.cooldown:
                print(
                    f"\r[{time.strftime('%H:%M:%S')}] UYANDIRMA SÖZCÜĞÜ ALGILANDI: "
                    f"{best_name} ({best_score:.3f}){' ' * 12}"
                )
                last_detection = now
                # Algılanan ifadenin ses/model bağlamı sonraki kararı etkilemesin.
                model.reset()
            elif not args.quiet:
                print(
                    f"\r{best_name}: {best_score:.3f}{' ' * 12}",
                    end="",
                    flush=True,
                )
        if args.wav:
            print("\nDosya taraması tamamlandı.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, wave.Error) as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nDinleme durduruldu.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
