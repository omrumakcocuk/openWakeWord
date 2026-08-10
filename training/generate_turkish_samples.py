#!/usr/bin/env python3
"""Üç Türkçe Piper sesiyle Hey Orbit eğitim WAV'ları üretir."""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

try:  # Hem ``python training/...`` hem modül olarak içe aktarma desteği.
    from .common import (
        DATA_DIR,
        ROOT,
        ManifestItem,
        assign_splits,
        atomic_write_json,
        load_manifest,
        validate_manifest_records,
    )
except ImportError:
    from common import (
        DATA_DIR,
        ROOT,
        ManifestItem,
        assign_splits,
        atomic_write_json,
        load_manifest,
        validate_manifest_records,
    )

if TYPE_CHECKING:
    from piper import PiperVoice, SynthesisConfig


VOICE_DIR = ROOT / "training" / "voices"
DATA_BACKUP_DIR = DATA_DIR.with_name(".data-previous")
SAMPLE_RATE = 16000
SEED = 20260808

VOICES = [
    "tr_TR-dfki-medium.onnx",
    "tr_TR-fahrettin-medium.onnx",
    "tr_TR-fettah-medium.onnx",
]

POSITIVE_PHRASES = [
    # (metin, en hızlı okuma, en yavaş okuma)
    ("HEY    ORBIT", 0.86, 1.22),
    ("HEY ORBIT", 0.68, 1.04),
    ("HeyOrbit", 0.68, 0.94),
    ("heyOrbit!", 0.68, 0.98),
    ("heyorbit", 0.68, 0.94),
    ("hey-orbit", 0.74, 1.08),
    ("Heyorbit", 0.68, 0.94),
    ("Hey, Orbit", 0.90, 1.25),
    ("Hey Orbit!", 0.86, 1.22),
]

NEGATIVE_PHRASES = [
    "hey",
    "orbit",
    "hey hey",
    "orbit orbit",
    "hey or",
    "or bit",
    "hey orbi",
    "ey orbit",
    "hey orbita",
    "hey orbital",
    "hey korbit",
    "hey gorbit",
    "hey kurbit",
    "hey kor bit",
    "hey korbi",
    "hey robert",
    "hey corbett",
    "hey orbitz",
    "orbital",
    "orbiting",
    "hey rabbit",
    "a orbit",
    "the orbit",
    "okay orbit",
    "hey robot",
    "hey orhan",
    "hey orkun",
    "hey orçun",
    "hey orhan bey",
    "hey oğlum",
    "hey orada",
    "hey otur",
    "hey organizasyon",
    "hey motor",
    "hey doruk",
    "hey bora",
    "haydi orbit",
    "orbit buraya gel",
    "merhaba orbit",
    "tamam orbit",
    "peki orbit",
    "sayın orbit",
    # Gerçek kullanımda yanlış tetikleme ürettiği bildirilen konuşma ailesi.
    "soru bankası",
    "günlük soru",
    "soru bankası günlük soru",
    "günlük soru bankası",
    "bugün soru bankasından kaç soru çözdün",
    "ders için günlük soru hedefini tamamladın mı",
]

GENERIC_PHRASES = [
    "Bugün hava oldukça güzel görünüyor.",
    "Saat kaçta yola çıkmamız gerekiyor?",
    "Mutfaktaki ışıkları kapatır mısın?",
    "Yarın sabah için bir alarm kur.",
    "Bir bardak su alabilir miyim?",
    "Televizyonun sesini biraz azalt.",
    "Bu akşam hangi filmi izleyelim?",
    "Pencereyi açınca oda serinledi.",
    "Telefonumu masanın üzerinde bıraktım.",
    "Yeni toplantı öğleden sonra başlayacak.",
    "Müzik listesini sırayla çalmaya devam et.",
    "Kahve hazır olunca bana haber ver.",
    "Dışarı çıkmadan önce kapıyı kilitle.",
    "Bilgisayar güncellemeyi tamamladı.",
    "Bugünkü işler beklediğimden erken bitti.",
    "Salondaki lambayı yüzde elliye getir.",
    "Hafta sonu sahilde yürüyüş yapabiliriz.",
    "Kitabın son bölümünü henüz okumadım.",
    "Akşam yemeği için ne hazırlayalım?",
    "Sessiz mod bir saat sonra kapansın.",
]


def resample(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == SAMPLE_RATE:
        return audio.astype(np.float32)
    new_length = max(1, round(len(audio) * SAMPLE_RATE / source_rate))
    old_x = np.linspace(0, 1, len(audio), endpoint=False)
    new_x = np.linspace(0, 1, new_length, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def synthesize(voice: "PiperVoice", text: str, config: "SynthesisConfig") -> np.ndarray:
    chunks = [chunk.audio_float_array for chunk in voice.synthesize(text, config)]
    if not chunks:
        raise RuntimeError(f"Piper ses üretemedi: {text}")
    return resample(np.concatenate(chunks), voice.config.sample_rate)


def save_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())


def validate_staged_dataset(staging_dir: Path, items: list[ManifestItem]) -> None:
    """Tamamlanmamış veri setinin canlı dizine geçmesini engelle."""

    data_prefix = DATA_DIR.relative_to(ROOT)
    expected_files: set[Path] = set()
    for item in items:
        relative_to_data = Path(item.path).relative_to(data_prefix)
        staged_path = staging_dir / relative_to_data
        expected_files.add(staged_path.resolve())
        if not staged_path.is_file():
            raise FileNotFoundError(f"Üretilen WAV bulunamadı: {staged_path}")
        with wave.open(str(staged_path), "rb") as wav_file:
            actual_format = (
                wav_file.getnchannels(),
                wav_file.getsampwidth(),
                wav_file.getframerate(),
            )
            if actual_format != (1, 2, SAMPLE_RATE) or wav_file.getnframes() <= 0:
                raise ValueError(f"Geçersiz üretilen WAV: {staged_path} ({actual_format})")

    actual_files = {path.resolve() for path in staging_dir.glob("*/*.wav")}
    if actual_files != expected_files:
        missing = expected_files - actual_files
        unexpected = actual_files - expected_files
        raise ValueError(
            "Geçici veri seti manifestle eşleşmiyor: "
            f"eksik={len(missing)}, fazladan={len(unexpected)}"
        )


def recover_interrupted_commit() -> None:
    """Bir önceki dizin değişiminin yarım kalmasını toparla."""

    if not DATA_BACKUP_DIR.exists():
        return
    if not DATA_DIR.exists():
        DATA_BACKUP_DIR.rename(DATA_DIR)
        print("Önceki yarım veri değişiminden eski veri seti geri yüklendi.", flush=True)
        return

    # Her iki dizin de varsa yeni dizinin taşınması tamamlanmış, yalnızca
    # eski yedeğin temizlenmesi yarım kalmış olabilir. Yeni set geçerli
    # değilse otomatik veri silmek yerine açık hata verilir.
    try:
        load_manifest(DATA_DIR / "manifest.json", allow_legacy_split=True)
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(
            f"Hem {DATA_DIR} hem {DATA_BACKUP_DIR} var ve yeni veri doğrulanamadı. "
            "Dizinleri elle inceleyin."
        ) from exc
    shutil.rmtree(DATA_BACKUP_DIR)


def commit_staged_dataset(staging_dir: Path) -> None:
    """Eksiksiz geçici veri setini geri alınabilir biçimde etkinleştir."""

    recover_interrupted_commit()
    had_previous_dataset = DATA_DIR.exists()
    if had_previous_dataset:
        DATA_DIR.rename(DATA_BACKUP_DIR)
    try:
        staging_dir.rename(DATA_DIR)
        # Dizin taşındıktan sonra gerçek yollar üzerinden de doğrula.
        load_manifest(DATA_DIR / "manifest.json", allow_legacy_split=False)
    except BaseException:
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
        if had_previous_dataset and DATA_BACKUP_DIR.exists():
            DATA_BACKUP_DIR.rename(DATA_DIR)
        raise
    else:
        if DATA_BACKUP_DIR.exists():
            shutil.rmtree(DATA_BACKUP_DIR)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Mevcut training/data setini yalnız yeni set tamamen doğrulanınca değiştir",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    recover_interrupted_commit()
    if DATA_DIR.exists() and not args.replace_existing:
        raise FileExistsError(
            f"Mevcut veri seti korunuyor: {DATA_DIR}. "
            "Bilinçli yeniden üretim için --replace-existing kullanın."
        )

    # Piper yalnızca veri üretirken gerekir; manifest ve test araçları bu
    # ağır bağımlılığı içe aktarmak zorunda kalmaz.
    from piper import PiperVoice, SynthesisConfig

    missing_voices = [str(VOICE_DIR / name) for name in VOICES if not (VOICE_DIR / name).is_file()]
    if missing_voices:
        raise FileNotFoundError("Piper ses modelleri bulunamadı:\n- " + "\n- ".join(missing_voices))

    rng = random.Random(SEED)
    staging_dir = Path(tempfile.mkdtemp(prefix=".data-build-", dir=DATA_DIR.parent))
    manifest_items: list[ManifestItem] = []
    try:
        for voice_index, voice_name in enumerate(VOICES):
            voice_path = VOICE_DIR / voice_name
            print(f"Yükleniyor: {voice_name}", flush=True)
            voice = PiperVoice.load(voice_path, use_cuda=False)
            voice_tag = voice_path.stem.replace("tr_TR-", "").replace("-medium", "")

            # Her varyasyondan ses başına 10 örnek üret. Bitişik ifadeler
            # hızlı, virgüllü/boşluklu ifadeler daha doğal aralıktadır.
            samples_per_phrase = 10
            for sample_index in range(len(POSITIVE_PHRASES) * samples_per_phrase):
                phrase, min_length_scale, max_length_scale = POSITIVE_PHRASES[
                    sample_index % len(POSITIVE_PHRASES)
                ]
                config = SynthesisConfig(
                    length_scale=rng.uniform(min_length_scale, max_length_scale),
                    noise_scale=rng.uniform(0.45, 0.9),
                    noise_w_scale=rng.uniform(0.45, 0.9),
                    volume=rng.uniform(0.82, 1.08),
                )
                filename = f"{voice_tag}_{sample_index:03d}.wav"
                save_wav(staging_dir / "positive" / filename, synthesize(voice, phrase, config))
                logical_path = (DATA_DIR / "positive" / filename).relative_to(ROOT).as_posix()
                manifest_items.append(
                    ManifestItem(
                        path=logical_path,
                        absolute_path=ROOT / logical_path,
                        label=1,
                        text=phrase,
                        split="",
                        group=logical_path.removesuffix(".wav"),
                        voice=voice_tag,
                    )
                )

            for sample_index, phrase in enumerate(NEGATIVE_PHRASES + GENERIC_PHRASES):
                config = SynthesisConfig(
                    length_scale=rng.uniform(0.82, 1.2),
                    noise_scale=rng.uniform(0.5, 0.85),
                    noise_w_scale=rng.uniform(0.5, 0.85),
                    volume=rng.uniform(0.82, 1.08),
                )
                filename = f"{voice_tag}_{sample_index:03d}.wav"
                save_wav(staging_dir / "negative" / filename, synthesize(voice, phrase, config))
                logical_path = (DATA_DIR / "negative" / filename).relative_to(ROOT).as_posix()
                manifest_items.append(
                    ManifestItem(
                        path=logical_path,
                        absolute_path=ROOT / logical_path,
                        label=0,
                        text=phrase,
                        split="",
                        group=logical_path.removesuffix(".wav"),
                        voice=voice_tag,
                    )
                )

            print(f"Tamamlandı: {voice_name} ({voice_index + 1}/{len(VOICES)})", flush=True)

        manifest_items = assign_splits(manifest_items, seed=SEED)
        manifest_payload = [item.as_json() for item in manifest_items]
        validate_manifest_records(
            manifest_payload,
            project_root=ROOT,
            data_dir=DATA_DIR,
            require_files=False,
            allow_legacy_split=False,
        )
        atomic_write_json(staging_dir / "manifest.json", manifest_payload)
        validate_staged_dataset(staging_dir, manifest_items)
        commit_staged_dataset(staging_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    print(f"Toplam {len(manifest_items)} temel ses örneği atomik olarak üretildi: {DATA_DIR}")


if __name__ == "__main__":
    main()
