#!/usr/bin/env python3
"""Üç Türkçe Piper sesiyle Hey Orbit eğitim WAV'ları üretir."""

from __future__ import annotations

import json
import random
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig


ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = ROOT / "training" / "voices"
DATA_DIR = ROOT / "training" / "data"
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


def synthesize(voice: PiperVoice, text: str, config: SynthesisConfig) -> np.ndarray:
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


def main() -> None:
    rng = random.Random(SEED)
    manifest: list[dict[str, object]] = []

    for voice_index, voice_name in enumerate(VOICES):
        voice_path = VOICE_DIR / voice_name
        print(f"Yükleniyor: {voice_name}", flush=True)
        voice = PiperVoice.load(voice_path, use_cuda=False)
        voice_tag = voice_path.stem.replace("tr_TR-", "").replace("-medium", "")

        # Her varyasyondan ses başına 10 örnek üret. Bitişik ifadeler hızlı,
        # virgüllü/boşluklu ifadeler daha doğal aralıkta sentezlenir. Yazım
        # varyasyonlarının yanında hız çeşitliliği akustik genellemeyi artırır.
        samples_per_phrase = 10
        for sample_index in range(len(POSITIVE_PHRASES) * samples_per_phrase):
            phrase, min_length_scale, max_length_scale = POSITIVE_PHRASES[
                sample_index % len(POSITIVE_PHRASES)
            ]
            length_scale = rng.uniform(min_length_scale, max_length_scale)
            config = SynthesisConfig(
                length_scale=length_scale,
                noise_scale=rng.uniform(0.45, 0.9),
                noise_w_scale=rng.uniform(0.45, 0.9),
                volume=rng.uniform(0.82, 1.08),
            )
            audio = synthesize(voice, phrase, config)
            path = DATA_DIR / "positive" / f"{voice_tag}_{sample_index:03d}.wav"
            save_wav(path, audio)
            manifest.append({"path": str(path.relative_to(ROOT)), "label": 1, "text": phrase})

        negative_texts = NEGATIVE_PHRASES + GENERIC_PHRASES
        for sample_index, phrase in enumerate(negative_texts):
            config = SynthesisConfig(
                length_scale=rng.uniform(0.82, 1.2),
                noise_scale=rng.uniform(0.5, 0.85),
                noise_w_scale=rng.uniform(0.5, 0.85),
                volume=rng.uniform(0.82, 1.08),
            )
            audio = synthesize(voice, phrase, config)
            path = DATA_DIR / "negative" / f"{voice_tag}_{sample_index:03d}.wav"
            save_wav(path, audio)
            manifest.append({"path": str(path.relative_to(ROOT)), "label": 0, "text": phrase})

        print(f"Tamamlandı: {voice_name} ({voice_index + 1}/{len(VOICES)})", flush=True)

    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Toplam {len(manifest)} temel ses örneği üretildi: {DATA_DIR}")


if __name__ == "__main__":
    main()
