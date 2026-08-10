#!/usr/bin/env python3
"""Doğrulanmış manifestten openWakeWord uyumlu Hey Orbit modeli eğitir."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import wave
from collections import defaultdict
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

try:  # Hem ``python training/...`` hem modül olarak içe aktarma desteği.
    from .common import (
        MANIFEST_PATH,
        ROOT,
        VALID_SPLITS,
        ManifestItem,
        atomic_write_json,
        evaluation_set_sha256,
        load_manifest,
        manifest_split_summary,
        sha256_file,
    )
except ImportError:
    from common import (
        MANIFEST_PATH,
        ROOT,
        VALID_SPLITS,
        ManifestItem,
        atomic_write_json,
        evaluation_set_sha256,
        load_manifest,
        manifest_split_summary,
        sha256_file,
    )


MODEL_DIR = ROOT / "models"
ACCEPTED_MODEL = MODEL_DIR / "hey_orbit.onnx"
OUTPUT_MODEL = MODEL_DIR / "candidates" / "hey_orbit_candidate.onnx"
METRICS_PATH = MODEL_DIR / "candidates" / "hey_orbit_candidate_metrics.json"
SAMPLE_RATE = 16000
WINDOW = 32000
POSITIVE_EDGE_MARGIN = 800  # Her iki tarafta en az 50 ms, yer varsa.
SEED = 20260808

HARD_NEGATIVE_TEXTS = {
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
    "soru bankası",
    "günlük soru",
    "soru bankası günlük soru",
    "günlük soru bankası",
    "bugün soru bankasından kaç soru çözdün",
    "ders için günlük soru hedefini tamamladın mı",
}

VERY_HARD_NEGATIVE_TEXTS = {"hey orbitz", "hey or", "orbit"}


class FeatureExtractor(Protocol):
    def extract_batch(self, audio: np.ndarray) -> np.ndarray: ...


class OpenWakeWordFeatureExtractor:
    """openWakeWord'ün kararlı, herkese açık toplu embedding API adaptörü."""

    def __init__(self, *, model_dir: Path, ncpu: int, batch_size: int) -> None:
        # Ağır bağımlılık yalnızca gerçek eğitimde yüklenir. Özel
        # _get_embeddings API'sine bağlanmak yerine public embed_clips kullanılır.
        from openwakeword.utils import AudioFeatures

        self._ncpu = ncpu
        self._batch_size = batch_size
        self._extractor = AudioFeatures(
            inference_framework="onnx",
            melspec_model_path=str(model_dir / "melspectrogram.onnx"),
            embedding_model_path=str(model_dir / "embedding_model.onnx"),
            ncpu=ncpu,
        )

    def extract_batch(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.int16)
        if audio.ndim != 2 or audio.shape[1] != WINDOW:
            raise ValueError(f"Embedding için beklenmeyen ses boyutu: {audio.shape}")
        feature = np.asarray(
            self._extractor.embed_clips(
                audio,
                batch_size=min(self._batch_size, len(audio)),
                ncpu=self._ncpu,
            ),
            dtype=np.float32,
        )
        if feature.shape != (len(audio), 16, 96):
            raise ValueError(f"Beklenmeyen öznitelik boyutu: {feature.shape}")
        return feature


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-model", type=Path, default=OUTPUT_MODEL)
    parser.add_argument("--metrics-output", type=Path, default=METRICS_PATH)
    parser.add_argument(
        "--train-splits",
        nargs="+",
        choices=VALID_SPLITS,
        default=["train"],
        help="Eğitimde kullanılacak bölmeler. Final model için 'train dev' verilebilir.",
    )
    parser.add_argument(
        "--eval-split",
        "--split",
        dest="eval_split",
        choices=VALID_SPLITS,
        default="dev",
        help="Metrik bölmesi. Final adayında bağımsız 'test' kullanın.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Eğitim-sonu sentetik metrik eşiği (varsayılan: runtime ile aynı 0.70)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ncpu", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold 0 ile 1 arasında olmalı")
    if args.ncpu < 1:
        parser.error("--ncpu en az 1 olmalı")
    if args.feature_batch_size < 1:
        parser.error("--feature-batch-size en az 1 olmalı")
    if len(set(args.train_splits)) != len(args.train_splits):
        parser.error("--train-splits aynı bölmeyi birden fazla içeremez")
    if args.eval_split in args.train_splits:
        parser.error("Değerlendirme bölmesi eğitim bölmelerinden bağımsız olmalı")
    if args.output_model.resolve() == args.metrics_output.resolve():
        parser.error("Model ve metrik çıktıları aynı dosya olamaz")
    if args.output_model.resolve() == ACCEPTED_MODEL.resolve():
        parser.error(
            "Kabul edilmiş models/hey_orbit.onnx eğitim betiğiyle doğrudan ezilemez; "
            "önce candidates/ altında üretip akış kabul kapısından geçirin"
        )
    if args.metrics_output.resolve() == (MODEL_DIR / "hey_orbit_metrics.json").resolve():
        parser.error(
            "Kabul edilmiş models/hey_orbit_metrics.json eğitim betiğiyle doğrudan ezilemez"
        )
    return args


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        if (wav_file.getnchannels(), wav_file.getsampwidth(), wav_file.getframerate()) != (
            1,
            2,
            SAMPLE_RATE,
        ):
            raise ValueError(f"Geçersiz WAV biçimi: {path}")
        audio = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    if audio.size == 0:
        raise ValueError(f"Boş WAV: {path}")
    return audio.astype(np.float32) / 32768


def change_speed(audio: np.ndarray, factor: float) -> np.ndarray:
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Ses tek boyutlu ve boş olmayan bir dizi olmalı")
    if factor <= 0:
        raise ValueError("Hız katsayısı sıfırdan büyük olmalı")
    new_length = max(1, round(len(audio) / factor))
    old_x = np.linspace(0, 1, len(audio), endpoint=False)
    new_x = np.linspace(0, 1, new_length, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def trim_positive_silence(audio: np.ndarray) -> np.ndarray:
    """Sentezleyicinin uzun kenar sessizliklerini hedefi kesmeden azalt."""

    peak = float(np.max(np.abs(audio), initial=0.0))
    if peak <= 0:
        return audio
    active = np.flatnonzero(np.abs(audio) >= max(0.002, peak * 0.02))
    if active.size == 0:
        return audio
    context = round(0.10 * SAMPLE_RATE)
    start = max(0, int(active[0]) - context)
    stop = min(len(audio), int(active[-1]) + context + 1)
    return audio[start:stop]


def fit_audio_window(
    audio: np.ndarray,
    rng: np.random.Generator,
    *,
    positive: bool,
) -> np.ndarray:
    """Sesi 2 saniyeye yerleştir; pozitif hedefin hiçbir hecesini kesme."""

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Ses tek boyutlu ve boş olmayan bir dizi olmalı")
    if positive:
        audio = trim_positive_silence(audio)

    audio = change_speed(audio, float(rng.uniform(0.84, 1.17)))
    if positive:
        maximum_phrase_length = WINDOW - 2 * POSITIVE_EDGE_MARGIN
        if len(audio) > maximum_phrase_length:
            # Eski rastgele crop uzun pozitif kliplerde "Hey" veya "Orbit"
            # hecesini atabiliyordu. Tüm hedefi pencereye sığdırmak recall'u korur.
            audio = change_speed(audio, len(audio) / maximum_phrase_length)

    if len(audio) > WINDOW:
        # Yalnızca negatifler buraya gelir; farklı bölümleri görmek yararlıdır.
        start = int(rng.integers(0, len(audio) - WINDOW + 1))
        return audio[start : start + WINDOW]
    if len(audio) == WINDOW:
        return audio

    available = WINDOW - len(audio)
    if positive:
        margin = min(POSITIVE_EDGE_MARGIN, available // 2)
        left = int(rng.integers(margin, available - margin + 1))
    else:
        left = int(rng.integers(0, available + 1))
    return np.pad(audio, (left, available - left))


def add_room_echo(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = audio.copy()
    for delay, base_strength in ((530, 0.12), (1100, 0.08), (1900, 0.04)):
        if delay < len(audio):
            strength = base_strength * rng.uniform(0.3, 1.2)
            result[delay:] += audio[:-delay] * strength
    return result


def augment(audio: np.ndarray, rng: np.random.Generator, positive: bool) -> np.ndarray:
    audio = fit_audio_window(audio, rng, positive=positive)
    audio = add_room_echo(audio, rng)
    audio *= rng.uniform(0.45, 1.15)
    signal_rms = max(float(np.sqrt(np.mean(audio**2))), 0.005)
    snr_db = rng.uniform(8, 32) if positive else rng.uniform(5, 35)
    noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise = rng.normal(0, noise_rms, WINDOW).astype(np.float32)

    if rng.random() < 0.35:
        frequency = rng.uniform(45, 350)
        phase = rng.uniform(0, 2 * np.pi)
        tone = np.sin(2 * np.pi * frequency * np.arange(WINDOW) / SAMPLE_RATE + phase)
        noise += tone.astype(np.float32) * rng.uniform(0.001, 0.012)
    return (np.clip(audio + noise, -1, 1) * 32767).astype(np.int16)


def noise_only(rng: np.random.Generator) -> np.ndarray:
    kind = int(rng.integers(0, 4))
    if kind == 0:
        audio = rng.normal(0, rng.uniform(0.002, 0.08), WINDOW)
    elif kind == 1:
        audio = np.zeros(WINDOW)
        for _ in range(int(rng.integers(1, 12))):
            index = int(rng.integers(0, WINDOW))
            audio[index : min(WINDOW, index + 100)] = rng.uniform(-0.3, 0.3)
    elif kind == 2:
        frequency = rng.uniform(40, 1000)
        audio = (
            np.sin(2 * np.pi * frequency * np.arange(WINDOW) / SAMPLE_RATE)
            * rng.uniform(0.002, 0.08)
        )
    else:
        audio = np.zeros(WINDOW)
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


def normalized_negative_text(text: str) -> str:
    return text.lower().strip(" !,.")


def augmentation_rounds(item: ManifestItem) -> int:
    if item.label == 1:
        return 32
    normalized = normalized_negative_text(item.text)
    if normalized in VERY_HARD_NEGATIVE_TEXTS:
        return 120
    if normalized in HARD_NEGATIVE_TEXTS:
        return 40
    return 8


def build_examples(
    items: Sequence[ManifestItem],
    *,
    extractor: FeatureExtractor,
    rng: np.random.Generator,
    noise_examples: int,
    progress_label: str,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    texts: list[str] = []
    pending_audio: list[np.ndarray] = []

    def flush_pending() -> None:
        if not pending_audio:
            return
        batch_features = extractor.extract_batch(np.asarray(pending_audio, dtype=np.int16))
        features.extend(batch_features)
        pending_audio.clear()

    def append_audio(audio: np.ndarray, label: int, text: str) -> None:
        pending_audio.append(audio)
        labels.append(label)
        texts.append(text)
        if len(pending_audio) >= batch_size:
            flush_pending()

    for item_index, item in enumerate(items):
        audio = read_wav(item.absolute_path)
        for _ in range(augmentation_rounds(item)):
            append_audio(augment(audio, rng, positive=bool(item.label)), item.label, item.text)
        if (item_index + 1) % 25 == 0 or item_index + 1 == len(items):
            print(f"{progress_label} öznitelikleri: {item_index + 1}/{len(items)}", flush=True)

    for _ in range(noise_examples):
        append_audio(noise_only(rng), 0, "[sentetik gürültü]")
    flush_pending()
    if not features:
        raise ValueError(f"{progress_label} için örnek bulunamadı")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        texts,
    )


def export_onnx(classifier: MLPClassifier, path: Path) -> None:
    nodes = [helper.make_node("Flatten", ["input"], ["flat"], axis=1)]
    initializers = []
    previous = "flat"
    for index, (weights, bias) in enumerate(zip(classifier.coefs_, classifier.intercepts_)):
        weight_name = f"weight_{index}"
        bias_name = f"bias_{index}"
        initializers.extend(
            [
                numpy_helper.from_array(weights.astype(np.float32), weight_name),
                numpy_helper.from_array(bias.astype(np.float32), bias_name),
            ]
        )
        is_last = index == len(classifier.coefs_) - 1
        gemm_output = "logits" if is_last else f"gemm_{index}"
        nodes.append(helper.make_node("Gemm", [previous, weight_name, bias_name], [gemm_output]))
        if is_last:
            nodes.append(helper.make_node("Sigmoid", [gemm_output], ["output"]))
        else:
            output_name = f"linear_{index}"
            nodes.append(helper.make_node("Relu", [gemm_output], [output_name]))
            previous = output_name

    graph = helper.make_graph(
        nodes,
        "hey_orbit_turkish_openwakeword",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 16, 96])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="uyandirma/openWakeWord",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    model.metadata_props.add(key="wake_phrase", value="Hey Orbit")
    model.metadata_props.add(key="language", value="tr-TR")
    onnx.checker.check_model(model)
    onnx.save(model, path)


def positive_probabilities(classifier: MLPClassifier, flat_features: np.ndarray) -> np.ndarray:
    positive_indices = np.flatnonzero(classifier.classes_ == 1)
    if positive_indices.size != 1:
        raise ValueError(f"Sınıflandırıcı 0/1 sınıflarını içermiyor: {classifier.classes_}")
    return classifier.predict_proba(flat_features)[:, int(positive_indices[0])]


def validate_onnx_parity(
    classifier: MLPClassifier,
    onnx_path: Path,
    feature_batch: np.ndarray,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, float | int]:
    """Dışa aktarılan ONNX skorlarını sklearn ile karşılaştır."""

    import onnxruntime as ort

    if feature_batch.ndim != 3 or feature_batch.shape[1:] != (16, 96):
        raise ValueError(f"Parity için beklenmeyen öznitelik boyutu: {feature_batch.shape}")
    if len(feature_batch) > 256:
        indices = np.linspace(0, len(feature_batch) - 1, 256, dtype=int)
        feature_batch = feature_batch[indices]
    sklearn_scores = positive_probabilities(classifier, feature_batch.reshape(len(feature_batch), -1))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_scores = np.asarray(
        session.run(None, {session.get_inputs()[0].name: feature_batch.astype(np.float32)})[0]
    ).reshape(-1)
    absolute_error = np.abs(sklearn_scores - onnx_scores)
    max_absolute_error = float(np.max(absolute_error, initial=0.0))
    if not np.allclose(sklearn_scores, onnx_scores, atol=atol, rtol=rtol):
        raise RuntimeError(
            "ONNX parity kontrolü başarısız: "
            f"maksimum mutlak fark={max_absolute_error:.8g}"
        )
    return {"checked_examples": int(len(feature_batch)), "max_absolute_error": max_absolute_error}


def export_candidate_atomically(
    classifier: MLPClassifier,
    output_path: Path,
    parity_features: np.ndarray,
) -> dict[str, float | int]:
    """Çıktıyı yalnızca geçerli ve parity-sağlanmış adayla atomik değiştir."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.candidate-",
        suffix=".onnx",
        dir=output_path.parent,
    )
    os.close(descriptor)
    candidate_path = Path(temporary_name)
    try:
        export_onnx(classifier, candidate_path)
        parity = validate_onnx_parity(classifier, candidate_path, parity_features)
        os.replace(candidate_path, output_path)
        return parity
    finally:
        candidate_path.unlink(missing_ok=True)


def phrase_metrics(
    labels: np.ndarray,
    texts: Sequence[str],
    probabilities: np.ndarray,
    threshold: float,
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for label, text, probability in zip(labels, texts, probabilities):
        grouped[(int(label), text)].append(float(probability))
    result = []
    for (label, text), scores in sorted(grouped.items(), key=lambda item: (-item[0][0], item[0][1])):
        hits = sum(score >= threshold for score in scores)
        result.append(
            {
                "label": label,
                "text": text,
                "examples": len(scores),
                "threshold_hits": hits,
                "minimum_score": float(min(scores)),
                "mean_score": float(np.mean(scores)),
                "maximum_score": float(max(scores)),
            }
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.resolve()
    manifest_hash = sha256_file(manifest_path)
    manifest = load_manifest(manifest_path)
    train_items = [item for item in manifest if item.split in args.train_splits]
    eval_items = [item for item in manifest if item.split == args.eval_split]
    for name, items in (("eğitim", train_items), ("değerlendirme", eval_items)):
        labels = {item.label for item in items}
        if labels != {0, 1}:
            raise ValueError(f"{name.capitalize()} verisi hem pozitif hem negatif içermeli: {labels}")
    training_set_hash = evaluation_set_sha256(train_items)
    evaluation_set_hash = evaluation_set_sha256(eval_items)
    feature_model_hashes = {
        name: sha256_file(MODEL_DIR / name)
        for name in ("melspectrogram.onnx", "embedding_model.onnx")
    }

    extractor = OpenWakeWordFeatureExtractor(
        model_dir=MODEL_DIR,
        ncpu=args.ncpu,
        batch_size=args.feature_batch_size,
    )
    train_rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1_000_003)
    train_features, train_labels, _ = build_examples(
        train_items,
        extractor=extractor,
        rng=train_rng,
        noise_examples=480,
        progress_label="Eğitim",
        batch_size=args.feature_batch_size,
    )
    eval_features, eval_labels, eval_texts = build_examples(
        eval_items,
        extractor=extractor,
        rng=eval_rng,
        noise_examples=120,
        progress_label="Değerlendirme",
        batch_size=args.feature_batch_size,
    )
    flat_train = train_features.reshape(len(train_features), -1)
    flat_eval = eval_features.reshape(len(eval_features), -1)
    order = train_rng.permutation(len(train_labels))

    classifier = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.002,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=args.seed,
        verbose=True,
    )
    classifier.fit(flat_train[order], train_labels[order])

    probabilities = positive_probabilities(classifier, flat_eval)
    predictions = (probabilities >= args.threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(eval_labels, predictions, labels=[0, 1]).ravel()
    if sha256_file(manifest_path) != manifest_hash:
        raise RuntimeError("Manifest eğitim sırasında değişti; aday model yazılmadı")
    if evaluation_set_sha256(train_items) != training_set_hash:
        raise RuntimeError("Eğitim WAV seti eğitim sırasında değişti; aday model yazılmadı")
    if evaluation_set_sha256(eval_items) != evaluation_set_hash:
        raise RuntimeError("Test WAV seti eğitim sırasında değişti; aday model yazılmadı")
    for name, expected_hash in feature_model_hashes.items():
        if sha256_file(MODEL_DIR / name) != expected_hash:
            raise RuntimeError(f"Özellik modeli eğitim sırasında değişti: {name}")
    parity = export_candidate_atomically(classifier, args.output_model.resolve(), eval_features)
    metrics = {
        "schema_version": 2,
        "wake_phrase": "Hey Orbit",
        "language": "tr-TR",
        "model": str(args.output_model.resolve()),
        "model_sha256": sha256_file(args.output_model.resolve()),
        "threshold": args.threshold,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "training_set_sha256": training_set_hash,
        "evaluation_set_sha256": evaluation_set_hash,
        "feature_model_sha256": feature_model_hashes,
        "manifest_split_summary": manifest_split_summary(manifest),
        "train_splits": list(args.train_splits),
        "evaluation_split": args.eval_split,
        "train_base_recordings": len(train_items),
        "evaluation_base_recordings": len(eval_items),
        "train_examples": int(len(train_labels)),
        "evaluation_examples": int(len(eval_labels)),
        "training_iterations": int(classifier.n_iter_),
        "accuracy": float(accuracy_score(eval_labels, predictions)),
        "precision": float(precision_score(eval_labels, predictions, zero_division=0)),
        "recall": float(recall_score(eval_labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(eval_labels, probabilities)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "onnx_parity": parity,
        "per_phrase": phrase_metrics(eval_labels, eval_texts, probabilities, args.threshold),
        "note": (
            "Manifestte kaynak-grubu sızdırmaz, ifade/etiket bazında katmanlı "
            "bölme üzerindeki sentetik sonuçlardır; gerçek insan sesi testi ayrıca gerekir."
        ),
    }
    atomic_write_json(args.metrics_output.resolve(), metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Model kaydedildi: {args.output_model.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
