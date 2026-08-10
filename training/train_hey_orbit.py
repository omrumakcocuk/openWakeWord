#!/usr/bin/env python3
"""Türkçe sentetik veriden openWakeWord uyumlu Hey Orbit ONNX modeli eğitir."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from openwakeword.utils import AudioFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training" / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_MODEL = MODEL_DIR / "hey_orbit.onnx"
METRICS_PATH = MODEL_DIR / "hey_orbit_metrics.json"
SAMPLE_RATE = 16000
WINDOW = 32000
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
}

VERY_HARD_NEGATIVE_TEXTS = {"hey orbitz", "hey or", "orbit"}


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        if (wav_file.getnchannels(), wav_file.getsampwidth(), wav_file.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError(f"Geçersiz WAV biçimi: {path}")
        return np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2").astype(np.float32) / 32768


def change_speed(audio: np.ndarray, factor: float) -> np.ndarray:
    new_length = max(1, round(len(audio) / factor))
    old_x = np.linspace(0, 1, len(audio), endpoint=False)
    new_x = np.linspace(0, 1, new_length, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def add_room_echo(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = audio.copy()
    for delay, strength in [(530, 0.12), (1100, 0.08), (1900, 0.04)]:
        strength *= rng.uniform(0.3, 1.2)
        result[delay:] += audio[:-delay] * strength
    return result


def augment(audio: np.ndarray, rng: np.random.Generator, positive: bool) -> np.ndarray:
    audio = change_speed(audio, rng.uniform(0.84, 1.17))
    if len(audio) >= WINDOW:
        start = int(rng.integers(0, len(audio) - WINDOW + 1))
        audio = audio[start:start + WINDOW]
    else:
        left = int(rng.integers(500, max(501, WINDOW - len(audio) - 500))) if positive else int(rng.integers(0, WINDOW - len(audio) + 1))
        audio = np.pad(audio, (left, WINDOW - len(audio) - left))

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
            ndx = int(rng.integers(0, WINDOW))
            audio[ndx:min(WINDOW, ndx + 100)] = rng.uniform(-0.3, 0.3)
    elif kind == 2:
        frequency = rng.uniform(40, 1000)
        audio = np.sin(2 * np.pi * frequency * np.arange(WINDOW) / SAMPLE_RATE) * rng.uniform(0.002, 0.08)
    else:
        audio = np.zeros(WINDOW)
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


def extract_feature(extractor: AudioFeatures, audio: np.ndarray) -> np.ndarray:
    feature = np.asarray(extractor._get_embeddings(audio), dtype=np.float32)
    if feature.shape != (16, 96):
        raise ValueError(f"Beklenmeyen öznitelik boyutu: {feature.shape}")
    return feature


def export_onnx(classifier: MLPClassifier, path: Path) -> None:
    nodes = [helper.make_node("Flatten", ["input"], ["flat"], axis=1)]
    initializers = []
    previous = "flat"
    for index, (weights, bias) in enumerate(zip(classifier.coefs_, classifier.intercepts_)):
        weight_name = f"weight_{index}"
        bias_name = f"bias_{index}"
        output_name = "output" if index == len(classifier.coefs_) - 1 else f"linear_{index}"
        initializers.extend([
            numpy_helper.from_array(weights.astype(np.float32), weight_name),
            numpy_helper.from_array(bias.astype(np.float32), bias_name),
        ])
        gemm_output = output_name if index == len(classifier.coefs_) - 1 else f"gemm_{index}"
        nodes.append(helper.make_node("Gemm", [previous, weight_name, bias_name], [gemm_output]))
        if index < len(classifier.coefs_) - 1:
            nodes.append(helper.make_node("Relu", [gemm_output], [output_name]))
            previous = output_name
        else:
            nodes[-1] = helper.make_node("Gemm", [previous, weight_name, bias_name], ["logits"])
            nodes.append(helper.make_node("Sigmoid", ["logits"], ["output"]))

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
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def main() -> None:
    rng = np.random.default_rng(SEED)
    positive_paths = sorted((DATA_DIR / "positive").glob("*.wav"))
    negative_paths = sorted((DATA_DIR / "negative").glob("*.wav"))
    if not positive_paths or not negative_paths:
        raise FileNotFoundError("Önce generate_turkish_samples.py çalıştırılmalı")

    extractor = AudioFeatures(
        inference_framework="onnx",
        melspec_model_path=str(MODEL_DIR / "melspectrogram.onnx"),
        embedding_model_path=str(MODEL_DIR / "embedding_model.onnx"),
        ncpu=8,
    )
    train_features: list[np.ndarray] = []
    train_labels: list[int] = []
    val_features: list[np.ndarray] = []
    val_labels: list[int] = []

    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    hard_negative_paths = {
        ROOT / item["path"]
        for item in manifest
        if item["label"] == 0 and item["text"].lower().strip(" !,.") in HARD_NEGATIVE_TEXTS
    }
    very_hard_negative_paths = {
        ROOT / item["path"]
        for item in manifest
        if item["label"] == 0
        and item["text"].lower().strip(" !,.") in VERY_HARD_NEGATIVE_TEXTS
    }
    jobs = (
        [(path, 1, 32) for path in positive_paths]
        + [
            (
                path,
                0,
                120 if path in very_hard_negative_paths else 40 if path in hard_negative_paths else 8,
            )
            for path in negative_paths
        ]
    )
    for job_index, (path, label, rounds) in enumerate(jobs):
        audio = read_wav(path)
        target_x, target_y = (val_features, val_labels) if job_index % 5 == 0 else (train_features, train_labels)
        for _ in range(rounds):
            target_x.append(extract_feature(extractor, augment(audio, rng, positive=bool(label))))
            target_y.append(label)
        if (job_index + 1) % 25 == 0:
            print(f"Öznitelikler: {job_index + 1}/{len(jobs)}", flush=True)

    for index in range(600):
        target_x, target_y = (val_features, val_labels) if index % 5 == 0 else (train_features, train_labels)
        target_x.append(extract_feature(extractor, noise_only(rng)))
        target_y.append(0)

    X_train = np.asarray(train_features, dtype=np.float32).reshape(len(train_features), -1)
    y_train = np.asarray(train_labels, dtype=np.int8)
    X_val = np.asarray(val_features, dtype=np.float32).reshape(len(val_features), -1)
    y_val = np.asarray(val_labels, dtype=np.int8)
    order = rng.permutation(len(y_train))

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
        random_state=SEED,
        verbose=True,
    )
    classifier.fit(X_train[order], y_train[order])

    threshold = 0.75
    probabilities = classifier.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_val, predictions, labels=[0, 1]).ravel()
    metrics = {
        "wake_phrase": "Hey Orbit",
        "language": "tr-TR",
        "threshold": threshold,
        "train_examples": int(len(y_train)),
        "validation_examples": int(len(y_val)),
        "training_iterations": int(classifier.n_iter_),
        "accuracy": float(accuracy_score(y_val, predictions)),
        "precision": float(precision_score(y_val, predictions, zero_division=0)),
        "recall": float(recall_score(y_val, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, probabilities)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "note": "Sentetik üç Türkçe Piper sesi üzerindeki ayrılmış doğrulama sonuçlarıdır.",
    }
    export_onnx(classifier, OUTPUT_MODEL)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Model kaydedildi: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
