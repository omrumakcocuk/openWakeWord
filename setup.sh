#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if (( $# != 0 )); then
    echo "Kullanım: PYTHON_BIN=python3.12 ./setup.sh" >&2
    exit 2
fi

python_bin="${PYTHON_BIN:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Hata: '$python_bin' bulunamadı." >&2
    exit 1
fi

if ! "$python_bin" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
    echo "Hata: Python 3.10 veya daha yeni bir sürüm gerekli." >&2
    exit 1
fi

for model in \
    models/hey_orbit.onnx \
    models/melspectrogram.onnx \
    models/embedding_model.onnx
do
    if [[ ! -f "$model" ]]; then
        echo "Hata: gerekli model dosyası eksik: $model" >&2
        echo "Eksik özel modeli otomatik indirmek yerine temiz bir depo kopyası kullanın." >&2
        exit 1
    fi
done

requested_version="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ -x .venv/bin/python ]]; then
    existing_version="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$existing_version" != "$requested_version" ]]; then
        echo "Hata: mevcut .venv Python $existing_version kullanıyor; istenen sürüm $requested_version." >&2
        echo ".venv klasörünü yedekleyin/kaldırın ve kurulumu yeniden çalıştırın." >&2
        exit 1
    fi
else
    "$python_bin" -m venv .venv
fi
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    if ! .venv/bin/python -m ensurepip --upgrade; then
        echo "Hata: sanal ortamda pip hazırlanamadı; sistemin python3-venv paketini kurun." >&2
        exit 1
    fi
fi
.venv/bin/python -m pip install -r requirements-runtime.txt

# openwakeword 0.6.0 paketinin normal Linux bağımlılıkları TFLite,
# SciPy ve scikit-learn'i de getirir. Bu proje bunları kullanmayan, ONNX-only
# bir yol izler; gerçekte gereken bağımlılıklar yukarıda açıkça kurulmuştur.
.venv/bin/python -m pip install --no-deps openwakeword==0.6.0

.venv/bin/python - <<'PY'
import numpy as np
from wake_word import CHUNK_SAMPLES, DEFAULT_MODEL, build_model, reset_and_prime_model

model = build_model(DEFAULT_MODEL)
reset_and_prime_model(model)
predictions = model.predict(np.zeros(CHUNK_SAMPLES, dtype=np.int16))
if not predictions:
    raise SystemExit("Model smoke testi tahmin üretmedi")
print("ONNX model smoke testi başarılı.")
PY

echo
echo "Kurulum tamamlandı. Başlatmak için:"
echo "  .venv/bin/python wake_word.py"
if ! command -v arecord >/dev/null 2>&1; then
    echo
    echo "Uyarı: 'arecord' bulunamadı. Mikrofon kullanımı için Linux'ta"
    echo "alsa-utils paketini kurun; WAV dosyası testi bundan etkilenmez."
fi
