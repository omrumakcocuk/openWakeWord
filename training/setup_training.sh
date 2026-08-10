#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if (( $# != 0 )); then
    echo "Kullanım: PYTHON_BIN=python3.12 training/setup_training.sh" >&2
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
    echo "Hata: sentetik veri üretimi için Python 3.10 veya daha yenisini kullanın." >&2
    echo "Farklı yorumlayıcı: PYTHON_BIN=python3.12 training/setup_training.sh" >&2
    exit 1
fi

for model in models/melspectrogram.onnx models/embedding_model.onnx; do
    if [[ ! -f "$model" ]]; then
        echo "Hata: eğitim için gereken özellik modeli eksik: $model" >&2
        exit 1
    fi
done

requested_version="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ -x .train-venv/bin/python ]]; then
    existing_version="$(.train-venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$existing_version" != "$requested_version" ]]; then
        echo "Hata: mevcut .train-venv Python $existing_version kullanıyor; istenen sürüm $requested_version." >&2
        echo ".train-venv klasörünü yedekleyin/kaldırın ve yeniden deneyin." >&2
        exit 1
    fi
else
    "$python_bin" -m venv .train-venv
fi
if ! .train-venv/bin/python -m pip --version >/dev/null 2>&1; then
    if ! .train-venv/bin/python -m ensurepip --upgrade; then
        echo "Hata: eğitim ortamında pip hazırlanamadı; python3-venv paketini kurun." >&2
        exit 1
    fi
fi
.train-venv/bin/python -m pip install -r requirements-training.txt
.train-venv/bin/python -m pip install --no-deps openwakeword==0.6.0

.train-venv/bin/python - <<'PY'
import onnx
from openwakeword.utils import AudioFeatures
from piper import PiperVoice, SynthesisConfig
from sklearn.neural_network import MLPClassifier

for path in ("models/melspectrogram.onnx", "models/embedding_model.onnx"):
    onnx.checker.check_model(onnx.load(path))
print("Eğitim bağımlılıkları ve özellik modelleri doğrulandı.")
PY

echo
echo "Eğitim Python ortamı hazır: .train-venv"
echo "Piper ses dosyaları lisans/provenans nedeniyle otomatik indirilmez."
echo "Gerekli dosyaları ve kısıtları THIRD_PARTY_NOTICES.md içinde kontrol edin."
echo "Resmî büyük-veri eğitimi için ayrıca OPENWAKEWORD_EGITIMI.md belgesine bakın."
