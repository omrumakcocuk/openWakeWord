#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps openwakeword==0.6.0
.venv/bin/python -c 'from openwakeword.utils import download_models; download_models(["__yalnizca_ozellik_modelleri__"], target_directory="models")'

echo
if [[ -f models/hey_orbit.onnx ]]; then
    echo "Kurulum tamamlandı. Başlatmak için:"
    echo "  .venv/bin/python wake_word.py"
else
    echo "Temel kurulum tamamlandı."
    echo "models/hey_orbit.onnx bulunamadı; MODEL_EGITIMI.md belgesine bakın."
fi
