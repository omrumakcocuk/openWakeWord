# openWakeWord mikrofon dinleyicisi

Bu uygulama, [openWakeWord](https://github.com/dscripka/openWakeWord) 0.6.0 ve
ONNX Runtime kullanarak mikrofonda Türkçe telaffuz edilen **“Hey Orbit”**
ifadesini algılamak üzere yapılandırılmıştır.

Türkçe model bu proje içinde eğitildi ve `models/hey_orbit.onnx` yoluna
yerleştirildi. Eğitim ayrıntıları ve ölçümler için
[MODEL_EGITIMI.md](MODEL_EGITIMI.md) belgesine bakabilirsiniz.

## Çalıştırma

Kurulum bu klasörde tamamlandı. Uygulamayı başlatın:

```bash
.venv/bin/python wake_word.py
```

Uygulama VAD, RMS veya ardışık kare filtresi kullanmaz. Tek bir model skoru
`0.70` eşiğini geçtiğinde uyanır ve başlangıç kalibrasyonu yapılmaz.

Mevcut kayıt aygıtlarını görmek ve farklı bir aygıt seçmek için:

```bash
.venv/bin/python wake_word.py --list-devices
.venv/bin/python wake_word.py --device hw:1,0
```

Varsayılan algılama eşiği `0.70` değeridir. Hassasiyeti değiştirmek için (düşük
değer daha hassastır):

```bash
.venv/bin/python wake_word.py --threshold 0.70
```

16 kHz, mono, 16-bit PCM bir WAV dosyasını test etmek için:

```bash
.venv/bin/python wake_word.py --wav test.wav
```

Baştan kurulum gerekirse:

```bash
./setup.sh
```

## Notlar

- `hey_orbit.onnx`, üç Türkçe Piper sesiyle üretilmiş sentetik örnekler ve
  openWakeWord ses öznitelikleri kullanılarak özel olarak eğitilmiştir.
- Eğitim verisi hem boşluklu “Hey Orbit” hem de hızlı/bitişik “Heyorbit”
  telaffuzlarını kapsar; tek başına “orbit” güçlü negatif olarak ağırlıklandırılır.
- Sentetik doğrulama başarısı gerçek mikrofon başarısıyla aynı değildir. Kendi
  sesinizle deneyip gerekirse eşiği küçük adımlarla (ör. `--threshold 0.65`)
  ayarlayın. Ardışık kare doğrulaması kullanılmadığı için eşik düşürüldükçe tek
  karelik yanlış tetikleme riski hızla artar.
- Mikrofon kaydı ALSA'nın `arecord` programını kullanır. Program yoksa sistemin
  paket yöneticisinden `alsa-utils` kurulmalıdır.
- Farklı bir openWakeWord ONNX modeli `--model models/model.onnx` ile seçilebilir.
- Resmî openWakeWord büyük-veri eğitim yapılandırması ve gereken harici
  datasetler [OPENWAKEWORD_EGITIMI.md](OPENWAKEWORD_EGITIMI.md) belgesindedir.
- Eski “Hey Jarvis” modeli hâlâ denenebilir:
  `.venv/bin/python wake_word.py --model models/hey_jarvis_v0.1.onnx`.
