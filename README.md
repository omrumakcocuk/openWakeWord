# Hey Orbit — yerel openWakeWord dinleyicisi

Bu proje, Türkçe söylenen **“Hey Orbit”** ve hızlı/bitişik
**“heyorbit”** telaffuzlarını yerel bir ONNX modeliyle algılar. Genel konuşmayı
metne çevirmez: mikrofon sesi 80 ms'lik PCM parçaları halinde openWakeWord
özellik modellerine ve ardından `hey_orbit.onnx` sınıflandırıcısına verilir.

Depoda çalıştırma için gereken üç ONNX dosyası bulunur:

- `models/melspectrogram.onnx`
- `models/embedding_model.onnx`
- `models/hey_orbit.onnx`

Model sentetik seslerle eğitilmiştir. Mevcut metrikler gerçek kişi, mikrofon
ve uzun süreli normal konuşma testi yerine geçmez; bir cihazda kullanmadan
önce hem kaçırma hem de saat başına yanlış tetikleme ölçülmelidir.

## Gereksinimler

- Python 3.10 veya daha yeni bir sürüm
- ONNX Runtime tarafından desteklenen bir platform
- Mikrofon kullanımı için Linux ve ALSA `arecord` (`alsa-utils` paketi)

Mikrofon girişi doğrudan ALSA'ya bağlıdır. macOS ve Windows için mikrofon
yakalama katmanı yoktur; uygun Python/ONNX kurulumu varsa `--wav` modu yine
kullanılabilir.

Debian/Ubuntu örneği:

```bash
sudo apt-get update
sudo apt-get install python3-venv alsa-utils
```

## Kurulum

Temiz bir klonda:

```bash
git clone https://github.com/omrumakcocuk/openWakeWord.git
cd openWakeWord
./setup.sh
```

Farklı bir Python yorumlayıcısı seçilebilir:

```bash
PYTHON_BIN=python3.12 ./setup.sh
```

Kurulum, paketleri indirmek için internete bağlanır; uygulamanın daha sonra
çalışması için ağ gerekmez. Betik gerekli model dosyalarından biri eksikse
hata verir ve kaynağı belirsiz bir özel modeli otomatik indirmez.

### Neden `openwakeword --no-deps` ile kuruluyor?

Bu proje **yalnızca ONNX inference** yolunu kullanır. openWakeWord 0.6.0'ın
standart paket tanımı TFLite runtime, SciPy, scikit-learn ve opsiyonel
işlevlerle ilgili bağımlılıkları da kurmaya çalışır. Bunlar bu uygulamanın
speaker-verifier kullanmayan ONNX akışında gerekli değildir ve bazı yeni
Python sürümlerinde kurulamaz.

`--no-deps`, uygulamanın bağımlılıksız olduğu anlamına gelmez.
[requirements-runtime.txt](requirements-runtime.txt) içinde gereken `numpy`,
`onnxruntime`, `requests` ve `tqdm` paketleri açıkça kurulur. Bu bilinçli
minimal kurulum yalnızca bu depodaki `wake_word.py` akışı için desteklenir;
openWakeWord'un VAD, TFLite, gürültü bastırma veya custom-verifier
özelliklerini etkinleştirmek ek bağımlılık gerektirir.

## Kullanım

Varsayılan mikrofonu dinleyin:

```bash
.venv/bin/python wake_word.py
```

ALSA aygıtlarını listeleyip belirli bir aygıtı seçin:

```bash
.venv/bin/python wake_word.py --list-devices
.venv/bin/python wake_word.py --device hw:1,0
```

16 kHz, mono, 16-bit PCM WAV dosyasını tarayın:

```bash
.venv/bin/python wake_word.py --wav test.wav
```

Varsayılan algılama eşiği `0.70`'tir:

```bash
.venv/bin/python wake_word.py --threshold 0.70
```

Duyarlılığı değiştirmemek için varsayılan karar tek eşik-üstü kareyle
verilir. İstenirse iki art arda karenin eşiği geçmesi zorunlu tutulabilir:

```bash
.venv/bin/python wake_word.py --confirmation-frames 2
```

Bu filtre anlık skor sıçramalarını azaltabilir, ancak kısa/hızlı gerçek
ifadeleri de kaçırabilir. Bu nedenle varsayılan olarak etkin değildir ve kendi
pozitif kayıtlarınızla recall testi yapmadan kalıcı ayar olarak kullanılmamalıdır.

Daha düşük eşik hedef ifadeyi yakalama olasılığını artırabilir, fakat
yanlış tetiklemeyi de artırır. Tetiklenmesi gereken sözleri korumak için
eşiği tek başına yükseltmek yerine gerçek pozitif kayıtlar ve uzun negatif
seslerle birlikte kalibre edin. Tüm seçenekler:

```bash
.venv/bin/python wake_word.py --help
```

## Gizlilik ve veri akışı

Normal mikrofon modunda uygulamanın veri yolu şöyledir:

```text
arecord -> 80 ms ham ses (RAM) -> yerel ONNX modelleri -> skor
```

Uygulama kendi başına:

- sesi diske yazmaz veya eğitim verisine eklemez,
- sesi bir API'ye/internete göndermez,
- konuşmayı metne çevirmez,
- skorları veya tetiklenmeleri bir log dosyasında saklamaz.

Ses parçaları ve model bağlamı işlem sırasında RAM'de geçici olarak bulunur.
Terminal çıktısını dosyaya yönlendirmeniz, kabuk/terminal kaydı, işletim
sistemi ses katmanı veya uygulamaya sonradan eklenen entegrasyonlar ayrı veri
saklama davranışları oluşturabilir.

## Test ve doğrulama

Geliştirme bağımlılıkları ve testler:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-deps openwakeword==0.6.0
.venv/bin/python -m unittest discover -s tests -v
```

GitHub Actions ayrıca Python sözdizimini, bütün izlenen ONNX dosyalarını,
model/rapor hash eşleşmesini ve sıfır sesle kısa inference smoke testini
denetler. Model sonuçları ve sınırlamalar için
[MODEL_EGITIMI.md](MODEL_EGITIMI.md) belgesine bakın.

## Hafif yerel eğitim

Eğitim ortamı runtime ortamından ayrıdır ve Python 3.10 veya yenisini ister:

```bash
PYTHON_BIN=python3.12 training/setup_training.sh
```

Sentetik veri üretimi ayrı Piper ses dosyaları gerektirir. Bu dosyalar ve
üretilen WAV'lar boyut/lisans nedenleriyle Git'e eklenmez ve otomatik
indirilmez. Betikteki üç ad mevcut modelin tarihsel girdilerini gösterir;
erişilebilirlik veya kullanım izni garantisi değildir. Yalnız kullanma
yetkisini doğruladığınız dosyaları `training/voices/` altında bulundurun.
Ardından:

```bash
.train-venv/bin/python training/generate_turkish_samples.py
.train-venv/bin/python training/train_hey_orbit.py \
  --train-splits train dev --split test
.venv/bin/python training/evaluate_streaming.py \
  --model models/candidates/hey_orbit_candidate.onnx \
  --split test --json-output /tmp/hey_orbit_candidate_test.json

.venv/bin/python training/compare_evaluations.py \
  models/evaluation/hey_orbit_test.json \
  /tmp/hey_orbit_candidate_test.json \
  --baseline-model models/hey_orbit.onnx \
  --candidate-model models/candidates/hey_orbit_candidate.onnx \
  --candidate-training-metrics models/candidates/hey_orbit_candidate_metrics.json
```

Mevcut `training/data/` setini bilinçli olarak yeniden üretmek için veri
üreticisine `--replace-existing` verilmelidir; bayrak olmadan eski set korunur.

Eğitim, kabul edilmiş `models/hey_orbit.onnx` dosyasını varsayılan olarak
ezmez; yeni modeli `models/candidates/` altına yazar. Adayın toplam veya ifade
bazında pozitif yakalaması gerilerse, bütün hedefleri yakalayamazsa ya da tek
bir negatif ifade bile kötüleşirse `training/compare_evaluations.py` adayı
reddeder. Ayrıntılı ve güncel kabul akışı
[MODEL_EGITIMI.md](MODEL_EGITIMI.md) belgesindedir.

Temiz klon, Piper sesleri, manifest ve üretilmiş WAV'ları içermediği için
eğitimi veya ayrıntılı akış raporlarını tek başına yeniden üretemez. Mevcut
`hey_orbit.onnx` modelini de sıfırdan birebir üretmez. openWakeWord'un yaklaşık
17 GB özellik verisi kullanan resmî büyük-veri akışı ayrıca
[OPENWAKEWORD_EGITIMI.md](OPENWAKEWORD_EGITIMI.md) belgesindedir.

## Lisans ve model kullanımı

Bu depo için proje genelinde bir açık kaynak lisansı henüz seçilmemiştir.
Özel modelin eğitiminde kullanılan seslerden biri CC BY-NC-SA kaynağına
dayanır; iki ses de upstream'den katkıcıların isteğiyle kaldırılmıştır.
Bu nedenle `hey_orbit.onnx` için ticari kullanım veya yeniden dağıtım hakkı
varsayılmamalıdır.

Ayrıntılar ve kaynak bağlantıları:
[LICENSE](LICENSE) ve [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
