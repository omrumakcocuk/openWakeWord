# Resmî openWakeWord eğitim planı

Hazırlanan yapılandırma: `training/hey_orbit_openwakeword.yml`.

Bu yapılandırma openWakeWord 0.6.0 kaynak kodundaki `train.py` ile doğrulandı.
Mevcut PyPI paketi inference sınıflarını içeriyor ancak `train.py`, PyTorch ve
ses augmentation bağımlılıklarını kurmuyor. Bilgisayardaki tek Python sürümü
3.14 olduğu için resmî eğitim zincirinin sabitlenmiş eski bağımlılıkları mevcut
`.venv` içine kurulmaya çalışılmamalıdır. Eğitim için ayrı, desteklenen bir
Python 3.10/3.11 ortamı veya uyumlu bir container kullanılmalıdır.

## Ayrıca gereken veriler

- `openwakeword_features_ACAV100M_2000_hrs_16bit.npy`: ACAV100M'den yaklaşık
  2.000 saatlik generic negative feature verisi, **17.280.000.128 bayt**.
- `validation_set_features.npy`: FP/saat hesabı için yaklaşık 11,3 saatlik
  doğrulama feature verisi, **184.836.608 bayt**.
- MIT Environmental Impulse Responses: `training/assets/rir/` altında WAV RIR
  dosyaları. Resmî eğitim notundaki kaynak:
  `https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses`.
- AudioSet/FSD50K gibi konuşma, fan, TV ve müzik içeren arka plan WAV'ları:
  `training/assets/background/`.
- openWakeWord'un kullandığı `piper-sample-generator`:
  `training/vendor/piper-sample-generator/`.

İlk iki dosyanın resmî proje dokümanında verilen kaynağı:
`https://huggingface.co/datasets/davidscripka/openwakeword_features`.

ACAV dosyası yaklaşık 17,3 GB olduğundan kullanıcı onayı olmadan indirilmez.
RIR ve background klasörleri boşken resmî `train.py` çalıştırılmamalıdır.
Resmî örnekteki Piper üreticisinin varsayılan İngilizce sesi Türkçe telaffuz için
körlemesine kullanılmamalıdır. Türkçe pozitifler mevcut DFKI/Fahrettin/Fettah
üreticisiyle hazırlanmalı veya Türkçe destekli eşdeğer bir üretici seçilmelidir.

## Nelerin faydası var?

- `custom_negative_phrases` ve `adversarial_negative`: Robert/rabbit/orbit gibi
  fonetik karışmaları doğrudan azaltır.
- ACAV100M generic negative verisi: sıradan konuşma, müzik ve gürültü kaynaklı
  false-positive sorununa en büyük katkıyı sağlar.
- RIR ve background augmentation: uzak/yankılı/gürültülü “Hey Orbit” recall
  değerini korumaya yardım eder.
- Ayrı 11,3 saatlik doğrulama verisi: accuracy yerine gerçek FP/saat seçimi
  yapılmasını sağlar.
- `max_negative_weight`: yalnız gerçek FP/saat verisiyle birlikte anlamlıdır;
  körlemesine yükseltilmemiş, resmî başlangıç değeri `1500` tutulmuştur.

`augmentation_rounds: 2`, `n_samples: 50000` ve `n_samples_val: 5000` yalnız
eğitim süresini ve disk kullanımını artırır. Dışa aktarılan model `dnn` ve
`layer_size: 32` kaldığı için Raspberry Pi inference maliyeti büyümez.

Custom verifier kullanılmaz. Mikrofon veya false-positive seslerini otomatik
kaydeden hiçbir sistem eklenmemiştir.
