# Türkçe “Hey Orbit” modeli

Model bu bilgisayarda gerçekten eğitildi; başka bir modelin adı değiştirilmedi.
Çıktı dosyası `models/hey_orbit.onnx` konumundadır.

## Eğitim yöntemi

- Üç yerel Türkçe Piper sesi kullanıldı: DFKI, Fahrettin ve Fettah.
- “Hey Orbit” için 270 temel olumlu sentetik kayıt üretildi. Dokuz yazım ve
  telaffuz varyasyonunun (`HEY    ORBIT`, `HEY ORBIT`, `HeyOrbit`, `heyorbit`,
  `hey-orbit`, `Heyorbit`, `heyOrbit!`, `Hey, Orbit`, `Hey Orbit!`) her biri üç
  seste 30'ar kayıtla eşit temsil edilir. Bitişik ifadeler ayrıca daha hızlı
  sentezlenir.
- Yalnız “hey” ve “orbit” ile “hey robot”, “hey robert”, “hey corbett”,
  “hey rabbit”, “hey or”, “orbital”, “orbiting”, “a/the/okay orbit” gibi
  karıştırılabilecek ifadeler özel olumsuz örnek olarak kullanıldı.
- Türkçe gündelik cümleler, sessizlik, ton, tıklama ve farklı gürültüler olumsuz
  veri olarak eklendi.
- Hız, ses seviyesi, oda yankısı, zaman konumu ve sinyal-gürültü oranı rastgele
  değiştirilerek toplam 15.480 öznitelik örneği hazırlandı. Yalnız “hey”, yalnız
  “orbit”, “hey korbit” ailesi ve eksik/benzer ifadeler özellikle fazla
  ağırlıklandırıldı.
- Resmî openWakeWord mel-spektrogram ve Google speech-embedding ONNX omurgası
  kullanıldı. Üst sınıflandırıcı iki gizli katmanlı küçük bir sinir ağıdır.

## Eğitim sırasındaki sentetik doğrulama sonucu

| Ölçüm | Sonuç |
|---|---:|
| Model değerlendirme eşiği | 0,75 |
| Eğitim örneği | 12.416 |
| Doğrulama örneği | 3.064 |
| Doğruluk | %92,36 |
| Kesinlik | %92,54 |
| Yakalama (recall) | %94,04 |
| ROC AUC | %97,26 |
| Yanlış olumlu / yanlış olumsuz | 131 / 103 |

Bu tablo model üretilirken kullanılan eski sabit-öznitelik ayrımına aittir.
Uygulamanın gerçek akış davranışını veya saat başına yanlış tetiklemeyi tek
başına göstermez.

## Deterministik 80 ms akış testi

openWakeWord'ün rastgele reset tamponu her dosyadan önce 2.080 ms sessizlikle
tamamen değiştirilir. Test `0,70` eşiği, tek-kare kararı ve WAV sonrası 400 ms
akış kuyruğuyla yapılır:

| Bölüm | Pozitif yakalama | Yanlış pozitif |
|---|---:|---:|
| Sabit manifest `test` regresyon seti | 27/27 | 7/62 |
| Bütün temel sentetik kayıtlar (tanısal) | 262/270 | 28/186 |

`test` setinde dokuz hedef varyasyonun her biri 3/3 algılandı. Bu WAV'lar kabul
edilmiş model eğitildikten sonra yeniden üretildi; ancak aynı üç sentetik Piper
sesi ve aynı ifade ailesi kullanıldı, split de modelden önce sabitlenmemişti.
Dolayısıyla `27/27` bağımsız genelleme sonucu değil, davranışın gerilemesini
önleyen **post-training sentetik regresyon referansıdır**. Yeni hatla eğitilen
aday için `test` temel kayıtları eğitimden ayrı tutulur; yine de aynı üç
sentetik konuşmacı splitler arasında bulunduğundan konuşmacı genellemesi
ölçülmez.

Bütün veride `HEY ORBIT`, `HEY    ORBIT`, `Hey, Orbit`, `Heyorbit`,
`hey-orbit` ve `heyorbit` grupları 30/30; `Hey Orbit!` 28/30, `HeyOrbit`
25/30 ve `heyOrbit!` 29/30 algılandı. Tek başına `orbit` 0/3 tetikleme verdi.
Ayrıntılı model, manifest ve WAV-içerik SHA-256 değerleri
`models/evaluation/` altındadır.

Bu sonuçlar sentetiktir; gerçek oda/mikrofon performansını veya saat başına
yanlış tetiklemeyi garanti etmez.

Akış testi yeniden çalıştırmak için:

```bash
.venv/bin/python training/evaluate_streaming.py --split test --threshold 0.70
```

Bu komut `training/data/manifest.json` ve onun WAV dosyalarını gerektirir.
Bunlar boyut/lisans nedeniyle Git'e eklenmediğinden temiz klonda bulunmaz.

Özet `models/hey_orbit_metrics.json`, ifade bazlı ayrıntılar ise
`models/evaluation/*.json` dosyalarındadır.

## Yeniden üretme

Türkçe temel sesler:

```bash
.train-venv/bin/python training/generate_turkish_samples.py
```

`training/data/` zaten varsa betik onu korur. Baştan üretmek istediğinizden
eminseniz `--replace-existing` ekleyin; yeni set tümüyle doğrulanmadan eski set
değiştirilmez.

Öznitelik çıkarma, eğitim ve ONNX adayı dışa aktarma:

```bash
.train-venv/bin/python training/train_hey_orbit.py \
  --train-splits train dev --split test
```

Eğitim varsayılan olarak kabul edilmiş modeli ezmez; adayı
`models/candidates/` altına yazar. Adayı aynı manifest ve ayarlarla test edip
recall-korumalı kapıdan geçirin:

```bash
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

Karşılaştırıcı toplam pozitif yakalama, her hedef varyasyonun yakalaması
ve yanlış pozitif sayısı gerilemediğinde başarılı olur. Ayrıca her negatif
ifadeyi ayrı denetler; raporun gerçek model, manifest ve WAV içerik hashleriyle
eşleşmesini zorunlu tutar. Kabul kapısı yalnız `test` splitini kabul eder.
`--split all` yalnız tanısal incelemedir; eğitim verisini içerdiği için kabul
kanıtı olarak kullanılmamalıdır.

Eğitim veri üreticisi `piper-tts`; sınıflandırma ortamı `openwakeword`,
`scikit-learn` ve `onnx` paketlerini gerektirir. Türkçe ses modelleri
`training/voices/` altında tutulur.

## Gerçek sesle iyileştirme

Bu model üç sentetik konuşmacıya dayanır. Farklı kişilerde ve uzak mikrofonda daha
iyi genelleme için gerçek “Hey Orbit” kayıtları olumlu veri havuzuna eklenip model
yeniden eğitilebilir. Uygulamanın varsayılan çalışma eşiği `0.70` değeridir:

```bash
.venv/bin/python wake_word.py --threshold 0.70
```

İlk tetiklemenin ardından art arda yanlış algılamayı önlemek için çalışma
zamanı kapısı, skor `0.30` altında beş kare kalmadan yeniden kurulmaz. Bu ayar
ilk tetiklemenin recall değerini değiştirmez; gerekirse
`--release-threshold` ve `--rearm-frames` ile kalibre edilebilir.

Eşiği düşürmek yanlış tetikleme riskini artırabilir; yükseltmek ise hedef
ifadeleri kaçırabilir. Kalıcı eşik değişikliği gerçek pozitif ve uzun süreli
negatif kayıtlarla birlikte ölçülmelidir.
