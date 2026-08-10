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

## Ayrılmış sentetik doğrulama sonucu

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

Uygulamanın güncel 80 ms akış yolunda VAD, RMS ve ardışık kare doğrulaması yoktur.
`0,70` eşiğinde ve WAV sonrası 400 ms akış kuyruğuyla pozitiflerin 258/270'i
algılandı. Yeni `heyOrbit!` grubu 27/30, `HEY ORBIT` ve `HEY    ORBIT`
gruplarının her biri 30/30 algılandı. Tek başına “orbit” 0/3 tetikleme verdi.
Toplam 186 sentetik negatif örneğin 19'u yanlış tetiklendi. Bu tek-kare test modu
recall'u yükseltirken false-positive riskini de artırır. Sentetik testler gerçek
oda/mikrofon performansını garanti etmez.

Akış testi yeniden çalıştırmak için:

```bash
.venv/bin/python training/evaluate_streaming.py --threshold 0.70
```

Makine tarafından yazılan ayrıntılı ölçümler `models/hey_orbit_metrics.json`
dosyasındadır.

## Yeniden üretme

Türkçe temel sesler:

```bash
.train-venv/bin/python training/generate_turkish_samples.py
```

Öznitelik çıkarma, eğitim ve ONNX dışa aktarma:

```bash
.venv/bin/python training/train_hey_orbit.py
```

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

Eşiği düşürmek eksik ifadelerin yeniden tetikleme riskini artırabilir.
