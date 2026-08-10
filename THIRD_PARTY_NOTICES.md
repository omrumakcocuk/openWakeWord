# Üçüncü taraf bildirimleri ve model provenansı

Bu belge hukuki tavsiye değildir. Depodaki özgün kod için proje genelinde bir
lisans seçilmemiştir; [LICENSE](LICENSE) dosyasına bakın. Bir bağımlılığın
burada anılması onun lisans metninin yerine geçmez.

## Runtime bileşenleri

- **openWakeWord 0.6.0**: David Scripka ve katkıcıları; kod Apache License
  2.0. Kaynak ve lisans:
  <https://github.com/dscripka/openWakeWord/tree/v0.6.0> ve
  <https://www.apache.org/licenses/LICENSE-2.0>.
- **openWakeWord wake-word modelleri**: upstream proje, birlikte yayımladığı
  önceden eğitilmiş wake-word modellerini CC BY-NC-SA 4.0 olarak belirtir.
  Bu depodaki `hey_jarvis_v0.1.*` bu gruptandır ve uygulamanın varsayılan
  modeli değildir. Dosyalar upstream release varlığından byte değişikliği
  yapılmadan alınmıştır; atıf David Scripka/openWakeWord projesinedir ve lisans
  <https://creativecommons.org/licenses/by-nc-sa/4.0/> adresindedir.
- **Google speech embedding omurgası**: openWakeWord belgeleri, temel
  embedding modelinin Google TFHub speech embedding modelinden geldiğini ve
  Apache License 2.0 altında olduğunu belirtir.
- **Silero VAD**: `models/silero_vad.onnx` depoda bulunsa da mevcut uygulama
  akışında etkin değildir. Dosya aşağıdaki openWakeWord v0.5.1 release
  varlığından gelir; kaynak Silero VAD projesi MIT lisansını yayımlar:
  <https://github.com/snakers4/silero-vad> ve
  <https://github.com/snakers4/silero-vad/blob/master/LICENSE>.
- Python paketlerinin (`numpy`, `onnxruntime`, `requests`, `tqdm` ve diğerleri)
  lisansları kendi dağıtımlarında bulunur. `pip` ile kurulmaları bu proje
  için yeniden lisanslandıkları anlamına gelmez.

### Depodaki upstream model varlıklarının kimliği

Aşağıdaki dosyalar openWakeWord 0.6.0'ın tanımladığı v0.5.1 release URL'lerinden
alınmıştır. SHA-256 değerleri bu depodaki byte içeriğini sabitler; bir lisans
izni yerine geçmez.

| Dosya | SHA-256 | Upstream release varlığı |
|---|---|---|
| `embedding_model.onnx` | `70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f` | [v0.5.1 ONNX](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx) |
| `embedding_model.tflite` | `c0aea21eb84a4ce90a08c870da41b7a7173b45269e6a3207c71d67c40f3a59d8` | [v0.5.1 TFLite](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite) |
| `melspectrogram.onnx` | `ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f` | [v0.5.1 ONNX](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx) |
| `melspectrogram.tflite` | `96fa0adccb6e8cf95cb14465409a1a2898ee4a96a85bb9ed3c7eb0e68bf163e8` | [v0.5.1 TFLite](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite) |
| `hey_jarvis_v0.1.onnx` | `94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb` | [v0.5.1 ONNX](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx) |
| `hey_jarvis_v0.1.tflite` | `14bff778604985e1b5c19f0f7bbe477a69cf281d8db34b232b3b972411f710e2` | [v0.5.1 TFLite](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.tflite) |
| `silero_vad.onnx` | `a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28` | [v0.5.1 ONNX](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/silero_vad.onnx) |

Mevcut `wake_word.py` yalnız iki ONNX özellik modeli ile özel
`hey_orbit.onnx` dosyasını yükler. Jarvis, TFLite ve Silero dosyaları runtime
yolunda kullanılmaz.

## `hey_orbit.onnx` eğitim provenansı

`models/hey_orbit.onnx`, openWakeWord mel-spektrogram/embedding özellikleri ve
üç Piper TTS sesiyle üretilen sentetik sesler kullanılarak eğitildi:

- `tr_TR-dfki-medium`: upstream model kartı, kaynak DFKI veri kümesini
  **CC BY-NC-SA 4.0** olarak belirtir.
- `tr_TR-fahrettin-medium` ve `tr_TR-fettah-medium`: eski upstream model
  kartları kaynak veri kümelerini **CC0** olarak belirtiyordu. Ancak Aralık
  2025'te bu kartlardaki veri kaynağının doğruluğu sorgulandı ve iki ses daha
  sonra katkıcıların isteğiyle kaldırıldı.

Fahrettin ve Fettah ses dosyaları, upstream `rhasspy/piper-voices` deposundan
30 Aralık 2025 tarihinde "katkıcıların isteğiyle" kaldırıldı. Bu depo söz
konusu Piper ses modellerini veya üretilen eğitim WAV'larını dağıtmaz ve
kurulum betiği bunları otomatik indirmez.

Bu kaynaklardan eğitilen `hey_orbit.onnx` için ticari kullanım veya yeniden
dağıtım izni bu depoda ileri sürülmemektedir. Özellikle CC BY-NC-SA girdisi
ve kaldırılan sesler nedeniyle, modeli dağıtmadan ya da ticari bir üründe
kullanmadan önce hak sahiplerinden/provenans kaynaklarından izin ve lisans
uyumluluğu doğrulanmalıdır.

## Eğitim aracı

Sentetik örnek üretiminde kullanılan `piper-tts` paketi, kurulan sürümün
kendi lisansına tabidir. Güncel paket `OHF-Voice/piper1-gpl` projesine dayanır
ve GPL-3.0-or-later olarak yayımlanır:
<https://github.com/OHF-Voice/piper1-gpl>. Piper motorunun lisansı ile her ses
modelinin/veri kümesinin lisansı ayrı ayrı kontrol edilmelidir.

## Kaynak bağlantıları

- openWakeWord lisans açıklaması:
  <https://github.com/dscripka/openWakeWord#license>
- DFKI ses model kartı:
  <https://huggingface.co/rhasspy/piper-voices/blob/main/tr/tr_TR/dfki/medium/MODEL_CARD>
- Fahrettin/Fettah kaldırma commit'i:
  <https://huggingface.co/rhasspy/piper-voices/commit/b145f2ac26522d6a2ccde0164b7b0e48b1e3199c>
- Türkçe model kartı veri kaynağı tartışması:
  <https://huggingface.co/rhasspy/piper-voices/discussions/58>
