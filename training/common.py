"""Eğitim veri manifesti için ortak, deterministik yardımcılar.

Eğitim ve değerlendirme betikleri dosya sistemini taramak yerine yalnızca
doğrulanmış manifest kayıtlarını kullanır. Bu, daha önce üretilmiş fakat
artık manifestte bulunmayan WAV dosyalarının sessizce eğitime karışmasını
engeller.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training" / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
VALID_SPLITS = ("train", "dev", "test")
DEFAULT_SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
SPLIT_SEED = 20260808


@dataclass(frozen=True)
class ManifestItem:
    """Doğrulanmış tek bir temel ses kaydı."""

    path: str
    absolute_path: Path
    label: int
    text: str
    split: str
    group: str
    voice: str | None = None

    def as_json(self) -> dict[str, object]:
        result: dict[str, object] = {
            "path": self.path,
            "label": self.label,
            "text": self.text,
            "split": self.split,
            "group": self.group,
        }
        if self.voice:
            result["voice"] = self.voice
        return result


def _stable_order_key(seed: int, stratum: tuple[int, str], group: str) -> bytes:
    value = f"{seed}\0{stratum[0]}\0{stratum[1]}\0{group}".encode("utf-8")
    return hashlib.sha256(value).digest()


def _split_counts(group_count: int, ratios: Mapping[str, float]) -> dict[str, int]:
    if group_count <= 0:
        return {split: 0 for split in VALID_SPLITS}
    if set(ratios) != set(VALID_SPLITS):
        raise ValueError(f"Bölme oranları tam olarak {VALID_SPLITS} anahtarlarını içermeli")
    if any(not isinstance(value, (int, float)) or value < 0 for value in ratios.values()):
        raise ValueError("Bölme oranları negatif olmayan sayılar olmalı")
    ratio_total = float(sum(ratios.values()))
    if ratio_total <= 0:
        raise ValueError("En az bir bölme oranı sıfırdan büyük olmalı")

    exact = {split: group_count * float(ratios[split]) / ratio_total for split in VALID_SPLITS}
    counts = {split: int(exact[split]) for split in VALID_SPLITS}
    remainder = group_count - sum(counts.values())
    remainder_order = sorted(
        VALID_SPLITS,
        key=lambda split: (exact[split] - counts[split], ratios[split], split == "train"),
        reverse=True,
    )
    for split in remainder_order[:remainder]:
        counts[split] += 1

    # Bir ifade en az üç bağımsız gruba sahipse her bölmede temsil
    # edilir. Özellikle her negatif ifadenin üç Piper sesi bulunuyor.
    nonzero_splits = [split for split in VALID_SPLITS if ratios[split] > 0]
    if group_count >= len(nonzero_splits):
        for missing_split in (split for split in nonzero_splits if counts[split] == 0):
            donor = max(nonzero_splits, key=lambda split: counts[split])
            if counts[donor] <= 1:
                raise ValueError("Bölme kapsamı için yeterli grup bulunamadı")
            counts[donor] -= 1
            counts[missing_split] += 1
    return counts


def assign_splits(
    items: Sequence[ManifestItem],
    *,
    seed: int = SPLIT_SEED,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
) -> list[ManifestItem]:
    """Kaynak gruplarını sızdırmadan ifade/etiket bazında böler.

    Aynı ``group`` değerindeki kayıtlar daima aynı bölmede kalır. Her
    etiket+ifade katmanı ayrı bölündüğü için küçük ama yeterli
    katmanlarda train/dev/test kapsamı korunur.
    """

    group_items: dict[str, list[ManifestItem]] = defaultdict(list)
    for item in items:
        group_items[item.group].append(item)

    strata: dict[tuple[int, str], list[str]] = defaultdict(list)
    for group, members in group_items.items():
        group_strata = {(item.label, item.text) for item in members}
        if len(group_strata) != 1:
            raise ValueError(
                f"Aynı kaynak grubu farklı etiket/ifadeler içeriyor: {group!r}"
            )
        strata[next(iter(group_strata))].append(group)

    group_split: dict[str, str] = {}
    for stratum, groups in sorted(strata.items()):
        ordered_groups = sorted(groups, key=lambda group: _stable_order_key(seed, stratum, group))
        counts = _split_counts(len(ordered_groups), ratios)
        offset = 0
        for split in VALID_SPLITS:
            for group in ordered_groups[offset : offset + counts[split]]:
                group_split[group] = split
            offset += counts[split]

    return [replace(item, split=group_split[item.group]) for item in items]


def _parse_manifest_payload(payload: object) -> list[object]:
    # Liste biçimi eski manifestlerle uyumluluğu korur. Gelecekte metadata
    # eklenebilmesi için {"samples": [...]} biçimi de kabul edilir.
    if isinstance(payload, dict):
        payload = payload.get("samples")
    if not isinstance(payload, list):
        raise ValueError("Manifest bir JSON listesi veya 'samples' listesi içeren nesne olmalı")
    return payload


def validate_manifest_records(
    records: Iterable[object],
    *,
    project_root: Path = ROOT,
    data_dir: Path = DATA_DIR,
    require_files: bool = True,
    allow_legacy_split: bool = True,
    split_seed: int = SPLIT_SEED,
) -> list[ManifestItem]:
    """Manifest yapısını, yollarını ve bölme tutarlılığını doğrula."""

    root_resolved = project_root.resolve()
    data_resolved = data_dir.resolve()
    items: list[ManifestItem] = []
    seen_paths: set[str] = set()
    explicit_split_count = 0

    for index, raw in enumerate(records):
        prefix = f"Manifest kaydı {index}"
        if not isinstance(raw, dict):
            raise ValueError(f"{prefix} bir JSON nesnesi olmalı")

        relative_path = raw.get("path")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError(f"{prefix} geçerli bir 'path' içermeli")
        pure_path = PurePosixPath(relative_path)
        if pure_path.is_absolute() or ".." in pure_path.parts or "\\" in relative_path:
            raise ValueError(f"{prefix} proje içinde güvenli, göreli bir yol içermeli")
        normalized_path = pure_path.as_posix()
        if normalized_path in seen_paths:
            raise ValueError(f"Manifestte yinelenen yol: {normalized_path}")
        seen_paths.add(normalized_path)

        absolute_path = (root_resolved / normalized_path).resolve()
        try:
            absolute_path.relative_to(data_resolved)
        except ValueError as exc:
            raise ValueError(f"Manifest yolu veri dizininin dışında: {normalized_path}") from exc
        if absolute_path.suffix.lower() != ".wav":
            raise ValueError(f"Manifest yolu WAV olmalı: {normalized_path}")
        if require_files and not absolute_path.is_file():
            raise FileNotFoundError(f"Manifestteki WAV bulunamadı: {absolute_path}")

        label = raw.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"{prefix} etiketi 0 veya 1 olmalı")
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{prefix} boş olmayan bir 'text' içermeli")

        split = raw.get("split")
        if split is not None:
            explicit_split_count += 1
            if split not in VALID_SPLITS:
                raise ValueError(f"{prefix} geçersiz bölme içeriyor: {split!r}")
        group = raw.get("group", normalized_path.removesuffix(".wav"))
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"{prefix} geçerli bir 'group' içermeli")
        voice = raw.get("voice")
        if voice is not None and (not isinstance(voice, str) or not voice.strip()):
            raise ValueError(f"{prefix} 'voice' alanı metin olmalı")

        items.append(
            ManifestItem(
                path=normalized_path,
                absolute_path=absolute_path,
                label=int(label),
                text=text,
                split=str(split or ""),
                group=group,
                voice=voice,
            )
        )

    if not items:
        raise ValueError("Manifest en az bir ses kaydı içermeli")
    if explicit_split_count not in (0, len(items)):
        raise ValueError("Manifestte ya tüm kayıtların 'split' alanı olmalı ya da hiçbirinde olmamalı")
    if explicit_split_count == 0:
        if not allow_legacy_split:
            raise ValueError("Manifest kayıtları açık bir 'split' alanı içermeli")
        warnings.warn(
            "Eski manifestte split alanı yok; deterministik train/dev/test bölmesi bellekte oluşturuldu. "
            "Kalıcı açık bölme için veri üreticisini yeniden çalıştırın.",
            UserWarning,
            stacklevel=2,
        )
        items = assign_splits(items, seed=split_seed)

    group_splits: dict[str, set[str]] = defaultdict(set)
    stratum_groups: dict[tuple[int, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for item in items:
        group_splits[item.group].add(item.split)
        stratum_groups[(item.label, item.text)][item.split].add(item.group)
    leaked_groups = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaked_groups:
        raise ValueError(f"Kaynak grupları bölmeler arasında sızıyor: {leaked_groups[:5]}")

    # Yeterli bağımsız grubu olan her hedef ve negatif ifade tüm bölmelerde
    # bulunmalı. Böylece bir hedef varyasyonunun recall kapısından sessizce
    # kaybolması da engellenir. Daha az kayıtta bu matematiksel olarak mümkün
    # olmayabilir.
    for (label, text), split_groups in stratum_groups.items():
        all_groups = set().union(*split_groups.values())
        if len(all_groups) >= len(VALID_SPLITS):
            missing = set(VALID_SPLITS) - set(split_groups)
            if missing:
                raise ValueError(
                    f"Etiket={label} ifadesi yeterli gruba sahip olduğu halde "
                    f"bölme kapsamı eksik: {text!r} -> {sorted(missing)}"
                )
    return items


def load_manifest(
    path: Path = MANIFEST_PATH,
    *,
    project_root: Path = ROOT,
    data_dir: Path = DATA_DIR,
    require_files: bool = True,
    allow_legacy_split: bool = True,
) -> list[ManifestItem]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Manifest bulunamadı: {path}. Önce generate_turkish_samples.py çalıştırılmalı."
        ) from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest geçerli JSON değil: {path}: {exc}") from exc
    return validate_manifest_records(
        _parse_manifest_payload(payload),
        project_root=project_root,
        data_dir=data_dir,
        require_files=require_files,
        allow_legacy_split=allow_legacy_split,
    )


def atomic_write_json(path: Path, payload: object) -> None:
    """JSON dosyasını aynı dizinde atomik olarak değiştir."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    """Dosya içeriğinin SHA-256 özetini akış halinde hesapla."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_set_sha256(items: Sequence[ManifestItem]) -> str:
    """Seçili kayıtların metadata ve gerçek WAV içeriklerini kimliklendir."""

    digest = hashlib.sha256(b"hey-orbit-evaluation-set-v1\0")
    for item in sorted(items, key=lambda entry: entry.path):
        metadata = json.dumps(
            item.as_json(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(bytes.fromhex(sha256_file(item.absolute_path)))
    return digest.hexdigest()


def manifest_split_summary(items: Sequence[ManifestItem]) -> dict[str, dict[str, int]]:
    return {
        split: {
            "positive": sum(item.label == 1 and item.split == split for item in items),
            "negative": sum(item.label == 0 and item.split == split for item in items),
        }
        for split in VALID_SPLITS
    }
