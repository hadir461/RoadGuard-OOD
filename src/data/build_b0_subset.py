"""Select whole train_id groups and package unchanged YOLO copies for Kaggle.

No model, raw data, test image or XML is opened. Selection uses train metadata
only. Existing v1 files/reports are read-only; destinations must be new.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import zipfile

from .audit import CLASSES
from .build_manifest import write_csv
from .prepare_yolo import inside, load_inventory, read_csv, sha, similarity_components, write_json


FIELDS = ("split", "group_id", "country", "category", "source_image", "image_path", "label_path",
          "image_sha256", "label_sha256", "D00", "D10", "D20", "D40", "width", "height")
REPORTS = ("manifest.csv", "selected_train.csv", "coverage.csv", "class_statistics.csv",
           "group_selection.csv", "summary.json", "KAGGLE.md")
CONFIG = {"train": "images/train", "val": "images/val", "names": list(CLASSES)}


def features(row):
    """Country/category/class coverage; standardized errors protect rare strata."""
    country = row["country"]
    result = Counter({"images": 1, f"country:{country}": 1,
                      f"category:{row['category']}": 1,
                      f"{country}:category:{row['category']}": 1,
                      f"dimensions:{row['width']}x{row['height']}": 1})
    for name in CLASSES:
        count = int(row[name])
        result[f"{country}:presence:{name}"] = int(count > 0)
        result[f"{country}:instances:{name}"] = count
    return +result


def select_groups(train, target=5000, seed=42):
    """Greedy balanced allocation with exact subset-sum feasibility, no split groups.

    Seeded ties, largest groups first then rare-feature coverage. Minimize the
    increment of weighted squared deviations / target count, for selected and
    remainder. Country/image weights=2, instances=0.5, other features=1.
    Suffix bitsets guarantee the closest attainable size (ties prefer smaller).
    This is not a global optimum of stratification or a random i.i.d. sample.
    """
    if not train or target < 1 or target > len(train):
        raise ValueError("Requested size must be between 1 and the train_id size")
    grouped = defaultdict(list)
    for row in sorted(train, key=lambda r: r["image_path"]):
        if row["split"] != "train_id" or row["country"] not in {"India", "Japan"}:
            raise ValueError("Selection accepts train_id India/Japan only")
        grouped[row["group_id"]].append(row)
    vectors = {group: sum((features(row) for row in members), Counter())
               for group, members in grouped.items()}
    totals = sum(vectors.values(), Counter())
    rng = random.Random(seed)
    order = sorted(grouped)
    rng.shuffle(order)
    order.sort(key=lambda group: (-len(grouped[group]),
                                 -sum(n / totals[k] for k, n in vectors[group].items())))
    reachable = [0] * (len(order) + 1)
    reachable[-1] = 1
    for i in range(len(order) - 1, -1, -1):
        bits = reachable[i + 1]
        reachable[i] = bits | (bits << len(grouped[order[i]]))
    attained = min((n for n in range(1, len(train) + 1) if (reachable[0] >> n) & 1),
                   key=lambda n: (abs(n - target), n))
    ratio = attained / len(train)
    actual = [Counter(), Counter()]
    selected = set()
    remaining = attained
    for i, group in enumerate(order):
        size, vector = len(grouped[group]), vectors[group]
        options = [side for side in (0, 1)
                   if remaining - (size if side == 0 else 0) >= 0
                   and (reachable[i + 1] >> (remaining - (size if side == 0 else 0))) & 1]
        rng.shuffle(options)

        def score(side):
            proportion = ratio if side == 0 else 1 - ratio
            if not proportion:
                return float("inf")
            return sum((2 if key == "images" or key.startswith("country:") else
                        0.5 if ":instances:" in key else 1) *
                       (2 * (actual[side][key] - proportion * totals[key]) * n + n * n)
                       / (proportion * totals[key]) for key, n in vector.items())

        side = min(options, key=score)
        actual[side].update(vector)
        if side == 0:
            selected.add(group)
            remaining -= size
    if remaining:
        raise ValueError("Internal cardinality error")
    return selected, attained


def load_sources(reports):
    """Reuse validated components; test metadata is used only for leakage checks."""
    reports = Path(reports)
    baseline = json.loads((reports / "yolo_preparation_summary.json").read_text(encoding="utf-8"))
    if not baseline.get("validation", {}).get("ok"):
        raise ValueError("Phase 2.2 validation has not passed")
    for name, digest in baseline["input_sha256"].items():
        if sha(inside(reports, name)) != digest:
            raise ValueError(f"Phase 2.2 input changed: {name}")
    original, pairs, inventory = load_inventory(reports)
    _, membership = similarity_components(original, pairs, inventory["near_duplicates"]["threshold"])
    original = {r["image_path"]: r for r in original}
    records = read_csv(reports / "split_manifest.csv")
    if len({r["source_image"] for r in records}) != len(records):
        raise ValueError("Repeated source in split manifest")
    groups, hashes = {}, {}
    for row in records:
        split = row["split"]
        if split not in baseline["splits"]:
            raise ValueError("Unexpected split")
        source = original[row["source_image"]]
        if row["group_id"] != membership[row["source_image"]] or row["country"] != source["country"]:
            raise ValueError("Component/country disagrees with the validated inventory")
        if row["category"] != source["category"] or row["image_sha256"] != source["sha256"]:
            raise ValueError("Category/image SHA disagrees with the validated inventory")
        if (row["country"] == "Czech") != (split == "test_ood"):
            raise ValueError("Czech contamination")
        for mapping, key in ((groups, row["group_id"]), (hashes, row["image_sha256"])):
            if key in mapping and mapping[key] != split:
                raise ValueError("Group/SHA crosses original splits")
            mapping[key] = split
        row.update(width=source["width"], height=source["height"])
    for split, stats in baseline["splits"].items():
        if sum(r["split"] == split for r in records) != stats["images"]:
            raise ValueError("Original split cardinality mismatch")
    result = {}
    for split in ("train_id", "val_id"):
        disk = read_csv(reports / f"{split}.csv")
        subset = [r for r in records if r["split"] == split]
        if disk != [{k: v for k, v in r.items() if k not in {"width", "height"}} for r in subset]:
            raise ValueError("Per-split manifest disagrees with global manifest")
        result[split] = subset
    return result["train_id"], result["val_id"]


def label_counts(text):
    counts = dict.fromkeys(CLASSES, 0)
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] not in {"0", "1", "2", "3"}:
            raise ValueError("Invalid YOLO class/line")
        x, y, w, h = map(float, fields[1:])
        if not all(math.isfinite(v) for v in (x, y, w, h)) or not (
            0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
        ) or min(x - w / 2, y - h / 2) < -1e-9 or max(x + w / 2, y + h / 2) > 1 + 1e-9:
            raise ValueError("YOLO box outside normalized bounds")
        counts[CLASSES[int(fields[0])]] += 1
    return counts


def verify_files(root, records, source=None, progress=False):
    """Check associations, exact byte digests and labels without image decoding."""
    root = Path(root).resolve()
    paths, boxes = set(), 0
    for index, row in enumerate(records, 1):
        folder = {"train_id": "train", "val_id": "val"}.get(row["split"])
        if folder is None or row["country"] not in {"India", "Japan"}:
            raise ValueError("Package accepts train/val ID only")
        if PurePosixPath(row["image_path"]).parts[:2] != ("images", folder):
            raise ValueError("Wrong image directory")
        label = str(PurePosixPath(row["image_path"]).with_suffix(".txt")).replace("images/", "labels/", 1)
        if label != row["label_path"]:
            raise ValueError("Wrong image/label association")
        for key, digest in (("image_path", "image_sha256"), ("label_path", "label_sha256")):
            if row[key] in paths:
                raise ValueError("Duplicate output path")
            paths.add(row[key])
            path = inside(root, row[key])
            if sha(path) != row[digest]:
                raise ValueError(f"SHA mismatch: {row[key]}")
            if source is not None and os.path.samefile(path, inside(source, row[key])):
                raise ValueError("Package files must be independent copies")
        counts = label_counts(inside(root, label).read_text(encoding="utf-8"))
        if counts != {name: int(row[name]) for name in CLASSES}:
            raise ValueError("Label counts differ from manifest")
        boxes += sum(counts.values())
        if progress and index % 3000 == 0:
            print(f"Verified {index}/{len(records)} image/label pairs: {root.name}", flush=True)
    return {"images": len(records), "labels": len(records), "boxes": boxes}


def coverage(train, selected):
    total = sum((features(row) for row in train), Counter())
    actual = sum((features(row) for row in selected), Counter())
    fraction = len(selected) / len(train)
    result = []
    for key, count in sorted(total.items()):
        value = actual[key]
        instance = ":instances:" in key
        # Denominators distinguish instance shares from image prevalence.
        country = key.split(":")[0]
        denominator = (sum(int(r[name]) for r in train if r["country"] == country for name in CLASSES)
                       if instance else len(train))
        selected_denominator = (sum(int(r[name]) for r in selected if r["country"] == country for name in CLASSES)
                                if instance else len(selected))
        before = count / denominator
        after = value / selected_denominator if selected_denominator else 0
        result.append({"feature": key, "full_count": count, "selected_count": value,
                       "expected_count": round(count * fraction, 6),
                       "retention_fraction": round(value / count, 8),
                       "full_share": round(before, 8), "selected_share": round(after, 8),
                       "share_delta_pp": round(100 * (after - before), 6),
                       "retention_delta_pp": round(100 * (value / count - fraction), 6),
                       "denominator": "country_target_instances" if instance else "all_train_images"})
    return result


def describe(rows):
    return {"images": len(rows), "countries": dict(Counter(r["country"] for r in rows)),
            "categories": dict(Counter(r["category"] for r in rows)),
            "empty_labels": sum(sum(int(r[n]) for n in CLASSES) == 0 for r in rows),
            "groups": len({r["group_id"] for r in rows}),
            "non_singleton_groups": sum(n > 1 for n in Counter(r["group_id"] for r in rows).values()),
            "largest_group": max(Counter(r["group_id"] for r in rows).values(), default=0),
            "instances": {n: sum(int(r[n]) for r in rows) for n in CLASSES},
            "presence": {n: sum(int(r[n]) > 0 for r in rows) for n in CLASSES}}


def kaggle_instructions():
    return '''# RoadGuard B0 5k — transfert et contrôles Kaggle

Charger le ZIP comme nouveau Dataset Kaggle privé, puis l'ajouter au Notebook
avec Add Input. Les archives Dataset sont décompressées en conservant leurs
sous-dossiers. Ne transférer aucun dataset test_id/test_ood. Aucun entraînement
n'est lancé par ce package. Respecter la licence RDD2022 et les droits d'accès.

Le YAML local est du JSON, sous-ensemble de YAML ; aucune dépendance YAML n'est
requise pour créer la configuration absolue. Exécuter cette cellule après avoir
attaché ce seul package (les montages peuvent varier) :

```python
from pathlib import Path
import csv, hashlib, json, math

configs = list(Path("/kaggle/input").rglob("data.yaml"))
roots = [p.parent for p in configs if (p.parent / "selected_train.csv").is_file()
         and (p.parent / "manifest.csv").is_file()]
assert len(roots) == 1, f"Choisir explicitement le package B0 : {roots}"
root = roots[0].resolve()
config = json.loads((root / "data.yaml").read_text())
assert config == {"train": "images/train", "val": "images/val",
                  "names": ["D00", "D10", "D20", "D40"]}
with (root / "manifest.csv").open(encoding="utf-8-sig") as stream:
    rows = list(csv.DictReader(stream))
for row in rows:
    assert row["split"] in {"train_id", "val_id"}
    assert row["country"] in {"India", "Japan"}
    for key, digest in [("image_path", "image_sha256"), ("label_path", "label_sha256")]:
        path = (root / row[key]).resolve()
        assert path.is_relative_to(root)
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == row[digest]
    for line in (root / row["label_path"]).read_text().splitlines():
        tokens = line.split()
        assert len(tokens) == 5 and tokens[0] in {"0", "1", "2", "3"}
        x, y, w, h = map(float, tokens[1:])
        assert all(math.isfinite(v) for v in [x, y, w, h])
        assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
        assert min(x-w/2, y-h/2) >= -1e-9 and max(x+w/2, y+h/2) <= 1+1e-9
counts = {s: sum(r["split"] == s for r in rows) for s in ["train_id", "val_id"]}
summary = json.loads((root / "summary.json").read_text())
assert counts == {"train_id": summary["selected_train"]["images"],
                  "val_id": summary["validation_id"]["images"]}
config["path"] = root.as_posix()
yaml_path = Path("/kaggle/working/roadguard_b0_5k.yaml")
yaml_path.write_text(json.dumps(config, indent=2) + "\\n")
print(counts, yaml_path)
```

Cette cellule ne télécharge ni modèle ni bibliothèque. Elle ne démarre aucun
entraînement. Ultralytics n'a pas été exécuté localement : le contrôle natif sera
fait après validation humaine et disponibilité de la version épinglée.

## Protocole et limites

Seed 42 par défaut. Seul train_id intervient dans l'objectif de sélection.
Groupes validés indivisibles, y compris leurs ponts exclus lors de la phase 2.2.
Objectif glouton : pays, présence des classes par pays, catégories globales et
par pays, dimensions et instances par classe/pays. Les erreurs standardisées
protègent notamment D10 en Inde ; les égalités sont départagées par le seed.
Un calcul de sommes de tailles de groupes garantit l'effectif le plus proche
possible de la cible ; en cas d'égalité, le plus petit effectif est retenu.
Le seed effectif et les écarts sont dans summary.json et coverage.csv.

La diversité est approchée par ces strates et par les groupes ; aucune couverture
de trajets n'est démontrée. Pas de représentativité parfaite ni de garantie
d'absence de fuite résiduelle : dHash peut manquer des similarités ou produire
des faux positifs. Aucun résultat de modèle/test n'intervient. La validation ID
est intégrale et inchangée. Les labels mixed ignorent toujours les objets hors
cible ; empty_xml reste une catégorie de négatifs annotés avec risque
d'annotations incomplètes. Les rares orientations EXIF restent inchangées.
Les sources et les labels sont copiés octet pour octet, sans lien ni conversion.

Références : https://www.kaggle.com/docs/datasets et
https://www.kaggle.com/docs/notebooks ;
https://docs.ultralytics.com/datasets/detect/ .
'''


def make_zip(output, archive):
    """Sorted entries, fixed timestamps/permissions; compression settings are fixed."""
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as package:
        for path in sorted(output.rglob("*")):
            if not path.is_file():
                continue
            name = "roadguard_b0_5k/" + path.relative_to(output).as_posix()
            info = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            # zipfile.open streams the file; no image re-encoding.
            info.compress_level = 1
            with path.open("rb") as source, package.open(info, "w", force_zip64=True) as dest:
                shutil.copyfileobj(source, dest)


def verify_zip(output, archive):
    expected = {"roadguard_b0_5k/" + p.relative_to(output).as_posix(): p
                for p in output.rglob("*") if p.is_file()}
    with zipfile.ZipFile(archive) as package:
        if len(package.namelist()) != len(expected) or set(package.namelist()) != set(expected):
            raise ValueError("ZIP entries differ from package")
        for name, path in expected.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError("Non-portable ZIP entry")
            with package.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != sha(path):
                    raise ValueError("ZIP bytes differ from package")
    return len(expected)


def validate_package(source, inventory, output, reports, archive):
    """Read-only replay of selection and verification of delivered files/ZIP."""
    source, inventory, output, reports, archive = map(Path, (source, inventory, output, reports, archive))
    summary = json.loads((reports / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("verification", {}).get("ok"):
        raise ValueError("Package preparation was not completed")
    for name, digest in summary["input_report_sha256"].items():
        if sha(inside(inventory, name)) != digest:
            raise ValueError("Previous input report changed")
    train, val = load_sources(inventory)
    groups, _ = select_groups(train, summary["requested_train_images"], summary["seed"])
    expected = sorted([r for r in train if r["group_id"] in groups] + val,
                      key=lambda r: (r["split"], r["image_path"]))
    if read_csv(reports / "manifest.csv") != [{k: str(r[k]) for k in FIELDS} for r in expected]:
        raise ValueError("Manifest differs from reproducible train selection/full validation")
    verify_files(source, expected)
    result = verify_files(output, expected, source=source, progress=True)
    if json.loads((output / "data.yaml").read_text(encoding="utf-8")) != CONFIG:
        raise ValueError("Unexpected package YAML")
    names = {"data.yaml", *REPORTS, *(r[k] for r in expected for k in ("image_path", "label_path"))}
    if {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} != names:
        raise ValueError("Unlisted file in package")
    for name in REPORTS:
        if sha(output / name) != sha(reports / name):
            raise ValueError("Packaged report differs from selection report")
    entries = verify_zip(output, archive)
    receipt = json.loads((reports / "zip_receipt.json").read_text(encoding="utf-8"))
    if sha(archive) != receipt["sha256"] or archive.stat().st_size != receipt["bytes"]:
        raise ValueError("ZIP receipt mismatch")
    return {"ok": True, **result, "zip_entries_verified": entries, "zip_sha256": receipt["sha256"]}


def build(source, inventory, output, reports, archive, target=5000, seed=42):
    source, inventory, output, reports, archive = (Path(p).resolve() for p in
                                                  (source, inventory, output, reports, archive))
    for dest in (output, reports, archive):
        if dest.is_relative_to(source) or source.is_relative_to(dest) or any(
            p.name.casefold() == "raw" for p in (dest, *dest.parents)
        ) or dest.is_relative_to(inventory):
            raise ValueError("Destinations must be outside source/raw/existing reports")
        if dest.exists():
            raise ValueError("Destination already exists; choose new paths, no automatic deletion")
    if output.is_relative_to(reports) or reports.is_relative_to(output) or archive.is_relative_to(output) or archive.is_relative_to(reports):
        raise ValueError("Package, reports and ZIP must have separate destinations")
    input_hashes = {p.name: sha(p) for p in inventory.iterdir() if p.is_file()}
    train, val = load_sources(inventory)
    selected_groups, attained = select_groups(train, target, seed)
    chosen = [r for r in train if r["group_id"] in selected_groups]
    records = sorted(chosen + val, key=lambda r: (r["split"], r["image_path"]))
    print(f"Selected {attained}/{len(train)} train images; {len(val)} validation images", flush=True)
    verify_files(source, train + val, progress=True)
    reports.mkdir(parents=True)
    output.mkdir(parents=True)
    for row in records:
        for key in ("image_path", "label_path"):
            dest = output / row[key]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inside(source, row[key]), dest)
    validation = verify_files(output, records, source=source, progress=True)
    if {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} != {
        r[key] for r in records for key in ("image_path", "label_path")
    }:
        raise ValueError("Unexpected package image/label")
    verify_files(source, train + val, progress=True)
    if input_hashes != {p.name: sha(p) for p in inventory.iterdir() if p.is_file()}:
        raise ValueError("Previous reports changed during packaging")
    rows = [{key: row[key] for key in FIELDS} for row in records]
    write_csv(reports / "manifest.csv", FIELDS, rows)
    write_csv(reports / "selected_train.csv", FIELDS, [r for r in rows if r["split"] == "train_id"])
    deviations = coverage(train, chosen)
    write_csv(reports / "coverage.csv", tuple(deviations[0]), deviations)
    stats = []
    for name, subset in (("full_train", train), ("selected_train", chosen), ("val_id", val)):
        for country in sorted({r["country"] for r in subset}):
            members = [r for r in subset if r["country"] == country]
            for label in CLASSES:
                stats.append({"population": name, "country": country, "class": label,
                              "images": len(members), "presence": sum(int(r[label]) > 0 for r in members),
                              "instances": sum(int(r[label]) for r in members),
                              "empty_xml": sum(r["category"] == "empty_xml" for r in members)})
    write_csv(reports / "class_statistics.csv", tuple(stats[0]), stats)
    sizes = Counter(r["group_id"] for r in train)
    group_rows = [{"group_id": group, "train_images": size, "selected": group in selected_groups}
                  for group, size in sorted(sizes.items())]
    write_csv(reports / "group_selection.csv", ("group_id", "train_images", "selected"), group_rows)
    summary = {"schema_version": 1, "seed": seed, "requested_train_images": target,
               "size_delta": attained - target, "method": "seeded greedy group balancing with suffix subset-sum feasibility",
               "full_train": describe(train), "selected_train": describe(chosen), "validation_id": describe(val),
               "input_report_sha256": input_hashes,
               "max_abs_image_share_delta_pp": max(abs(r["share_delta_pp"]) for r in deviations if ":instances:" not in r["feature"]),
               "coverage_warnings": [r for r in deviations if abs(r["retention_delta_pp"]) > 2],
               "verification": {"ok": True, **validation, "full_train_val_sources_verified_before_after": len(train) + len(val),
                                "previous_reports_unchanged": True, "groups_split_by_selection": 0,
                                "test_images_in_package": 0, "ultralytics_runtime_tested": False}}
    write_json(reports / "summary.json", summary)
    (reports / "KAGGLE.md").write_text(kaggle_instructions(), encoding="utf-8")
    write_json(output / "data.yaml", CONFIG)
    for name in REPORTS:
        shutil.copyfile(reports / name, output / name)
    archive.parent.mkdir(parents=True, exist_ok=True)
    print("Writing portable ZIP...", flush=True)
    make_zip(output, archive)
    print("Verifying every ZIP entry...", flush=True)
    count = verify_zip(output, archive)
    receipt = {"filename": archive.name, "bytes": archive.stat().st_size,
               "sha256": sha(archive), "entries_verified": count}
    write_json(reports / "zip_receipt.json", receipt)
    return {"summary": summary, "zip": receipt, "zip_absolute_path": str(archive)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/roadguard_v1"))
    parser.add_argument("--inventory", type=Path, default=Path("reports/data_preparation"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/roadguard_b0_5k"))
    parser.add_argument("--reports", type=Path, default=Path("reports/b0_5k"))
    parser.add_argument("--zip", type=Path, default=Path("data/processed/roadguard_b0_5k.zip"))
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true", help="Verify existing package without writing files")
    args = parser.parse_args(argv)
    try:
        result = (validate_package(args.input, args.inventory, args.output, args.reports, args.zip)
                  if args.validate_only else
                  build(args.input, args.inventory, args.output, args.reports, args.zip, args.target, args.seed))
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"B0 preparation failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
