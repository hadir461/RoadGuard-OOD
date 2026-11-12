"""Prepare RoadGuard v1 group-aware ID splits and a strictly held-out OOD test.

Only new copies/labels/reports are written. Existing inventory/audit artifacts
are never changed. JSON serialization is used for data.yaml: valid YAML 1.2,
portable POSIX paths, no optional YAML dependency. No Ultralytics import/run.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageOps

from .audit import CLASSES, box_error, coordinates, dimensions
from .build_manifest import CLASS_IDS, cross_id_ood, write_csv


SPLITS = {"train_id": "train", "val_id": "val", "test_id": "test_id", "test_ood": "test_ood"}
RATIOS = {"train_id": 0.8, "val_id": 0.1, "test_id": 0.1}
INCLUDED = {"target_only", "mixed", "empty_xml"}
RECORD_FIELDS = ("split", "group_id", "country", "category", "source_image", "source_xml",
                 "image_path", "label_path", "image_sha256", "xml_sha256", "label_sha256",
                 "D00", "D10", "D20", "D40", "empty_label", "original_empty_xml")
NEW_REPORTS = ("split_manifest.csv", "train_id.csv", "val_id.csv", "test_id.csv", "test_ood.csv",
               "split_statistics.csv", "stratification.csv", "similarity_groups.csv",
               "similarity_group_summary.csv", "yolo_exclusions.csv", "split_policy.md",
               "similarity_review.html", "cross_country_review.jpg", "yolo_preparation_summary.json")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def inside(root, relative):
    """Reject traversal, absolute paths and external symlink targets."""
    value = PurePosixPath(relative)
    if not relative or value.is_absolute() or ".." in value.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Non-portable dataset path: {relative}")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset: {relative}")
    return path


def load_inventory(inventory):
    inventory = Path(inventory)
    summary = json.loads((inventory / "summary.json").read_text(encoding="utf-8"))
    for name, digest in summary["output_sha256"].items():
        if sha(inside(inventory.resolve(), name)) != digest:
            raise ValueError(f"Phase 2.1 artifact changed: {name}")
    rows = read_csv(inventory / "manifest.csv")
    pairs = read_csv(inventory / "near_duplicate_candidates.csv")
    if len({r["image_path"] for r in rows}) != len(rows) or len(rows) != summary["image_count"]:
        raise ValueError("Manifest cardinality/uniqueness mismatch")
    if len(pairs) != summary["near_duplicates"]["candidate_pairs"]:
        raise ValueError("Candidate count mismatch")
    return rows, pairs, summary


def similarity_components(rows, pairs, threshold=4):
    """Connect ALL inventory rows, including excluded-image bridges and exact SHA copies."""
    by_path = {r["image_path"]: r for r in rows}
    parent = {name: name for name in by_path}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    seen = set()
    for pair in pairs:
        a, b = pair["image_a"], pair["image_b"]
        if a == b or a not in by_path or b not in by_path or tuple(sorted((a, b))) in seen:
            raise ValueError("Invalid or repeated candidate edge")
        seen.add(tuple(sorted((a, b))))
        distance = (int(by_path[a]["dhash64"], 16) ^ int(by_path[b]["dhash64"], 16)).bit_count()
        if distance != int(pair["hamming_distance"]) or distance > threshold:
            raise ValueError("Candidate Hamming distance inconsistent with inventory")
        if pair["country_a"] != by_path[a]["country"] or pair["country_b"] != by_path[b]["country"]:
            raise ValueError("Candidate country inconsistent with inventory")
        union(a, b)
    hashes = {}
    for row in rows:
        digest = row["sha256"]
        if digest in hashes:
            union(row["image_path"], hashes[digest])
        hashes[digest] = row["image_path"]
    components = defaultdict(list)
    for name in sorted(by_path):
        components[find(name)].append(name)
    groups, membership = {}, {}
    for members in components.values():
        group = "g_" + hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]
        groups[group] = members
        for name in members:
            membership[name] = group
        if cross_id_ood({by_path[name]["country"] for name in members}):
            raise ValueError(f"ID/OOD similarity component {group}: human resolution required before preparation")
    return groups, membership


def convert_annotation(content, width, height):
    """Return target YOLO lines/counts and object exclusions; never clamp or add 1."""
    root = ET.fromstring(content)
    if root.tag != "annotation" or dimensions(root) != (width, height):
        raise ValueError("XML root/dimensions are inconsistent with the audited image")
    lines, counts, exclusions = [], dict.fromkeys(CLASSES, 0), []
    for index, obj in enumerate(root.findall("object"), 1):
        name = obj.findtext("name", "").strip()
        error = box_error(root, obj, "rdd2022")
        if name not in CLASS_IDS or error:
            exclusions.append({"object_index": index, "class": name,
                               "reason": "non_target_class" if name not in CLASS_IDS else "invalid_target_box",
                               "detail": error or "Outside the four-class protocol"})
            continue
        xmin, ymin, xmax, ymax = coordinates(obj)
        values = ((xmin + xmax) / (2 * width), (ymin + ymax) / (2 * height),
                  (xmax - xmin) / width, (ymax - ymin) / height)
        lines.append(str(CLASS_IDS[name]) + " " + " ".join(f"{value:.12f}" for value in values))
        counts[name] += 1
    return "".join(line + "\n" for line in lines), counts, exclusions


def feature_counts(group):
    counts = Counter()
    for row in group:
        country = row["country"]
        counts["all_images"] += 1
        counts[f"country:{country}"] += 1
        counts[f"{country}:empty_xml"] += int(row["category"] == "empty_xml")
        for name in CLASSES:
            counts[f"{country}:presence:{name}"] += int(row["counts"][name] > 0)
    return +counts


def assign_id_groups(retained, seed=42):
    """Deterministic greedy multilabel balancing; OOD rows never enter the objective.

    Largest groups first, rare-feature tie-break, seeded order for equal scores.
    Minimize the increment of sum((actual-target)^2 / split_feature_target), with a
    weight of 2 for country sizes, 1 otherwise. No learned/model criterion.
    """
    grouped = defaultdict(list)
    assignments = {}
    for row in sorted(retained, key=lambda r: r["image_path"]):
        if row["country"] == "Czech":
            assignments[row["group_id"]] = "test_ood"
        elif row["country"] in {"India", "Japan"}:
            grouped[row["group_id"]].append(row)
        else:
            raise ValueError(f"Country outside the approved protocol: {row['country']}")
    vectors = {group: feature_counts(members) for group, members in grouped.items()}
    totals = sum(vectors.values(), Counter())
    rng = random.Random(seed)
    order = sorted(grouped)
    rng.shuffle(order)
    order.sort(key=lambda group: (-len(grouped[group]), -sum(n / totals[k] for k, n in vectors[group].items())))
    actual = {split: Counter() for split in RATIOS}
    for group in order:
        vector = vectors[group]
        options = list(RATIOS)
        rng.shuffle(options)

        def delta(split):
            return sum((2 if key.startswith("country:") else 1) *
                       (2 * (actual[split][key] - RATIOS[split] * totals[key]) * count + count * count)
                       / (RATIOS[split] * totals[key]) for key, count in vector.items())
        chosen = min(options, key=delta)
        assignments[group] = chosen
        actual[chosen].update(vector)
    deviations = []
    for split, ratio in RATIOS.items():
        for key, total in sorted(totals.items()):
            value = actual[split][key]
            deviations.append({"split": split, "feature": key, "population": total,
                               "expected": round(ratio * total, 4), "actual": value,
                               "actual_fraction": round(value / total, 8),
                               "delta_percentage_points": round(100 * (value / total - ratio), 6)})
    return assignments, deviations


def review_gallery(root, reports, rows, pairs, membership, groups, assignments):
    """Publish every candidate with thumbnails, without claiming visual confirmation."""
    folder = reports / "similarity_review"
    folder.mkdir()
    paths = sorted({p[key] for p in pairs for key in ("image_a", "image_b")})
    thumbnails = {}
    for path in paths:
        name = hashlib.sha256(path.encode()).hexdigest()[:20] + ".jpg"
        with Image.open(inside(root, path)) as source:
            thumb = ImageOps.contain(source.convert("RGB"), (320, 240))
            thumb.save(folder / name, quality=82)
            thumb.close()
        thumbnails[path] = "similarity_review/" + name
    cards = []
    cross = [p for p in pairs if p["country_a"] != p["country_b"]]
    for pair in sorted(pairs, key=lambda p: (p["country_a"] == p["country_b"], membership[p["image_a"]], p["image_a"], p["image_b"])):
        group = membership[pair["image_a"]]
        sides = "".join(f'<div><img loading="lazy" src="{thumbnails[pair[key]]}" alt="Image candidate"><p>{escape(pair[key])}</p></div>'
                        for key in ("image_a", "image_b"))
        cards.append(f'<article class="{"cross" if pair in cross else ""}"><h2>{group} · dH={pair["hamming_distance"]} · {escape(assignments.get(group, "excluded"))}</h2><section>{sides}</section></article>')
    (reports / "similarity_review.html").write_text(
        '<!doctype html><html lang="fr"><meta charset="utf-8"><title>Revue des similarités</title>'
        '<style>body{font:15px sans-serif;margin:24px;color:#142d45}article{border:1px solid #ccd6df;padding:14px;margin:16px 0;break-inside:avoid}.cross{border:3px solid #b78031}section{display:flex;gap:20px}section div{width:48%}img{max-width:100%;height:240px;object-fit:contain}p{overflow-wrap:anywhere}h2{font-size:16px}</style>'
        f'<h1>{len(pairs)} paires à inspecter</h1><p>Toutes les distances ont été revérifiées. Les {len(cross)} paires inter-pays sont en tête. '
        'Aucune paire n’est déclarée doublon réel. Les composantes sont des groupes de précaution, pas des trajets validés.</p>' + "".join(cards) + '</html>', encoding="utf-8")
    sheet = Image.new("RGB", (900, max(1, len(cross)) * 300), "white")
    draw = ImageDraw.Draw(sheet)
    if not cross:
        draw.text((12, 12), "No cross-country candidate pairs", fill="black")
    for index, pair in enumerate(cross):
        for column, key in enumerate(("image_a", "image_b")):
            with Image.open(reports / thumbnails[pair[key]]) as image:
                sheet.paste(image, (column * 450 + 30, index * 300 + 45))
            draw.text((column * 450 + 10, index * 300 + 8), Path(pair[key]).name, fill="black")
        draw.text((10, index * 300 + 275), f'dH={pair["hamming_distance"]} | {membership[pair["image_a"]]} | inspection required', fill="black")
    sheet.save(reports / "cross_country_review.jpg", quality=90)
    sheet.close()


def split_policy(summary):
    return f"""# RoadGuard v1 — protocole de préparation et limites

Seed {summary['seed']}. Sources ID : India/Japan. Czech = test_ood exclusivement.
Les {summary['components']['candidate_edges_reviewed_structurally']} liens de la phase 2.1
sont contrôlés structurellement ; `similarity_review.html` présente toutes les
paires pour inspection humaine, notamment les liens India/Japan. Aucune paire
n'est certifiée doublon réel par cette préparation.

## Groupes et stratification

Composantes connexes construites AVANT filtrage, avec les images exclues comme
ponts éventuels ; les identités SHA-256 sont également réunies. Les composantes
de similarité restent indivisibles, même en présence de faux positifs ou d'un
effet de chaîne A–B–C. Elles ne constituent pas une preuve de trajet commun.
Un composant reliant ID à Czech bloque la préparation pour décision humaine.
Les noms différents ne sont pas une garantie d'indépendance.

Découpage ID visé 80/10/10 : affectation gloutonne des groupes, gros groupes
d'abord, objectif d'écart quadratique divisé par l'effectif cible de chaque
split/variable (pour ne pas remplir le train avant les autres), sur les images, le pays,
la présence de chaque classe PAR PAYS et les XML vides par pays. Les poids pays
valent 2, les autres 1. Random(42) départage les égalités. Aucune information
de Czech ne participe à l'objectif ID. Les écarts réels sont consignés dans
stratification.csv ; un écart > 2 points est signalé, sans modifier le test OOD.

## Politique autorisée à l'étape 2.2

- target_only : inclure. mixed : inclure mais retirer les objets hors D00/D10/D20/D40.
- Les objets hors cible des images mixed sont donc ignorés par le détecteur à
  quatre classes ; ils peuvent être vus comme arrière-plan pour ces classes.
  Aucun remappage n'est effectué. Cette décision ne prouve pas la complétude
  sémantique des labels et ne constitue pas une détection open-set.
- non_target_only : exclure du protocole principal, sans supprimer les fichiers.
- empty_xml : inclure comme négatifs annotés selon la politique humaine demandée.
  Leur nombre/proportion reste séparé des autres labels vides. Cela ne prouve pas
  qu'une image soit sans dommage : des annotations incomplètes peuvent introduire
  de faux négatifs. Aucun échantillonnage n'est ajusté à des résultats de modèle.
- Boîte cible invalide : retirer seulement cet objet ; garder les autres objets
  et l'image. Un label devenu vide est signalé séparément. Un XML illisible ou
  manquant bloque la préparation : il n'est jamais transformé en négatif.

## Conversion et Windows

RDD2022 : 0 <= xmin < xmax <= W et 0 <= ymin < ymax <= H. Centre =
((xmin+xmax)/(2W), (ymin+ymax)/(2H)) ; taille = ((xmax-xmin)/W, (ymax-ymin)/H).
Aucun +1, clamp, augmentation ou réencodage des images. IDs D00=0/D10=1/D20=2/D40=3.
Les fichiers image sont copiés indépendamment, sans lien symbolique ou hardlink.
{len(summary['negative_exif_orientations'])} copie(s) de négatifs empty_xml portent une orientation EXIF différente
de 1, détaillées dans negative_exif_orientations du résumé. Elles restent copiées
octet pour octet : une auto-orientation par le futur loader ne déplace aucune
boîte puisqu'il n'y a aucun objet XML. Toute image annotée avec une telle
orientation bloque l'export jusqu'à définition explicite du repère des boîtes.

data.yaml et data_ood.yaml sont sérialisés en JSON, sous-ensemble valide de YAML
1.2. POSIX relatif ; clé path omise pour résolution depuis le parent du YAML.
data.yaml : train/val ID et test=test_id. data_ood.yaml garde train/val ID et
test=test_ood ; réservé à l'évaluation FINALE, jamais au tuning. Passer au futur
loader le chemin ABSOLU du YAML pour éviter toute ambiguïté de répertoire courant.
Références vérifiées : https://docs.ultralytics.com/datasets/detect/ et
https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py
(check_det_dataset : fallback sur le parent de yaml_file lorsque path est absent).
Ultralytics n'est pas installé/exécuté à cette étape ; la validation est notre
contrôle structurel et numérique. Revalider avec la version épinglée lors de B0.

## Reproduction, validation et limites

Inventaire, candidats et politique d'entrée sont verrouillés par leurs SHA-256.
Les XML et images brutes sont comparés au manifeste avant/après export ; les
copies sont vérifiées octet par octet via SHA-256. Les rapports antérieurs sont
préservés. Une destination non vide est refusée ; une interruption laisse ses
fichiers partiels, sans suppression automatique, et ne produit pas de validation
réussie. Pour une nouvelle préparation utiliser un nouveau nom de version.

Validation sans entraînement ni téléchargement :
`.venv/Scripts/python.exe -m src.data.prepare_yolo --validate-only --input data/raw/RDD2022 --output data/processed/roadguard_v1`

Les contrôles prouvent l'absence de chevauchement et de lien détecté traversant
les splits, PAS l'élimination de tous les risques de fuite. Le dHash peut manquer
des recadrages, changements de point de vue ou frames proches et produire des
collisions. Les {summary['sequence_metadata']['exif_context_fields_image_counts'].get('DateTime', 0)} champs DateTime précédemment observés ne sont pas des
identifiants de trajets validés. Aucune indépendance de séquences n'est certifiée.
"""


def validate_dataset(root, output, reports, inventory, allow_unchecked=False):
    root, output, reports = Path(root).resolve(), Path(output).resolve(), Path(reports).resolve()
    original, pairs, baseline = load_inventory(inventory)
    _, membership = similarity_components(original, pairs, baseline["near_duplicates"]["threshold"])
    records = read_csv(reports / "split_manifest.csv")
    summary = json.loads((reports / "yolo_preparation_summary.json").read_text(encoding="utf-8"))
    if not allow_unchecked and not summary.get("validation", {}).get("ok"):
        raise ValueError("Dataset has no completed preparation validation")
    for filename, split_key in (("data.yaml", "test_id"), ("data_ood.yaml", "test_ood")):
        config = json.loads((output / filename).read_text(encoding="utf-8"))
        expected = {"train": "images/train", "val": "images/val", "test": f"images/{split_key}", "names": list(CLASSES)}
        if config != expected:
            raise ValueError(f"Unexpected or non-portable YAML configuration: {filename}")
        for key in ("train", "val", "test"):
            if not inside(output, config[key]).is_dir():
                raise ValueError(f"Missing YAML directory: {config[key]}")
    by_source = {r["source_image"]: r for r in records}
    expected_sources = {r["image_path"] for r in original if r["category"] in INCLUDED}
    if len(by_source) != len(records) or set(by_source) != expected_sources:
        raise ValueError("Source overlap, omission or unexpected retained image")
    groups, hashes, labels_count, source_count = {}, {}, 0, 0
    for index, row in enumerate(original, 1):
        source, xml = inside(root, row["image_path"]), inside(root, row["xml_path"])
        if sha(source) != row["sha256"] or sha(xml) != row["xml_sha256"]:
            raise ValueError(f"Raw source changed since phase 2.1: {row['image_path']}")
        source_count += 1
        record = by_source.get(row["image_path"])
        if record is None:
            continue
        split = record["split"]
        if split not in SPLITS or (row["country"] == "Czech") != (split == "test_ood"):
            raise ValueError("Czech contamination or wrong split")
        if record["group_id"] != membership[row["image_path"]]:
            raise ValueError("Wrong component assignment")
        for store, key in ((groups, record["group_id"]), (hashes, row["sha256"])):
            if key in store and store[key] != split:
                raise ValueError("Similarity component or SHA identity crosses splits")
            store[key] = split
        image_path, label_path = inside(output, record["image_path"]), inside(output, record["label_path"])
        if record["image_path"].split("/")[:2] != ["images", SPLITS[split]]:
            raise ValueError("Image in wrong processed directory")
        if record["label_path"] != str(PurePosixPath(record["image_path"]).with_suffix(".txt")).replace("images/", "labels/", 1):
            raise ValueError("Image/label association mismatch")
        if image_path.is_symlink() or label_path.is_symlink() or os.path.samefile(source, image_path):
            raise ValueError("Processed image must be an independent copy")
        if sha(image_path) != row["sha256"] or record["image_sha256"] != row["sha256"]:
            raise ValueError("Processed image hash differs from raw")
        expected, counts, _ = convert_annotation(xml.read_bytes(), int(row["width"]), int(row["height"]))
        text = label_path.read_text(encoding="utf-8")
        if text != expected or sha(label_path) != record["label_sha256"]:
            raise ValueError("YOLO labels differ from source conversion")
        for name in CLASSES:
            if int(record[name]) != counts[name]:
                raise ValueError("Split class count inconsistent with labels")
        for line in text.splitlines():
            tokens = line.split()
            if len(tokens) != 5 or tokens[0] not in {"0", "1", "2", "3"}:
                raise ValueError("Invalid YOLO class ID or line length")
            x, y, w, h = map(float, tokens[1:])
            if not all(math.isfinite(value) for value in (x, y, w, h)) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("Non-normalized YOLO coordinates")
            if min(x - w / 2, y - h / 2) < -1e-9 or max(x + w / 2, y + h / 2) > 1 + 1e-9:
                raise ValueError("Normalized box extends outside image")
            labels_count += 1
        if index % 5000 == 0:
            print(f"Validation: {index}/{len(original)} source pairs", flush=True)
    actual_images = {p.relative_to(output).as_posix() for p in (output / "images").rglob("*") if p.is_file()}
    actual_labels = {p.relative_to(output).as_posix() for p in (output / "labels").rglob("*") if p.is_file()}
    if actual_images != {r["image_path"] for r in records} or actual_labels != {r["label_path"] for r in records}:
        raise ValueError("Unlisted or missing processed file")
    for split in SPLITS:
        if read_csv(reports / f"{split}.csv") != [r for r in records if r["split"] == split]:
            raise ValueError("Per-split manifest inconsistent with global manifest")
    return {"ok": True, "raw_images_verified": source_count, "raw_xml_verified": source_count,
            "copied_images_verified": len(records), "label_files_verified": len(records),
            "normalized_boxes_verified": labels_count, "cross_split_components": 0,
            "cross_split_exact_hashes": 0, "czech_in_id": 0,
            "ultralytics_runtime_tested": False}


def prepare(root, inventory, output, reports, seed=42):
    root, inventory, output, reports = map(lambda p: Path(p).resolve(), (root, inventory, output, reports))
    protected = [root, *(p for p in root.parents if p.name.casefold() == "raw")]
    if any(output.is_relative_to(p) or reports.is_relative_to(p) for p in protected):
        raise ValueError("Outputs/reports must be outside raw data")
    if output.exists() or any((reports / name).exists() for name in (*NEW_REPORTS, "similarity_review")):
        raise ValueError("Destination/version already exists; use a NEW output/report version (no automatic deletion)")
    rows, pairs, baseline = load_inventory(inventory)
    if baseline["scan_errors"] or baseline["unpaired_xml_paths"] or not rows:
        raise ValueError("Unresolved/empty source inventory")
    groups, membership = similarity_components(rows, pairs, baseline["near_duplicates"]["threshold"])
    exclusions, retained, negative_exif = [], [], []
    for index, row in enumerate(rows, 1):
        image = inside(root, row["image_path"])
        xml = inside(root, row["xml_path"])
        if row["country"] not in {"India", "Japan", "Czech"} or row["xml_status"] != "parsed":
            raise ValueError("Unknown country or unreadable annotation requires human decision")
        if sha(image) != row["sha256"]:
            raise ValueError(f"Source image changed: {row['image_path']}")
        xml_bytes = xml.read_bytes()
        if hashlib.sha256(xml_bytes).hexdigest() != row["xml_sha256"]:
            raise ValueError(f"Source XML changed: {row['xml_path']}")
        with Image.open(image) as source:
            orientation = source.getexif().get(274, 1)
            if source.size != (int(row["width"]), int(row["height"])) or orientation not in range(1, 9) or (
                orientation != 1 and row["category"] != "empty_xml"
            ):
                raise ValueError("Image dimensions/EXIF orientation require an explicit conversion policy")
            if orientation != 1:
                negative_exif.append({"source_image": row["image_path"], "orientation": orientation,
                                      "policy": "bit_exact_copy_of_empty_xml_negative_no_box_to_transform"})
        if any(issue["code"] != "invalid_box" for issue in json.loads(row["issues"])):
            raise ValueError(f"Unresolved source issue: {row['image_path']}")
        if row["category"] == "non_target_only":
            exclusions.append({"kind": "image", "source_image": row["image_path"], "source_xml": row["xml_path"],
                               "object_index": "", "class": "", "reason": "non_target_only",
                               "detail": f"{row['object_count']} non-target objects excluded with this image"})
            continue
        if row["category"] not in INCLUDED:
            raise ValueError(f"Unusable category: {row['image_path']}")
        label, counts, removed = convert_annotation(xml_bytes, int(row["width"]), int(row["height"]))
        if sum(counts.values()) != int(row["valid_target_count"]):
            raise ValueError("Converted target counts disagree with the inventory")
        for item in removed:
            exclusions.append({"kind": "box", "source_image": row["image_path"], "source_xml": row["xml_path"], **item})
        retained.append({**row, "group_id": membership[row["image_path"]], "counts": counts, "label": label})
        if index % 5000 == 0:
            print(f"Source verification: {index}/{len(rows)}", flush=True)
    assignments, deviations = assign_id_groups(retained, seed)
    output.mkdir(parents=True)
    reports.mkdir(parents=True, exist_ok=True)
    for split in SPLITS.values():
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    records, used_names = [], set()
    for index, row in enumerate(retained, 1):
        split = assignments[row["group_id"]]
        source = PurePosixPath(row["image_path"])
        token = hashlib.sha256(row["image_path"].encode()).hexdigest()[:12]
        name = f"{row['country']}__{source.stem}__{token}{source.suffix.lower()}"
        if name.casefold() in used_names:
            raise ValueError("Processed filename collision")
        used_names.add(name.casefold())
        image = f"images/{SPLITS[split]}/{name}"
        label = f"labels/{SPLITS[split]}/{Path(name).stem}.txt"
        shutil.copyfile(inside(root, row["image_path"]), output / image)
        with (output / label).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(row["label"])
        records.append({"split": split, "group_id": row["group_id"], "country": row["country"], "category": row["category"],
                        "source_image": row["image_path"], "source_xml": row["xml_path"], "image_path": image, "label_path": label,
                        "image_sha256": row["sha256"], "xml_sha256": row["xml_sha256"],
                        "label_sha256": sha(output / label), **row["counts"],
                        "empty_label": not row["label"], "original_empty_xml": row["category"] == "empty_xml"})
        if index % 3000 == 0:
            print(f"Independent copies and labels: {index}/{len(retained)}", flush=True)
    for filename, test in (("data.yaml", "test_id"), ("data_ood.yaml", "test_ood")):
        write_json(output / filename, {"train": "images/train", "val": "images/val", "test": f"images/{test}", "names": list(CLASSES)})
    write_csv(reports / "split_manifest.csv", RECORD_FIELDS, records)
    for split in SPLITS:
        write_csv(reports / f"{split}.csv", RECORD_FIELDS, [r for r in records if r["split"] == split])
    write_csv(reports / "yolo_exclusions.csv", ("kind", "source_image", "source_xml", "object_index", "class", "reason", "detail"), exclusions)
    write_csv(reports / "stratification.csv", ("split", "feature", "population", "expected", "actual", "actual_fraction", "delta_percentage_points"), deviations)
    statistics, split_summary = [], {}
    for split in SPLITS:
        selected = [r for r in records if r["split"] == split]
        split_summary[split] = {"images": len(selected), "countries": dict(Counter(r["country"] for r in selected)),
                                "categories": dict(Counter(r["category"] for r in selected)),
                                "negative_images": sum(r["original_empty_xml"] for r in selected),
                                "empty_labels_from_invalid_targets": sum(r["empty_label"] and not r["original_empty_xml"] for r in selected),
                                "instances": {name: sum(r[name] for r in selected) for name in CLASSES}}
        for country in sorted({r["country"] for r in selected}):
            subset = [r for r in selected if r["country"] == country]
            for name in CLASSES:
                statistics.append({"split": split, "country": country, "class": name,
                                   "images": len(subset), "images_with_class": sum(r[name] > 0 for r in subset),
                                   "instances": sum(r[name] for r in subset),
                                   "negative_images": sum(r["original_empty_xml"] for r in subset)})
    write_csv(reports / "split_statistics.csv", ("split", "country", "class", "images", "images_with_class", "instances", "negative_images"), statistics)
    kept_by_source = {r["source_image"]: r for r in records}
    members, component_rows = [], []
    edges = Counter(membership[p["image_a"]] for p in pairs)
    by_path = {r["image_path"]: r for r in rows}
    for group, paths in sorted(groups.items()):
        for path in paths:
            record = kept_by_source.get(path)
            members.append({"group_id": group, "component_size": len(paths), "source_image": path,
                            "country": by_path[path]["country"], "category": by_path[path]["category"],
                            "split": record["split"] if record else "excluded"})
        if len(paths) > 1:
            component_rows.append({"group_id": group, "component_size": len(paths),
                                   "retained_images": sum(path in kept_by_source for path in paths),
                                   "countries": json.dumps(sorted({by_path[p]["country"] for p in paths})),
                                   "candidate_edges": edges[group], "assigned_split": assignments.get(group, "excluded"),
                                   "review_status": "human_visual_review_pending"})
    write_csv(reports / "similarity_groups.csv", ("group_id", "component_size", "source_image", "country", "category", "split"), members)
    write_csv(reports / "similarity_group_summary.csv", ("group_id", "component_size", "retained_images", "countries", "candidate_edges", "assigned_split", "review_status"), component_rows)
    review_gallery(root, reports, rows, pairs, membership, groups, assignments)
    summary = {"schema_version": 1, "seed": seed, "target_id_ratios": RATIOS, "splits": split_summary,
               "components": {"all": len(groups), "non_singleton": len(component_rows), "largest": max(map(len, groups.values())),
                              "cross_country_components": sum(len({by_path[p]["country"] for p in paths}) > 1 for paths in groups.values()),
                              "candidate_edges_reviewed_structurally": len(pairs)},
               "exclusions": {"images": sum(e["kind"] == "image" for e in exclusions),
                              "objects_in_excluded_images": sum(int(r["object_count"]) for r in rows if r["category"] == "non_target_only"),
                              "ignored_non_target_boxes_in_retained_images": sum(e["reason"] == "non_target_class" for e in exclusions),
                              "invalid_target_boxes": sum(e["reason"] == "invalid_target_box" for e in exclusions)},
               "stratification_warnings": [r for r in deviations if abs(r["delta_percentage_points"]) > 2],
               "input_sha256": {name: sha(inventory / name) for name in ("manifest.csv", "near_duplicate_candidates.csv", "summary.json")},
               "sequence_metadata": baseline["sequence_metadata"],
               "negative_exif_orientations": negative_exif,
               "yaml_portability": "YAML 1.2 JSON subset, omit path, resolve relative entries against absolute YAML parent",
               "validation": {"ok": False}}
    (reports / "split_policy.md").write_text(split_policy(summary), encoding="utf-8")
    write_json(reports / "yolo_preparation_summary.json", summary)
    summary["validation"] = validate_dataset(root, output, reports, inventory, allow_unchecked=True)
    write_json(reports / "yolo_preparation_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/RDD2022"))
    parser.add_argument("--inventory", type=Path, default=Path("reports/data_preparation"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/roadguard_v1"))
    parser.add_argument("--reports", type=Path, default=Path("reports/data_preparation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = (validate_dataset(args.input, args.output, args.reports, args.inventory) if args.validate_only
                  else prepare(args.input, args.inventory, args.output, args.reports, args.seed))
    except (ValueError, OSError, ET.ParseError) as error:
        parser.exit(1, f"Preparation/validation failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
