"""Reproducible, read-only RDD2022 manifest and duplicate-risk inventory.

Image/XML paths are POSIX paths relative to --input. SHA-256 compares file
bytes. dHash compares horizontal luminance gradients after LANCZOS resizing
to 9x8, with no EXIF rotation: 64 bits, Hamming threshold 4 by default.
Threshold+1 disjoint bit bands retrieve every hash pair within the threshold
(pigeonhole principle), followed by an exact Hamming check. This is not an
exhaustive image-pair scan. Similarity is not proof of a shared scene/route.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO
from itertools import combinations, product
import json
import os
from pathlib import Path
import platform
import re
import xml.etree.ElementTree as ET

try:
    from PIL import Image, __version__ as PILLOW_VERSION
except ImportError:
    Image, PILLOW_VERSION = None, None

from .audit import CLASSES, IMAGE_EXTENSIONS, box_error, dimensions, file_context


CLASS_IDS = {name: index for index, name in enumerate(CLASSES)}
CATEGORIES = ("target_only", "mixed", "non_target_only", "empty_xml", "unusable_annotation")
SEQUENCE_TAGS = {"sequence", "sequence_id", "route_id", "trip_id", "trajectory_id", "video_id"}
TIME_TAGS = {"timestamp", "frame_id", "frame_index", "gps", "latitude", "longitude"}
OUTPUT_NAMES = ("manifest.csv", "exact_duplicates.csv", "near_duplicate_candidates.csv", "summary.json", "data_policy.md")
MANIFEST_FIELDS = (
    "country", "domain_role", "image_path", "xml_path", "xml_status", "width", "height",
    "xml_width", "xml_height", "object_count", "target_count", "valid_target_count",
    "non_target_count", "D00", "D10", "D20", "D40", "classes_present",
    "non_target_counts", "category", "invalid_box_count", "invalid_target_box_count",
    "sha256", "xml_sha256", "dhash64", "xml_sequence_metadata", "exif_context_fields", "issues",
)
EXACT_FIELDS = ("group_id", "sha256", "group_size", "country", "image_path", "xml_path", "cross_id_ood")
NEAR_FIELDS = ("image_a", "country_a", "xml_a", "image_b", "country_b", "xml_b",
               "hamming_distance", "cross_country", "cross_id_ood")


def compact(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def domain_role(country):
    return "ID_source" if country in {"India", "Japan"} else "OOD_test_only" if country == "Czech" else "unassigned"


def cross_id_ood(countries):
    return "Czech" in countries and bool({"India", "Japan"}.intersection(countries))


def classify(target_count, other_count):
    if target_count:
        return "mixed" if other_count else "target_only"
    return "non_target_only" if other_count else "empty_xml"


def dhash(image):
    """64 row-major bits, each bit 1 iff the left grayscale pixel is brighter."""
    with image.convert("L") as gray:
        with gray.resize((9, 8), Image.Resampling.LANCZOS) as small:
            pixels = list(small.get_flattened_data())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def band_keys(value, threshold):
    count = threshold + 1
    width, extra = divmod(64, count)
    offset = 0
    for band in range(count):
        size = width + (band < extra)
        yield band, (value >> offset) & ((1 << size) - 1)
        offset += size


def near_pairs(rows, threshold=4, stats=None, progress=False):
    """Yield unordered image pairs once, excluding byte-identical SHA groups.

    Index distinct hashes first. Identical perceptual hashes still expand to
    candidate pairs if the file bytes differ. Dense buckets can remain costly;
    the number of actual candidate output pairs can itself be quadratic.
    """
    if not 0 <= threshold <= 8:
        raise ValueError("Hamming threshold must be between 0 and 8")
    stats = stats if stats is not None else {}
    grouped = defaultdict(list)
    for row in rows:
        if row["dhash64"]:
            grouped[int(row["dhash64"], 16)].append(row)
    stats.update(unique_hashes=len(grouped), full_hamming_comparisons=0,
                 exhaustive_unique_hash_pairs=len(grouped) * (len(grouped) - 1) // 2,
                 largest_equal_hash_group=max(map(len, grouped.values()), default=0))
    index = defaultdict(list)

    def pair(a, b, distance):
        if a["sha256"] == b["sha256"]:
            return None
        a, b = sorted((a, b), key=lambda row: row["image_path"])
        return {"image_a": a["image_path"], "country_a": a["country"], "xml_a": a["xml_path"],
                "image_b": b["image_path"], "country_b": b["country"], "xml_b": b["xml_path"],
                "hamming_distance": distance, "cross_country": a["country"] != b["country"],
                "cross_id_ood": cross_id_ood({a["country"], b["country"]})}

    for number, value in enumerate(sorted(grouped), 1):
        for a, b in combinations(grouped[value], 2):
            candidate = pair(a, b, 0)
            if candidate is not None:
                yield candidate
        keys = list(band_keys(value, threshold))
        hits = set()
        for key in keys:
            hits.update(index[key])
        for previous in sorted(hits):
            stats["full_hamming_comparisons"] += 1
            distance = (value ^ previous).bit_count()
            if distance <= threshold:
                for a, b in product(grouped[previous], grouped[value]):
                    candidate = pair(a, b, distance)
                    if candidate is not None:
                        yield candidate
        for key in keys:
            index[key].append(value)
        if progress and number % 5000 == 0:
            print(f"Perceptual index: {number}/{len(grouped)} unique hashes", flush=True)


def collect_manifest(root, progress=False):
    """Read each image and its associated XML; retain failed rows explicitly."""
    root = Path(root).resolve()
    files, sidecars, scan_errors = [], [], []
    for directory, folders, names in os.walk(root, onerror=lambda error: scan_errors.append(str(error).replace(str(root), "."))):
        folders.sort()
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix.lower() in IMAGE_EXTENSIONS | {".xml"}:
                if not path.resolve().is_relative_to(root):
                    scan_errors.append(f"External link excluded: {path.relative_to(root).as_posix()}")
                    continue
                files.append(path)
            elif path.suffix.lower() in {".csv", ".json", ".txt", ".gpx", ".kml"}:
                sidecars.append(path.relative_to(root).as_posix())
    if not root.is_dir():
        scan_errors.append("Input directory is missing or is not a directory")
    groups = defaultdict(lambda: {"images": [], "xmls": []})
    for path in files:
        country, scope, _ = file_context(path, [root])
        kind = "images" if path.suffix.lower() in IMAGE_EXTENSIONS else "xmls"
        groups[(country, scope, path.stem.casefold())][kind].append(path)
    rows, used_xmls = [], set()
    tags, sequence_fields, exif_fields = Counter(), Counter(), Counter()
    images = sorted(path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS)
    for number, path in enumerate(images, 1):
        country, scope, _ = file_context(path, [root])
        candidates = groups[(country, scope, path.stem.casefold())]
        row = dict.fromkeys(MANIFEST_FIELDS, "")
        row.update(country=country, domain_role=domain_role(country), image_path=path.relative_to(root).as_posix(),
                   category="unusable_annotation", xml_status="missing")
        for key in ("object_count", "target_count", "valid_target_count", "non_target_count",
                    "invalid_box_count", "invalid_target_box_count", *CLASSES):
            row[key] = 0
        issues = []
        other, labels, sequence = Counter(), Counter(), {}

        def issue(code, error=""):
            detail = str(error).replace(str(root), ".")
            # Pillow may include a BytesIO memory address in decoding errors.
            # Such runtime addresses must not break reproducible serialization.
            detail = re.sub(r"\bat 0x[0-9a-fA-F]+", "at <address>", detail)
            issues.append({"code": code, "detail": detail})

        try:
            content = path.read_bytes()
            row["sha256"] = hashlib.sha256(content).hexdigest()
            with Image.open(BytesIO(content)) as image:
                image.load()
                row["width"], row["height"] = image.size
                row["dhash64"] = f"{dhash(image):016x}"
                try:
                    exif = image.getexif()
                    present = set(exif.keys())
                    if 34665 in present:
                        present.update(exif.get_ifd(34665).keys())
                    fields = sorted(name for key, name in ((306, "DateTime"), (36867, "DateTimeOriginal"),
                                    (36868, "DateTimeDigitized"), (34853, "GPSInfo")) if key in present)
                    row["exif_context_fields"] = compact(fields)
                    exif_fields.update(fields)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    issue("exif_metadata_error", error)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            issue("image_read_or_decode_error", error)

        xmls = candidates["xmls"]
        if len(xmls) != 1 or len(candidates["images"]) != 1:
            row["xml_status"] = "ambiguous" if xmls else "missing"
            issue("xml_ambiguous" if xmls else "xml_missing", compact([p.relative_to(root).as_posix() for p in xmls]))
        else:
            xml = xmls[0]
            used_xmls.add(xml)
            row["xml_path"] = xml.relative_to(root).as_posix()
            try:
                xml_bytes = xml.read_bytes()
                row["xml_sha256"] = hashlib.sha256(xml_bytes).hexdigest()
                annotation = ET.fromstring(xml_bytes)
                if annotation.tag != "annotation":
                    raise ValueError("Expected PASCAL VOC annotation root")
                row["xml_status"] = "parsed"
                tags.update({child.tag for child in annotation})
                for element in annotation.iter():
                    tag = str(element.tag).rsplit("}", 1)[-1].casefold()
                    if tag in SEQUENCE_TAGS | TIME_TAGS and element.text and element.text.strip():
                        sequence.setdefault(tag, set()).add(element.text.strip())
                sequence_fields.update(sequence.keys())
                size = dimensions(annotation)
                if size is None:
                    issue("invalid_xml_dimensions")
                else:
                    row["xml_width"], row["xml_height"] = size
                    if row["width"] and size != (row["width"], row["height"]):
                        issue("image_xml_dimension_mismatch")
                filename = annotation.findtext("filename", "").strip().replace("\\", "/").rsplit("/", 1)[-1]
                if not filename or filename.casefold() != path.name.casefold():
                    issue("xml_filename_mismatch", filename)
                for index, obj in enumerate(annotation.findall("object"), 1):
                    name = obj.findtext("name", "").strip()
                    labels[name] += 1
                    row["object_count"] += 1
                    if name in CLASSES:
                        row[name] += 1
                        row["target_count"] += 1
                    else:
                        other[name] += 1
                        row["non_target_count"] += 1
                    error = box_error(annotation, obj, "rdd2022")
                    if error:
                        row["invalid_box_count"] += 1
                        row["invalid_target_box_count"] += int(name in CLASSES)
                        issue("invalid_box", f"Object {index} ({name}): {error}")
                    elif name in CLASSES:
                        row["valid_target_count"] += 1
                row["category"] = classify(row["target_count"], row["non_target_count"])
            except (OSError, ET.ParseError, ValueError, LookupError) as error:
                row["xml_status"] = "invalid"
                issue("xml_invalid", error)
        row["classes_present"] = compact(sorted(labels))
        row["non_target_counts"] = compact(dict(other))
        row["xml_sequence_metadata"] = compact({key: sorted(values) for key, values in sequence.items()})
        row["issues"] = compact(issues)
        rows.append(row)
        if progress and number % 1000 == 0:
            print(f"Manifest: {number}/{len(images)} images", flush=True)
    countries = {}
    for country in sorted({row["country"] for row in rows}):
        selected = [row for row in rows if row["country"] == country]
        countries[country] = {
            "image_count": len(selected), "categories": {name: sum(row["category"] == name for row in selected) for name in CATEGORIES},
            "object_count": sum(row["object_count"] for row in selected),
            "target_count": sum(row["target_count"] for row in selected),
            "valid_target_count": sum(row["valid_target_count"] for row in selected),
            "non_target_count": sum(row["non_target_count"] for row in selected),
            "instances_per_class": {name: sum(row[name] for row in selected) for name in CLASSES},
        }
    summary = {
        "schema_version": 1, "dataset_root": ".", "paths": "POSIX, relative to the dataset root passed as --input",
        "python_version": platform.python_version(), "pillow_version": PILLOW_VERSION,
        "class_ids": CLASS_IDS, "image_count": len(rows), "xml_count": sum(p.suffix.lower() == ".xml" for p in files),
        "countries": countries, "scan_errors": scan_errors,
        "unpaired_xml_paths": sorted(p.relative_to(root).as_posix() for p in files if p.suffix.lower() == ".xml" and p not in used_xmls),
        "issue_counts": dict(sorted(Counter(item["code"] for row in rows for item in json.loads(row["issues"])).items())),
        "sequence_metadata": {
            "validated_route_or_sequence_groups": False,
            "xml_top_level_tag_counts": dict(sorted(tags.items())),
            "xml_context_fields_image_counts": dict(sorted(sequence_fields.items())),
            "exif_context_fields_image_counts": dict(sorted(exif_fields.items())),
            "sidecar_files_to_review": sorted(sidecars),
            "limitation": "No groups inferred from filenames or numeric suffixes. Explicit metadata, if present, still requires semantic validation; timestamps/GPS alone are not validated trip identifiers.",
        },
    }
    return rows, summary


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def policy_text(summary):
    table = "\n".join(
        f"| {country} | {values['image_count']} | " + " | ".join(str(values["categories"][name]) for name in CATEGORIES) + " |"
        for country, values in summary["countries"].items()
    )
    exact, near = summary["exact_duplicates"], summary["near_duplicates"]
    metadata = summary["sequence_metadata"]
    invalid_targets = sum(values["target_count"] - values["valid_target_count"] for values in summary["countries"].values())
    return f"""# Politique de préparation avant partitions — RoadGuard-OOD

Inventaire déterministe en lecture seule. Aucun split, filtrage, déplacement,
suppression, conversion YOLO ou entraînement n'est effectué. Aucun tirage aléatoire.
Les chemins image/XML sont relatifs à la racine RDD2022 et utilisent `/`.

## Catégories : images, pas instances

| Pays | Images | Cibles seules | Mixed | Hors cible seules | XML sans objet | Annotation inexploitable |
|---|---:|---:|---:|---:|---:|---:|
{table}

`target_only` : au moins un objet cible et aucun autre ; `mixed` : les deux ;
`non_target_only` : objets exclusivement hors cible ; `empty_xml` : XML PASCAL VOC
analysé avec zéro objet. Une annotation manquante, ambiguë ou invalide est
`unusable_annotation`, jamais un XML vide. Les catégories comptent les objets
déclarés, même si leur boîte est invalide ; consulter `valid_target_count`,
`invalid_target_box_count`, `xml_status` et `issues` avant toute utilisation.
`classes_present`, `non_target_counts`, métadonnées et `issues` sont du JSON dans
les cellules CSV. Les dimensions `width/height` sont celles décodées de l'image ;
`xml_width/xml_height` sont celles déclarées. Classes : D00=0, D10=1, D20=2, D40=3.

## Doublons et risques de fuite

- SHA-256 des octets complets : {exact['group_count']} groupes, {exact['member_images']} images
  membres, {exact['redundant_images']} copies au-delà d'un représentant par groupe,
  {exact['pair_count']} paires. Le CSV exact contient une ligne par membre de groupe.
- Groupes exacts reliant India/Japan à Czech : {exact['cross_id_ood_groups']} ;
  paires exactes ID/OOD : {exact['cross_id_ood_pairs']}.
- Candidats perceptuels : {near['candidate_pairs']} paires d'octets différents,
  dont {near['cross_id_ood_pairs']} reliant ID à Czech. Le CSV near contient une
  ligne par paire non ordonnée ; les doublons SHA-256 en sont exclus.
- Les colonnes `cross_id_ood` signalent séparément les liens India/Japan–Czech.
  Les nombres de groupes, membres, copies et paires ne sont pas interchangeables.

dHash64 : image telle que stockée (sans correction d'orientation EXIF), conversion
en niveaux de gris, redimensionnement LANCZOS à 9×8, 64 comparaisons horizontales
gauche > droite, bits dans l'ordre des lignes. Distance de Hamming <= {near['threshold']}.
L'index utilise {near['bands']} bandes disjointes : toute paire à cette distance
partage au moins une bande. Les distances sont ensuite vérifiées exactement.
Recherche sur les hashes distincts, sans balayage exhaustif N² ; les buckets
très denses et le volume des paires réellement candidates peuvent toutefois être
quadratiques dans des cas dégénérés. Comparaisons complètes observées :
{near['full_hamming_comparisons']} contre {near['exhaustive_unique_hash_pairs']} paires de hashes distincts possibles.

Le seuil 4 est un choix conservateur de présélection, pas un seuil validé sur
RDD2022. Il peut rater un recadrage, une rotation, un déplacement de caméra ou
deux frames proches ; des routes sans texture peuvent produire des collisions.
Une distance zéro ne prouve pas l'identité. Aucune paire perceptuelle n'autorise
une suppression ou un regroupement automatique.

## Recommandations à valider humainement pour B0

1. India/Japan restent sources ID ; Czech reste réservé au test OOD final.
   Aucun apprentissage, augmentation, réglage ou choix de modèle sur Czech.
2. Commencer l'examen par les liens exacts ou perceptuels ID/OOD. Un doublon
   confirmé ne doit pas traverser apprentissage et évaluation. Vérifier la
   provenance ; prévoir l'exclusion d'un représentant ID des futurs manifestes
   dérivés ou une quarantaine du composant ambigu, avec journal de décision.
   Ne pas altérer les bruts et ne pas décider à partir de scores de modèles.
3. Garder les copies exactes confirmées dans une même unité de partition.
   Examiner aussi leurs annotations ; des XML de noms différents peuvent avoir
   des SHA différents sans contradiction sémantique.
4. Revoir visuellement les candidats perceptuels avant de décider des groupes.
   Une chaîne A–B–C ne prouve pas que A et C représentent la même scène.
5. `target_only` avec annotations cohérentes : candidats prioritaires pour B0.
   `mixed` : conserver les cibles valides seulement après décision explicite de
   taxonomie et vérification des annotations. Ne pas transformer une image mixed
   en image négative. Les labels hors cible ne sont jamais remappés implicitement.
6. `non_target_only` : écarter provisoirement de B0, sans suppression, en attendant
   une revue de complétude. Ce ne sont pas automatiquement des négatifs cibles.
   `empty_xml` : n'admettre comme négatifs qu'après revue d'échantillons et
   validation de la politique ; zéro objet XML ne prouve pas zéro dommage.
7. Annotations inexploitables ou boîtes invalides : revue séparée. Cet inventaire
   compte {invalid_targets} boîte(s) cible(s) invalide(s), à examiner via le manifeste.
   Toute exclusion/correction future sera appliquée seulement à une copie dérivée
   et tracée, sans éliminer arbitrairement les autres objets valides de l'image.
8. Après ces décisions seulement : figer les unités de séparation, les manifestes
   train/validation/test ID et le périmètre OOD, avant toute conversion/expérience.
   L'absence de candidat détecté n'est pas une preuve d'indépendance.

## Trajets et séquences

Aucun groupe par trajet/séquence n'est inventé. `sequence_metadata` du JSON décrit
les tags XML, champs explicites de séquence/temps/GPS, champs EXIF contextuels et
éventuels fichiers annexes trouvés. Les champs XML explicites sont conservés dans
`xml_sequence_metadata` ; leur sémantique reste à valider. Des numéros voisins
dans les noms ne prouvent ni un trajet commun ni une indépendance.

Constats de cette exécution :
- Tags XML de premier niveau (nombre de XML) : `{compact(metadata['xml_top_level_tag_counts'])}`.
- Champs XML de contexte (nombre d'images par champ) : `{compact(metadata['xml_context_fields_image_counts'])}`.
- Champs EXIF de contexte (nombre d'images par champ) : `{compact(metadata['exif_context_fields_image_counts'])}`.
- Annexes potentielles à examiner : `{compact(metadata['sidecar_files_to_review'])}`.

Ces contrôles ne fournissent aucun regroupement par trajet validé. En l'absence
d'identifiants explicites vérifiables, signaler cette limite dans l'étude ID/OOD
et ne pas prétendre que des noms distincts assurent l'indépendance des images.

## Reproductibilité et limites

Ordre des chemins, groupes et paires fixé, sérialisation JSON stable, aucune date
variable ni chemin absolu dans les rapports. Versions Python/Pillow et SHA-256
des quatre autres livrables figurent dans summary.json. XML et images ont aussi
leurs hashes dans le manifeste. Relancer avec les mêmes octets, versions et seuil
doit produire les mêmes fichiers. Le décodage ne certifie pas l'exhaustivité des
annotations, les groupes de trajets ou l'absence de toutes les fuites possibles.
"""


def write_reports(rows, summary, output, threshold=4, progress=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "manifest.csv", MANIFEST_FIELDS, rows)
    grouped = defaultdict(list)
    for row in rows:
        if row["sha256"]:
            grouped[row["sha256"]].append(row)
    exact = {"group_count": 0, "member_images": 0, "redundant_images": 0,
             "pair_count": 0, "cross_id_ood_groups": 0, "cross_id_ood_pairs": 0}
    members = []
    for digest, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        size = len(group)
        cross = cross_id_ood({row["country"] for row in group})
        exact["group_count"] += 1
        exact["member_images"] += size
        exact["redundant_images"] += size - 1
        exact["pair_count"] += size * (size - 1) // 2
        exact["cross_id_ood_groups"] += int(cross)
        exact["cross_id_ood_pairs"] += sum(row["domain_role"] == "ID_source" for row in group) * sum(row["country"] == "Czech" for row in group)
        for row in group:
            members.append({"group_id": f"sha256:{digest}", "sha256": digest, "group_size": size,
                            **{key: row[key] for key in ("country", "image_path", "xml_path")}, "cross_id_ood": cross})
    write_csv(output / "exact_duplicates.csv", EXACT_FIELDS, members)
    near = {"algorithm": "dhash64-horizontal-LANCZOS-9x8-no-exif-transpose-v1", "threshold": threshold,
            "bands": threshold + 1, "candidate_pairs": 0, "cross_id_ood_pairs": 0, "cross_country_pairs": 0}
    pair_countries = Counter()
    involved = set()

    def candidates():
        for pair in near_pairs(rows, threshold, near, progress):
            near["candidate_pairs"] += 1
            near["cross_id_ood_pairs"] += int(pair["cross_id_ood"])
            near["cross_country_pairs"] += int(pair["cross_country"])
            pair_countries["|".join(sorted((pair["country_a"], pair["country_b"])))] += 1
            involved.update((pair["image_a"], pair["image_b"]))
            yield pair
    write_csv(output / "near_duplicate_candidates.csv", NEAR_FIELDS, candidates())
    near["images_in_candidates"] = len(involved)
    near["pairs_by_country_pair"] = dict(sorted(pair_countries.items()))
    summary["exact_duplicates"], summary["near_duplicates"] = exact, near
    (output / "data_policy.md").write_text(policy_text(summary), encoding="utf-8")
    summary["output_sha256"] = {}
    for name in OUTPUT_NAMES:
        if name != "summary.json":
            with (output / name).open("rb") as stream:
                summary["output_sha256"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Dataset root")
    parser.add_argument("--output", required=True, type=Path, help="Report directory outside raw data")
    parser.add_argument("--hamming-threshold", type=int, default=4, choices=range(9))
    args = parser.parse_args(argv)
    if Image is None:
        parser.error("Pillow is required; use the existing project .venv")
    root, output = args.input.resolve(), args.output.resolve()
    protected = [root, *(path for path in root.parents if path.name.casefold() == "raw")]
    if any(output.is_relative_to(path) or any((output / name).resolve().is_relative_to(path) for name in OUTPUT_NAMES) for path in protected):
        parser.error("Output must be outside the dataset and its raw-data directory")
    try:
        rows, summary = collect_manifest(root, progress=True)
        write_reports(rows, summary, output, args.hamming_threshold, progress=True)
    except OSError as error:
        parser.exit(1, f"Cannot build manifest reports: {error}\n")
    print(f"Reports: {args.output}")
    print(f"Images: {summary['image_count']}; exact groups: {summary['exact_duplicates']['group_count']}; "
          f"near candidate pairs: {summary['near_duplicates']['candidate_pairs']}")
    hard_errors = {"image_read_or_decode_error", "xml_missing", "xml_ambiguous", "xml_invalid"}
    return int(not rows or bool(summary["scan_errors"]) or bool(hard_errors.intersection(summary["issue_counts"])))


if __name__ == "__main__":
    raise SystemExit(main())
