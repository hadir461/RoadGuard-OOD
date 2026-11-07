"""Read-only RDD2022 inventory, associations, XML statistics and optional decoding.

The CLI uses empirical RDD2022 bounds: 0 <= xmin < xmax <= width (and y).
Box statistics use geometric spans xmax-xmin and ymax-ymin, not pixel counts.
This policy accepts observed zero borders without asserting whether the
original annotations use inclusive or exclusive pixel endpoints. The legacy
one-based inclusive VOC mode remains the default for the Python audit API.
"""

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

try:
    from PIL import Image
except ImportError:
    Image = None


CLASSES = ("D00", "D10", "D20", "D40")
COUNTRIES = {name.casefold(): name for name in ("India", "Japan", "Czech")}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BOX_CONVENTION = (
    "Integer, one-based, inclusive coordinates: "
    "1 <= xmin <= xmax <= width; 1 <= ymin <= ymax <= height. "
    "Positive integer XML width and height are required. "
    "A one-pixel box is valid; box statistics use inclusive spans (+1)."
)
RDD_BOX_CONVENTION = (
    "Integer coordinates with observed RDD2022 bounds: "
    "0 <= xmin < xmax <= width; 0 <= ymin < ymax <= height. "
    "Positive integer XML width and height are required. "
    "Statistics use geometric spans (xmax-xmin, ymax-ymin), not pixel counts. "
    "Pixel endpoint inclusivity cannot be inferred from these bounds alone."
)
ERROR_CATEGORIES = (
    "xml_errors", "unknown_classes", "invalid_boxes", "scan_errors",
    "invalid_dimensions", "filename_errors", "image_decode_errors",
    "image_dimension_mismatches", "ambiguous_associations",
)
CSV_NAMES = (
    "country_summary.csv", "class_counts.csv", "dimension_statistics.csv",
    "anomalies.csv", "missing_associations.csv",
)


def dimensions(root: ET.Element) -> tuple[int, int] | None:
    try:
        width = int(root.findtext("size/width", ""))
        height = int(root.findtext("size/height", ""))
    except ValueError:
        return None
    return (width, height) if width > 0 and height > 0 else None


def coordinates(obj: ET.Element) -> tuple[int, ...] | None:
    try:
        return tuple(int(obj.findtext(f"bndbox/{name}", ""))
                     for name in ("xmin", "ymin", "xmax", "ymax"))
    except ValueError:
        return None


def box_error(root: ET.Element, obj: ET.Element, convention: str = "voc") -> str | None:
    """Return the reason a box is invalid, or None when it is valid."""
    size, box = dimensions(root), coordinates(obj)
    if size is None or box is None:
        return "Missing or non-integer dimensions/coordinates"
    width, height = size
    xmin, ymin, xmax, ymax = box
    if convention == "rdd2022":
        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
            return "Coordinates violate RDD2022 bounds or have non-positive extent"
    elif convention == "voc":
        if 1 <= xmin <= xmax <= width and 1 <= ymin <= ymax <= height:
            return None
        return "Coordinates violate the one-based, inclusive box convention"
    else:
        raise ValueError(f"Unknown box convention: {convention}")
    return None


def new_summary() -> dict:
    return {
        **dict.fromkeys((
            "image_count", "xml_count", "parsed_xml_count", "object_count",
            "empty_xml_count", "valid_box_count", "invalid_box_count",
            "unknown_class_count", "matched_pair_count", "missing_image_count",
            "missing_annotation_count", "ambiguous_association_count",
            "image_decode_attempted", "image_decode_succeeded", "image_decode_failed",
        ), 0),
        "instances_per_class": dict.fromkeys(CLASSES, 0),
        "valid_boxes_per_class": dict.fromkeys(CLASSES, 0),
        "unknown_class_counts": {},
    }


def file_context(path: Path, inputs: list[Path]) -> tuple[str, str, str]:
    """Pair within the same dataset scope, never across countries or splits.

    RDD images/ and annotations/ share a parent (normally train/). For generic
    layouts, use the country folder or the containing input root as a scope.
    """
    country, country_root = "unassigned", None
    for index, part in enumerate(path.parts[:-1]):
        if part.casefold() in COUNTRIES:
            country = COUNTRIES[part.casefold()]
            country_root = Path(*path.parts[:index + 1])
    markers = [index for index, part in enumerate(path.parts[:-1])
               if part.casefold() in {"images", "annotations"}]
    fallback = next((root for root in inputs if path.is_relative_to(root)), path.parent)
    scope = Path(*path.parts[:markers[-1]]) if markers else country_root or fallback
    split = scope.name if scope.name.casefold() in {"train", "val", "validation", "test"} else "unspecified"
    return country, str(scope), split


def describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {"count": len(values), "min": min(values), "max": max(values),
            "mean": statistics.fmean(values), "median": statistics.median(values)}


def audit(inputs: list[Path], *, convention: str = "voc", verify_images: bool = False,
          progress: bool = False) -> dict:
    """Inventory without writes. Defaults preserve the initial Python API.

    Counts include known-class objects with invalid boxes. Box statistics exclude
    invalid boxes; overall box statistics also include unknown classes. Image
    dimension statistics come only from valid XML size fields, including empty
    annotations. Decoding is optional and never repairs or saves an image.
    """
    if convention not in {"voc", "rdd2022"}:
        raise ValueError(f"Unknown box convention: {convention}")
    report = {
        **new_summary(),
        "input_paths": [str(Path(path)) for path in inputs],
        "image_integrity_checked": False,
        "box_convention_name": convention,
        "box_convention": RDD_BOX_CONVENTION if convention == "rdd2022" else BOX_CONVENTION,
        "countries": {},
        **{key: [] for key in ERROR_CATEGORIES},
        "missing_associations": [],
        "coordinate_evidence": {
            "zero_min_boxes": 0, "max_at_dimension_boxes": 0,
            "equal_edge_boxes": 0, "examples": {},
        },
        "statistics_notes": (
            "Image sizes are XML metadata, not decoded-image statistics. "
            "Box statistics exclude invalid boxes; all-class statistics include unknown labels. "
            "Known-class instance counts include invalid boxes. Empty XML has no objects. "
            "Pairing uses case-insensitive stems within a dataset scope; only unique 1:1 matches count."
        ),
    }
    roots = sorted({Path(path).resolve() for path in inputs}, key=lambda path: (len(path.parts), str(path)))
    samples = defaultdict(lambda: defaultdict(list))
    per_class_samples = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    files = set()

    def scan_error(error: OSError) -> None:
        report["scan_errors"].append(
            {"path": str(error.filename), "error": str(error)}
        )

    for path in roots:
        if not path.is_dir():
            report["scan_errors"].append(
                {"path": str(path), "error": "Input directory does not exist or is not a directory"}
            )
            continue
        for directory, folders, names in os.walk(path, onerror=scan_error):
            folders.sort()
            for name in names:
                candidate = Path(directory) / name
                if candidate.suffix.lower() in IMAGE_EXTENSIONS | {".xml"}:
                    files.add(candidate.resolve())

    contexts = {path: file_context(path, roots) for path in files}
    groups = defaultdict(lambda: {"images": [], "xmls": []})

    def summaries(country):
        return report, report["countries"].setdefault(country, new_summary())

    def increment(country, key, amount=1):
        for summary in summaries(country):
            summary[key] += amount

    for path in sorted(files):
        country, scope, _ = contexts[path]
        kind = "images" if path.suffix.lower() in IMAGE_EXTENSIONS else "xmls"
        increment(country, "image_count" if kind == "images" else "xml_count")
        groups[(country, scope, path.stem.casefold())][kind].append(path)

    pairs = {}
    for (country, scope, stem), group in sorted(groups.items()):
        images, xmls = group["images"], group["xmls"]
        if len(images) == len(xmls) == 1:
            pairs[xmls[0]] = images[0]
            increment(country, "matched_pair_count")
        if not images or not xmls:
            kind = "missing_image" if not images else "missing_annotation"
            for path in xmls if not images else images:
                report["missing_associations"].append({
                    "country": country, "scope": scope, "split": contexts[path][2],
                    "type": kind, "stem": stem, "path": str(path),
                })
                increment(country, f"{kind}_count")
        if len(images) > 1 or len(xmls) > 1:
            report["ambiguous_associations"].append({
                "country": country, "scope": scope, "stem": stem,
                "images": [str(path) for path in images], "xmls": [str(path) for path in xmls],
                "error": "Duplicate stem: no unique image/XML association",
            })
            increment(country, "ambiguous_association_count")

    xml_sizes = {}
    xml_paths = sorted(path for path in files if path.suffix.lower() == ".xml")
    for number, path in enumerate(xml_paths, start=1):
        country, scope, split = contexts[path]
        context = {"path": str(path), "country": country, "scope": scope, "split": split}
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError, ValueError, LookupError) as error:
            report["xml_errors"].append({**context, "error": str(error)})
            continue
        if root.tag != "annotation":
            report["xml_errors"].append(
                {**context, "error": "Expected PASCAL VOC annotation root"}
            )
            continue
        increment(country, "parsed_xml_count")
        size = dimensions(root)
        if size is None:
            report["invalid_dimensions"].append({**context, "error": "Missing or invalid positive integer XML dimensions"})
        else:
            xml_sizes[path] = size
            for label in ("all", country):
                for metric, value in zip(("image_width", "image_height", "image_aspect_ratio"), (*size, size[0] / size[1])):
                    samples[label][metric].append(value)
        filename = root.findtext("filename", "").strip().replace("\\", "/").rsplit("/", 1)[-1]
        paired_image = pairs.get(path)
        if not filename or Path(filename).stem.casefold() != path.stem.casefold() or (
            paired_image is not None and filename.casefold() != paired_image.name.casefold()
        ):
            report["filename_errors"].append({**context, "filename": filename,
                "error": "Missing XML filename or filename inconsistent with XML/image basename"})
        objects = root.findall("object")
        increment(country, "empty_xml_count", int(not objects))
        for index, obj in enumerate(objects, start=1):
            name = obj.findtext("name", "").strip()
            object_context = {**context, "object_index": index, "class": name}
            increment(country, "object_count")
            if name in CLASSES:
                for summary in summaries(country):
                    summary["instances_per_class"][name] += 1
            else:
                report["unknown_classes"].append(object_context)
                increment(country, "unknown_class_count")
                for summary in summaries(country):
                    summary["unknown_class_counts"][name] = summary["unknown_class_counts"].get(name, 0) + 1
            box = coordinates(obj)
            if box is not None and size is not None:
                evidence = report["coordinate_evidence"]
                checks = {"zero_min_boxes": box[0] == 0 or box[1] == 0,
                          "max_at_dimension_boxes": box[2] == size[0] or box[3] == size[1],
                          "equal_edge_boxes": box[0] == box[2] or box[1] == box[3]}
                for key, present in checks.items():
                    evidence[key] += int(present)
                    if present:
                        examples = evidence["examples"].setdefault(key, [])
                        if len(examples) < 3:
                            examples.append({**object_context, "size": size, "coordinates": box})
            error = box_error(root, obj, convention)
            if error is not None:
                report["invalid_boxes"].append({**object_context, "coordinates": box, "error": error})
                increment(country, "invalid_box_count")
                continue
            increment(country, "valid_box_count")
            offset = 1 if convention == "voc" else 0
            width, height = box[2] - box[0] + offset, box[3] - box[1] + offset
            metrics = {"box_width": width, "box_height": height, "box_area": width * height,
                       "box_aspect_ratio": width / height, "box_relative_area": width * height / (size[0] * size[1])}
            for label in ("all", country):
                for metric, value in metrics.items():
                    samples[label][metric].append(value)
                    if name in CLASSES:
                        per_class_samples[label][name][metric].append(value)
            if name in CLASSES:
                for summary in summaries(country):
                    summary["valid_boxes_per_class"][name] += 1
        if progress and number % 5000 == 0:
            print(f"XML metadata: {number}/{len(xml_paths)}", flush=True)

    verification = {
        "requested": verify_images, "pillow_available": Image is not None,
        "status": "not_requested", "attempted": 0, "succeeded": 0, "failed": 0,
        "method": "Pillow Image.open followed by load(), no image writes",
    }
    image_sizes = {}
    if verify_images and Image is None:
        verification["status"] = "not_performed_pillow_unavailable"
    elif verify_images:
        verification["status"] = "completed" if report["image_count"] else "no_images"
        image_paths = sorted(path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS)
        for number, path in enumerate(image_paths, start=1):
            country, scope, split = contexts[path]
            increment(country, "image_decode_attempted")
            try:
                with Image.open(path) as image:
                    image.load()
                    image_sizes[path] = image.size
                increment(country, "image_decode_succeeded")
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                report["image_decode_errors"].append({"path": str(path), "country": country,
                    "scope": scope, "split": split, "error": str(error)})
                increment(country, "image_decode_failed")
            if progress and number % 1000 == 0:
                print(f"Image decoding: {number}/{len(image_paths)}", flush=True)
        report["image_integrity_checked"] = bool(image_paths)
    for key in ("attempted", "succeeded", "failed"):
        verification[key] = report[f"image_decode_{key}"]
    report["image_verification"] = verification
    for xml, image in pairs.items():
        if xml in xml_sizes and image in image_sizes and xml_sizes[xml] != image_sizes[image]:
            report["image_dimension_mismatches"].append({
                "path": str(xml), "image_path": str(image), "country": contexts[xml][0],
                "xml_size": xml_sizes[xml], "decoded_size": image_sizes[image],
                "error": "XML dimensions differ from decoded image dimensions",
            })

    metric_names = ("image_width", "image_height", "image_aspect_ratio", "box_width",
                    "box_height", "box_area", "box_aspect_ratio", "box_relative_area")
    for label, summary in [("all", report), *sorted(report["countries"].items())]:
        summary["dimension_statistics"] = {metric: describe(samples[label][metric]) for metric in metric_names}
        summary["box_statistics_per_class"] = {
            name: {metric: describe(per_class_samples[label][name][metric])
                   for metric in metric_names if metric.startswith("box_")}
            for name in CLASSES
        }
        summary["anomaly_counts"] = {
            category: sum(label == "all" or item.get("country") == label for item in report[category])
            for category in ERROR_CATEGORIES
        }
    report["anomaly_counts"]["missing_associations"] = len(report["missing_associations"])

    report["dataset_found"] = bool(report["image_count"] or report["xml_count"])
    return report


def write_csv(path: Path, fields: list[str], rows) -> None:
    """UTF-8 BOM and explicit headers make even empty CSVs usable in Excel."""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_reports(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    counts = ["image_count", "xml_count", "parsed_xml_count", "object_count", "empty_xml_count",
              "valid_box_count", "invalid_box_count", "unknown_class_count", "matched_pair_count",
              "missing_image_count", "missing_annotation_count", "ambiguous_association_count",
              "image_decode_attempted", "image_decode_succeeded", "image_decode_failed"]
    write_csv(output / "country_summary.csv", ["country", *counts],
              ({"country": country, **summary} for country, summary in sorted(report["countries"].items())))
    class_rows, dimension_rows = [], []
    for country, summary in [("all", report), *sorted(report["countries"].items())]:
        for name in CLASSES:
            class_rows.append({"country": country, "class": name, "known_class": True,
                               "instances": summary["instances_per_class"][name],
                               "valid_boxes": summary["valid_boxes_per_class"][name]})
        for name, count in sorted(summary["unknown_class_counts"].items()):
            class_rows.append({"country": country, "class": name, "known_class": False,
                               "instances": count, "valid_boxes": ""})
        for metric, values in summary["dimension_statistics"].items():
            dimension_rows.append({"country": country, "class": "all", "metric": metric, **values})
        for name, metrics in summary["box_statistics_per_class"].items():
            for metric, values in metrics.items():
                dimension_rows.append({"country": country, "class": name, "metric": metric, **values})
    write_csv(output / "class_counts.csv", ["country", "class", "known_class", "instances", "valid_boxes"], class_rows)
    write_csv(output / "dimension_statistics.csv", ["country", "class", "metric", "count", "min", "max", "mean", "median"], dimension_rows)
    anomaly_rows = []
    for category in (*ERROR_CATEGORIES, "missing_associations"):
        for item in report[category]:
            anomaly_rows.append({"type": category, **item, "details": json.dumps(item, ensure_ascii=False)})
    write_csv(output / "anomalies.csv", ["type", "country", "path", "object_index", "class", "error", "details"], anomaly_rows)
    write_csv(output / "missing_associations.csv", ["country", "scope", "split", "type", "stem", "path"], report["missing_associations"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Input directories")
    parser.add_argument("--output", type=Path, required=True, help="Report directory")
    parser.add_argument("--box-convention", choices=("rdd2022", "voc"), default="rdd2022")
    parser.add_argument("--skip-image-verification", action="store_true", help="Skip optional Pillow decoding")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    destinations = [output / name for name in ("summary.json", *CSV_NAMES)]
    if any(
        output.is_relative_to(path.resolve())
        or any(destination.resolve().is_relative_to(path.resolve()) for destination in destinations)
        for path in args.input
    ):
        parser.error("--output must be outside all input directories to preserve source data")

    print("Scanning inputs and XML metadata...", flush=True)
    report = audit(args.input, convention=args.box_convention,
                   verify_images=not args.skip_image_verification, progress=True)
    try:
        write_reports(report, output)
    except OSError as error:
        parser.exit(1, f"Cannot write audit reports: {error}\n")
    print(f"Reports: {output}")
    print(f"Images: {report['image_count']}; XML: {report['xml_count']}")
    print(f"Image verification: {report['image_verification']['status']}")
    for country, summary in sorted(report["countries"].items()):
        print(f"{country}: {summary['image_count']} images; {summary['xml_count']} XML; "
              f"classes={summary['instances_per_class']}")
    if not report["dataset_found"]:
        print("No dataset files found.")
    return 1 if report["scan_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
