"""Inventory RDD2022 files without modifying or decoding the source data.

Bounding boxes use integer, one-based, inclusive PASCAL VOC coordinates:
1 <= xmin <= xmax <= width and 1 <= ymin <= ymax <= height. A one-pixel
box is valid. Each object's XML must supply positive integer size/width and
size/height values. Missing or invalid dimensions/coordinates are reported
as invalid boxes; image dimensions are not checked against image contents.
"""

import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


CLASSES = ("D00", "D10", "D20", "D40")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BOX_CONVENTION = (
    "Integer, one-based, inclusive coordinates: "
    "1 <= xmin <= xmax <= width; 1 <= ymin <= ymax <= height. "
    "Positive integer XML width and height are required. "
    "A one-pixel box is valid. Image contents are not decoded."
)


def box_error(root: ET.Element, obj: ET.Element) -> str | None:
    """Return the reason a box is invalid, or None when it is valid."""
    try:
        width = int(root.findtext("size/width", ""))
        height = int(root.findtext("size/height", ""))
        xmin, ymin, xmax, ymax = (
            int(obj.findtext(f"bndbox/{name}", ""))
            for name in ("xmin", "ymin", "xmax", "ymax")
        )
    except ValueError:
        return "Missing or non-integer dimensions/coordinates"
    if width <= 0 or height <= 0:
        return "Image width and height must be positive"
    if not (1 <= xmin <= xmax <= width and 1 <= ymin <= ymax <= height):
        return "Coordinates violate the one-based, inclusive box convention"
    return None


def audit(inputs: list[Path]) -> dict:
    """Recursively inventory directories; report errors without changing inputs.

    Files in overlapping input paths are counted once. Known-class instances
    include objects with invalid boxes. Unknown classes are reported per object
    and excluded from instances_per_class. Object indexes are one-based.
    """
    report = {
        "input_paths": [str(Path(path)) for path in inputs],
        "image_count": 0,
        "xml_count": 0,
        "parsed_xml_count": 0,
        "instances_per_class": dict.fromkeys(CLASSES, 0),
        "image_integrity_checked": False,
        "box_convention": BOX_CONVENTION,
        "xml_errors": [],
        "unknown_classes": [],
        "invalid_boxes": [],
        "scan_errors": [],
    }
    files = set()

    def scan_error(error: OSError) -> None:
        report["scan_errors"].append(
            {"path": str(error.filename), "error": str(error)}
        )

    for input_path in inputs:
        path = Path(input_path)
        if not path.is_dir():
            report["scan_errors"].append(
                {"path": str(path), "error": "Input directory does not exist or is not a directory"}
            )
            continue
        for directory, _, names in os.walk(path, onerror=scan_error):
            for name in names:
                candidate = Path(directory) / name
                if candidate.suffix.lower() in IMAGE_EXTENSIONS | {".xml"}:
                    files.add(candidate.resolve())

    for path in sorted(files):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            report["image_count"] += 1
            continue
        report["xml_count"] += 1
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError, ValueError, LookupError) as error:
            report["xml_errors"].append({"path": str(path), "error": str(error)})
            continue
        if root.tag != "annotation":
            report["xml_errors"].append(
                {"path": str(path), "error": "Expected PASCAL VOC annotation root"}
            )
            continue
        report["parsed_xml_count"] += 1
        for index, obj in enumerate(root.findall("object"), start=1):
            name = obj.findtext("name", "").strip()
            context = {"path": str(path), "object_index": index, "class": name}
            if name in CLASSES:
                report["instances_per_class"][name] += 1
            else:
                report["unknown_classes"].append(context)
            error = box_error(root, obj)
            if error is not None:
                report["invalid_boxes"].append({**context, "error": error})

    report["dataset_found"] = bool(report["image_count"] or report["xml_count"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Input directories")
    parser.add_argument("--output", type=Path, required=True, help="Report directory")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    destination = output / "summary.json"
    if any(
        output.is_relative_to(path.resolve())
        or destination.resolve().is_relative_to(path.resolve())
        for path in args.input
    ):
        parser.error("--output must be outside all input directories to preserve source data")

    report = audit(args.input)
    try:
        output.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except OSError as error:
        parser.exit(1, f"Cannot write summary.json: {error}\n")
    print(f"Report: {destination}")
    print(f"Images: {report['image_count']}; XML: {report['xml_count']}")
    if not report["dataset_found"]:
        print("No dataset files found.")
    return 1 if report["scan_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
