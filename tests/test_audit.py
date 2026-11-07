"""Synthetic fixtures only: these tests contain no real RDD2022 data.

The initial 16 tests keep image decoding disabled. Additional tests use
explicitly synthetic images or placeholders to exercise optional decoding.
"""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from src.data.audit import Image, audit, box_error, main


def synthetic_xml(name="D00", coordinates=(1, 1, 100, 80), size=(100, 80)):
    """Build a synthetic PASCAL VOC annotation, unrelated to RDD2022 files."""
    xmin, ymin, xmax, ymax = coordinates
    width, height = size
    return (
        f"<annotation><size><width>{width}</width><height>{height}</height></size>"
        f"<object><name>{name}</name><bndbox>"
        f"<xmin>{xmin}</xmin><ymin>{ymin}</ymin>"
        f"<xmax>{xmax}</xmax><ymax>{ymax}</ymax>"
        "</bndbox></object></annotation>"
    )


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "synthetic_dataset"
        self.input.mkdir()

    def write_xml(self, content):
        path = self.input / "synthetic_annotation.xml"
        path.write_text(content, encoding="utf-8")
        return path

    def test_valid_d00_and_recursive_image_counts(self):
        self.write_xml(synthetic_xml())
        nested = self.input / "arbitrary" / "nested"
        nested.mkdir(parents=True)
        for extension in ("JPG", "jpeg", "PNG", "gif"):
            (nested / f"synthetic_placeholder.{extension}").write_bytes(b"synthetic, not an image")
        report = audit([self.input, nested])
        self.assertEqual(report["image_count"], 3)
        self.assertEqual(report["xml_count"], 1)
        self.assertEqual(report["parsed_xml_count"], 1)
        self.assertEqual(report["instances_per_class"], {"D00": 1, "D10": 0, "D20": 0, "D40": 0})
        self.assertEqual(report["invalid_boxes"], [])
        self.assertFalse(report["image_integrity_checked"])

    def test_all_known_classes(self):
        for name in ("D00", "D10", "D20", "D40"):
            (self.input / f"synthetic_{name}.XML").write_text(synthetic_xml(name), encoding="utf-8")
        self.assertEqual(audit([self.input])["instances_per_class"], dict.fromkeys(("D00", "D10", "D20", "D40"), 1))

    def test_malformed_xml(self):
        self.write_xml("<annotation><object>")
        report = audit([self.input])
        self.assertEqual(report["xml_count"], 1)
        self.assertEqual(report["parsed_xml_count"], 0)
        self.assertEqual(len(report["xml_errors"]), 1)

    def test_unsupported_xml_encoding(self):
        self.write_xml('<?xml version="1.0" encoding="SYNTHETIC_UNKNOWN"?>' + synthetic_xml())
        report = audit([self.input])
        self.assertEqual(len(report["xml_errors"]), 1)
        self.assertEqual(report["parsed_xml_count"], 0)

    def test_multiple_instances_in_one_xml(self):
        content = synthetic_xml()
        self.write_xml(content.replace(
            "</annotation>",
            "<object><name>D00</name><bndbox>"
            "<xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax>"
            "</bndbox></object></annotation>",
        ))
        report = audit([self.input])
        self.assertEqual(report["xml_count"], 1)
        self.assertEqual(report["instances_per_class"]["D00"], 2)

    def test_unreadable_xml(self):
        self.write_xml(synthetic_xml())
        with patch("src.data.audit.ET.parse", side_effect=PermissionError("synthetic access denial")):
            report = audit([self.input])
        self.assertEqual(len(report["xml_errors"]), 1)
        self.assertIn("synthetic access denial", report["xml_errors"][0]["error"])

    def test_non_voc_xml(self):
        self.write_xml("<synthetic/>")
        self.assertEqual(len(audit([self.input])["xml_errors"]), 1)

    def test_unknown_class(self):
        self.write_xml(synthetic_xml("SYNTHETIC_UNKNOWN"))
        report = audit([self.input])
        self.assertEqual(report["unknown_classes"][0]["class"], "SYNTHETIC_UNKNOWN")
        self.assertEqual(sum(report["instances_per_class"].values()), 0)

    def test_invalid_boxes(self):
        cases = (
            ((0, 1, 50, 50), (100, 80)),
            ((50, 1, 10, 50), (100, 80)),
            ((1, 50, 50, 10), (100, 80)),
            ((1, 1, 101, 80), (100, 80)),
            ((1, 1, 100, 81), (100, 80)),
            ((1, 1, "NaN", 80), (100, 80)),
            ((1, 1, 1.5, 80), (100, 80)),
            ((1, 1, 50, 50), (0, 80)),
            ((1, 1, 50, 50), (100, "")),
        )
        for coordinates, size in cases:
            with self.subTest(coordinates=coordinates, size=size):
                self.write_xml(synthetic_xml(coordinates=coordinates, size=size))
                report = audit([self.input])
                self.assertEqual(len(report["invalid_boxes"]), 1)
                self.assertEqual(report["instances_per_class"]["D00"], 1)

    def test_missing_box(self):
        self.write_xml("<annotation><size><width>100</width><height>80</height></size>"
                       "<object><name>D00</name></object></annotation>")
        self.assertEqual(len(audit([self.input])["invalid_boxes"]), 1)

    def test_one_pixel_box_at_boundary(self):
        self.write_xml(synthetic_xml(coordinates=(100, 80, 100, 80)))
        self.assertEqual(audit([self.input])["invalid_boxes"], [])

    def test_directory_without_dataset(self):
        (self.input / "synthetic_note.txt").write_text("No dataset", encoding="utf-8")
        report = audit([self.input])
        self.assertFalse(report["dataset_found"])
        self.assertEqual(report["image_count"], 0)
        self.assertEqual(report["xml_count"], 0)
        self.assertEqual(report["scan_errors"], [])

    def test_missing_directory(self):
        report = audit([self.input / "missing"])
        self.assertFalse(report["dataset_found"])
        self.assertEqual(len(report["scan_errors"]), 1)

    def test_cli_writes_json_and_preserves_sources(self):
        source = self.write_xml(synthetic_xml())
        before = source.read_bytes()
        output = self.root / "report"
        with contextlib.redirect_stdout(io.StringIO()):
            result = main(["--input", str(self.input), "--output", str(output)])
        report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(report["instances_per_class"]["D00"], 1)
        self.assertEqual(source.read_bytes(), before)

    def test_cli_rejects_output_inside_sources(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["--input", str(self.input), "--output", str(self.input / "report")])
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse((self.input / "report").exists())

    def test_cli_refreshes_existing_report(self):
        output = self.root / "report"
        output.mkdir()
        destination = output / "summary.json"
        destination.write_text("synthetic existing content", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            result = main(["--input", str(self.input), "--output", str(output)])
        self.assertEqual(result, 0)
        self.assertFalse(json.loads(destination.read_text(encoding="utf-8"))["dataset_found"])


class RDD2022AuditTests(unittest.TestCase):
    """Additional synthetic fixtures for country-aware, read-only auditing."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "RDD2022"
        self.input.mkdir()

    def image_placeholder(self, country, stem, split="train", extension="jpg"):
        path = self.input / country / split / "images" / f"{stem}.{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SYNTHETIC: not a decodable image")
        return path

    def annotation(self, country, stem, split="train", name="D00",
                   coordinates=(0, 0, 50, 20), size=(100, 80), filename=None):
        path = self.input / country / split / "annotations" / "xmls" / f"{stem}.xml"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = synthetic_xml(name, coordinates, size).replace(
            "<annotation>", f"<annotation><filename>{filename or stem + '.jpg'}</filename>"
        )
        path.write_text(content, encoding="utf-8")
        return path

    def test_rdd_bounds_and_legacy_convention(self):
        root = ET.fromstring(synthetic_xml(coordinates=(0, 0, 100, 80)))
        obj = root.find("object")
        self.assertIsNone(box_error(root, obj, "rdd2022"))
        self.assertIsNotNone(box_error(root, obj, "voc"))
        for box in ((-1, 1, 50, 20), (1, 1, 101, 20), (1, 1, 50, 81),
                    (50, 1, 1, 20), (1, 1, 1, 20), (1, 20, 50, 20)):
            with self.subTest(box=box):
                root = ET.fromstring(synthetic_xml(coordinates=box))
                self.assertIsNotNone(box_error(root, root.find("object"), "rdd2022"))

    def test_country_counts_metadata_statistics_and_empty_xml(self):
        self.image_placeholder("India", "synthetic_a")
        self.annotation("India", "synthetic_a")
        empty = self.annotation("India", "synthetic_empty", size=(200, 100))
        root = ET.parse(empty).getroot()
        root.remove(root.find("object"))
        empty.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
        self.image_placeholder("Czech", "synthetic_c")
        self.annotation("Czech", "synthetic_c", name="D10")
        report = audit([self.input], convention="rdd2022")
        india = report["countries"]["India"]
        self.assertEqual((india["image_count"], india["xml_count"], india["empty_xml_count"]), (1, 2, 1))
        self.assertEqual(india["instances_per_class"]["D00"], 1)
        self.assertEqual(report["countries"]["Czech"]["instances_per_class"]["D10"], 1)
        stats = india["dimension_statistics"]
        self.assertEqual(stats["image_width"], {"count": 2, "min": 100, "max": 200, "mean": 150, "median": 150})
        self.assertEqual(stats["box_width"]["mean"], 50)
        self.assertEqual(stats["box_height"]["mean"], 20)
        self.assertEqual(stats["box_area"]["mean"], 1000)
        self.assertEqual(stats["box_relative_area"]["mean"], 0.125)
        self.assertEqual(report["coordinate_evidence"]["zero_min_boxes"], 2)

    def test_missing_associations_never_cross_countries_or_splits(self):
        self.image_placeholder("India", "synthetic_same")
        self.annotation("Czech", "synthetic_same")
        self.annotation("India", "synthetic_same", split="test")
        report = audit([self.input])
        self.assertEqual(report["matched_pair_count"], 0)
        self.assertEqual(report["missing_image_count"], 2)
        self.assertEqual(report["missing_annotation_count"], 1)
        self.assertEqual(len(report["missing_associations"]), 3)

    def test_case_insensitive_pairing_and_overlapping_inputs(self):
        self.image_placeholder("Japan", "SYNTHETIC_PAIR", extension="JPG")
        self.annotation("Japan", "synthetic_pair")
        report = audit([self.input, self.input / "Japan"], convention="rdd2022")
        self.assertEqual(report["image_count"], 1)
        self.assertEqual(report["xml_count"], 1)
        self.assertEqual(report["matched_pair_count"], 1)
        self.assertEqual(report["filename_errors"], [])

    def test_duplicate_stems_are_ambiguous(self):
        for extension in ("jpg", "png"):
            self.image_placeholder("Japan", "synthetic_duplicate", extension=extension)
        self.annotation("Japan", "synthetic_duplicate")
        report = audit([self.input])
        self.assertEqual(report["matched_pair_count"], 0)
        self.assertEqual(report["ambiguous_association_count"], 1)
        self.assertEqual(len(report["ambiguous_associations"][0]["images"]), 2)

    def test_filename_mismatch(self):
        self.image_placeholder("India", "synthetic_a")
        self.annotation("India", "synthetic_a", filename="synthetic_wrong.jpg")
        report = audit([self.input])
        self.assertEqual(report["matched_pair_count"], 1)
        self.assertEqual(len(report["filename_errors"]), 1)

    def test_invalid_dimensions_without_objects(self):
        path = self.annotation("India", "synthetic_empty")
        path.write_text("<annotation><filename>synthetic_empty.jpg</filename></annotation>", encoding="utf-8")
        report = audit([self.input])
        self.assertEqual(len(report["invalid_dimensions"]), 1)
        self.assertEqual(report["dimension_statistics"]["image_width"]["count"], 0)
        self.assertIsNone(report["dimension_statistics"]["image_width"]["mean"])

    def test_invalid_and_unknown_boxes_are_counted_but_statistics_are_filtered(self):
        self.annotation("Japan", "synthetic_invalid", coordinates=(5, 5, 5, 10))
        self.annotation("Japan", "synthetic_unknown", name="SYNTHETIC_UNKNOWN")
        report = audit([self.input], convention="rdd2022")
        self.assertEqual(report["instances_per_class"]["D00"], 1)
        self.assertEqual(report["valid_boxes_per_class"]["D00"], 0)
        self.assertEqual(report["countries"]["Japan"]["unknown_class_counts"], {"SYNTHETIC_UNKNOWN": 1})
        self.assertEqual(report["dimension_statistics"]["box_width"]["count"], 1)
        self.assertEqual(report["box_statistics_per_class"]["D00"]["box_width"]["count"], 0)

    def test_pillow_unavailable_is_explicit(self):
        image = self.image_placeholder("India", "synthetic_a")
        before = image.read_bytes()
        with patch("src.data.audit.Image", None):
            report = audit([self.input], verify_images=True)
        self.assertFalse(report["image_integrity_checked"])
        self.assertEqual(report["image_verification"]["status"], "not_performed_pillow_unavailable")
        self.assertEqual(report["image_verification"]["attempted"], 0)
        self.assertEqual(image.read_bytes(), before)

    @unittest.skipUnless(Image is not None, "Pillow is optional")
    def test_real_synthetic_decoding_corruption_and_size_mismatch(self):
        image = self.image_placeholder("India", "synthetic_valid", extension="png")
        with Image.new("RGB", (12, 8), color="white") as synthetic_image:
            synthetic_image.save(image)
        self.annotation("India", "synthetic_valid", filename="synthetic_valid.png")
        self.image_placeholder("Japan", "synthetic_corrupt")
        before = image.read_bytes()
        report = audit([self.input], convention="rdd2022", verify_images=True)
        self.assertTrue(report["image_integrity_checked"])
        self.assertEqual(report["image_verification"]["attempted"], 2)
        self.assertEqual(report["image_verification"]["succeeded"], 1)
        self.assertEqual(report["image_verification"]["failed"], 1)
        self.assertEqual(len(report["image_decode_errors"]), 1)
        self.assertEqual(len(report["image_dimension_mismatches"]), 1)
        self.assertEqual(report["image_dimension_mismatches"][0]["decoded_size"], (12, 8))
        self.assertEqual(image.read_bytes(), before)

    def test_cli_csv_exports_and_source_preservation(self):
        image = self.image_placeholder("India", "synthetic_a")
        xml = self.annotation("India", "synthetic_a")
        self.annotation("Japan", "synthetic_orphan", name="SYNTHETIC_UNKNOWN")
        before = {path: path.read_bytes() for path in (image, xml)}
        output = self.root / "reports"
        with contextlib.redirect_stdout(io.StringIO()):
            result = main(["--input", str(self.input), "--output", str(output), "--skip-image-verification"])
        self.assertEqual(result, 0)
        expected = {"summary.json", "country_summary.csv", "class_counts.csv",
                    "dimension_statistics.csv", "anomalies.csv", "missing_associations.csv"}
        self.assertEqual({path.name for path in output.iterdir()}, expected)
        def rows(name):
            with (output / name).open(encoding="utf-8-sig", newline="") as stream:
                return list(csv.DictReader(stream))
        self.assertEqual(len(rows("country_summary.csv")), 2)
        self.assertTrue(any(row["country"] == "India" and row["class"] == "D00" and row["instances"] == "1"
                            for row in rows("class_counts.csv")))
        self.assertEqual(rows("missing_associations.csv")[0]["type"], "missing_image")
        self.assertTrue(any(row["type"] == "unknown_classes" for row in rows("anomalies.csv")))
        self.assertTrue(rows("dimension_statistics.csv"))
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
