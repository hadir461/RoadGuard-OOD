"""Synthetic fixtures only: these tests contain no real RDD2022 data.

Image placeholders deliberately contain no decoded image data. The audit
counts extensions only and must never claim to validate image integrity.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.data.audit import audit, main


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


if __name__ == "__main__":
    unittest.main()
