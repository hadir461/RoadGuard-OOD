"""Synthetic data only: no real RDD2022 files are used or changed here."""

import contextlib
import csv
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from PIL import Image, PngImagePlugin

from src.data.build_manifest import collect_manifest, dhash, main, near_pairs, write_reports


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "raw" / "RDD2022"
        self.dataset.mkdir(parents=True)

    def fixture(self, country, name, labels=(), boxes=None, metadata=None):
        """Create explicitly synthetic PNG and VOC XML with a fixed gradient."""
        folder = self.dataset / country / "train"
        image_path = folder / "images" / f"{name}.png"
        xml_path = folder / "annotations" / "xmls" / f"{name}.xml"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.new("RGB", (24, 16)) as image:
            image.putdata([((x * 11) % 256, (y * 17) % 256, (x * y * 3) % 256)
                           for y in range(16) for x in range(24)])
            image.save(image_path)
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = image_path.name
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = "24"
        ET.SubElement(size, "height").text = "16"
        for key, value in (metadata or {}).items():
            ET.SubElement(root, key).text = value
        for index, name in enumerate(labels):
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = name
            box = ET.SubElement(obj, "bndbox")
            coordinates = boxes[index] if boxes else (0, 0, 12, 8)
            for key, value in zip(("xmin", "ymin", "xmax", "ymax"), coordinates):
                ET.SubElement(box, key).text = str(value)
        xml_path.write_bytes(ET.tostring(root, encoding="utf-8"))
        return image_path, xml_path

    def test_four_categories_counts_and_portable_paths(self):
        self.fixture("India", "synthetic_target", ("D00", "D10"))
        self.fixture("Japan", "synthetic_mixed", ("D20", "D44"))
        self.fixture("India", "synthetic_other", ("D50", "D50"))
        self.fixture("Czech", "synthetic_empty")
        rows, summary = collect_manifest(self.dataset)
        self.assertEqual({row["category"] for row in rows}, {"target_only", "mixed", "non_target_only", "empty_xml"})
        by_name = {Path(row["image_path"]).stem: row for row in rows}
        self.assertEqual(by_name["synthetic_target"]["target_count"], 2)
        self.assertEqual(by_name["synthetic_other"]["non_target_count"], 2)
        self.assertEqual(json.loads(by_name["synthetic_mixed"]["classes_present"]), ["D20", "D44"])
        self.assertEqual(by_name["synthetic_empty"]["domain_role"], "OOD_test_only")
        self.assertEqual(summary["countries"]["India"]["image_count"], 2)
        for row in rows:
            for key in ("image_path", "xml_path"):
                self.assertFalse(Path(row[key]).is_absolute())
                self.assertNotIn("\\", row[key])
            self.assertEqual((row["width"], row["height"]), (24, 16))

    def test_invalid_or_missing_xml_is_not_empty(self):
        image_a, xml_a = self.fixture("India", "synthetic_invalid")
        xml_a.write_text("<annotation>", encoding="utf-8")
        self.fixture("Japan", "synthetic_orphan")
        # A different image filename has no XML; no source fixture is deleted.
        source = self.dataset / "Japan/train/images/synthetic_orphan.png"
        source.rename(source.with_name("synthetic_missing.png"))
        rows, summary = collect_manifest(self.dataset)
        self.assertTrue(all(row["category"] == "unusable_annotation" for row in rows))
        self.assertEqual(summary["issue_counts"]["xml_invalid"], 1)
        self.assertEqual(summary["issue_counts"]["xml_missing"], 1)
        self.assertEqual(len(summary["unpaired_xml_paths"]), 1)

    def test_invalid_box_counts_preserve_declared_target_class(self):
        self.fixture("Japan", "synthetic_box", ("D20",), boxes=[(5, 1, 5, 8)])
        rows, _ = collect_manifest(self.dataset)
        self.assertEqual(rows[0]["category"], "target_only")
        self.assertEqual(rows[0]["D20"], 1)
        self.assertEqual(rows[0]["valid_target_count"], 0)
        self.assertEqual(rows[0]["invalid_target_box_count"], 1)

    def test_ambiguous_image_stems_are_not_paired(self):
        image, _ = self.fixture("India", "synthetic_ambiguous", ("D00",))
        with Image.open(image) as source:
            source.save(image.with_suffix(".jpg"))
        rows, summary = collect_manifest(self.dataset)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["xml_status"] == "ambiguous" for row in rows))
        self.assertEqual(summary["issue_counts"]["xml_ambiguous"], 2)

    def test_exact_groups_separate_members_copies_pairs_and_id_ood(self):
        for country in ("India", "Japan", "Czech"):
            self.fixture(country, "synthetic_copy", ("D00",))
        rows, summary = collect_manifest(self.dataset)
        output = self.root / "report"
        write_reports(rows, summary, output)
        exact = summary["exact_duplicates"]
        self.assertEqual((exact["group_count"], exact["member_images"], exact["redundant_images"], exact["pair_count"]), (1, 3, 2, 3))
        self.assertEqual(exact["cross_id_ood_groups"], 1)
        self.assertEqual(exact["cross_id_ood_pairs"], 2)
        self.assertEqual(summary["near_duplicates"]["candidate_pairs"], 0)
        with (output / "exact_duplicates.csv").open(encoding="utf-8-sig", newline="") as stream:
            members = list(csv.DictReader(stream))
        self.assertEqual(len(members), 3)
        self.assertTrue(all(row["cross_id_ood"] == "True" for row in members))

    def test_dhash_bit_order_and_gradient_definition(self):
        with Image.new("L", (9, 8)) as image:
            image.putdata([x * 28 for _ in range(8) for x in range(9)])
            self.assertEqual(dhash(image), 0)
            image.putdata([(8 - x) * 28 for _ in range(8) for x in range(9)])
            self.assertEqual(dhash(image), (1 << 64) - 1)

    def test_same_pixels_different_bytes_are_perceptual_candidates(self):
        image_a, _ = self.fixture("India", "synthetic_pixels", ("D00",))
        image_b, _ = self.fixture("Czech", "synthetic_pixels", ("D10",))
        text = PngImagePlugin.PngInfo()
        text.add_text("Fixture", "SYNTHETIC re-encoded copy")
        with Image.open(image_a) as image:
            image.save(image_b, pnginfo=text, compress_level=0)
        rows, summary = collect_manifest(self.dataset)
        write_reports(rows, summary, self.root / "report")
        self.assertEqual(summary["exact_duplicates"]["group_count"], 0)
        self.assertEqual(summary["near_duplicates"]["candidate_pairs"], 1)
        self.assertEqual(summary["near_duplicates"]["cross_id_ood_pairs"], 1)
        pair = list(near_pairs(rows))[0]
        self.assertEqual(pair["hamming_distance"], 0)

    def test_index_matches_small_exhaustive_oracle_including_threshold(self):
        hashes = [0, 15, 31, (1 << 63) | (1 << 40) | (1 << 20) | 1]
        hashes += [int.from_bytes(hashlib.sha256(f"synthetic-{i}".encode()).digest()[:8], "big") for i in range(64)]
        hashes += [hashes[5], hashes[10] ^ ((1 << 60) | (1 << 45) | (1 << 30) | (1 << 15))]
        rows = [{"image_path": f"India/train/images/synthetic_{i:03}.png", "country": "India", "xml_path": "",
                 "dhash64": f"{value:016x}", "sha256": f"synthetic-distinct-bytes-{i}"} for i, value in enumerate(hashes)]
        expected = {(a["image_path"], b["image_path"]) for a, b in combinations(rows, 2)
                    if (int(a["dhash64"], 16) ^ int(b["dhash64"], 16)).bit_count() <= 4}
        actual_list = list(near_pairs(rows, 4))
        actual = {(pair["image_a"], pair["image_b"]) for pair in actual_list}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(actual_list))
        self.assertIn((rows[0]["image_path"], rows[1]["image_path"]), actual)
        self.assertNotIn((rows[0]["image_path"], rows[2]["image_path"]), actual)
        stats = {}
        list(near_pairs(rows, 4, stats))
        self.assertLess(stats["full_hamming_comparisons"], stats["exhaustive_unique_hash_pairs"] // 4)
        for bad in (-1, 9):
            with self.assertRaises(ValueError):
                list(near_pairs(rows, bad))

    def test_numeric_filenames_and_context_fields_do_not_invent_routes(self):
        self.fixture("India", "India_000001", metadata={"timestamp": "synthetic-time"})
        self.fixture("India", "India_000002", metadata={"sequence_id": "synthetic-explicit-id"})
        rows, summary = collect_manifest(self.dataset)
        metadata = summary["sequence_metadata"]
        self.assertFalse(metadata["validated_route_or_sequence_groups"])
        self.assertEqual(metadata["xml_context_fields_image_counts"], {"sequence_id": 1, "timestamp": 1})
        values = [json.loads(row["xml_sequence_metadata"]) for row in rows]
        self.assertIn({"sequence_id": ["synthetic-explicit-id"]}, values)

    def test_exif_time_is_reported_without_becoming_a_sequence_group(self):
        image, _ = self.fixture("Japan", "synthetic_exif")
        exif = Image.Exif()
        exif[306] = "2000:01:01 00:00:00"
        with Image.open(image) as source:
            source.save(image, exif=exif)
        _, summary = collect_manifest(self.dataset)
        self.assertEqual(summary["sequence_metadata"]["exif_context_fields_image_counts"], {"DateTime": 1})
        self.assertFalse(summary["sequence_metadata"]["validated_route_or_sequence_groups"])

    def test_decode_failure_retains_row_and_raw_sha(self):
        image, _ = self.fixture("India", "synthetic_corrupt", ("D00",))
        image.write_bytes(b"SYNTHETIC corrupt image fixture")
        rows, summary = collect_manifest(self.dataset)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sha256"], hashlib.sha256(image.read_bytes()).hexdigest())
        self.assertEqual(rows[0]["dhash64"], "")
        self.assertEqual(summary["issue_counts"]["image_read_or_decode_error"], 1)
        self.assertNotRegex(rows[0]["issues"], r"at 0x[0-9a-fA-F]+")
        again, _ = collect_manifest(self.dataset)
        self.assertEqual(rows, again)

    def test_cli_portable_reproducible_reports_and_source_preservation(self):
        self.fixture("India", "synthetic_data", ("D00", "D44"))
        self.fixture("Czech", "synthetic_empty")
        before = {path.relative_to(self.dataset): path.read_bytes() for path in self.dataset.rglob("*") if path.is_file()}
        copied = self.root / "relocated" / "RDD2022"
        shutil.copytree(self.dataset, copied)
        outputs = [self.root / "reports_a", self.root / "reports_b"]
        for root, output in zip((self.dataset, copied), outputs):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--input", str(root), "--output", str(output)]), 0)
        expected = {"manifest.csv", "exact_duplicates.csv", "near_duplicate_candidates.csv", "summary.json", "data_policy.md"}
        self.assertEqual({path.name for path in outputs[0].iterdir()}, expected)
        for name in expected:
            self.assertEqual((outputs[0] / name).read_bytes(), (outputs[1] / name).read_bytes(), name)
        for path, content in before.items():
            self.assertEqual((self.dataset / path).read_bytes(), content)
        summary = json.loads((outputs[0] / "summary.json").read_text(encoding="utf-8"))
        for name, digest in summary["output_sha256"].items():
            self.assertEqual(hashlib.sha256((outputs[0] / name).read_bytes()).hexdigest(), digest)

    def test_output_cannot_write_inside_any_raw_directory(self):
        for output in (self.dataset / "reports", self.dataset.parent / "reports"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--input", str(self.dataset), "--output", str(output)])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(output.exists())

    def test_no_dataset_produces_empty_headers_and_nonzero_cli_exit(self):
        output = self.root / "reports"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--input", str(self.dataset), "--output", str(output)]), 1)
        with (output / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertIn("category", reader.fieldnames)
            self.assertEqual(list(reader), [])


if __name__ == "__main__":
    unittest.main()
