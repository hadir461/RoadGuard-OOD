"""Synthetic images/XML only. No real RDD2022 sources are modified by tests."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
import xml.etree.ElementTree as ET

from PIL import Image

from src.data.build_manifest import collect_manifest, write_reports
from src.data.prepare_yolo import (
    assign_id_groups, convert_annotation, prepare, read_csv, similarity_components,
    validate_dataset,
)


def annotation(filename="synthetic.png", labels=("D00",), boxes=None):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "32"
    ET.SubElement(size, "height").text = "24"
    for index, label in enumerate(labels):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        box = ET.SubElement(obj, "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), boxes[index] if boxes else (0, 0, 16, 12)):
            ET.SubElement(box, key).text = str(value)
    return ET.tostring(root, encoding="utf-8")


class YoloPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name) / "synthetic Windows path with spaces"
        self.root = self.base / "raw" / "RDD2022"
        self.inventory = self.base / "inventory"
        self.output = self.base / "processed" / "roadguard_v1"
        self.reports = self.base / "reports"
        self.root.mkdir(parents=True)

    def image(self, country, name, labels=("D00",), boxes=None):
        image = self.root / country / "train" / "images" / f"{name}.png"
        xml = self.root / country / "train" / "annotations" / "xmls" / f"{name}.xml"
        image.parent.mkdir(parents=True, exist_ok=True)
        xml.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random(int(hashlib.sha256((country + name).encode()).hexdigest()[:16], 16))
        with Image.new("RGB", (32, 24)) as pixels:
            pixels.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(32 * 24)])
            pixels.save(image)
        xml.write_bytes(annotation(image.name, labels, boxes))
        return image, xml

    def fixtures(self):
        for country in ("India", "Japan"):
            for index in range(20):
                labels = () if index % 5 == 0 else (("D20", "D44") if index % 5 == 1 else ("D00", "D10", "D20", "D40"))
                self.image(country, f"synthetic_{index:02}", labels)
        for index in range(4):
            self.image("Czech", f"synthetic_ood_{index}", () if index == 0 else ("D00", "D40"))
        self.image("India", "synthetic_other_only", ("D44",))
        self.image("Japan", "Japan_001265", ("D00", "D20"), [(0, 0, 16, 12), (5, 1, 5, 8)])
        self.image("Japan", "synthetic_invalid_only", ("D20",), [(5, 1, 5, 8)])
        # A byte-exact pair must remain indivisible even with no perceptual CSV edge.
        first, _ = self.image("India", "synthetic_exact_a", ("D00",))
        second, _ = self.image("India", "synthetic_exact_b", ("D10",))
        second.write_bytes(first.read_bytes())
        rows, summary = collect_manifest(self.root)
        write_reports(rows, summary, self.inventory)
        return rows

    def run_prepare(self, output=None, reports=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare(self.root, self.inventory, output or self.output, reports or self.reports, seed=42)

    def test_conversion_exact_formula_all_class_ids_no_plus_one(self):
        text, counts, excluded = convert_annotation(annotation(labels=("D00", "D10", "D20", "D40")), 32, 24)
        self.assertEqual(counts, dict.fromkeys(("D00", "D10", "D20", "D40"), 1))
        self.assertEqual(excluded, [])
        for class_id, line in enumerate(text.splitlines()):
            tokens = line.split()
            self.assertEqual(int(tokens[0]), class_id)
            self.assertEqual(list(map(float, tokens[1:])), [0.25, 0.25, 0.5, 0.5])
        full, _, _ = convert_annotation(annotation(boxes=[(0, 0, 32, 24)]), 32, 24)
        self.assertEqual(list(map(float, full.split()[1:])), [0.5, 0.5, 1.0, 1.0])

    def test_invalid_and_unknown_objects_only_are_excluded(self):
        text, counts, excluded = convert_annotation(annotation(labels=("D00", "D20", "D44"), boxes=[(0, 0, 16, 12), (5, 1, 5, 8), (0, 0, 10, 10)]), 32, 24)
        self.assertEqual(len(text.splitlines()), 1)
        self.assertEqual(counts["D00"], 1)
        self.assertEqual(counts["D20"], 0)
        self.assertEqual([r["reason"] for r in excluded], ["invalid_target_box", "non_target_class"])

    def test_dimension_mismatch_and_invalid_root_fail(self):
        for content, width in ((annotation(), 31), (b"<synthetic/>", 32)):
            with self.assertRaises(ValueError):
                convert_annotation(content, width, 24)

    def test_components_are_transitive_and_keep_excluded_bridges(self):
        rows = [{"image_path": name, "country": "India", "sha256": name, "dhash64": f"{value:016x}",
                 "category": "non_target_only" if name == "bridge" else "target_only"}
                for name, value in (("a", 0), ("bridge", 1), ("c", 3))]
        pairs = [{"image_a": "a", "image_b": "bridge", "hamming_distance": 1, "country_a": "India", "country_b": "India"},
                 {"image_a": "bridge", "image_b": "c", "hamming_distance": 1, "country_a": "India", "country_b": "India"}]
        groups, membership = similarity_components(rows, pairs)
        self.assertEqual(len(groups), 1)
        self.assertEqual(membership["a"], membership["c"])

    def test_id_ood_component_blocks_preparation(self):
        rows = [{"image_path": name, "country": country, "sha256": "same-bytes", "dhash64": "0" * 16}
                for name, country in (("a", "India"), ("b", "Czech"))]
        with self.assertRaisesRegex(ValueError, "ID/OOD"):
            similarity_components(rows, [])

    def test_seed_and_input_order_are_reproducible_ood_not_used(self):
        rows = [{"image_path": f"synthetic/{i:02}", "country": "India" if i % 2 else "Japan", "group_id": f"g{i}",
                 "category": "target_only", "counts": {"D00": 1, "D10": i % 3, "D20": 0, "D40": i % 2}}
                for i in range(40)]
        a, deviations = assign_id_groups(rows, 42)
        b, _ = assign_id_groups(list(reversed(rows)), 42)
        ood = {"image_path": "Czech/extra", "country": "Czech", "group_id": "ood", "category": "empty_xml", "counts": dict.fromkeys(("D00", "D10", "D20", "D40"), 500)}
        c, _ = assign_id_groups([ood, *rows], 42)
        self.assertEqual(a, b)
        self.assertEqual(a, {key: value for key, value in c.items() if key != "ood"})
        self.assertEqual(c["ood"], "test_ood")
        self.assertTrue(all(abs(row["delta_percentage_points"]) <= 5 for row in deviations))

    def test_whole_preparation_integrity_categories_groups_and_ood(self):
        rows = self.fixtures()
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        inventory_before = {path: path.read_bytes() for path in self.inventory.iterdir() if path.is_file()}
        summary = self.run_prepare()
        self.assertTrue(summary["validation"]["ok"])
        records = read_csv(self.reports / "split_manifest.csv")
        self.assertEqual(len(records), len(rows) - 1)
        grouped = {}
        for row in records:
            self.assertEqual(row["country"] == "Czech", row["split"] == "test_ood")
            grouped.setdefault(row["group_id"], set()).add(row["split"])
            self.assertFalse(os.path.samefile(self.root / row["source_image"], self.output / row["image_path"]))
            if row["category"] == "empty_xml":
                self.assertEqual((self.output / row["label_path"]).read_text(), "")
            if "Japan_001265" in row["source_image"]:
                self.assertEqual(len((self.output / row["label_path"]).read_text().splitlines()), 1)
        self.assertTrue(all(len(splits) == 1 for splits in grouped.values()))
        self.assertEqual(summary["exclusions"]["images"], 1)
        self.assertEqual(summary["exclusions"]["invalid_target_boxes"], 2)
        self.assertEqual(sum(s["empty_labels_from_invalid_targets"] for s in summary["splits"].values()), 1)
        for path, content in {**before, **inventory_before}.items():
            self.assertEqual(path.read_bytes(), content)
        config = json.loads((self.output / "data.yaml").read_text())
        self.assertNotIn("path", config)
        self.assertEqual(config["test"], "images/test_id")
        self.assertEqual(json.loads((self.output / "data_ood.yaml").read_text())["test"], "images/test_ood")

    def test_complete_reproduction_same_seed(self):
        self.fixtures()
        self.run_prepare()
        second = self.base / "processed other path"
        second_reports = self.base / "reports other path"
        self.run_prepare(second, second_reports)
        for first, other in ((self.output, second), (self.reports, second_reports)):
            for path in first.rglob("*"):
                if path.is_file():
                    self.assertEqual(path.read_bytes(), (other / path.relative_to(first)).read_bytes(), str(path))

    def test_validator_rejects_wrong_ids_out_of_bounds_and_nonfinite_labels(self):
        self.fixtures()
        self.run_prepare()
        record = next(r for r in read_csv(self.reports / "split_manifest.csv") if r["D00"] != "0")
        path = self.output / record["label_path"]
        for bad in ("9 0.5 0.5 0.2 0.2\n", "0 1.1 0.5 0.2 0.2\n", "0 nan 0.5 0.2 0.2\n"):
            path.write_text(bad)
            with self.assertRaises(ValueError):
                validate_dataset(self.root, self.output, self.reports, self.inventory)

    def test_validator_rejects_cross_split_assignment(self):
        self.fixtures()
        self.run_prepare()
        path = self.reports / "split_manifest.csv"
        text = path.read_text(encoding="utf-8-sig")
        path.write_text(text.replace("test_ood,", "train_id,", 1), encoding="utf-8-sig")
        with self.assertRaises(ValueError):
            validate_dataset(self.root, self.output, self.reports, self.inventory)

    def test_raw_change_is_detected_before_writing_outputs(self):
        self.fixtures()
        path = next(self.root.rglob("*.png"))
        path.write_bytes(b"SYNTHETIC unexpected change")
        with self.assertRaisesRegex(ValueError, "Source image changed"):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_existing_destination_and_raw_outputs_are_refused(self):
        self.fixtures()
        self.output.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_prepare()
        with self.assertRaisesRegex(ValueError, "outside raw"):
            self.run_prepare(output=self.root / "bad_output")

    def test_exif_orientation_on_empty_negative_is_preserved_and_reported(self):
        path, _ = self.image("India", "synthetic_exif_negative", ())
        exif = Image.Exif()
        exif[274] = 6
        with Image.open(path) as image:
            image.save(path, exif=exif)
        before = path.read_bytes()
        rows, summary = collect_manifest(self.root)
        write_reports(rows, summary, self.inventory)
        result = self.run_prepare()
        self.assertEqual(len(result["negative_exif_orientations"]), 1)
        self.assertEqual(result["negative_exif_orientations"][0]["orientation"], 6)
        self.assertEqual(path.read_bytes(), before)
        row = read_csv(self.reports / "split_manifest.csv")[0]
        self.assertEqual((self.output / row["image_path"]).read_bytes(), before)
        self.assertEqual((self.output / row["label_path"]).read_bytes(), b"")

    def test_exif_orientation_on_annotated_image_requires_explicit_policy(self):
        path, _ = self.image("India", "synthetic_exif_annotated", ("D00",))
        exif = Image.Exif()
        exif[274] = 6
        with Image.open(path) as image:
            image.save(path, exif=exif)
        rows, summary = collect_manifest(self.root)
        write_reports(rows, summary, self.inventory)
        with self.assertRaisesRegex(ValueError, "orientation"):
            self.run_prepare()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
