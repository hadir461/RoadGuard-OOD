"""Synthetic metadata, images and XML only; no real RoadGuard dataset is used."""

import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
import zipfile

from PIL import Image

from src.data.build_manifest import collect_manifest, write_reports
from src.data.prepare_yolo import prepare, read_csv, sha
from src.data.build_b0_subset import (
    CONFIG, build, coverage, label_counts, load_sources, select_groups, validate_package, verify_files, verify_zip,
)


def synthetic_row(index, country="India", group=None, category="target_only"):
    return {"split": "train_id", "image_path": f"images/train/synthetic_{index}.png",
            "group_id": group or f"group_{index}", "country": country, "category": category,
            "width": "32", "height": "24", "D00": "1", "D10": "0", "D20": "0", "D40": "0"}


class B0SubsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / "SYNTHETIC Windows path with spaces"
        self.raw = self.base / "raw" / "RDD2022"
        self.inventory = self.base / "inventory"
        self.source = self.base / "roadguard_v1"
        self.output = self.base / "roadguard_b0_5k"
        self.reports = self.base / "reports_b0"
        self.archive = self.base / "package.zip"

    def fixtures(self):
        for country, size in (("India", 35), ("Japan", 35), ("Czech", 5)):
            for index in range(size):
                name = f"SYNTHETIC_{country}_{index:03}"
                image = self.raw / country / "train" / "images" / (name + ".png")
                xml = self.raw / country / "train" / "annotations" / "xmls" / (name + ".xml")
                image.parent.mkdir(parents=True, exist_ok=True)
                xml.parent.mkdir(parents=True, exist_ok=True)
                rng = random.Random(country + str(index))
                with Image.new("RGB", (32, 24)) as pixels:
                    pixels.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(768)])
                    pixels.save(image)
                names = () if index % 5 == 0 else (("D10", "D44") if index % 5 == 1 else ("D00", "D20", "D40"))
                objects = "".join(f"<object><name>{label}</name><bndbox><xmin>0</xmin><ymin>0</ymin>"
                                  "<xmax>16</xmax><ymax>12</ymax></bndbox></object>" for label in names)
                xml.write_text(f"<annotation><filename>{image.name}</filename><size><width>32</width>"
                               f"<height>24</height></size>{objects}</annotation>", encoding="utf-8")
        # One explicit byte-identical group, with separate XML and names.
        images = sorted((self.raw / "India").rglob("*.png"))
        images[2].write_bytes(images[1].read_bytes())
        rows, summary = collect_manifest(self.raw)
        write_reports(rows, summary, self.inventory)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.raw, self.inventory, self.source, self.inventory)

    def run_build(self, output=None, reports=None, archive=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return build(self.source, self.inventory, output or self.output, reports or self.reports,
                         archive or self.archive, target=24, seed=42)

    def test_seed_and_input_order_reproduce_selection(self):
        rows = [synthetic_row(i, "India" if i % 2 else "Japan", group=f"g_{i // 3}") for i in range(60)]
        self.assertEqual(select_groups(rows, 24, 42), select_groups(list(reversed(rows)), 24, 42))

    def test_exact_size_with_indivisible_groups(self):
        rows = [synthetic_row(i, group=f"g_{i // 3}") for i in range(60)]
        groups, size = select_groups(rows, 24)
        selected = [r for r in rows if r["group_id"] in groups]
        self.assertEqual(size, 24)
        self.assertEqual(len(selected), 24)
        self.assertTrue(all(sum(r["group_id"] == group for r in selected) == 3 for group in groups))

    def test_unattainable_target_uses_nearest_size_and_smaller_tie(self):
        rows = [synthetic_row(i, group=f"g_{i // 3}") for i in range(6)]
        self.assertEqual(select_groups(rows, 5)[1], 6)
        pairs = [synthetic_row(i, group=f"g_{i // 2}") for i in range(8)]
        self.assertEqual(select_groups(pairs, 5)[1], 4)
        self.assertEqual(select_groups(rows, 6)[1], 6)

    def test_only_train_id_india_japan_and_valid_sizes(self):
        for row in (dict(synthetic_row(0), split="val_id"), dict(synthetic_row(0), country="Czech")):
            with self.assertRaises(ValueError):
                select_groups([row], 1)
        for target in (0, -1, 11):
            with self.assertRaises(ValueError):
                select_groups([synthetic_row(i) for i in range(10)], target)

    def test_rare_india_d10_categories_and_country_are_covered(self):
        rows = [synthetic_row(i, "India" if i < 100 else "Japan",
                              category=("empty_xml", "mixed", "target_only")[i % 3]) for i in range(200)]
        for i, row in enumerate(rows):
            row["D00"] = "0" if row["category"] == "empty_xml" else "1"
            row["D10"] = "1" if i in (2, 5, 8, 11, 14, 17) else "0"
        groups, _ = select_groups(rows, 100)
        chosen = [r for r in rows if r["group_id"] in groups]
        self.assertLessEqual(abs(sum(r["country"] == "India" for r in chosen) - 50), 1)
        self.assertLessEqual(abs(sum(int(r["D10"]) for r in chosen) - 3), 1)
        for category in ("empty_xml", "mixed", "target_only"):
            self.assertLessEqual(abs(sum(r["category"] == category for r in chosen) -
                                     sum(r["category"] == category for r in rows) / 2), 1)
        self.assertTrue(coverage(rows, chosen))

    def test_label_bounds_classes_empty_and_nonfinite(self):
        self.assertEqual(sum(label_counts("").values()), 0)
        self.assertEqual(label_counts("0 0.5 0.5 1 1\n")["D00"], 1)
        for bad in ("4 .5 .5 .2 .2", "0 nan .5 .2 .2", "0 .95 .5 .2 .2",
                    "0 .5 .5 0 .2", "0 .5 .5 1.01 .2", "0 .5 .5 .2"):
            with self.assertRaises(ValueError):
                label_counts(bad)

    def test_full_package_keeps_val_sources_groups_and_no_tests(self):
        self.fixtures()
        before = {p: sha(p) for folder in (self.raw, self.source, self.inventory)
                  for p in folder.rglob("*") if p.is_file()}
        result = self.run_build()
        rows = read_csv(self.output / "manifest.csv")
        train, val = load_sources(self.inventory)
        self.assertEqual(len([r for r in rows if r["split"] == "train_id"]), 24)
        self.assertEqual({r["image_path"] for r in rows if r["split"] == "val_id"},
                         {r["image_path"] for r in val})
        groups = {r["group_id"] for r in rows if r["split"] == "train_id"}
        self.assertEqual({r["image_path"] for r in train if r["group_id"] in groups},
                         {r["image_path"] for r in rows if r["split"] == "train_id"})
        self.assertTrue(all(r["country"] != "Czech" and r["split"] in {"train_id", "val_id"} for r in rows))
        self.assertEqual(json.loads((self.output / "data.yaml").read_text()), CONFIG)
        self.assertEqual(before, {p: sha(p) for folder in (self.raw, self.source, self.inventory)
                                  for p in folder.rglob("*") if p.is_file()})
        self.assertTrue(result["summary"]["verification"]["ok"])
        with zipfile.ZipFile(self.archive) as package:
            self.assertTrue(all(name.startswith("roadguard_b0_5k/") and "\\" not in name and ":" not in name
                                and "/test_id/" not in name and "/test_ood/" not in name for name in package.namelist()))
            self.assertEqual(package.read("roadguard_b0_5k/data.yaml"), (self.output / "data.yaml").read_bytes())
        verify_zip(self.output, self.archive)

    def test_same_seed_produces_identical_zip_and_manifests(self):
        self.fixtures()
        self.run_build()
        second, reports, archive = self.base / "second dataset", self.base / "second reports", self.base / "second.zip"
        self.run_build(second, reports, archive)
        self.assertEqual(sha(self.archive), sha(archive))
        for first in self.output.rglob("*"):
            if first.is_file():
                self.assertEqual(sha(first), sha(second / first.relative_to(self.output)))

    def test_source_corruption_stops_before_output(self):
        self.fixtures()
        train, _ = load_sources(self.inventory)
        (self.source / train[0]["label_path"]).write_text("SYNTHETIC corrupted label")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.run_build()
        self.assertFalse(self.output.exists())

    def test_copy_hash_and_independence_checks(self):
        self.fixtures()
        self.run_build()
        rows = read_csv(self.output / "manifest.csv")
        with self.assertRaisesRegex(ValueError, "independent"):
            verify_files(self.source, rows, source=self.source)
        (self.output / rows[0]["image_path"]).write_bytes(b"SYNTHETIC corruption")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            verify_files(self.output, rows)

    def test_original_component_tampering_is_rejected(self):
        self.fixtures()
        path = self.inventory / "split_manifest.csv"
        text = path.read_text(encoding="utf-8-sig")
        records = read_csv(path)
        path.write_text(text.replace(records[0]["group_id"], "SYNTHETIC_invalid_group", 1), encoding="utf-8-sig")
        with self.assertRaisesRegex(ValueError, "Component"):
            load_sources(self.inventory)

    def test_existing_raw_and_source_destinations_are_refused(self):
        self.fixtures()
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_build()
        for dest in (self.source / "bad", self.raw / "bad", self.inventory / "bad"):
            with self.assertRaisesRegex(ValueError, "outside"):
                self.run_build(output=dest)

    def test_zip_with_unlisted_test_entry_is_rejected(self):
        self.fixtures()
        self.run_build()
        with zipfile.ZipFile(self.archive, "a") as package:
            package.writestr("roadguard_b0_5k/images/test_ood/SYNTHETIC.txt", "Not real data")
        with self.assertRaisesRegex(ValueError, "ZIP entries"):
            verify_zip(self.output, self.archive)

    def test_read_only_validation_replays_selection(self):
        self.fixtures()
        self.run_build()
        before = {p: sha(p) for p in self.output.rglob("*") if p.is_file()}
        result = validate_package(self.source, self.inventory, self.output, self.reports, self.archive)
        self.assertTrue(result["ok"])
        self.assertEqual(before, {p: sha(p) for p in self.output.rglob("*") if p.is_file()})

    def test_validation_rejects_incomplete_validation_set(self):
        self.fixtures()
        self.run_build()
        path = self.reports / "manifest.csv"
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8-sig")
        with self.assertRaisesRegex(ValueError, "Manifest differs"):
            validate_package(self.source, self.inventory, self.output, self.reports, self.archive)

    def test_documented_kaggle_cell_builds_runtime_absolute_yaml(self):
        self.fixtures()
        self.run_build()
        runtime_yaml = self.base / "SYNTHETIC_kaggle_working.yaml"
        cell = (self.output / "KAGGLE.md").read_text(encoding="utf-8").split("```python\n", 1)[1].split("```", 1)[0]
        cell = cell.replace('Path("/kaggle/input")', f"Path({str(self.base)!r})")
        cell = cell.replace('Path("/kaggle/working/roadguard_b0_5k.yaml")', f"Path({str(runtime_yaml)!r})")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cell, "SYNTHETIC_KAGGLE_CELL", "exec"), {})
        config = json.loads(runtime_yaml.read_text())
        self.assertEqual(config, {**CONFIG, "path": self.output.resolve().as_posix()})
        self.assertNotIn("path", json.loads((self.output / "data.yaml").read_text()))


if __name__ == "__main__":
    unittest.main()
