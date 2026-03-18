from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from soundspaces_mp3d_demo import (
    PlacedSource,
    ScenePoint,
    ScenePlan,
    TopDownBounds,
    build_source_map_svg,
    clip_name_for_candidate,
    discover_scene_sources,
    make_synthetic_sources,
    object_aabb_center,
    object_label,
    save_plan_artifacts,
    save_source_map_artifacts,
    scene_assets,
    source_label_counts,
    source_priority,
)


@dataclass
class _FakeCategory:
    value: str

    def name(self) -> str:
        return self.value


class _FakeAabbProperty:
    def __init__(self, center):
        self.center = center


class _FakeAabbMethod:
    def center(self):
        return [1.0, 2.0, 3.0]


class _FakeObject:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeSemanticScene:
    def __init__(self, objects):
        self.objects = objects


class SoundSpacesDemoTests(unittest.TestCase):
    def test_scene_assets(self):
        assets = scene_assets(Path("data/mp3d"), "scene123")
        self.assertEqual(assets["glb"], Path("data/mp3d/scene123.glb"))
        self.assertEqual(assets["semantic"], Path("data/mp3d/scene123_semantic.ply"))

    def test_source_priority_prefers_known_appliances(self):
        self.assertEqual(source_priority("Kitchen Sink"), 0)
        self.assertLess(source_priority("washing machine"), source_priority("chair"))
        self.assertGreater(source_priority("wall"), 1000)

    def test_clip_name_for_candidate_uses_generic_pool_for_unknown_objects(self):
        label = clip_name_for_candidate("bookshelf", 1000)
        self.assertIn(label, {"lamp_buzz", "chair_creak", "table_clatter"})

    def test_object_label_handles_category_name_method(self):
        obj = _FakeObject(category=_FakeCategory("sink"))
        self.assertEqual(object_label(obj), "sink")

    def test_object_label_handles_string_category(self):
        obj = _FakeObject(category_name="chair")
        self.assertEqual(object_label(obj), "chair")

    def test_object_aabb_center_handles_property_and_method(self):
        self.assertEqual(object_aabb_center(_FakeObject(aabb=_FakeAabbProperty([1, 2, 3]))), [1.0, 2.0, 3.0])
        self.assertEqual(object_aabb_center(_FakeObject(aabb=_FakeAabbMethod())), [1.0, 2.0, 3.0])

    def test_discover_scene_sources_prefers_real_fixtures(self):
        sim = _FakeObject(
            semantic_scene=_FakeSemanticScene(
                [
                    _FakeObject(category_name="wall", aabb=_FakeAabbProperty([0, 0, 0])),
                    _FakeObject(category_name="sink", aabb=_FakeAabbProperty([1, 0, 1])),
                    _FakeObject(category_name="washing machine", aabb=_FakeAabbProperty([2, 0, 1])),
                    _FakeObject(category_name="freezer", aabb=_FakeAabbProperty([3, 0, 1])),
                    _FakeObject(category_name="aircon", aabb=_FakeAabbProperty([4, 0, 1])),
                    _FakeObject(category_name="cooktop", aabb=_FakeAabbProperty([5, 0, 1])),
                ]
            )
        )

        sources = discover_scene_sources(sim, 5)
        self.assertEqual(
            [source.label for source in sources],
            ["tap_water", "washing_machine", "fridge_hum", "fan_noise", "kettle_hiss"],
        )
        self.assertEqual(
            [source.object_name for source in sources],
            ["sink", "washing machine", "freezer", "aircon", "cooktop"],
        )

    def test_source_label_counts(self):
        sources = [
            PlacedSource("tap_water", "sink", [0.0, 0.0, 0.0], "tap.wav"),
            PlacedSource("fan_noise", "fan", [1.0, 0.0, 0.0], "fan.wav"),
            PlacedSource("tap_water", "sink_2", [2.0, 0.0, 0.0], "tap.wav"),
        ]
        self.assertEqual(source_label_counts(sources)["tap_water"], 2)
        self.assertEqual(source_label_counts(sources)["fan_noise"], 1)

    def test_save_plan_artifacts_writes_json_and_csv(self):
        sources = [
            PlacedSource("tap_water", "sink", [1.1, 2.2, 3.3], "tap.wav"),
            PlacedSource("fan_noise", "fan", [4.4, 5.5, 6.6], "fan.wav"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            plan_path, csv_path = save_plan_artifacts(output_dir, "scene", sources)

            self.assertTrue(plan_path.exists())
            self.assertTrue(csv_path.exists())

            with plan_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(len(payload), 2)
            self.assertEqual(payload[0]["label"], "tap_water")

            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["object_name"], "fan")
            self.assertEqual(rows[0]["x"], "1.100000")

    def test_build_source_map_svg_includes_sources_and_objects(self):
        plan = ScenePlan(
            sources=[
                PlacedSource("tap_water", "sink", [1.0, 1.0, 2.0], "tap.wav"),
                PlacedSource("fan_noise", "fan", [4.0, 1.0, 5.0], "fan.wav"),
            ],
            object_points=[
                ScenePoint("sink", [1.0, 1.0, 2.0]),
                ScenePoint("counter", [4.0, 1.0, 5.0]),
            ],
            bounds=TopDownBounds(0.0, 6.0, 0.0, 6.0),
        )
        svg = build_source_map_svg("scene", plan)
        self.assertIn("scene source map", svg)
        self.assertIn("tap_water", svg)
        self.assertIn("fan_noise", svg)
        self.assertIn("object centroids", svg)

    def test_save_source_map_artifacts_writes_svg_and_html(self):
        plan = ScenePlan(
            sources=[
                PlacedSource("tap_water", "sink", [1.0, 1.0, 2.0], "tap.wav"),
                PlacedSource("fan_noise", "fan", [4.0, 1.0, 5.0], "fan.wav"),
            ],
            object_points=[
                ScenePoint("sink", [1.0, 1.0, 2.0]),
                ScenePoint("counter", [4.0, 1.0, 5.0]),
            ],
            bounds=TopDownBounds(0.0, 6.0, 0.0, 6.0),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            svg_path, html_path = save_source_map_artifacts(output_dir, "scene", plan)

            self.assertTrue(svg_path.exists())
            self.assertTrue(html_path.exists())
            self.assertIn("tap_water", svg_path.read_text(encoding="utf-8"))
            self.assertIn("source map", html_path.read_text(encoding="utf-8"))

    def test_make_synthetic_sources_uses_requested_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            clips = {
                name: audio_dir / f"{name}.wav"
                for name in ["tap_water", "washing_machine", "fridge_hum", "fan_noise", "kettle_hiss"]
            }
            sources = make_synthetic_sources(clips, 3, "dry_run")
            self.assertEqual(len(sources), 3)
            self.assertEqual(sources[0].object_name, "dry_run_0")
            self.assertEqual(sources[2].label, "fridge_hum")


if __name__ == "__main__":
    unittest.main()
