#!/usr/bin/env python3
"""Matterport3D + SoundSpaces 2.0 demo scaffold.

This script is designed to:
1. Load a single Matterport3D scene in Habitat-Sim.
2. Discover semantically labeled objects in the scene.
3. Attach 4-5 spatial audio sources to objects such as taps, sinks,
   washing machines, fridges, fans, lamps, tables, or chairs.
4. Render binaural impulse responses with SoundSpaces 2.0 and convolve
   them with short source clips.
5. Save a rendered mix so the result can be checked offline or played back.

The workspace currently contains a Matterport3D scene asset bundle at:
    data/mp3d/5LpN3gDmAk7.{glb,house,navmesh,semantic.ply}

If habitat_sim is not installed yet, the script still runs in dry-run mode and
prints the exact scene and source plan it would use.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import hashlib
import html
import math
import json
import os
import wave
from copy import deepcopy
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np


SCENE_ID = "5LpN3gDmAk7"
DEFAULT_SCENE_DIR = Path("data/mp3d")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_AUDIO_DIR = Path("assets/audio")
DEFAULT_MATERIALS_JSON = Path("data/mp3d_material_config.json")
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_LISTENER_HEIGHT = 1.5


def configure_habitat_editable_skip() -> None:
    """Avoid editable-install rebuilds when importing Habitat-Sim locally."""
    if os.environ.get("SKBUILD_EDITABLE_SKIP"):
        return

    build_root = Path(__file__).resolve().parent / "third_party" / "habitat-sim" / "build"
    if not build_root.exists():
        return

    build_dirs = sorted(
        str(path)
        for path in build_root.iterdir()
        if path.is_dir() and path.name.startswith("cp")
    )
    if build_dirs:
        os.environ["SKBUILD_EDITABLE_SKIP"] = os.pathsep.join(build_dirs)


configure_habitat_editable_skip()


SOUND_LIBRARY = [
    {
        "name": "tap_water",
        "keywords": ["tap", "faucet", "sink", "washbasin"],
        "kind": "water_noise",
        "duration_s": 2.5,
    },
    {
        "name": "washing_machine",
        "keywords": ["washing machine", "laundry", "washer"],
        "kind": "low_rumble",
        "duration_s": 3.0,
    },
    {
        "name": "fridge_hum",
        "keywords": ["fridge", "refrigerator"],
        "kind": "hum",
        "duration_s": 3.0,
    },
    {
        "name": "fan_noise",
        "keywords": ["fan", "vent", "air conditioner"],
        "kind": "broadband_fan",
        "duration_s": 2.5,
    },
    {
        "name": "kettle_hiss",
        "keywords": ["kettle", "stove", "oven", "microwave"],
        "kind": "hiss",
        "duration_s": 2.0,
    },
    {
        "name": "lamp_buzz",
        "keywords": ["lamp", "light", "bulb"],
        "kind": "buzz",
        "duration_s": 2.0,
    },
    {
        "name": "chair_creak",
        "keywords": ["chair", "stool"],
        "kind": "creak",
        "duration_s": 1.5,
    },
    {
        "name": "table_clatter",
        "keywords": ["table", "counter", "desk"],
        "kind": "clatter",
        "duration_s": 1.5,
    },
]


@dataclass
class PlacedSource:
    label: str
    object_name: str
    position: list[float]
    audio_clip: str


@dataclass
class ScenePoint:
    label: str
    position: list[float]


@dataclass(frozen=True)
class TopDownBounds:
    min_x: float
    max_x: float
    min_z: float
    max_z: float


@dataclass(frozen=True)
class MapLayout:
    width: int
    height: int
    margin: int
    scale: float
    offset_x: float
    offset_y: float


@dataclass
class ScenePlan:
    sources: list[PlacedSource]
    object_points: list[ScenePoint]
    bounds: TopDownBounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoundSpaces 2.0 MP3D demo")
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--scene-id", default=SCENE_ID)
    parser.add_argument(
        "--scene-dataset-config",
        default="default",
        help="Habitat scene dataset config to use when loading the scene.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--max-sources", type=int, default=5)
    parser.add_argument("--render-steps", type=int, default=4)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Load the scene and export the object-to-sound plan, but skip audio rendering.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not require habitat_sim.")
    return parser.parse_args()


def scene_assets(scene_dir: Path, scene_id: str) -> dict[str, Path]:
    return {
        "glb": scene_dir / f"{scene_id}.glb",
        "house": scene_dir / f"{scene_id}.house",
        "navmesh": scene_dir / f"{scene_id}.navmesh",
        "semantic": scene_dir / f"{scene_id}_semantic.ply",
    }


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1 if pcm16.ndim == 1 else pcm16.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def fft_convolve(signal: np.ndarray, ir: np.ndarray) -> np.ndarray:
    n = signal.shape[0] + ir.shape[0] - 1
    fft_len = 1 << (n - 1).bit_length()
    s = np.fft.rfft(signal, fft_len)
    h = np.fft.rfft(ir, fft_len)
    out = np.fft.irfft(s * h, fft_len)[:n]
    return out.astype(np.float32)


def normalize_audio(samples: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(samples)))
    if max_abs <= 1e-9:
        return samples
    return (samples / max_abs) * peak


def synth_clip(kind: str, duration_s: float, sample_rate: int) -> np.ndarray:
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    rng = np.random.default_rng(abs(hash(kind)) & 0xFFFFFFFF)

    if kind == "water_noise":
        noise = rng.normal(0.0, 0.35, size=n).astype(np.float32)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.35 * t)
        clip = 0.55 * noise * envelope + 0.12 * np.sin(2 * np.pi * 480 * t)
    elif kind == "low_rumble":
        clip = 0.42 * np.sin(2 * np.pi * 48 * t) + 0.18 * np.sin(2 * np.pi * 96 * t)
        clip += 0.12 * rng.normal(0.0, 1.0, size=n)
    elif kind == "hum":
        clip = 0.36 * np.sin(2 * np.pi * 60 * t) + 0.11 * np.sin(2 * np.pi * 120 * t)
        clip += 0.05 * rng.normal(0.0, 1.0, size=n)
    elif kind == "broadband_fan":
        noise = rng.normal(0.0, 0.3, size=n).astype(np.float32)
        clip = 0.35 * noise + 0.18 * np.sin(2 * np.pi * 220 * t)
    elif kind == "hiss":
        noise = rng.normal(0.0, 0.4, size=n).astype(np.float32)
        clip = noise + 0.04 * np.sin(2 * np.pi * 1800 * t)
    elif kind == "buzz":
        clip = 0.18 * np.sin(2 * np.pi * 100 * t) + 0.12 * np.sin(2 * np.pi * 200 * t)
        clip += 0.07 * rng.normal(0.0, 1.0, size=n)
    elif kind == "creak":
        clip = 0.18 * np.sin(2 * np.pi * 280 * t) * np.sin(2 * np.pi * 1.8 * t)
        clip += 0.12 * rng.normal(0.0, 1.0, size=n)
    elif kind == "clatter":
        pulse = (np.sin(2 * np.pi * 7 * t) > 0.7).astype(np.float32)
        clip = 0.24 * pulse * rng.normal(0.0, 1.0, size=n)
    else:
        clip = rng.normal(0.0, 0.15, size=n).astype(np.float32)

    return normalize_audio(clip.astype(np.float32))


def ensure_source_clips(audio_dir: Path, sample_rate: int) -> dict[str, Path]:
    ensure_dirs(audio_dir)
    clips: dict[str, Path] = {}
    for spec in SOUND_LIBRARY:
        path = audio_dir / f"{spec['name']}.wav"
        if not path.exists():
            samples = synth_clip(spec["kind"], float(spec["duration_s"]), sample_rate)
            write_wav(path, samples, sample_rate)
        clips[spec["name"]] = path
    return clips


STRUCTURAL_OBJECT_KEYWORDS = [
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "beam",
    "column",
    "railing",
    "stairs",
    "stair",
    "room",
    "hallway",
    "corridor",
    "unknown",
    "misc",
    "opening",
    "arch",
]


def source_priority(object_name: str) -> int:
    lowered = object_name.lower()
    for idx, spec in enumerate(SOUND_LIBRARY):
        if any(keyword in lowered for keyword in spec["keywords"]):
            return idx
    if any(keyword in lowered for keyword in STRUCTURAL_OBJECT_KEYWORDS):
        return 10_000
    return 1_000


def clip_name_for_candidate(object_name: str, priority: int) -> str:
    if priority < len(SOUND_LIBRARY):
        return SOUND_LIBRARY[priority]["name"]

    generic_pool = ["lamp_buzz", "chair_creak", "table_clatter"]
    score = sum(ord(ch) for ch in object_name)
    return generic_pool[score % len(generic_pool)]


def object_label(obj: object) -> str:
    for attr in ("category", "category_name", "semantic_id", "id"):
        value = getattr(obj, attr, None)
        if value is None:
            continue
        name_attr = getattr(value, "name", None)
        if callable(name_attr):
            try:
                return str(name_attr())
            except TypeError:
                pass
        if isinstance(name_attr, str):
            return name_attr
        if isinstance(value, str):
            return value
        if callable(value):
            try:
                value = value()
            except TypeError:
                pass
        return str(value)
    return "unknown"


def object_aabb_center(obj: object) -> list[float] | None:
    aabb = getattr(obj, "aabb", None)
    if aabb is None:
        return None
    center = getattr(aabb, "center", None)
    if callable(center):
        try:
            center = center()
        except TypeError:
            pass
    if center is None:
        center = getattr(aabb, "center_", None)
        if callable(center):
            try:
                center = center()
            except TypeError:
                pass
    if center is None and hasattr(aabb, "min") and hasattr(aabb, "max"):
        mn = np.array(aabb.min, dtype=np.float32)
        mx = np.array(aabb.max, dtype=np.float32)
        center = (mn + mx) / 2.0
    if center is None:
        return None
    try:
        return [float(x) for x in np.asarray(center, dtype=np.float32).tolist()]
    except TypeError:
        return None


def iter_semantic_objects(semantic_scene: object) -> list[object]:
    seen: set[int] = set()
    objects: list[object] = []

    for obj in getattr(semantic_scene, "objects", []):
        ident = id(obj)
        if ident in seen:
            continue
        seen.add(ident)
        objects.append(obj)

    for level in getattr(semantic_scene, "levels", []):
        for obj in getattr(level, "objects", []):
            ident = id(obj)
            if ident in seen:
                continue
            seen.add(ident)
            objects.append(obj)

    return objects


def build_scene_cfg(scene_glb: Path, scene_dataset_config: str, create_renderer: bool):
    import habitat_sim
    from habitat_sim.utils.settings import default_sim_settings, make_cfg

    settings = deepcopy(default_sim_settings)
    settings.update(
        {
            "scene_dataset_config_file": scene_dataset_config,
            "scene": str(scene_glb),
            "color_sensor": False,
            "depth_sensor": False,
            "semantic_sensor": False,
            "ortho_rgba_sensor": False,
            "ortho_depth_sensor": False,
            "ortho_semantic_sensor": False,
            "fisheye_rgba_sensor": False,
            "fisheye_depth_sensor": False,
            "fisheye_semantic_sensor": False,
            "equirect_rgba_sensor": False,
            "equirect_depth_sensor": False,
            "equirect_semantic_sensor": False,
            "enable_physics": False,
        }
    )

    cfg = make_cfg(settings)
    cfg.sim_cfg.load_semantic_mesh = True
    cfg.sim_cfg.allow_sliding = True
    cfg.sim_cfg.create_renderer = create_renderer
    return cfg


def build_scene_sim(scene_glb: Path, scene_dir: Path, scene_dataset_config: str, create_renderer: bool):
    import habitat_sim

    cfg = build_scene_cfg(scene_glb, scene_dataset_config, create_renderer)
    sim = habitat_sim.Simulator(cfg)
    navmesh = scene_dir / f"{scene_glb.stem}.navmesh"
    if navmesh.exists() and hasattr(sim, "pathfinder") and hasattr(sim.pathfinder, "load_nav_mesh"):
        try:
            sim.pathfinder.load_nav_mesh(str(navmesh))
        except Exception:
            pass
    return sim


def make_synthetic_sources(clips: dict[str, Path], max_sources: int, prefix: str) -> list[PlacedSource]:
    sources: list[PlacedSource] = []
    for i, spec in enumerate(SOUND_LIBRARY[:max_sources]):
        sources.append(
            PlacedSource(
                label=spec["name"],
                object_name=f"{prefix}_{i}",
                position=[1.0 + i * 0.5, DEFAULT_LISTENER_HEIGHT, 1.0 + i * 0.25],
                audio_clip=str(clips[spec["name"]]),
            )
        )
    return sources


def collect_scene_object_points(sim) -> list[ScenePoint]:
    semantic_scene = getattr(sim, "semantic_scene", None)
    if semantic_scene is None:
        return []

    points: list[ScenePoint] = []
    seen: set[tuple[int, int, int]] = set()
    for obj in iter_semantic_objects(semantic_scene):
        label = object_label(obj)
        center = object_aabb_center(obj)
        if center is None:
            continue
        key = tuple(int(round(v * 100)) for v in center)
        if key in seen:
            continue
        seen.add(key)
        points.append(ScenePoint(label=label, position=center))

    points.sort(key=lambda item: (item.label.lower(), item.position[0], item.position[2]))
    return points


def pathfinder_topdown_bounds(sim) -> TopDownBounds | None:
    pathfinder = getattr(sim, "pathfinder", None)
    if pathfinder is None or not hasattr(pathfinder, "get_bounds"):
        return None

    try:
        bounds = np.asarray(pathfinder.get_bounds(), dtype=np.float32)
    except Exception:
        return None

    if bounds.shape[0] < 2:
        return None

    lower = bounds[0]
    upper = bounds[1]
    return TopDownBounds(
        min_x=float(min(lower[0], upper[0])),
        max_x=float(max(lower[0], upper[0])),
        min_z=float(min(lower[2], upper[2])),
        max_z=float(max(lower[2], upper[2])),
    )


def _expand_bounds(
    bounds: TopDownBounds | None,
    points: Sequence[Sequence[float]],
    padding_ratio: float = 0.08,
    min_padding: float = 0.75,
) -> TopDownBounds:
    xs: list[float] = []
    zs: list[float] = []

    if bounds is not None:
        xs.extend([bounds.min_x, bounds.max_x])
        zs.extend([bounds.min_z, bounds.max_z])

    for point in points:
        xs.append(float(point[0]))
        zs.append(float(point[2]))

    if not xs:
        return TopDownBounds(-2.0, 2.0, -2.0, 2.0)

    min_x = min(xs)
    max_x = max(xs)
    min_z = min(zs)
    max_z = max(zs)

    span_x = max(max_x - min_x, 0.5)
    span_z = max(max_z - min_z, 0.5)
    pad_x = max(span_x * padding_ratio, min_padding)
    pad_z = max(span_z * padding_ratio, min_padding)
    return TopDownBounds(min_x - pad_x, max_x + pad_x, min_z - pad_z, max_z + pad_z)


def build_scene_bounds(
    sim,
    source_positions: Sequence[Sequence[float]],
    object_points: Sequence[ScenePoint] | Sequence[PlacedSource] | None = None,
) -> TopDownBounds:
    points: list[Sequence[float]] = [list(point) for point in source_positions]
    if object_points is not None:
        for item in object_points:
            points.append(item.position if hasattr(item, "position") else item)
    return _expand_bounds(pathfinder_topdown_bounds(sim), points)


def build_map_layout(bounds: TopDownBounds, width: int = 960, height: int = 720, margin: int = 64) -> MapLayout:
    scene_width = max(bounds.max_x - bounds.min_x, 0.5)
    scene_height = max(bounds.max_z - bounds.min_z, 0.5)
    inner_width = max(width - 2 * margin, 1)
    inner_height = max(height - 2 * margin, 1)
    scale = min(inner_width / scene_width, inner_height / scene_height)
    content_width = scene_width * scale
    content_height = scene_height * scale
    offset_x = margin + (inner_width - content_width) / 2.0
    offset_y = margin + (inner_height - content_height) / 2.0
    return MapLayout(
        width=width,
        height=height,
        margin=margin,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def world_to_topdown(
    position: Sequence[float],
    bounds: TopDownBounds,
    layout: MapLayout,
) -> tuple[float, float]:
    x = float(position[0])
    z = float(position[2])
    px = layout.offset_x + (x - bounds.min_x) * layout.scale
    py = layout.offset_y + (bounds.max_z - z) * layout.scale
    return px, py


def stable_label_color(label: str) -> str:
    palette = [
        "#67E8F9",
        "#F59E0B",
        "#A78BFA",
        "#34D399",
        "#F472B6",
        "#F87171",
        "#38BDF8",
        "#FBBF24",
    ]
    digest = hashlib.sha1(label.encode("utf-8")).digest()
    return palette[digest[0] % len(palette)]


def nice_grid_step(bounds: TopDownBounds) -> float:
    span = max(bounds.max_x - bounds.min_x, bounds.max_z - bounds.min_z)
    if span <= 8:
        return 0.5
    if span <= 16:
        return 1.0
    if span <= 32:
        return 2.0
    if span <= 64:
        return 5.0
    return 10.0


def format_point(position: Sequence[float]) -> str:
    return f"[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]"


def build_scene_plan(
    scene_glb: Path,
    scene_dir: Path,
    scene_dataset_config: str,
    max_sources: int,
) -> ScenePlan:
    sim = build_scene_sim(scene_glb, scene_dir, scene_dataset_config, create_renderer=False)
    try:
        sources = discover_scene_sources(sim, max_sources)
        if len(sources) < min(4, max_sources):
            fallback = fallback_sources(sim, max_sources)
            if len(sources) < max_sources:
                sources.extend(fallback[: max_sources - len(sources)])
        object_points = collect_scene_object_points(sim)
        bounds = build_scene_bounds(
            sim,
            [source.position for source in sources],
            object_points,
        )
        return ScenePlan(sources=sources, object_points=object_points, bounds=bounds)
    finally:
        try:
            sim.close()
        except Exception:
            pass


def make_schematic_plan(sources: Sequence[PlacedSource]) -> ScenePlan:
    source_list = list(sources)
    bounds = build_scene_bounds(None, [source.position for source in source_list], None)
    return ScenePlan(sources=source_list, object_points=[], bounds=bounds)


def build_source_map_svg(
    scene_id: str,
    plan: ScenePlan,
    width: int = 960,
    height: int = 720,
) -> str:
    layout = build_map_layout(plan.bounds, width=width, height=height)
    grid_step = nice_grid_step(plan.bounds)

    x_start = math.floor(plan.bounds.min_x / grid_step) * grid_step
    x_stop = math.ceil(plan.bounds.max_x / grid_step) * grid_step
    z_start = math.floor(plan.bounds.min_z / grid_step) * grid_step
    z_stop = math.ceil(plan.bounds.max_z / grid_step) * grid_step

    scene_width = max(plan.bounds.max_x - plan.bounds.min_x, 0.5)
    scene_height = max(plan.bounds.max_z - plan.bounds.min_z, 0.5)
    content_width = scene_width * layout.scale
    content_height = scene_height * layout.scale
    map_x = layout.offset_x
    map_y = layout.offset_y

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {layout.width} {layout.height}" '
        f'width="{layout.width}" height="{layout.height}" role="img" aria-label="Top-down source map for {html.escape(scene_id)}">'
    )
    parts.append(
        "<defs>"
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#000000" flood-opacity="0.35"/>'
        "</filter>"
        '<style><![CDATA['
        ".grid { stroke: #2a3342; stroke-width: 1; opacity: 0.55; }"
        ".axis { stroke: #475569; stroke-width: 1.25; opacity: 0.7; }"
        ".bounds { fill: rgba(20,24,31,0.78); stroke: #93c5fd; stroke-width: 2; stroke-dasharray: 7 6; }"
        ".objects { fill: #cbd5e1; opacity: 0.18; }"
        ".source-label { font: 600 14px/1.2 system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #e5e7eb; paint-order: stroke; stroke: #0f172a; stroke-width: 3px; stroke-linejoin: round; }"
        ".title { font: 700 28px/1.2 system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #f8fafc; }"
        ".subtitle { font: 400 13px/1.4 system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #cbd5e1; }"
        ".tiny { font: 400 11px/1.4 system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #94a3b8; }"
        "]]></style>"
        "</defs>"
    )
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="#0b1020"/>')
    parts.append('<rect x="24" y="24" width="{0}" height="{1}" rx="20" fill="#111827" stroke="#243043"/>'.format(layout.width - 48, layout.height - 48))
    parts.append(f'<text x="48" y="62" class="title">{html.escape(scene_id)} source map</text>')
    parts.append(
        f'<text x="48" y="86" class="subtitle">Gray dots are semantic object centroids. Colored markers are planned sound sources.</text>'
    )
    parts.append(
        f'<text x="48" y="108" class="tiny">Bounds: x {plan.bounds.min_x:.2f}..{plan.bounds.max_x:.2f} m, z {plan.bounds.min_z:.2f}..{plan.bounds.max_z:.2f} m | '
        f'{len(plan.object_points)} object centroids | {len(plan.sources)} sources</text>'
    )

    # Grid and axes.
    for x in np.arange(x_start, x_stop + grid_step * 0.5, grid_step):
        px, _ = world_to_topdown((x, 0.0, plan.bounds.min_z), plan.bounds, layout)
        parts.append(
            f'<line x1="{px:.2f}" y1="{map_y:.2f}" x2="{px:.2f}" y2="{map_y + content_height:.2f}" class="grid"/>'
        )
        if abs(x - round(x)) < 1e-6 or grid_step >= 1:
            parts.append(
                f'<text x="{px + 2:.2f}" y="{map_y + content_height + 18:.2f}" class="tiny">{x:.0f}</text>'
            )

    for z in np.arange(z_start, z_stop + grid_step * 0.5, grid_step):
        _, py = world_to_topdown((plan.bounds.min_x, 0.0, z), plan.bounds, layout)
        parts.append(
            f'<line x1="{map_x:.2f}" y1="{py:.2f}" x2="{map_x + content_width:.2f}" y2="{py:.2f}" class="grid"/>'
        )
        if abs(z - round(z)) < 1e-6 or grid_step >= 1:
            parts.append(
                f'<text x="{map_x - 28:.2f}" y="{py + 4:.2f}" class="tiny">{z:.0f}</text>'
            )

    parts.append(
        f'<rect x="{map_x:.2f}" y="{map_y:.2f}" width="{content_width:.2f}" height="{content_height:.2f}" rx="14" class="bounds"/>'
    )
    parts.append(
        f'<text x="{map_x + content_width - 30:.2f}" y="{map_y + content_height + 34:.2f}" class="tiny">x</text>'
    )
    parts.append(f'<text x="{map_x - 18:.2f}" y="{map_y + 14:.2f}" class="tiny">z</text>')

    for point in plan.object_points:
        px, py = world_to_topdown(point.position, plan.bounds, layout)
        dot_size = 2.0 if source_priority(point.label) >= 10000 else 2.7
        dot_opacity = 0.08 if source_priority(point.label) >= 10000 else 0.18
        parts.append(
            f'<g class="objects"><title>{html.escape(point.label)} {html.escape(format_point(point.position))}</title>'
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{dot_size:.2f}" fill="#cbd5e1" opacity="{dot_opacity:.2f}"/></g>'
        )

    for idx, source in enumerate(plan.sources):
        px, py = world_to_topdown(source.position, plan.bounds, layout)
        color = stable_label_color(source.label)
        label_text = f"{source.label} -> {source.object_name}"
        label_dx = 16 if idx % 2 == 0 else -16
        text_anchor = "start" if label_dx > 0 else "end"
        text_x = px + label_dx
        text_y = py - 12 if idx % 2 == 0 else py + 20
        parts.append(
            f'<g filter="url(#shadow)"><title>{html.escape(label_text)} {html.escape(format_point(source.position))}</title>'
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="11" fill="#0f172a" opacity="0.85"/>'
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="8.5" fill="{color}" stroke="#f8fafc" stroke-width="2"/>'
            f'<text x="{text_x:.2f}" y="{text_y:.2f}" text-anchor="{text_anchor}" class="source-label">{html.escape(source.label)}</text>'
            f"</g>"
        )

    parts.append(
        f'<text x="{layout.width - 46}" y="{layout.height - 34}" text-anchor="end" class="tiny">Top-down view | x/z plane</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def build_source_map_html(scene_id: str, plan: ScenePlan, svg: str) -> str:
    summary_items = []
    for source in plan.sources:
        summary_items.append(
            "<li>"
            f'<span class="swatch" style="background:{stable_label_color(source.label)}"></span>'
            f"<strong>{html.escape(source.label)}</strong>"
            f" <span class=\"muted\">on {html.escape(source.object_name)}</span>"
            f"<code>{html.escape(format_point(source.position))}</code>"
            "</li>"
        )

    note = (
        "This is a schematic dry-run view." if not plan.object_points else "Gray points are the semantic object centroids from the Matterport scene."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(scene_id)} source map</title>
<style>
  :root {{
    color-scheme: dark;
    --bg: #0b1020;
    --panel: #111827;
    --panel-border: #223046;
    --text: #e5e7eb;
    --muted: #94a3b8;
  }}
  body {{
    margin: 0;
    background:
      radial-gradient(circle at top left, rgba(103, 232, 249, 0.12), transparent 30%),
      radial-gradient(circle at bottom right, rgba(167, 139, 250, 0.12), transparent 26%),
      var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  .wrap {{
    max-width: 1240px;
    margin: 0 auto;
    padding: 24px;
  }}
  .hero {{
    display: flex;
    justify-content: space-between;
    gap: 20px;
    align-items: end;
    margin-bottom: 18px;
  }}
  h1 {{
    margin: 0;
    font-size: 32px;
    line-height: 1.1;
  }}
  .sub {{
    margin: 8px 0 0;
    color: var(--muted);
    max-width: 68ch;
  }}
  .card {{
    background: rgba(17, 24, 39, 0.92);
    border: 1px solid var(--panel-border);
    border-radius: 22px;
    overflow: hidden;
    box-shadow: 0 24px 72px rgba(0, 0, 0, 0.35);
  }}
  .meta {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 18px 0;
  }}
  .stat {{
    border: 1px solid var(--panel-border);
    background: rgba(17, 24, 39, 0.74);
    border-radius: 16px;
    padding: 12px 14px;
  }}
  .stat .label {{
    display: block;
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
  }}
  .stat .value {{
    font-size: 18px;
    font-weight: 700;
  }}
  .legend {{
    margin-top: 18px;
    display: grid;
    gap: 12px;
  }}
  .legend h2 {{
    margin: 0 0 6px;
    font-size: 18px;
  }}
  .legend ul {{
    list-style: none;
    padding: 0;
    margin: 0;
    display: grid;
    gap: 8px;
  }}
  .legend li {{
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 10px;
    align-items: center;
    padding: 10px 12px;
    background: rgba(17, 24, 39, 0.72);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
  }}
  .swatch {{
    width: 14px;
    height: 14px;
    border-radius: 999px;
    display: inline-block;
    box-shadow: 0 0 0 3px rgba(255,255,255,0.08);
  }}
  .muted {{
    color: var(--muted);
  }}
  code {{
    display: inline-block;
    margin-left: 8px;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.06);
    color: #dbeafe;
  }}
  .note {{
    color: var(--muted);
    font-size: 14px;
    margin-top: 10px;
  }}
  @media (max-width: 760px) {{
    .wrap {{ padding: 16px; }}
    h1 {{ font-size: 24px; }}
  }}
</style>
</head>
<body>
  <main class="wrap">
    <div class="hero">
      <div>
        <h1>{html.escape(scene_id)} source map</h1>
        <p class="sub">Inspect how the planned sound sources land in the room. Open this file in a browser, then use the CSV/JSON plan alongside it if you want to replay the setup elsewhere.</p>
      </div>
    </div>

    <section class="meta" aria-label="Scene summary">
      <div class="stat"><span class="label">Sources</span><span class="value">{len(plan.sources)}</span></div>
      <div class="stat"><span class="label">Semantic objects</span><span class="value">{len(plan.object_points)}</span></div>
      <div class="stat"><span class="label">Bounds</span><span class="value">x {plan.bounds.min_x:.1f}..{plan.bounds.max_x:.1f} m</span></div>
      <div class="stat"><span class="label">Bounds</span><span class="value">z {plan.bounds.min_z:.1f}..{plan.bounds.max_z:.1f} m</span></div>
    </section>

    <section class="card">
      {svg}
    </section>

    <p class="note">{html.escape(note)}</p>

    <section class="legend">
      <h2>Planned Sources</h2>
      <ul>
        {''.join(summary_items)}
      </ul>
    </section>
  </main>
</body>
</html>
"""


def save_source_map_artifacts(output_dir: Path, scene_id: str, plan: ScenePlan) -> tuple[Path, Path]:
    ensure_dirs(output_dir)
    svg_path = output_dir / f"{scene_id}_source_map.svg"
    html_path = output_dir / f"{scene_id}_source_map.html"
    svg = build_source_map_svg(scene_id, plan)
    html_doc = build_source_map_html(scene_id, plan, svg)
    svg_path.write_text(svg, encoding="utf-8")
    html_path.write_text(html_doc, encoding="utf-8")
    return svg_path, html_path


def source_label_counts(sources: Sequence[PlacedSource]) -> Counter[str]:
    return Counter(source.label for source in sources)


def discover_scene_sources(sim, max_sources: int) -> list[PlacedSource]:
    semantic_scene = getattr(sim, "semantic_scene", None)
    candidates: list[tuple[int, str, list[float]]] = []

    if semantic_scene is not None:
        for obj in iter_semantic_objects(semantic_scene):
            name = object_label(obj)
            center = object_aabb_center(obj)
            if center is None:
                continue
            score = source_priority(name)
            candidates.append((score, name, center))

    candidates.sort(key=lambda item: (item[0], item[1]))

    placed: list[PlacedSource] = []
    deferred: list[PlacedSource] = []
    used_labels: set[str] = set()
    used_positions: set[tuple[int, int, int]] = set()
    for score, name, center in candidates:
        if len(placed) >= max_sources:
            break
        key = tuple(int(round(v * 100)) for v in center)
        if key in used_positions:
            continue
        used_positions.add(key)
        clip_name = clip_name_for_candidate(name, score)
        source = PlacedSource(
            label=clip_name,
            object_name=name,
            position=center,
            audio_clip=str(DEFAULT_AUDIO_DIR / f"{clip_name}.wav"),
        )
        if clip_name in used_labels:
            deferred.append(source)
            continue
        used_labels.add(clip_name)
        placed.append(source)

    for source in deferred:
        if len(placed) >= max_sources:
            break
        placed.append(source)

    return placed


def fallback_sources(sim, max_sources: int) -> list[PlacedSource]:
    pathfinder = getattr(sim, "pathfinder", None)
    positions: list[list[float]] = []
    if pathfinder is not None and hasattr(pathfinder, "get_random_navigable_point"):
        for _ in range(max_sources):
            try:
                pos = pathfinder.get_random_navigable_point()
                positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
            except Exception:
                break

    if not positions:
        positions = [[1.0 + i * 0.5, DEFAULT_LISTENER_HEIGHT, 1.0] for i in range(max_sources)]

    out: list[PlacedSource] = []
    for i, pos in enumerate(positions[:max_sources]):
        spec = SOUND_LIBRARY[i % len(SOUND_LIBRARY)]
        out.append(
            PlacedSource(
                label=spec["name"],
                object_name=f"fallback_{i}",
                position=pos,
                audio_clip=str(DEFAULT_AUDIO_DIR / f"{spec['name']}.wav"),
            )
        )
    return out


def mix_binaural_sources(ir_map: Sequence[np.ndarray], source_clips: Sequence[np.ndarray]) -> np.ndarray:
    assert len(ir_map) == len(source_clips)
    mixed = None
    for ir, clip in zip(ir_map, source_clips):
        if ir.ndim == 1:
            ir = ir[None, :]
        channels = []
        for ch in range(ir.shape[0]):
            channels.append(fft_convolve(clip, ir[ch]))
        src_mix = np.stack(channels, axis=1)
        if mixed is None:
            mixed = src_mix
            continue

        if mixed.shape[0] < src_mix.shape[0]:
            pad = np.zeros((src_mix.shape[0] - mixed.shape[0], mixed.shape[1]), dtype=np.float32)
            mixed = np.concatenate([mixed, pad], axis=0)
        elif src_mix.shape[0] < mixed.shape[0]:
            pad = np.zeros((mixed.shape[0] - src_mix.shape[0], src_mix.shape[1]), dtype=np.float32)
            src_mix = np.concatenate([src_mix, pad], axis=0)
        mixed = mixed + src_mix
    if mixed is None:
        return np.zeros((1, 2), dtype=np.float32)
    return normalize_audio(mixed.astype(np.float32))


def add_audio_sensor(sim, output_dir: Path, materials_json: Path):
    import habitat_sim

    acoustic_cfg = habitat_sim.sensor.RLRAudioPropagationConfiguration()
    acoustic_cfg.sampleRate = DEFAULT_SAMPLE_RATE
    acoustic_cfg.enableMaterials = materials_json.exists()

    channel_layout = habitat_sim.sensor.RLRAudioPropagationChannelLayout()
    channel_layout.channelType = (
        habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    )
    channel_layout.channelCount = 2

    audio_spec = habitat_sim.AudioSensorSpec()
    audio_spec.uuid = "audio_sensor"
    audio_spec.sensor_type = habitat_sim.SensorType.AUDIO
    audio_spec.position = [0.0, 0.0, 0.0]
    audio_spec.enableMaterials = materials_json.exists()
    audio_spec.acousticsConfig = acoustic_cfg
    audio_spec.channelLayout = channel_layout
    audio_spec.outputDirectory = str(output_dir)

    try:
        sim.add_sensor(audio_spec)
    except Exception:
        # Older/newer Habitat-Sim builds vary slightly here; fall back to the
        # agent-based sensor attachment when direct registration is unavailable.
        from habitat_sim.agent import AgentConfiguration

        agent_cfg = AgentConfiguration()
        agent_cfg.sensor_specifications = [audio_spec]
        sim.add_agent(agent_cfg)

    agent = sim.get_agent(0)
    audio_sensor = getattr(agent, "sensors", {}).get("audio_sensor")
    if audio_sensor is None and hasattr(agent, "_sensors"):
        audio_sensor = agent._sensors.get("audio_sensor")
    if audio_sensor is None:
        raise RuntimeError("Could not attach the Habitat audio sensor.")

    attach_audio_materials(audio_sensor, materials_json)
    return audio_sensor


def sample_listener_positions(sim, render_steps: int) -> list[list[float]]:
    pathfinder = getattr(sim, "pathfinder", None)
    listener_positions: list[list[float]] = []
    if pathfinder is not None and hasattr(pathfinder, "get_random_navigable_point"):
        for _ in range(render_steps):
            try:
                p = pathfinder.get_random_navigable_point()
                listener_positions.append([float(p[0]), float(p[1]), float(p[2])])
            except Exception:
                break
    if not listener_positions:
        listener_positions = [
            [1.0, DEFAULT_LISTENER_HEIGHT, 1.0 + 0.5 * i] for i in range(render_steps)
        ]
    return listener_positions


def attach_audio_materials(audio_sensor: object, materials_json: Path) -> None:
    if materials_json.exists() and hasattr(audio_sensor, "setAudioMaterialsJSON"):
        audio_sensor.setAudioMaterialsJSON(str(materials_json))


def plan_scene_sources(
    scene_glb: Path,
    scene_dir: Path,
    scene_dataset_config: str,
    max_sources: int,
) -> list[PlacedSource]:
    return build_scene_plan(scene_glb, scene_dir, scene_dataset_config, max_sources).sources


def render_audio_for_sources(
    scene_glb: Path,
    scene_dir: Path,
    scene_dataset_config: str,
    output_dir: Path,
    materials_json: Path,
    scene_id: str,
    sources: Sequence[PlacedSource],
    render_steps: int,
) -> list[Path]:
    sim = build_scene_sim(scene_glb, scene_dir, scene_dataset_config, create_renderer=True)
    try:
        audio_sensor = add_audio_sensor(sim, output_dir, materials_json)
        listener_positions = sample_listener_positions(sim, render_steps)

        rendered_paths: list[Path] = []
        for step_idx, listener_pos in enumerate(listener_positions):
            agent = sim.get_agent(0)
            if hasattr(agent, "get_state") and hasattr(agent, "set_state"):
                state = agent.get_state()
                state.position = np.array(listener_pos, dtype=np.float32)
                if hasattr(state, "sensor_states"):
                    state.sensor_states = {}
                agent.set_state(state, True)

            irs: list[np.ndarray] = []
            clips_for_mix: list[np.ndarray] = []
            for source in sources:
                audio_sensor.setAudioSourceTransform(np.array(source.position, dtype=np.float32))
                obs = sim.get_sensor_observations()["audio_sensor"]
                ir = np.asarray(obs, dtype=np.float32)
                if ir.ndim == 1:
                    ir = ir[None, :]
                irs.append(ir)
                spec = next(spec for spec in SOUND_LIBRARY if spec["name"] == source.label)
                clips_for_mix.append(
                    synth_clip(spec["kind"], float(spec["duration_s"]), DEFAULT_SAMPLE_RATE)
                )

            mix = mix_binaural_sources(irs, clips_for_mix)
            out_path = output_dir / f"{scene_id}_listener_{step_idx:02d}.wav"
            write_wav(out_path, mix, DEFAULT_SAMPLE_RATE)
            rendered_paths.append(out_path)

        return rendered_paths
    finally:
        try:
            sim.close()
        except Exception:
            pass


def print_scene_plan(
    scene_paths: dict[str, Path],
    sources: Sequence[PlacedSource],
    object_points: Sequence[ScenePoint] | None = None,
) -> None:
    print("Scene assets:")
    for key, path in scene_paths.items():
        print(f"  {key}: {path} {'(missing)' if not path.exists() else ''}")
    print("Planned audio sources:")
    for idx, src in enumerate(sources, 1):
        pos = ", ".join(f"{v:.3f}" for v in src.position)
        print(f"  {idx}. {src.label} on {src.object_name} at [{pos}] -> {src.audio_clip}")
    if object_points is not None:
        print(f"Semantic object centroids: {len(object_points)}")
    counts = source_label_counts(sources)
    if counts:
        print("Source summary:")
        for label, count in counts.most_common():
            print(f"  {label}: {count}")


def save_plan(output_dir: Path, scene_id: str, sources: Sequence[PlacedSource]) -> Path:
    ensure_dirs(output_dir)
    plan_path = output_dir / f"{scene_id}_audio_plan.json"
    with plan_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(src) for src in sources], f, indent=2)
    return plan_path


def save_plan_csv(output_dir: Path, scene_id: str, sources: Sequence[PlacedSource]) -> Path:
    ensure_dirs(output_dir)
    csv_path = output_dir / f"{scene_id}_audio_plan.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "label", "object_name", "x", "y", "z", "audio_clip"],
        )
        writer.writeheader()
        for idx, src in enumerate(sources, 1):
            writer.writerow(
                {
                    "index": idx,
                    "label": src.label,
                    "object_name": src.object_name,
                    "x": f"{src.position[0]:.6f}",
                    "y": f"{src.position[1]:.6f}",
                    "z": f"{src.position[2]:.6f}",
                    "audio_clip": src.audio_clip,
                }
            )
    return csv_path


def save_plan_artifacts(output_dir: Path, scene_id: str, sources: Sequence[PlacedSource]) -> tuple[Path, Path]:
    plan_path = save_plan(output_dir, scene_id, sources)
    csv_path = save_plan_csv(output_dir, scene_id, sources)
    return plan_path, csv_path


def save_scene_artifacts(
    output_dir: Path,
    scene_id: str,
    plan: ScenePlan,
) -> tuple[Path, Path, Path, Path]:
    plan_path, csv_path = save_plan_artifacts(output_dir, scene_id, plan.sources)
    svg_path, html_path = save_source_map_artifacts(output_dir, scene_id, plan)
    return plan_path, csv_path, svg_path, html_path


def run_demo(args: argparse.Namespace) -> int:
    scene_paths = scene_assets(args.scene_dir, args.scene_id)
    ensure_dirs(args.output_dir, args.audio_dir)
    clips = ensure_source_clips(args.audio_dir, DEFAULT_SAMPLE_RATE)
    materials_json = DEFAULT_MATERIALS_JSON

    if args.dry_run:
        sources = make_synthetic_sources(clips, args.max_sources, "dry_run_object")
        plan = make_schematic_plan(sources)
        print_scene_plan(scene_paths, plan.sources, plan.object_points)
        plan_path, csv_path, svg_path, html_path = save_scene_artifacts(
            args.output_dir,
            args.scene_id,
            plan,
        )
        print(f"Saved source plan: {plan_path}")
        print(f"Saved source CSV: {csv_path}")
        print(f"Saved source map: {svg_path}")
        print(f"Saved source map HTML: {html_path}")
        return 0

    try:
        import habitat_sim  # noqa: F401
    except Exception as exc:
        print(
            "habitat_sim is not available in this environment, so I can only do a dry-run.\n"
            f"Import error: {exc}\n"
            "Install Habitat-Sim + SoundSpaces 2.0, then rerun without --dry-run."
        )
        sources = make_synthetic_sources(clips, args.max_sources, "dry_run_object")
        plan = make_schematic_plan(sources)
        print_scene_plan(scene_paths, plan.sources, plan.object_points)
        plan_path, csv_path, svg_path, html_path = save_scene_artifacts(
            args.output_dir,
            args.scene_id,
            plan,
        )
        print(f"Saved source plan: {plan_path}")
        print(f"Saved source CSV: {csv_path}")
        print(f"Saved source map: {svg_path}")
        print(f"Saved source map HTML: {html_path}")
        return 1

    scene_glb = scene_paths["glb"]
    if not scene_glb.exists():
        print(f"Missing scene GLB at {scene_glb}")
        return 1

    planning_only = args.plan_only or not getattr(habitat_sim, "audio_enabled", False)
    if planning_only:
        if args.plan_only:
            print(
                "Planning mode: loading the scene and exporting the object-to-sound map "
                "without trying to render binaural audio."
            )
        else:
            print(
                "Habitat-Sim is installed, but audio is disabled on this machine.\n"
                "This Mac can still load the scene and export the object-linked source plan."
            )
        plan = build_scene_plan(
            scene_glb,
            args.scene_dir,
            args.scene_dataset_config,
            args.max_sources,
        )
        print_scene_plan(scene_paths, plan.sources, plan.object_points)
        plan_path, csv_path, svg_path, html_path = save_scene_artifacts(
            args.output_dir,
            args.scene_id,
            plan,
        )
        print(f"Saved source plan: {plan_path}")
        print(f"Saved source CSV: {csv_path}")
        print(f"Saved source map: {svg_path}")
        print(f"Saved source map HTML: {html_path}")
        return 0

    plan = build_scene_plan(
        scene_glb,
        args.scene_dir,
        args.scene_dataset_config,
        args.max_sources,
    )
    print_scene_plan(scene_paths, plan.sources, plan.object_points)
    plan_path, csv_path, svg_path, html_path = save_scene_artifacts(
        args.output_dir,
        args.scene_id,
        plan,
    )

    rendered_paths = render_audio_for_sources(
        scene_glb=scene_glb,
        scene_dir=args.scene_dir,
        scene_dataset_config=args.scene_dataset_config,
        output_dir=args.output_dir,
        materials_json=materials_json,
        scene_id=args.scene_id,
        sources=plan.sources,
        render_steps=args.render_steps,
    )

    print(f"Saved source plan: {plan_path}")
    print(f"Saved source CSV: {csv_path}")
    print(f"Saved source map: {svg_path}")
    print(f"Saved source map HTML: {html_path}")
    for path in rendered_paths:
        print(f"Saved binaural render: {path}")
    print(
        "If you open the WAVs with headphones, you should hear the sources separated in space.\n"
        "Next step is to replace the synthesized stand-ins with real tap / washer / fan samples."
    )
    return 0


def main() -> int:
    args = parse_args()
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
