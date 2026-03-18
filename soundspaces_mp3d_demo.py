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
import json
import wave
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoundSpaces 2.0 MP3D demo")
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--scene-id", default=SCENE_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--max-sources", type=int, default=5)
    parser.add_argument("--render-steps", type=int, default=4)
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


def source_priority(object_name: str) -> int:
    lowered = object_name.lower()
    for idx, spec in enumerate(SOUND_LIBRARY):
        if any(keyword in lowered for keyword in spec["keywords"]):
            return idx
    return 999


def object_label(obj: object) -> str:
    for attr in ("category", "category_name", "semantic_id", "id"):
        value = getattr(obj, attr, None)
        if value is None:
            continue
        if hasattr(value, "name"):
            return str(value.name)
        return str(value)
    return "unknown"


def object_aabb_center(obj: object) -> list[float] | None:
    aabb = getattr(obj, "aabb", None)
    if aabb is None:
        return None
    center = getattr(aabb, "center", None)
    if center is None:
        center = getattr(aabb, "center_", None)
    if center is None and hasattr(aabb, "min") and hasattr(aabb, "max"):
        mn = np.array(aabb.min, dtype=np.float32)
        mx = np.array(aabb.max, dtype=np.float32)
        center = (mn + mx) / 2.0
    if center is None:
        return None
    return [float(x) for x in np.array(center, dtype=np.float32).tolist()]


def discover_scene_sources(sim, max_sources: int) -> list[PlacedSource]:
    semantic_scene = getattr(sim, "semantic_scene", None)
    candidates: list[tuple[int, str, list[float]]] = []

    if semantic_scene is not None:
        levels = getattr(semantic_scene, "levels", [])
        for level in levels:
            for obj in getattr(level, "objects", []):
                name = object_label(obj)
                center = object_aabb_center(obj)
                if center is None:
                    continue
                score = source_priority(name)
                candidates.append((score, name, center))

        # Some builds expose objects directly on the scene.
        for obj in getattr(semantic_scene, "objects", []):
            name = object_label(obj)
            center = object_aabb_center(obj)
            if center is None:
                continue
            score = source_priority(name)
            candidates.append((score, name, center))

    candidates.sort(key=lambda item: (item[0], item[1]))

    placed: list[PlacedSource] = []
    used_positions: set[tuple[int, int, int]] = set()
    for score, name, center in candidates:
        if len(placed) >= max_sources:
            break
        key = tuple(int(round(v * 100)) for v in center)
        if key in used_positions:
            continue
        used_positions.add(key)
        library_idx = min(score, len(SOUND_LIBRARY) - 1)
        clip_name = SOUND_LIBRARY[library_idx]["name"]
        placed.append(
            PlacedSource(
                label=clip_name,
                object_name=name,
                position=center,
                audio_clip=str(DEFAULT_AUDIO_DIR / f"{clip_name}.wav"),
            )
        )

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


def build_habitat_sim(scene_glb: Path, scene_dir: Path):
    import habitat_sim
    from habitat_sim.sim import SimulatorConfiguration

    sim_cfg = SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = str(scene_glb)
    sim_cfg.scene_dataset_config_file = ""
    sim_cfg.load_semantic_mesh = True
    sim_cfg.allow_sliding = True
    sim_cfg.create_renderer = True
    sim_cfg.enable_physics = False

    acoustic_cfg = habitat_sim.RLRAudioPropagationConfiguration()
    acoustic_cfg.sampleRate = DEFAULT_SAMPLE_RATE
    acoustic_cfg.frequencyBands = 4
    acoustic_cfg.directSHOrder = 3
    acoustic_cfg.indirectSHOrder = 1
    acoustic_cfg.threadCount = 1
    acoustic_cfg.updateDt = 0.02
    acoustic_cfg.irTime = 4.0
    acoustic_cfg.unitScale = 1.0
    acoustic_cfg.globalVolume = 0.25
    acoustic_cfg.indirectRayCount = 2500
    acoustic_cfg.indirectRayDepth = 100
    acoustic_cfg.sourceRayCount = 200
    acoustic_cfg.sourceRayDepth = 10
    acoustic_cfg.maxDiffractionOrder = 8
    acoustic_cfg.direct = True
    acoustic_cfg.indirect = True
    acoustic_cfg.diffraction = True
    acoustic_cfg.transmission = True
    acoustic_cfg.meshSimplification = False
    acoustic_cfg.temporalCoherence = False
    acoustic_cfg.dumpWaveFiles = False
    acoustic_cfg.enableMaterials = True
    acoustic_cfg.writeIrToFile = False

    channel_layout = habitat_sim.RLRAudioPropagationChannelLayout()
    channel_layout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    channel_layout.channelCount = 2

    audio_spec = habitat_sim.AudioSensorSpec()
    audio_spec.uuid = "audio_sensor"
    audio_spec.sensor_type = habitat_sim.SensorType.AUDIO
    audio_spec.position = [0.0, DEFAULT_LISTENER_HEIGHT, 0.0]
    audio_spec.enableMaterials = DEFAULT_MATERIALS_JSON.exists()
    audio_spec.acousticsConfig = acoustic_cfg
    audio_spec.channelLayout = channel_layout
    audio_spec.outputDirectory = str(scene_dir)

    sim = habitat_sim.Simulator(sim_cfg)
    try:
        sim.add_sensor(audio_spec)
    except Exception:
        # Older/newer Habitat-Sim builds vary slightly here; fall back to the
        # agent-based sensor attachment when direct registration is unavailable.
        from habitat_sim.agent import AgentConfiguration

        agent_cfg = AgentConfiguration()
        agent_cfg.sensor_specifications = [audio_spec]
        sim.add_agent(agent_cfg)
    if hasattr(sim, "pathfinder") and hasattr(sim.pathfinder, "load_nav_mesh"):
        navmesh = scene_dir / f"{SCENE_ID}.navmesh"
        if navmesh.exists():
            try:
                sim.pathfinder.load_nav_mesh(str(navmesh))
            except Exception:
                pass

    return sim


def attach_audio_materials(audio_sensor: object, materials_json: Path) -> None:
    if materials_json.exists() and hasattr(audio_sensor, "setAudioMaterialsJSON"):
        audio_sensor.setAudioMaterialsJSON(str(materials_json))


def print_scene_plan(scene_paths: dict[str, Path], sources: Sequence[PlacedSource]) -> None:
    print("Scene assets:")
    for key, path in scene_paths.items():
        print(f"  {key}: {path} {'(missing)' if not path.exists() else ''}")
    print("Planned audio sources:")
    for idx, src in enumerate(sources, 1):
        pos = ", ".join(f"{v:.3f}" for v in src.position)
        print(f"  {idx}. {src.label} on {src.object_name} at [{pos}] -> {src.audio_clip}")


def save_plan(output_dir: Path, scene_id: str, sources: Sequence[PlacedSource]) -> Path:
    ensure_dirs(output_dir)
    plan_path = output_dir / f"{scene_id}_audio_plan.json"
    with plan_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(src) for src in sources], f, indent=2)
    return plan_path


def run_demo(args: argparse.Namespace) -> int:
    scene_paths = scene_assets(args.scene_dir, args.scene_id)
    ensure_dirs(args.output_dir, args.audio_dir)
    clips = ensure_source_clips(args.audio_dir, DEFAULT_SAMPLE_RATE)
    materials_json = DEFAULT_MATERIALS_JSON

    if args.dry_run:
        sources = []
        for i, spec in enumerate(SOUND_LIBRARY[: args.max_sources]):
            sources.append(
                PlacedSource(
                    label=spec["name"],
                    object_name=f"dry_run_object_{i}",
                    position=[1.0 + i * 0.5, DEFAULT_LISTENER_HEIGHT, 1.0 + i * 0.25],
                    audio_clip=str(clips[spec["name"]]),
                )
            )
        print_scene_plan(scene_paths, sources)
        save_plan(args.output_dir, args.scene_id, sources)
        return 0

    try:
        import habitat_sim  # noqa: F401
    except Exception as exc:
        print(
            "habitat_sim is not installed in this environment, so I can only do a dry-run.\n"
            f"Import error: {exc}\n"
            "Install Habitat-Sim + SoundSpaces 2.0, then rerun without --dry-run."
        )
        sources = []
        for i, spec in enumerate(SOUND_LIBRARY[: args.max_sources]):
            sources.append(
                PlacedSource(
                    label=spec["name"],
                    object_name=f"dry_run_object_{i}",
                    position=[1.0 + i * 0.5, DEFAULT_LISTENER_HEIGHT, 1.0 + i * 0.25],
                    audio_clip=str(clips[spec["name"]]),
                )
            )
        print_scene_plan(scene_paths, sources)
        save_plan(args.output_dir, args.scene_id, sources)
        return 1

    if not getattr(habitat_sim, "audio_enabled", False):
        print(
            "Habitat-Sim is installed, but it was built without audio support on this machine.\n"
            "This Mac can run the scene and source planning step, but not the SoundSpaces 2.0\n"
            "live binaural render path because the upstream audio propagation library is Linux x64 only."
        )
        sources = []
        for i, spec in enumerate(SOUND_LIBRARY[: args.max_sources]):
            sources.append(
                PlacedSource(
                    label=spec["name"],
                    object_name=f"dry_run_object_{i}",
                    position=[1.0 + i * 0.5, DEFAULT_LISTENER_HEIGHT, 1.0 + i * 0.25],
                    audio_clip=str(clips[spec["name"]]),
                )
            )
        print_scene_plan(scene_paths, sources)
        save_plan(args.output_dir, args.scene_id, sources)
        return 0

    missing = [name for name, path in scene_paths.items() if name in {"glb", "semantic", "house", "navmesh"} and not path.exists()]
    if "glb" in missing:
        print(f"Missing scene GLB at {scene_paths['glb']}")
        return 1

    sim = build_habitat_sim(scene_paths["glb"], args.scene_dir)
    try:
        sources = discover_scene_sources(sim, args.max_sources)
        if len(sources) < min(4, args.max_sources):
            fallback = fallback_sources(sim, args.max_sources)
            if len(sources) < args.max_sources:
                sources.extend(fallback[: args.max_sources - len(sources)])
        print_scene_plan(scene_paths, sources)
        plan_path = save_plan(args.output_dir, args.scene_id, sources)

        # Collect a few listener positions so we can hear the scene from multiple viewpoints.
        pathfinder = getattr(sim, "pathfinder", None)
        listener_positions: list[list[float]] = []
        if pathfinder is not None and hasattr(pathfinder, "get_random_navigable_point"):
            for _ in range(args.render_steps):
                try:
                    p = pathfinder.get_random_navigable_point()
                    listener_positions.append([float(p[0]), float(p[1]), float(p[2])])
                except Exception:
                    break
        if not listener_positions:
            listener_positions = [[1.0, DEFAULT_LISTENER_HEIGHT, 1.0 + 0.5 * i] for i in range(args.render_steps)]

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
            audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
            attach_audio_materials(audio_sensor, materials_json)
            for source in sources:
                audio_sensor.setAudioSourceTransform(np.array(source.position, dtype=np.float32))
                obs = sim.get_sensor_observations()["audio_sensor"]
                ir = np.asarray(obs, dtype=np.float32)
                if ir.ndim == 1:
                    ir = ir[None, :]
                irs.append(ir)
                spec = next(spec for spec in SOUND_LIBRARY if spec["name"] == source.label)
                clips_for_mix.append(synth_clip(spec["kind"], float(spec["duration_s"]), DEFAULT_SAMPLE_RATE))

            mix = mix_binaural_sources(irs, clips_for_mix)
            out_path = args.output_dir / f"{args.scene_id}_listener_{step_idx:02d}.wav"
            write_wav(out_path, mix, DEFAULT_SAMPLE_RATE)
            rendered_paths.append(out_path)

        print(f"Saved source plan: {plan_path}")
        for path in rendered_paths:
            print(f"Saved binaural render: {path}")
        print(
            "If you open the WAVs with headphones, you should hear the sources separated in space.\n"
            "Next step is to replace the synthesized stand-ins with real tap / washer / fan samples."
        )
        return 0
    finally:
        try:
            sim.close()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
