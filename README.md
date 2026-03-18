# Habitat + SoundSpaces MP3D Demo

This workspace now contains a single Matterport3D scene asset bundle:

- `data/mp3d/5LpN3gDmAk7.glb`
- `data/mp3d/5LpN3gDmAk7.house`
- `data/mp3d/5LpN3gDmAk7.navmesh`
- `data/mp3d/5LpN3gDmAk7_semantic.ply`

The main demo entrypoint is `soundspaces_mp3d_demo.py`.
`run_demo.sh` is a convenience launcher that sets up the Habitat env `PATH`
before invoking the demo.
`Makefile` provides one-word shortcuts for the common repo actions.

Other useful docs:

- `SETUP.md`
- `CONTRIBUTING.md`

## What it does

1. Loads the Matterport3D scene in Habitat-Sim when the simulator is available.
2. Tries to discover semantic objects in the room.
3. Places up to 5 audio sources on those objects.
4. Uses SoundSpaces-style binaural rendering to produce spatial audio.
5. Saves a source plan JSON and one or more rendered WAV files.
6. Writes a tiny top-down source-map visualizer with hover, click-to-pin, and fixture filters so you can inspect the layout.

If Habitat-Sim is not installed yet, the script still runs in `--dry-run` mode
and generates a source plan plus synthetic stand-in audio clips.

## Quick Start

On macOS, this is the best path:

```bash
./run_demo.sh --plan-only
```

or:

```bash
make plan
```

That loads the scene, finds semantic room objects, and exports the
object-to-sound plan without trying to render binaural audio.

It also writes a tiny source-map visualizer:

- `outputs/5LpN3gDmAk7_source_map.svg`
- `outputs/5LpN3gDmAk7_source_map.html`

If you do not have Habitat installed yet, use:

```bash
./run_demo.sh --dry-run
```

or:

```bash
make dry-run
```

That will create:

- `assets/audio/*.wav`
- `outputs/5LpN3gDmAk7_audio_plan.json`
- `outputs/5LpN3gDmAk7_audio_plan.csv`
- `outputs/5LpN3gDmAk7_source_map.svg`
- `outputs/5LpN3gDmAk7_source_map.html`

## Real Habitat Run

After installing Habitat-Sim and SoundSpaces 2.0 on Linux x64, run:

```bash
./run_demo.sh
```

Expected outputs:

- `outputs/5LpN3gDmAk7_audio_plan.json`
- `outputs/5LpN3gDmAk7_audio_plan.csv`
- `outputs/5LpN3gDmAk7_source_map.svg`
- `outputs/5LpN3gDmAk7_source_map.html`
- `outputs/5LpN3gDmAk7_listener_00.wav`
- `outputs/5LpN3gDmAk7_listener_01.wav`
- `outputs/5LpN3gDmAk7_listener_02.wav`
- `outputs/5LpN3gDmAk7_listener_03.wav`

## Tests

The pure Python planning helpers are covered with unit tests:

```bash
./habitat-env/bin/python -m unittest discover -s tests
```

or:

```bash
make test
```

## Notes

- The current audio clips are synthetic stand-ins so the pipeline can be tested
  without external media assets.
- Once the simulator is live, the next improvement is to replace those clips
  with real tap, washer, fridge, fan, and kettle recordings.
- The source-placement logic is fixture-first: it prefers sink, washer, fridge,
  fan, kettle, lamp, chair, and table-like labels while skipping structural
  surfaces such as walls, floors, and ceilings. It still falls back to
  navigable points if the scene labels are sparse.
- `run_demo.sh` also skips the editable Habitat-Sim rebuild step so the local
  Mac workflow stays stable.
- True SoundSpaces 2.0 audio rendering requires the upstream RLRAudioPropagation
  binary, which is Linux x64 only. On this macOS arm64 machine, Habitat-Sim can
  run and the demo can plan sources, but the live binaural render path is not
  available without moving the stack to Linux.
