# Setup

This repo currently contains a single Matterport3D scene asset bundle and a
SoundSpaces demo scaffold.

## Local run

On macOS, the recommended path is the scene-planning mode:

```bash
./run_demo.sh --plan-only
```

If you prefer a shortcut:

```bash
make plan
```

That loads the Matterport scene, finds semantic objects, and saves the
object-to-sound plan to `outputs/` as JSON and CSV, plus a source map SVG/HTML
pair you can open in a browser.

If you want a no-Habitat fallback:

```bash
./run_demo.sh --dry-run
```

Shortcut:

```bash
make dry-run
```

This prints the scene assets and the planned 4-5 audio sources.

## Verify

Run the lightweight unit tests with:

```bash
./habitat-env/bin/python -m unittest discover -s tests
```

or:

```bash
make test
```

## Dependencies

- Python 3.10 Conda env at `./habitat-env`
- `Habitat-Sim` built from source in that env
- `SoundSpaces` sources in `third_party/sound-spaces`

## Important note

True SoundSpaces 2.0 binaural rendering requires the upstream
`RLRAudioPropagation` binary, which is Linux x64 only. On this macOS arm64
machine, the repo can still plan sources and run the dry-run demo, but the live
spatial audio render path is not available.
