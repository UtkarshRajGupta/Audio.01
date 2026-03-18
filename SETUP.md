# Setup

This repo currently contains a single Matterport3D scene asset bundle and a
SoundSpaces demo scaffold.

## Local run

```bash
./run_demo.sh --dry-run
```

This prints the scene assets and the planned 4-5 audio sources.

## Dependencies

- Python 3.10 Conda env at `./habitat-env`
- `Habitat-Sim` built from source in that env
- `SoundSpaces` sources in `third_party/sound-spaces`

## Important note

True SoundSpaces 2.0 binaural rendering requires the upstream
`RLRAudioPropagation` binary, which is Linux x64 only. On this macOS arm64
machine, the repo can still plan sources and run the dry-run demo, but the live
spatial audio render path is not available.
