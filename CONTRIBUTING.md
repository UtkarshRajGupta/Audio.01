# Contributing

Thanks for helping improve this demo.

## What to keep in mind

- Keep the Matterport3D scene path and source plan workflow working.
- Avoid committing generated artifacts from `outputs/`, local environments, or third-party checkouts.
- If you touch the audio path, call out whether it works on macOS or requires Linux x64.

## Suggested workflow

1. Make your changes on a branch.
2. Run the dry-run demo:
   ```bash
   ./run_demo.sh --dry-run
   ```
3. Run the unit tests:
   ```bash
   ./habitat-env/bin/python -m unittest discover -s tests
   ```
4. If you have a Linux audio-enabled environment, test the live SoundSpaces path there too.
5. Keep commit messages short and descriptive.

## Helpful references

- `README.md`
- `SETUP.md`
- `soundspaces_mp3d_demo.py`
