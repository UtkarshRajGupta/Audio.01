#!/usr/bin/env python3
"""Preflight check for the SoundSpaces render host.

This repo can do the planning and source-map work on macOS, but the actual
SoundSpaces 2.0 binaural render path needs Linux x86_64. This helper gives a
quick yes/no style report for a Mac host, an Ubuntu VM, or a remote Linux box.
"""

from __future__ import annotations

import argparse
import platform
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class HostReport:
    system: str
    machine: str
    distro_id: str
    pretty_name: str
    is_linux: bool
    is_x86_64: bool
    is_ubuntu: bool
    soundspaces_ready: bool


def _load_os_release() -> dict[str, str]:
    if not hasattr(platform, "freedesktop_os_release"):
        return {}
    try:
        return dict(platform.freedesktop_os_release())
    except Exception:
        return {}


def detect_host_environment(
    system: str | None = None,
    machine: str | None = None,
    os_release: Mapping[str, str] | None = None,
) -> HostReport:
    """Return a small report for the current host."""
    system = system or platform.system()
    machine = machine or platform.machine()
    release = dict(os_release or _load_os_release())

    distro_id = release.get("ID", "").strip()
    pretty_name = release.get("PRETTY_NAME", "").strip()
    system_lower = system.lower()
    machine_lower = machine.lower()
    is_linux = system_lower == "linux"
    is_x86_64 = machine_lower in {"x86_64", "amd64"}
    is_ubuntu = distro_id.lower() == "ubuntu" or "ubuntu" in pretty_name.lower()
    soundspaces_ready = is_linux and is_x86_64

    return HostReport(
        system=system,
        machine=machine,
        distro_id=distro_id,
        pretty_name=pretty_name,
        is_linux=is_linux,
        is_x86_64=is_x86_64,
        is_ubuntu=is_ubuntu,
        soundspaces_ready=soundspaces_ready,
    )


def guidance_for_report(report: HostReport) -> str:
    if report.soundspaces_ready:
        if report.is_ubuntu:
            return "This Ubuntu x86_64 host is ready for the SoundSpaces render path."
        return "This Linux x86_64 host should be ready for the SoundSpaces render path."

    system_lower = report.system.lower()
    if system_lower == "darwin":
        return "Use an Ubuntu x86_64 VM on Mac, or a remote Linux x86_64 machine."
    if system_lower == "windows":
        return "Windows is fine for planning, but the render path still needs Linux x86_64."
    if report.is_linux and not report.is_x86_64:
        return "You are on Linux, but not x86_64. Use a Linux x86_64 VM or host."
    return "Use a Linux x86_64 host for the final SoundSpaces audio render."


def format_host_report(report: HostReport) -> str:
    distro = report.pretty_name or report.distro_id or "unknown"
    status = "ready" if report.soundspaces_ready else "not ready"
    lines = [
        "SoundSpaces host check",
        f"  system: {report.system}",
        f"  machine: {report.machine}",
        f"  distro: {distro}",
        f"  soundspaces_ready: {status}",
        f"  guidance: {guidance_for_report(report)}",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether this host can run SoundSpaces audio rendering.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when the host is not Linux x86_64.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = detect_host_environment()
    print(format_host_report(report))
    if args.strict and not report.soundspaces_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
