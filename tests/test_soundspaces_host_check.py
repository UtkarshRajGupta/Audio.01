from __future__ import annotations

import unittest

from soundspaces_host_check import (
    detect_host_environment,
    format_host_report,
    guidance_for_report,
)


class HostCheckTests(unittest.TestCase):
    def test_detect_host_environment_marks_macos_as_not_ready(self):
        report = detect_host_environment(
            system="Darwin",
            machine="arm64",
            os_release={"ID": "darwin", "PRETTY_NAME": "macOS 15"},
        )
        self.assertFalse(report.soundspaces_ready)
        self.assertFalse(report.is_linux)
        self.assertFalse(report.is_ubuntu)
        self.assertIn("Ubuntu x86_64 VM", guidance_for_report(report))

    def test_detect_host_environment_marks_ubuntu_x86_64_as_ready(self):
        report = detect_host_environment(
            system="Linux",
            machine="x86_64",
            os_release={"ID": "ubuntu", "PRETTY_NAME": "Ubuntu 24.04.2 LTS"},
        )
        self.assertTrue(report.soundspaces_ready)
        self.assertTrue(report.is_linux)
        self.assertTrue(report.is_x86_64)
        self.assertTrue(report.is_ubuntu)
        self.assertIn("ready for the SoundSpaces render path", guidance_for_report(report))

    def test_format_host_report_is_readable(self):
        report = detect_host_environment(
            system="Linux",
            machine="x86_64",
            os_release={"ID": "ubuntu", "PRETTY_NAME": "Ubuntu 24.04.2 LTS"},
        )
        text = format_host_report(report)
        self.assertIn("SoundSpaces host check", text)
        self.assertIn("soundspaces_ready: ready", text)
        self.assertIn("Ubuntu 24.04.2 LTS", text)


if __name__ == "__main__":
    unittest.main()
