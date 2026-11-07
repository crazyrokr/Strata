"""packaging/build_dsv41_hip_linux.sh: the one variable a Linux Strix Halo (gfx1151, Ryzen AI Max) package is built with.

No ROCm is on the machine that runs this test, so it reads the script's own VARIANT contract instead of running the build: `VARIANT=halo`
must fill ARCHS with gfx1151, exactly as the Qwen engine's `halo` variant does, and must not disturb the RDNA4 / RX 7000 builds.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "packaging" / "build_dsv41_hip_linux.sh"


class HaloVariant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_halo_fills_archs_with_gfx1151(self):
        self.assertRegex(self.text, r"(?m)^\s*halo\)\s+ARCHS=\$\{ARCHS:-gfx1151\}\s+;;\s*$")

    def test_the_other_variants_keep_their_archs(self):
        self.assertRegex(self.text, r"(?m)^\s*rx9070\)\s+ARCHS=\$\{ARCHS:-gfx1200;gfx1201\}\s+;;\s*$")
        self.assertRegex(self.text, r"(?m)^\s*rx7900\)\s+ARCHS=\$\{ARCHS:-gfx1100;gfx1101;gfx1102\}\s+;;\s*$")

    def test_the_variant_contract_names_halo(self):
        self.assertRegex(self.text, r"VARIANT=rx9070\|rx7900\|halo\|<name> with ARCHS=")


if __name__ == "__main__":
    unittest.main()
