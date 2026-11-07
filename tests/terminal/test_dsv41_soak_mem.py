"""tools/dsv41/soak_mem.py: the pure helpers of the memory soak (no server, no engine).  The harness itself drives a running
OpenAI-compatible server and samples /proc while it answers; those parts need a gfx1151 box, so only the rules below are tested here."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dsv41"))
import soak_mem  # noqa: E402

GIB = 1 << 30


class Reserve(unittest.TestCase):
    def test_an_explicit_reserve_wins(self):
        self.assertEqual(soak_mem.reserve_gib_from(6.0), 6.0)
        self.assertEqual(soak_mem.reserve_gib_from("12"), 12.0)
        self.assertEqual(soak_mem.reserve_gib_from(0.0), 0.0)

    def test_the_default_is_eight_gib(self):
        self.assertEqual(soak_mem.reserve_gib_from(None), 8.0)

    def test_a_large_pc_keeps_the_larger_eight_percent(self):
        self.assertAlmostEqual(soak_mem.reserve_gib_from(None, 128.0), 10.24, places=2)
        self.assertEqual(soak_mem.reserve_gib_from(None, 64.0), 8.0)      # 8% of 64 GiB is below the 8 GiB floor

    def test_a_negative_or_unreadable_reserve_is_refused(self):
        for bad in (-1, "x"):
            with self.assertRaises(ValueError):
                soak_mem.reserve_gib_from(bad)


class Summarize(unittest.TestCase):
    def test_the_floor_is_kept(self):
        s = soak_mem.summarize([10 * GIB, 9 * GIB, 11 * GIB], [], 8 * GIB)
        self.assertEqual(s["min_available"], 9 * GIB)
        self.assertIsNone(s["peak_rss_shmem"])
        self.assertTrue(s["kept"])

    def test_exactly_at_the_floor_is_kept(self):
        self.assertTrue(soak_mem.summarize([8 * GIB], [], 8 * GIB)["kept"])

    def test_a_crossed_floor_is_named(self):
        s = soak_mem.summarize([10 * GIB, 7 * GIB], [100 << 20, 300 << 20], 8 * GIB)
        self.assertEqual(s["min_available"], 7 * GIB)
        self.assertEqual(s["peak_rss_shmem"], 300 << 20)
        self.assertFalse(s["kept"])

    def test_no_reads_is_unknown_not_a_pass(self):
        s = soak_mem.summarize([], [], 8 * GIB)
        self.assertIsNone(s["min_available"])
        self.assertIsNone(s["kept"])


class PromptText(unittest.TestCase):
    def test_the_nonce_is_there_and_makes_each_prompt_cold(self):
        a = soak_mem.prompt_text(42000, "run-1")
        b = soak_mem.prompt_text(42000, "run-2")
        self.assertIn("run-1", a)
        self.assertNotEqual(a, b)

    def test_the_length_follows_the_token_count(self):
        for tokens in (1000, 42000, 200000):
            text = soak_mem.prompt_text(tokens, "x")
            want = tokens * soak_mem.CHARS_PER_TOKEN
            self.assertLessEqual(abs(len(text) - want), want * 0.1, tokens)

    def test_a_tiny_prompt_is_still_a_prompt(self):
        text = soak_mem.prompt_text(0, "zero")
        self.assertIn("zero", text)
        self.assertTrue(text.strip())


class Parsing(unittest.TestCase):
    def test_the_prompt_sizes(self):
        self.assertEqual(soak_mem.parse_prompts("42000,72000,100000,200000"), [42000, 72000, 100000, 200000])
        self.assertEqual(soak_mem.parse_prompts("1000"), [1000])

    def test_meminfo_gives_memavailable_in_bytes(self):
        text = "MemTotal:       131072000 kB\nMemFree:          1000000 kB\nMemAvailable:    65536000 kB\n"
        self.assertEqual(soak_mem.parse_meminfo_available(text), 65536000 * 1024)
        self.assertIsNone(soak_mem.parse_meminfo_available("MemTotal: 1 kB\n"))

    def test_status_gives_rssshmem_in_bytes(self):
        text = "Name:\tstrata-dsv41\nVmRSS:\t  200000 kB\nRssShmem:\t  300000 kB\n"
        self.assertEqual(soak_mem.parse_rss_shmem(text), 300000 * 1024)
        self.assertIsNone(soak_mem.parse_rss_shmem("Name:\tx\n"))

    def test_the_help_runs(self):
        with self.assertRaises(SystemExit) as caught:
            soak_mem.main(["--help"])
        self.assertEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
