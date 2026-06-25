"""
Unit tests for srt_utils — the pure, failure-prone subtitle logic.

Run from the project root:

    python -m unittest tests.test_srt_utils -v
"""
import unittest

from src.config import SubtitleStyle
from src.srt_utils import (
    Cue,
    compute_cps,
    cues_to_srt,
    cues_to_vtt,
    enforce_timing,
    format_timestamp,
    split_long_cues,
    wrap_text,
)

STYLE = SubtitleStyle()


class TestTimestamps(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(format_timestamp(0), "00:00:00,000")

    def test_negative_clamped(self):
        self.assertEqual(format_timestamp(-5), "00:00:00,000")

    def test_milliseconds_and_carry(self):
        self.assertEqual(format_timestamp(3661.5), "01:01:01,500")

    def test_vtt_uses_period(self):
        self.assertEqual(format_timestamp(1.25, vtt=True), "00:00:01.250")

    def test_srt_uses_comma(self):
        self.assertIn(",", format_timestamp(1.25))
        self.assertNotIn(".", format_timestamp(1.25))


class TestWrapping(unittest.TestCase):
    def test_short_unchanged(self):
        self.assertEqual(wrap_text("Hello world", 42, 2), "Hello world")

    def test_respects_line_length_when_feasible(self):
        # Text that genuinely fits within two 42-char lines must never overflow.
        text = "The quick brown fox jumps over the lazy dog while the sun sets"
        wrapped = wrap_text(text, 42, 2)
        self.assertLessEqual(len(wrapped.split("\n")), 2)
        for line in wrapped.split("\n"):
            self.assertLessEqual(len(line), 42)
        # And no words are lost in the process.
        self.assertEqual(wrapped.replace("\n", " ").split(), text.split())

    def test_never_exceeds_max_lines(self):
        text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        self.assertLessEqual(len(wrap_text(text, 12, 2).split("\n")), 2)

    def test_no_words_dropped(self):
        text = "one two three four five six seven eight nine ten eleven twelve"
        wrapped = wrap_text(text, 15, 2).replace("\n", " ")
        self.assertEqual(set(wrapped.split()), set(text.split()))

    def test_preserves_unicode(self):
        text = "नमस्ते दुनिया यह एक परीक्षण है"
        self.assertIn("नमस्ते", wrap_text(text, 42, 2))


class TestTiming(unittest.TestCase):
    def test_min_duration_extended(self):
        cues = [Cue(0.0, 0.2, "hi")]
        enforce_timing(cues, STYLE)
        self.assertAlmostEqual(cues[0].end, STYLE.min_duration, places=3)

    def test_max_duration_capped(self):
        cues = [Cue(0.0, 20.0, "long")]
        enforce_timing(cues, STYLE)
        self.assertAlmostEqual(cues[0].end, STYLE.max_duration, places=3)

    def test_negative_start_clamped(self):
        cues = [Cue(-2.0, 3.0, "x")]
        enforce_timing(cues, STYLE)
        self.assertGreaterEqual(cues[0].start, 0.0)

    def test_gap_enforced_between_cues(self):
        cues = [Cue(0.0, 2.0, "a"), Cue(2.0, 4.0, "b")]
        enforce_timing(cues, STYLE)
        self.assertLessEqual(cues[0].end, cues[1].start - STYLE.min_gap + 1e-6)

    def test_short_cue_does_not_collide_with_next(self):
        cues = [Cue(0.0, 0.1, "a"), Cue(0.5, 2.0, "b")]
        enforce_timing(cues, STYLE)
        self.assertLessEqual(cues[0].end, cues[1].start - STYLE.min_gap + 1e-6)


class TestSerialisation(unittest.TestCase):
    def setUp(self):
        self.cues = [
            Cue(0.0, 2.0, "First line", translation="पहली पंक्ति"),
            Cue(2.5, 4.0, "Second line", translation="दूसरी पंक्ति"),
        ]

    def test_srt_structure(self):
        srt = cues_to_srt(self.cues, STYLE)
        blocks = srt.strip().split("\n\n")
        self.assertEqual(len(blocks), 2)
        # First block: index, timestamp line, text.
        lines = blocks[0].split("\n")
        self.assertEqual(lines[0], "1")
        self.assertIn(" --> ", lines[1])
        self.assertIn(",", lines[1])  # SRT comma timestamps
        self.assertEqual(lines[2], "First line")

    def test_srt_sequential_indices(self):
        srt = cues_to_srt(self.cues, STYLE)
        self.assertTrue(srt.startswith("1\n"))
        self.assertIn("\n2\n", srt)

    def test_translated_srt_uses_translation(self):
        srt = cues_to_srt(self.cues, STYLE, use_translation=True)
        self.assertIn("पहली पंक्ति", srt)
        self.assertNotIn("First line", srt)

    def test_bilingual_srt_has_both(self):
        srt = cues_to_srt(self.cues, STYLE, bilingual=True)
        self.assertIn("First line", srt)
        self.assertIn("पहली पंक्ति", srt)

    def test_vtt_header_and_period(self):
        vtt = cues_to_vtt(self.cues, STYLE)
        self.assertTrue(vtt.startswith("WEBVTT"))
        # Find the timestamp line and confirm it uses VTT period separators.
        stamp_lines = [ln for ln in vtt.split("\n") if " --> " in ln]
        self.assertTrue(stamp_lines)
        self.assertIn(".", stamp_lines[0])
        self.assertNotIn(",", stamp_lines[0])


class TestSplitting(unittest.TestCase):
    def test_short_cue_unchanged(self):
        cues = [Cue(0.0, 2.0, "Short line")]
        out = split_long_cues(cues, STYLE)
        self.assertEqual(len(out), 1)

    def test_long_cue_is_split(self):
        long_text = (
            "This is a deliberately long sentence that cannot possibly fit inside "
            "two subtitle lines of forty-two characters each so it must be split "
            "into several consecutive cues by the formatter."
        )
        cues = [Cue(0.0, 12.0, long_text)]
        out = split_long_cues(cues, STYLE)
        self.assertGreater(len(out), 1)
        budget = STYLE.max_line_length * STYLE.max_lines
        for cue in out:
            self.assertLessEqual(len(cue.text), budget)

    def test_split_preserves_words_and_span(self):
        long_text = " ".join(f"word{i}" for i in range(60))
        cues = [Cue(1.0, 13.0, long_text)]
        out = split_long_cues(cues, STYLE)
        rejoined = " ".join(c.text for c in out).split()
        self.assertEqual(rejoined, long_text.split())          # no words lost
        self.assertAlmostEqual(out[0].start, 1.0, places=3)    # span preserved
        self.assertAlmostEqual(out[-1].end, 13.0, places=3)


class TestCps(unittest.TestCase):
    def test_cps_basic(self):
        self.assertAlmostEqual(compute_cps("abcde", 0.0, 1.0), 5.0, places=3)

    def test_cps_ignores_newlines(self):
        self.assertAlmostEqual(compute_cps("ab\ncd", 0.0, 1.0), 5.0, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
