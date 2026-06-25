"""
Unit tests for src.summary — bullet parsing and the summarize flow (mocked LLM).

Run from the project root:

    python -m unittest tests.test_summary -v
"""
import unittest
from unittest import mock

from src import summary


# ---- Minimal fake of the Groq client (OpenAI-compatible shape) -------------- #
class _Msg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _Resp:
    def __init__(self, content):
        self.choices = [_Msg(content)]


class _Completions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return _Resp(self._content)


class FakeGroq:
    def __init__(self, content):
        self.chat = type("Chat", (), {"completions": _Completions(content)})


class TestParseBullets(unittest.TestCase):
    def test_dash_markers(self):
        self.assertEqual(summary._parse_bullets("- a\n- b\n- c"), ["a", "b", "c"])

    def test_mixed_markers(self):
        self.assertEqual(
            summary._parse_bullets("1. a\n* b\n• c\n2) d"), ["a", "b", "c", "d"]
        )

    def test_strips_preamble_when_markers_present(self):
        text = "Here are the key points:\n- first\n- second"
        self.assertEqual(summary._parse_bullets(text), ["first", "second"])

    def test_no_markers_keeps_lines(self):
        self.assertEqual(
            summary._parse_bullets("just one line\nand another"),
            ["just one line", "and another"],
        )

    def test_blank_lines_skipped(self):
        self.assertEqual(summary._parse_bullets("\n- a\n\n- b\n"), ["a", "b"])

    def test_unicode_devanagari(self):
        # Search/summary must be robust to non-Latin scripts.
        self.assertEqual(
            summary._parse_bullets("- नमस्ते दुनिया\n- यह एक परीक्षण है"),
            ["नमस्ते दुनिया", "यह एक परीक्षण है"],
        )


class TestSummarizeTranscript(unittest.TestCase):
    def test_parses_into_list(self):
        client = FakeGroq("- point one\n- point two\n- point three")
        with mock.patch.object(summary.auth, "get_groq_client", return_value=client):
            out = summary.summarize_transcript("some transcript text")
        self.assertEqual(out, ["point one", "point two", "point three"])

    def test_unicode_summary(self):
        client = FakeGroq("- मुख्य बात एक\n- मुख्य बात दो")
        with mock.patch.object(summary.auth, "get_groq_client", return_value=client):
            out = summary.summarize_transcript("हिंदी ट्रांसक्रिप्ट")
        self.assertEqual(out, ["मुख्य बात एक", "मुख्य बात दो"])

    def test_empty_transcript_raises(self):
        with self.assertRaises(summary.SummaryError):
            summary.summarize_transcript("   ")

    def test_missing_key_raises_summary_error(self):
        with mock.patch.object(
            summary.auth, "get_groq_client",
            side_effect=summary.auth.AuthError("no key"),
        ):
            with self.assertRaises(summary.SummaryError):
                summary.summarize_transcript("text")


if __name__ == "__main__":
    unittest.main(verbosity=2)
