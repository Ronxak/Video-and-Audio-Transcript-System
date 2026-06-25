"""
Unit tests for src.translation — tolerant response parsing and the translate
flow, across the Groq (default) and Anthropic (backwards-compat) paths, with the
network client mocked.

Run from the project root:

    python -m unittest tests.test_translation -v
"""
import json
import unittest
from unittest import mock

from src import translation
from src.srt_utils import Cue


# --------------------------------------------------------------------------- #
# Fakes for the Groq client (OpenAI-compatible chat completions)
# --------------------------------------------------------------------------- #
class _GroqResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})})]


def _make_groq(create):
    """Build a fake Groq client whose .chat.completions.create == create."""
    cmp = type("Cmp", (), {"create": staticmethod(create)})
    chat = type("Chat", (), {"completions": cmp})
    return type("FakeGroq", (), {"chat": chat})()


def _user_lines(kwargs):
    """The list of source lines the code sent in the user message."""
    return json.loads(kwargs["messages"][-1]["content"])


# --------------------------------------------------------------------------- #
# _parse_translations — the heart of the bug fix
# --------------------------------------------------------------------------- #
class TestParseTranslations(unittest.TestCase):
    def test_bare_array(self):
        self.assertEqual(translation._parse_translations('["a", "b"]', 2), ["a", "b"])

    def test_translations_object(self):
        self.assertEqual(
            translation._parse_translations('{"translations": ["a", "b"]}', 2),
            ["a", "b"],
        )

    def test_singular_key_via_single_list_value(self):
        # This is the shape that caused the English-twice bug.
        self.assertEqual(
            translation._parse_translations('{"translation": ["a", "b"]}', 2),
            ["a", "b"],
        )

    def test_index_keyed_object(self):
        self.assertEqual(
            translation._parse_translations('{"0": "a", "1": "b"}', 2), ["a", "b"]
        )

    def test_chatty_text_with_embedded_json(self):
        self.assertEqual(
            translation._parse_translations('Sure! {"translations": ["a","b"]} ', 2),
            ["a", "b"],
        )

    def test_length_mismatch_returns_none(self):
        self.assertIsNone(translation._parse_translations('["a"]', 2))

    def test_unparseable_returns_none(self):
        self.assertIsNone(translation._parse_translations("nope", 2))


# --------------------------------------------------------------------------- #
# translate_cues — Groq path
# --------------------------------------------------------------------------- #
class TestTranslateGroq(unittest.TestCase):
    def _translate(self, create, cues, target="Hindi"):
        with mock.patch.object(translation.config, "LLM_PROVIDER", "groq"), \
                mock.patch.object(
                    translation.auth, "get_groq_client",
                    return_value=_make_groq(create)):
            translation.translate_cues(cues, target, model="x")
        return cues

    def test_translates_in_order(self):
        def create(**kw):
            lines = _user_lines(kw)
            return _GroqResp(json.dumps({"translations": ["t:" + s for s in lines]}))

        cues = self._translate(create, [Cue(0.0, 1.0, "hello"), Cue(1.0, 2.0, "world")])
        self.assertEqual([c.translation for c in cues], ["t:hello", "t:world"])

    def test_unicode_translation(self):
        def create(**kw):
            return _GroqResp(json.dumps({"translations": ["नमस्ते", "दुनिया"]},
                                        ensure_ascii=False))

        cues = self._translate(create, [Cue(0.0, 1.0, "hello"), Cue(1.0, 2.0, "world")])
        self.assertEqual([c.translation for c in cues], ["नमस्ते", "दुनिया"])

    def test_singular_key_shape_is_handled(self):
        # The reported bug: Groq returned a non-"translations" key → English twice.
        def create(**kw):
            lines = _user_lines(kw)
            return _GroqResp(json.dumps({"translation": ["t:" + s for s in lines]}))

        cues = self._translate(create, [Cue(0.0, 1.0, "hello"), Cue(1.0, 2.0, "world")])
        self.assertEqual([c.translation for c in cues], ["t:hello", "t:world"])

    def test_count_mismatch_self_heals_by_splitting(self):
        # Wrong count for multi-line requests, correct for single lines.
        def create(**kw):
            lines = _user_lines(kw)
            if len(lines) > 1:
                return _GroqResp(json.dumps({"translations": ["WRONG"]}))
            return _GroqResp(json.dumps({"translations": ["t:" + lines[0]]}))

        cues = self._translate(create, [Cue(0.0, 1.0, "hello"), Cue(1.0, 2.0, "world")])
        self.assertEqual([c.translation for c in cues], ["t:hello", "t:world"])

    def test_unparseable_keeps_originals(self):
        def create(**kw):
            return _GroqResp("I'm sorry, I can't do that.")

        cues = self._translate(create, [Cue(0.0, 1.0, "hello"), Cue(1.0, 2.0, "world")])
        self.assertEqual([c.translation for c in cues], ["hello", "world"])


# --------------------------------------------------------------------------- #
# translate_cues — Anthropic (backwards-compat) path
# --------------------------------------------------------------------------- #
class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class TestTranslateAnthropic(unittest.TestCase):
    def test_anthropic_path(self):
        def create(**kw):
            return type("R", (), {"content": [_Block('["bonjour"]')]})

        client = type("FakeAnthropic", (), {
            "messages": type("Msgs", (), {"create": staticmethod(create)})
        })()
        with mock.patch.object(translation.config, "LLM_PROVIDER", "anthropic"), \
                mock.patch.object(
                    translation.auth, "get_anthropic_client", return_value=client):
            cues = [Cue(0.0, 1.0, "hello")]
            translation.translate_cues(cues, "French", model="claude-x")
        self.assertEqual(cues[0].translation, "bonjour")


if __name__ == "__main__":
    unittest.main(verbosity=2)
