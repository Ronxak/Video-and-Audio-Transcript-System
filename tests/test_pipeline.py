"""
Integration test for the pipeline's new summarize wiring, with transcription
and the Groq client mocked (so it runs without models, ffmpeg, or an API key).

Run from the project root:

    python -m unittest tests.test_pipeline -v
"""
import os
import tempfile
import unittest
from unittest import mock

from src import pipeline, summary
from src.srt_utils import Cue


class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})})]


class FakeGroq:
    def __init__(self, content):
        create = lambda **kw: _Resp(content)  # noqa: E731
        self.chat = type("Chat", (), {"completions": type("Cmp", (), {"create": staticmethod(create)})})


class TestPipelineSummarize(unittest.TestCase):
    def setUp(self):
        # A tiny real file so process()'s existence/size checks pass.
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(b"\x00" * 32)
        tmp.close()
        self.media = tmp.name
        self.addCleanup(os.unlink, self.media)
        self.cues = [Cue(0.0, 2.0, "hello world"), Cue(2.0, 4.0, "second line here")]

    def test_summarize_populates_result(self):
        client = FakeGroq("- key point one\n- key point two")
        with mock.patch.object(
            pipeline.transcription, "transcribe",
            return_value=(self.cues, "en", 4.0),
        ), mock.patch.object(summary.auth, "get_groq_client", return_value=client):
            result = pipeline.process(self.media, summarize=True)
        self.assertEqual(result.summary, ["key point one", "key point two"])

    def test_no_summary_by_default(self):
        with mock.patch.object(
            pipeline.transcription, "transcribe",
            return_value=(self.cues, "en", 4.0),
        ):
            result = pipeline.process(self.media)
        self.assertIsNone(result.summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
