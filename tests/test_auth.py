"""
Unit tests for src.auth — API key resolution (env vs services.json).

Run from the project root:

    python -m unittest tests.test_auth -v
"""
import json
import os
import tempfile
import unittest
from unittest import mock

from src import auth


class TestResolveKey(unittest.TestCase):
    def setUp(self):
        auth._load_services.cache_clear()
        self.addCleanup(auth._load_services.cache_clear)

    def _services_file(self, data) -> str:
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(data, tmp)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def test_environment_wins(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "envkey"}, clear=False):
            self.assertEqual(auth._resolve_key("GROQ_API_KEY", "groq"), "envkey")

    def test_services_flat(self):
        path = self._services_file({"GROQ_API_KEY": "filekey"})
        with mock.patch.object(auth, "_SERVICES_JSON", path), \
                mock.patch.dict(os.environ, {}, clear=True):
            auth._load_services.cache_clear()
            self.assertEqual(auth._resolve_key("GROQ_API_KEY", "groq"), "filekey")

    def test_services_nested(self):
        path = self._services_file({"groq": {"api_key": "nestedkey"}})
        with mock.patch.object(auth, "_SERVICES_JSON", path), \
                mock.patch.dict(os.environ, {}, clear=True):
            auth._load_services.cache_clear()
            self.assertEqual(auth._resolve_key("GROQ_API_KEY", "groq"), "nestedkey")

    def test_missing_returns_none(self):
        path = self._services_file({"unrelated": "x"})
        with mock.patch.object(auth, "_SERVICES_JSON", path), \
                mock.patch.dict(os.environ, {}, clear=True):
            auth._load_services.cache_clear()
            self.assertIsNone(auth._resolve_key("GROQ_API_KEY", "groq"))

    def test_get_groq_client_raises_without_key(self):
        path = self._services_file({})
        with mock.patch.object(auth, "_SERVICES_JSON", path), \
                mock.patch.dict(os.environ, {}, clear=True):
            auth._load_services.cache_clear()
            with self.assertRaises(auth.AuthError):
                auth.get_groq_client()


if __name__ == "__main__":
    unittest.main(verbosity=2)
