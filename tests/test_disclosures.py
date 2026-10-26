import ast
import inspect
import unittest
from datetime import date
from unittest.mock import patch

import app as api
from fastapi.testclient import TestClient


class DisclosureTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def tearDown(self):
        self.client.close()

    def test_separate_endpoint(self):
        payload = {"available": True, "items": [{"report_name": "Test report"}]}
        with patch.object(api, "recent_disclosures", return_value=payload) as fetch:
            response = self.client.get("/api/disclosures/005930?base_date=2026-10-26&risk_probability=0.7")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), payload)
            fetch.assert_called_once_with("005930", date(2026, 10, 26), risk_probability=0.7)

    def test_invalid_inputs_do_not_call_provider(self):
        with patch.object(api, "recent_disclosures") as fetch:
            for url in ("/api/disclosures/invalid", "/api/disclosures/005930?base_date=bad", "/api/disclosures/005930?risk_probability=2"):
                self.assertEqual(self.client.get(url).status_code, 422)
            fetch.assert_not_called()

    def test_missing_key_and_disabled_states(self):
        with patch.object(api, "DART_ENABLED", True), patch.object(api, "dart_api_key", return_value=""):
            result = self.client.get("/api/disclosures/005930").json()
            self.assertFalse(result["available"])
            self.assertIn("DART_API_KEY", result["reason"])
        with patch.object(api, "DART_ENABLED", False):
            self.assertFalse(self.client.get("/api/disclosures/005930").json()["available"])

    def test_provider_errors_do_not_expose_credentials(self):
        with patch.object(api, "DART_ENABLED", True), patch.object(api, "dart_api_key", return_value="secret"), patch.object(api, "dart_corp_codes", side_effect=RuntimeError("url?crtfc_key=secret")):
            response = self.client.get("/api/disclosures/005930")
            self.assertNotIn("secret", response.text)
            self.assertFalse(response.json()["available"])

    def test_prediction_no_longer_calls_dart(self):
        tree = ast.parse(inspect.getsource(api.predict))
        names = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertNotIn("recent_disclosures", names)


if __name__ == "__main__":
    unittest.main()
