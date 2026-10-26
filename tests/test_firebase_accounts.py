import os
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from database.firebase_accounts import COOKIE, FirebaseAccounts, create_firebase_router
from database.firebase_client import WEB_CONFIG
from database.publish_firestore import build_snapshot
from database import stock_repository

HEADERS = {"X-Requested-With": "stock-risk", "Origin": "http://testserver"}


class FakeAccounts:
    def __init__(self):
        self.sessions = {}
        self.lists = {}

    def ready(self):
        pass

    def user_for(self, token):
        return self.sessions.get(token)

    def start(self, token, old):
        if token not in {"alice", "bob"}:
            raise HTTPException(401, "invalid")
        self.logout(old)
        cookie = f"session-{token}"
        user = {"id": token, "username": token + "@example.com"}
        self.sessions[cookie] = user
        return cookie, user

    def logout(self, token):
        self.sessions.pop(token, None)

    def items(self, uid):
        return self.lists.get(uid, [])

    def change(self, uid, ticker, remove):
        values = self.lists.setdefault(uid, [])
        if remove:
            self.lists[uid] = [value for value in values if value != ticker]
        elif ticker not in values:
            values.append(ticker)


class FirebaseRouteTests(unittest.TestCase):
    def setUp(self):
        self.store = FakeAccounts()
        self.app = FastAPI()
        self.env = patch.dict(os.environ, {"AUTH_PRODUCTION": "0", "RENDER": "false"})
        self.env.start()
        self.app.include_router(create_firebase_router(lambda ticker: {"ticker": ticker, "name": "Samsung"} if ticker == "005930.KS" else None, self.store))
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.env.stop()

    def login(self, client, uid):
        return client.post("/api/auth/session", headers=HEADERS, json={"id_token": uid})

    def test_config_and_session_cookie(self):
        config = self.client.get("/api/auth/me").json()
        self.assertEqual(config["firebase_config"], WEB_CONFIG)
        self.assertNotIn("private_key", str(config))
        response = self.login(self.client, "alice")
        self.assertEqual(response.status_code, 200)
        self.assertIn("HttpOnly", response.headers["set-cookie"])
        self.assertEqual(self.client.get("/api/auth/me").json()["user"]["id"], "alice")

    def test_isolation_idempotence_and_logout(self):
        self.login(self.client, "alice")
        for _ in range(2):
            self.assertEqual(self.client.put("/api/watchlist/005930.KS", headers=HEADERS).status_code, 200)
        with TestClient(self.app) as bob:
            self.login(bob, "bob")
            self.assertEqual(bob.get("/api/watchlist").json()["items"], [])
            bob.delete("/api/watchlist/005930.KS", headers=HEADERS)
        self.assertEqual(len(self.client.get("/api/watchlist").json()["items"]), 1)
        token = self.client.cookies.get(COOKIE)
        self.client.post("/api/auth/logout", headers=HEADERS)
        self.client.cookies.set(COOKIE, token)
        self.assertEqual(self.client.get("/api/watchlist").status_code, 401)
        self.login(self.client, "alice")
        self.assertEqual(len(self.client.get("/api/watchlist").json()["items"]), 1)
        self.client.delete("/api/watchlist/005930.KS", headers=HEADERS)
        self.assertEqual(self.client.get("/api/watchlist").json()["items"], [])

    def test_auth_csrf_and_validation(self):
        self.assertEqual(self.client.get("/api/watchlist").status_code, 401)
        self.assertEqual(self.login(self.client, "bad").status_code, 401)
        self.assertEqual(self.client.post("/api/auth/session", json={"id_token": "alice"}).status_code, 403)
        self.assertEqual(self.client.post("/api/auth/session", headers={**HEADERS, "Origin": "https://evil.example"}, json={"id_token": "alice"}).status_code, 403)
        self.login(self.client, "alice")
        self.assertEqual(self.client.put("/api/watchlist/FAKE", headers=HEADERS).status_code, 404)

    def test_storage_failure_is_redacted(self):
        with patch.object(self.store, "ready", side_effect=RuntimeError("private_key secret")):
            response = self.client.get("/api/auth/me")
            self.assertFalse(response.json()["available"])
            self.assertNotIn("private_key", response.text)

    def test_secure_cookie_in_production(self):
        with patch.dict(os.environ, {"AUTH_PRODUCTION": "1"}):
            app = FastAPI()
            app.include_router(create_firebase_router(lambda _: None, self.store))
            with TestClient(app, base_url="https://testserver") as client:
                response = client.post("/api/auth/session", headers={**HEADERS, "Origin": "https://testserver"}, json={"id_token": "alice"})
                self.assertIn("Secure", response.headers["set-cookie"])


class FirebaseStoreTests(unittest.TestCase):
    def test_recent_login_required(self):
        store = FirebaseAccounts()
        with patch.object(store, "claims", return_value={"uid": "alice", "auth_time": time.time() - 600}):
            with self.assertRaises(HTTPException) as caught:
                store.start("token", None)
            self.assertEqual(caught.exception.status_code, 401)

    def test_session_membership_and_expiration(self):
        store = FirebaseAccounts()
        db = MagicMock()
        document = db.collection.return_value.document.return_value.get.return_value
        with patch.object(store, "claims", return_value={"uid": "alice", "email": "alice@example.com"}), patch("database.firebase_accounts.firestore_db", return_value=db):
            for row in (None, {"uid": "bob", "expires": time.time() + 60}, {"uid": "alice", "expires": 0}):
                document.to_dict.return_value = row
                self.assertIsNone(store.user_for("cookie"))
            document.to_dict.return_value = {"uid": "alice", "expires": time.time() + 60}
            self.assertEqual(store.user_for("cookie")["id"], "alice")

    def test_snapshot_cache_and_fallback(self):
        stock_repository._cache.clear()
        db = MagicMock()
        get = db.collection.return_value.document.return_value.get
        get.return_value.to_dict.return_value = {"latest_date": "2026-10-23", "stocks": [], "predictions": []}
        with patch("database.stock_repository.firestore_db", return_value=db):
            stock_repository.snapshot()
            stock_repository.snapshot()
            self.assertEqual(get.call_count, 1)
        stock_repository._cache.clear()
        with patch("database.stock_repository.firestore_db", side_effect=RuntimeError("credentials")):
            with self.assertRaisesRegex(RuntimeError, "Firestore is unavailable"):
                stock_repository.snapshot()
        stock_repository._cache.clear()

    def test_real_snapshot_dry_run(self):
        payload, size = build_snapshot()
        self.assertLess(size, 700000)
        self.assertGreater(len(payload["stocks"]), 0)
        probabilities = [row["risk_probability"] for row in payload["predictions"]]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))


if __name__ == "__main__":
    unittest.main()
