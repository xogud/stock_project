import hashlib
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from database.accounts import COOKIE, create_account_router

HEADERS = {"X-Requested-With": "stock-risk", "Origin": "http://testserver"}
PASSWORD = "test-password-2026"


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "accounts.sqlite3"
        self.environment = patch.dict(os.environ, {
            "AUTH_DB_BACKEND": "sqlite", "AUTH_SQLITE_PATH": str(self.path),
            "AUTH_PRODUCTION": "0", "RENDER": "false", "AUTH_COOKIE_SECURE": "0",
        })
        self.environment.start()
        self.client = self.make_client()

    def tearDown(self):
        self.client.close()
        self.environment.stop()
        self.directory.cleanup()

    def make_client(self):
        app = FastAPI()
        app.include_router(create_account_router(lambda ticker: {
            "ticker": ticker, "name": "Samsung", "code": "005930", "market": "KOSPI",
        } if ticker == "005930.KS" else None))
        return TestClient(app)

    def register(self, username="alice", client=None):
        client = client or self.client
        return client.post("/api/auth/register", headers=HEADERS, json={"username": username, "password": PASSWORD})

    def test_session_password_logout_and_restart(self):
        response = self.register()
        self.assertEqual(response.status_code, 201)
        self.assertIn("HttpOnly", response.headers["set-cookie"])
        self.assertIn("SameSite=lax", response.headers["set-cookie"])
        token = self.client.cookies.get(COOKIE)
        with closing(sqlite3.connect(self.path)) as db, db:
            hashed = db.execute("SELECT password_hash FROM account_users").fetchone()[0]
            self.assertNotIn(PASSWORD, hashed)
            self.assertEqual(db.execute("SELECT token_hash FROM account_sessions").fetchone()[0], hashlib.sha256(token.encode()).hexdigest())
        with self.make_client() as restarted:
            restarted.cookies.set(COOKIE, token)
            self.assertEqual(restarted.get("/api/auth/me").json()["user"]["username"], "alice")
            self.assertEqual(restarted.post("/api/auth/logout", headers=HEADERS).status_code, 200)
            restarted.cookies.set(COOKIE, token)
            self.assertEqual(restarted.get("/api/watchlist").status_code, 401)

    def test_watchlist_isolation_and_idempotency(self):
        self.assertEqual(self.client.get("/api/watchlist").status_code, 401)
        self.register()
        for _ in range(2):
            self.assertEqual(self.client.put("/api/watchlist/005930.KS", headers=HEADERS).status_code, 200)
        self.assertEqual(len(self.client.get("/api/watchlist").json()["items"]), 1)
        with self.make_client() as bob:
            self.register("bob", bob)
            self.assertEqual(bob.get("/api/watchlist").json()["items"], [])
            bob.delete("/api/watchlist/005930.KS", headers=HEADERS)
            self.assertEqual(len(self.client.get("/api/watchlist").json()["items"]), 1)
        with self.make_client() as returning:
            returning.post("/api/auth/login", headers=HEADERS, json={"username": "ALICE", "password": PASSWORD})
            self.assertEqual(len(returning.get("/api/watchlist").json()["items"]), 1)
            returning.delete("/api/watchlist/005930.KS", headers=HEADERS)
        self.assertEqual(self.client.get("/api/watchlist").json()["items"], [])

    def test_validation_duplicate_and_bad_login(self):
        self.register()
        self.assertEqual(self.register("ALICE").status_code, 409)
        self.assertEqual(self.client.put("/api/watchlist/FAKE", headers=HEADERS).status_code, 404)
        self.assertEqual(self.client.post("/api/auth/login", headers=HEADERS, json={"username": "alice", "password": "wrong-password"}).status_code, 401)
        self.assertEqual(self.client.post("/api/auth/register", headers=HEADERS, json={"username": "new_user", "password": "1234"}).status_code, 422)

    def test_csrf_and_expired_session(self):
        self.register()
        self.assertEqual(self.client.put("/api/watchlist/005930.KS").status_code, 403)
        self.assertEqual(self.client.put("/api/watchlist/005930.KS", headers={**HEADERS, "Origin": "https://evil.example"}).status_code, 403)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE account_sessions SET expires=0")
        self.assertIsNone(self.client.get("/api/auth/me").json()["user"])
        self.assertEqual(self.client.put("/api/watchlist/005930.KS", headers=HEADERS).status_code, 401)

    def test_rate_limit(self):
        self.client.get("/api/auth/me")
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO account_attempts VALUES (?, 15, 9999999999)", (hashlib.sha256(b"user:alice").hexdigest(),))
        self.assertEqual(self.register().status_code, 429)
        self.assertEqual(self.register().status_code, 429)

    def test_production_rejects_ephemeral_storage(self):
        with patch.dict(os.environ, {"AUTH_PRODUCTION": "1"}):
            with self.make_client() as production:
                self.assertFalse(production.get("/api/auth/me").json()["available"])
                self.assertEqual(self.register(client=production).status_code, 503)


if __name__ == "__main__":
    unittest.main()
