"""Account storage, opaque cookie sessions, and per-user watchlists."""

import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pymysql
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

COOKIE = "stock_session"
SESSION_SECONDS = 7 * 24 * 3600
SCHEMA = (
    "CREATE TABLE IF NOT EXISTS account_users (id VARCHAR(32) PRIMARY KEY, username VARCHAR(32) NOT NULL UNIQUE, password_hash VARCHAR(256) NOT NULL)",
    "CREATE TABLE IF NOT EXISTS account_sessions (token_hash VARCHAR(64) PRIMARY KEY, user_id VARCHAR(32) NOT NULL, expires BIGINT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS account_watchlist (user_id VARCHAR(32) NOT NULL, ticker VARCHAR(16) NOT NULL, created BIGINT NOT NULL, PRIMARY KEY(user_id, ticker))",
    "CREATE TABLE IF NOT EXISTS account_attempts (bucket VARCHAR(64) PRIMARY KEY, attempts INTEGER NOT NULL, expires BIGINT NOT NULL)",
)


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_]+$")
    password: str = Field(min_length=1)


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    value = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=5, dklen=32)
    return f"scrypt${salt}${value.hex()}"


def password_matches(password, stored):
    _, salt, _ = stored.split("$")
    return hmac.compare_digest(password_hash(password, salt), stored)


class AccountStore:
    def __init__(self):
        self.backend = os.getenv("AUTH_DB_BACKEND", "sqlite")
        self.production = os.getenv("AUTH_PRODUCTION") == "1" or os.getenv("RENDER") == "true"
        self.path = Path(os.getenv("AUTH_SQLITE_PATH", "data/accounts/accounts.sqlite3"))
        self.ready = False
        self.lock = threading.Lock()

    @contextmanager
    def connection(self):
        if self.backend not in {"sqlite", "mysql"} or (self.production and self.backend != "mysql"):
            raise HTTPException(503, "계정 저장소가 아직 준비되지 않았습니다.")
        connection = None
        try:
            if self.backend == "mysql":
                if not os.getenv("AUTH_MYSQL_HOST") or not os.getenv("AUTH_MYSQL_PASSWORD"):
                    raise HTTPException(503, "계정 저장소가 아직 준비되지 않았습니다.")
                connection = pymysql.connect(
                    host=os.environ["AUTH_MYSQL_HOST"], port=int(os.getenv("AUTH_MYSQL_PORT", "3306")),
                    user=os.getenv("AUTH_MYSQL_USER", "stock_accounts"), password=os.environ["AUTH_MYSQL_PASSWORD"],
                    database=os.getenv("AUTH_MYSQL_DATABASE", "stock_risk_project"), charset="utf8mb4",
                    connect_timeout=5, read_timeout=5, write_timeout=5,
                    ssl_ca=os.getenv("AUTH_MYSQL_SSL_CA"),
                    ssl_verify_cert=self.production, ssl_verify_identity=self.production,
                    ssl_disabled=not self.production and os.getenv("AUTH_MYSQL_TLS", "1") == "0",
                )
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(self.path, timeout=10)
            with self.lock:
                if not self.ready:
                    for statement in SCHEMA:
                        self.execute(connection, statement)
                    connection.commit()
                    self.ready = True
            yield connection
            connection.commit()
        except (sqlite3.IntegrityError, pymysql.IntegrityError):
            if connection:
                connection.rollback()
            raise HTTPException(409, "이미 사용 중인 아이디입니다.") from None
        except (sqlite3.Error, pymysql.MySQLError):
            if connection:
                connection.rollback()
            raise HTTPException(503, "계정 저장소에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.") from None
        finally:
            if connection:
                connection.close()

    def execute(self, connection, sql, params=()):
        cursor = connection.cursor()
        cursor.execute(sql.replace("?", "%s") if self.backend == "mysql" else sql, params)
        rows = cursor.fetchall() if cursor.description else []
        cursor.close()
        return rows

    def throttle(self, request, username):
        now = int(time.time())
        # Atomic upserts count attempts across processes, including unknown users.
        keys = [("ip:" + (request.client.host if request.client else "unknown"), 40), ("user:" + username, 15)]
        denied = False
        with self.connection() as connection:
            self.execute(connection, "DELETE FROM account_attempts WHERE expires <= ?", (now,))
            for key, limit in keys:
                bucket = hashlib.sha256(key.encode()).hexdigest()
                suffix = (" ON DUPLICATE KEY UPDATE attempts=attempts+1" if self.backend == "mysql"
                          else " ON CONFLICT(bucket) DO UPDATE SET attempts=attempts+1")
                self.execute(connection, "INSERT INTO account_attempts VALUES (?, 1, ?)" + suffix, (bucket, now + 600))
                attempts = self.execute(connection, "SELECT attempts FROM account_attempts WHERE bucket=?", (bucket,))[0][0]
                denied = denied or attempts > limit
        if denied:
            raise HTTPException(429, "시도가 너무 많습니다. 10분 후 다시 시도해 주세요.", headers={"Retry-After": "600"})


def create_account_router(stock_lookup):
    if os.getenv("AUTH_DB_BACKEND") == "firebase":
        from database.firebase_accounts import create_firebase_router
        return create_firebase_router(stock_lookup)
    store = AccountStore()
    router = APIRouter(prefix="/api")
    dummy_hash = password_hash(secrets.token_urlsafe(24))
    secure = store.production or os.getenv("AUTH_COOKIE_SECURE") == "1"

    def require_write(request: Request):
        origins = {str(request.base_url).rstrip("/")}
        origins.update(filter(None, os.getenv("AUTH_ALLOWED_ORIGINS", "").split(",")))
        if not store.production:
            origins.update({"http://localhost:5173", "http://127.0.0.1:5173"})
        origin = request.headers.get("origin")
        if request.headers.get("x-requested-with") != "stock-risk" or (origin and origin not in origins):
            raise HTTPException(403, "허용되지 않은 요청입니다.")

    def user_for(request):
        token = request.cookies.get(COOKIE)
        if not token or len(token) > 128:
            return None
        with store.connection() as connection:
            rows = store.execute(connection,
                "SELECT u.id, u.username FROM account_users u JOIN account_sessions s ON s.user_id=u.id WHERE s.token_hash=? AND s.expires>?",
                (hashlib.sha256(token.encode()).hexdigest(), int(time.time())))
        return {"id": rows[0][0], "username": rows[0][1]} if rows else None

    def require_user(request: Request):
        user = user_for(request)
        if not user:
            raise HTTPException(401, "로그인이 필요합니다.")
        return user

    def start_session(connection, request, response, user_id):
        old = request.cookies.get(COOKIE, "")
        now = int(time.time())
        store.execute(connection, "DELETE FROM account_sessions WHERE expires<=? OR token_hash=?",
                      (now, hashlib.sha256(old.encode()).hexdigest()))
        token = secrets.token_urlsafe(32)
        store.execute(connection, "INSERT INTO account_sessions VALUES (?, ?, ?)",
                      (hashlib.sha256(token.encode()).hexdigest(), user_id, now + SESSION_SECONDS))
        response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, httponly=True, secure=secure, samesite="lax", path="/")
        response.headers["Cache-Control"] = "no-store"

    @router.get("/auth/me")
    def me(request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        # Verify storage even when no session exists so the UI can disable signup.
        try:
            with store.connection():
                pass
            return {"user": user_for(request), "available": True}
        except HTTPException as error:
            if error.status_code != 503:
                raise
            return {"user": None, "available": False, "message": error.detail}

    @router.post("/auth/register", dependencies=[Depends(require_write)], status_code=201)
    def register(body: Credentials, request: Request, response: Response):
        username = body.username.lower()
        store.throttle(request, username)
        user = {"id": secrets.token_hex(16), "username": username}
        hashed = password_hash(body.password)
        with store.connection() as connection:
            store.execute(connection, "INSERT INTO account_users VALUES (?, ?, ?)", (user["id"], username, hashed))
            start_session(connection, request, response, user["id"])
        return {"user": user}

    @router.post("/auth/login", dependencies=[Depends(require_write)])
    def login(body: Credentials, request: Request, response: Response):
        username = body.username.lower()
        store.throttle(request, username)
        with store.connection() as connection:
            rows = store.execute(connection, "SELECT id, password_hash FROM account_users WHERE username=?", (username,))
            matches = password_matches(body.password, rows[0][1] if rows else dummy_hash)
            if not rows or not matches:
                raise HTTPException(401, "아이디 또는 비밀번호가 올바르지 않습니다.")
            user = {"id": rows[0][0], "username": username}
            start_session(connection, request, response, user["id"])
        return {"user": user}

    @router.post("/auth/logout", dependencies=[Depends(require_write)])
    def logout(request: Request, response: Response):
        token = request.cookies.get(COOKIE, "")
        with store.connection() as connection:
            store.execute(connection, "DELETE FROM account_sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))
        response.delete_cookie(COOKIE, path="/", secure=secure, httponly=True, samesite="lax")
        response.headers["Cache-Control"] = "no-store"
        return {"ok": True}

    @router.get("/watchlist")
    def watchlist(response: Response, user=Depends(require_user)):
        response.headers["Cache-Control"] = "no-store"
        with store.connection() as connection:
            rows = store.execute(connection, "SELECT ticker FROM account_watchlist WHERE user_id=? ORDER BY created DESC, ticker", (user["id"],))
        return {"items": [stock_lookup(row[0]) or {"ticker": row[0], "name": row[0], "code": row[0].split('.')[0], "unavailable": True} for row in rows]}

    @router.put("/watchlist/{ticker}", dependencies=[Depends(require_write)])
    def add_watch(ticker: str, user=Depends(require_user)):
        stock = stock_lookup(ticker)
        if not stock:
            raise HTTPException(404, "등록되지 않은 종목입니다.")
        with store.connection() as connection:
            suffix = " ON DUPLICATE KEY UPDATE ticker=ticker" if store.backend == "mysql" else " ON CONFLICT(user_id, ticker) DO NOTHING"
            store.execute(connection, "INSERT INTO account_watchlist VALUES (?, ?, ?)" + suffix, (user["id"], ticker, int(time.time())))
        return {"item": stock}

    @router.delete("/watchlist/{ticker}", dependencies=[Depends(require_write)])
    def delete_watch(ticker: str, user=Depends(require_user)):
        with store.connection() as connection:
            store.execute(connection, "DELETE FROM account_watchlist WHERE user_id=? AND ticker=?", (user["id"], ticker))
        return {"ok": True}

    return router
