"""Firebase Auth sessions and server-authorized Firestore watchlists."""

import hashlib
import os
import re
import time
from contextlib import contextmanager
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from database.firebase_client import WEB_CONFIG, firebase_app, firestore_db

COOKIE = "stock_firebase_session"
SESSION_SECONDS = 7 * 24 * 3600


class SessionInput(BaseModel):
    id_token: str = Field(min_length=1, max_length=16000)


@contextmanager
def firebase_errors():
    try:
        yield
    except HTTPException:
        raise
    except Exception:
        # Never expose credential paths, access tokens or provider error payloads.
        raise HTTPException(503, "Firebase 연결을 확인해 주세요. 잠시 후 다시 시도할 수 있습니다.") from None


def token_key(token):
    return hashlib.sha256(token.encode()).hexdigest()


class FirebaseAccounts:
    def ready(self):
        firestore_db().collection("app_metadata").document("health").get(timeout=5, retry=None)

    def claims(self, token, session=False):
        from firebase_admin import auth
        try:
            verify = auth.verify_session_cookie if session else auth.verify_id_token
            return verify(token, app=firebase_app(), check_revoked=True)
        except (auth.InvalidIdTokenError, auth.InvalidSessionCookieError,
                auth.RevokedIdTokenError, auth.RevokedSessionCookieError,
                auth.UserDisabledError, auth.UserNotFoundError, ValueError):
            raise HTTPException(401, "로그인이 만료되었습니다. 다시 로그인해 주세요.") from None

    def user_for(self, token):
        if not token or len(token) > 16000:
            return None
        try:
            claims = self.claims(token, session=True)
        except HTTPException as error:
            if error.status_code == 401:
                return None
            raise
        row = firestore_db().collection("account_sessions").document(token_key(token)).get(timeout=5, retry=None).to_dict()
        if not row or row.get("uid") != claims["uid"] or row.get("expires", 0) <= time.time():
            return None
        return {"id": claims["uid"], "username": claims.get("email", claims["uid"])}

    def start(self, token, old_token):
        from firebase_admin import auth
        claims = self.claims(token)
        if not 0 <= time.time() - claims.get("auth_time", 0) < 300:
            raise HTTPException(401, "다시 로그인해 주세요.")
        cookie = auth.create_session_cookie(token, expires_in=timedelta(seconds=SESSION_SECONDS), app=firebase_app())
        db = firestore_db()
        batch = db.batch()
        if old_token:
            batch.delete(db.collection("account_sessions").document(token_key(old_token)))
        batch.set(db.collection("account_sessions").document(token_key(cookie)), {
            "uid": claims["uid"], "expires": int(time.time()) + SESSION_SECONDS,
        })
        batch.commit(timeout=5, retry=None)
        return cookie, {"id": claims["uid"], "username": claims.get("email", claims["uid"])}

    def logout(self, token):
        if token:
            firestore_db().collection("account_sessions").document(token_key(token)).delete(timeout=5, retry=None)

    def items(self, uid):
        row = firestore_db().collection("account_watchlists").document(uid).get(timeout=5, retry=None).to_dict() or {}
        return row.get("tickers", [])

    def change(self, uid, ticker, remove):
        from google.cloud import firestore
        db = firestore_db()
        ref = db.collection("account_watchlists").document(uid)

        @firestore.transactional
        def update(transaction):
            values = (ref.get(transaction=transaction, timeout=5, retry=None).to_dict() or {}).get("tickers", [])
            if remove:
                values = [value for value in values if value != ticker]
            elif ticker not in values:
                if len(values) >= 100:
                    raise HTTPException(409, "관심종목은 최대 100개까지 저장할 수 있습니다.")
                values = [ticker, *values]
            transaction.set(ref, {"tickers": values})

        update(db.transaction(max_attempts=3))


def create_firebase_router(stock_lookup, store=None):
    store = store or FirebaseAccounts()
    router = APIRouter(prefix="/api")
    production = os.getenv("AUTH_PRODUCTION") == "1" or os.getenv("RENDER") == "true"
    secure = production or os.getenv("AUTH_COOKIE_SECURE") == "1"

    def require_write(request: Request):
        origins = {str(request.base_url).rstrip("/")}
        origins.update(x.strip() for x in os.getenv("AUTH_ALLOWED_ORIGINS", "").split(",") if x.strip())
        if not production:
            origins.update({"http://localhost:5173", "http://127.0.0.1:5173"})
        origin = request.headers.get("origin")
        if request.headers.get("x-requested-with") != "stock-risk" or (origin and origin not in origins):
            raise HTTPException(403, "허용되지 않은 요청입니다.")

    def require_user(request: Request):
        with firebase_errors():
            user = store.user_for(request.cookies.get(COOKIE))
        if not user:
            raise HTTPException(401, "로그인이 필요합니다.")
        return user

    @router.get("/auth/me")
    def me(request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        base = {"provider": "firebase", "firebase_config": WEB_CONFIG}
        try:
            with firebase_errors():
                store.ready()
                user = store.user_for(request.cookies.get(COOKIE))
            return {**base, "user": user, "available": True}
        except HTTPException as error:
            if error.status_code != 503:
                raise
            return {**base, "user": None, "available": False, "message": error.detail}

    @router.post("/auth/session", dependencies=[Depends(require_write)])
    def session(body: SessionInput, request: Request, response: Response):
        with firebase_errors():
            cookie, user = store.start(body.id_token, request.cookies.get(COOKIE))
        response.set_cookie(COOKIE, cookie, max_age=SESSION_SECONDS, httponly=True, secure=secure, samesite="lax", path="/")
        response.headers["Cache-Control"] = "no-store"
        return {"user": user}

    @router.post("/auth/logout", dependencies=[Depends(require_write)])
    def logout(request: Request, response: Response):
        with firebase_errors():
            store.logout(request.cookies.get(COOKIE))
        response.delete_cookie(COOKIE, path="/", secure=secure, httponly=True, samesite="lax")
        response.headers["Cache-Control"] = "no-store"
        return {"ok": True}

    @router.get("/watchlist")
    def watchlist(response: Response, user=Depends(require_user)):
        response.headers["Cache-Control"] = "no-store"
        with firebase_errors():
            tickers = store.items(user["id"])
        return {"items": [stock_lookup(ticker) or {"ticker": ticker, "name": ticker, "code": ticker.split('.')[0], "unavailable": True} for ticker in tickers]}

    def change(ticker, user, remove):
        if not re.fullmatch(r"[A-Za-z0-9^._-]{1,20}", ticker):
            raise HTTPException(404, "등록되지 않은 종목입니다.")
        stock = stock_lookup(ticker)
        if not remove and not stock:
            raise HTTPException(404, "등록되지 않은 종목입니다.")
        with firebase_errors():
            store.change(user["id"], ticker, remove)
        return {"ok": True} if remove else {"item": stock}

    @router.put("/watchlist/{ticker}", dependencies=[Depends(require_write)])
    def add(ticker: str, user=Depends(require_user)):
        return change(ticker, user, False)

    @router.delete("/watchlist/{ticker}", dependencies=[Depends(require_write)])
    def delete(ticker: str, user=Depends(require_user)):
        return change(ticker, user, True)

    return router
