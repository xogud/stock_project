"""Server-only Firebase access. Credentials are never sent to the browser."""

import os
import threading
from functools import lru_cache

PROJECT_ID = "stockproject-736df"
WEB_CONFIG = {
    "apiKey": "AIzaSyAlACMQNCiOQTnuPBBbnC1AgEFR8LYizKk",
    "authDomain": "stockproject-736df.firebaseapp.com",
    "projectId": PROJECT_ID,
    "storageBucket": "stockproject-736df.firebasestorage.app",
    "messagingSenderId": "313619364749",
    "appId": "1:313619364749:web:c99e87257b0c2ddc451bc4",
}
_lock = threading.Lock()


@lru_cache(maxsize=1)
def firebase_app():
    import firebase_admin
    from firebase_admin import credentials

    # An explicit secret file avoids accidentally using another local project.
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not path:
        raise RuntimeError("Firebase service account is not configured")
    credential = credentials.Certificate(path)
    if credential.project_id != PROJECT_ID:
        raise RuntimeError("Firebase credential project mismatch")
    if os.getenv("RENDER") == "true" or os.getenv("AUTH_PRODUCTION") == "1":
        if os.getenv("FIRESTORE_EMULATOR_HOST") or os.getenv("FIREBASE_AUTH_EMULATOR_HOST"):
            raise RuntimeError("Emulators are not allowed in production")
    with _lock:
        try:
            return firebase_admin.get_app("stock-risk")
        except ValueError:
            return firebase_admin.initialize_app(credential, {"projectId": PROJECT_ID, "httpTimeout": 10}, name="stock-risk")


@lru_cache(maxsize=1)
def firestore_db():
    from firebase_admin import firestore
    return firestore.client(app=firebase_app())
