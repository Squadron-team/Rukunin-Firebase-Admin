import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv

_app = None
_db = None

load_dotenv()


def init_firebase():
    global _app, _db

    if _app:
        return _db

    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set")

    cred = credentials.Certificate(cred_path)
    _app = firebase_admin.initialize_app(cred)
    _db = firestore.client()

    return _db


def get_db():
    if not _db:
        return init_firebase()
    return _db
