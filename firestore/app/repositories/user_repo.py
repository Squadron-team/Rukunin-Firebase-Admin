from app.core.firebase import get_db
from app.schemas.user import UserSchema

COLLECTION = "users"


def create_user(uid: str, user: UserSchema) -> None:
    db = get_db()
    
    if db is None:
        raise Exception("Database not loaded properly!")
    
    db.collection(COLLECTION).document(uid).set(user.model_dump(exclude_none=True))


def update_user(uid: str, data: dict) -> None:
    """
    Partial update (e.g. onboarding flow)
    """
    db = get_db()
    
    if db is None:
        raise Exception("Database not loaded properly!")
    
    db.collection(COLLECTION).document(uid).update(data)
