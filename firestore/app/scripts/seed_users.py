import os
import uuid
from datetime import datetime, time
from faker import Faker
from app.schemas.user import UserSchema
from app.repositories.user_repo import create_user, update_user
from app.core.firebase import get_db

fake = Faker()


def generate_base_user(uid: str) -> UserSchema:
    """
    Simulates initial user creation (Flutter signup)
    """
    now = datetime.utcnow()

    return UserSchema(
        name=fake.name(),
        email=fake.email(),
        role="resident",
        createdAt=now,
        updatedAt=now,
        onboardingCompleted=False,
    )


def generate_onboarding_data() -> dict:
    """
    Simulates onboarding completion (Flutter profile update)
    """
    birth_date = fake.date_of_birth(minimum_age=18, maximum_age=70)
    
    return {
        "nik": fake.numerify("################"),
        "birthPlace": fake.city(),
        "birthdate": datetime.combine(birth_date, time.min),
        "gender": fake.random_element(["male", "female"]),
        "address": fake.address(),
        "rt": fake.numerify("0##"),
        "rw": fake.numerify("0##"),
        "kelurahan": fake.city_suffix(),
        "kecamatan": fake.city(),
        "religion": fake.random_element(
            ["Islam", "Christian", "Catholic", "Hindu", "Buddha"]
        ),
        "maritalStatus": fake.random_element(
            ["single", "married", "divorced"]
        ),
        "occupation": fake.job(),
        "education": fake.random_element(
            ["SD", "SMP", "SMA", "D3", "S1", "S2"]
        ),
        "kkNumber": fake.numerify("################"),
        "headOfFamily": fake.name(),
        "relationToHead": fake.random_element(
            ["self", "spouse", "child"]
        ),
        "onboardingCompleted": True,
        "updatedAt": datetime.utcnow(),
    }


def seed_users(count: int = 20, onboarding_ratio: float = 0.7) -> None:
    """
    Seeds Firestore users collection.

    onboarding_ratio:
        0.0 = no users completed onboarding
        1.0 = all users completed onboarding
    """
    if os.getenv("ENV") == "production":
        raise RuntimeError("❌ Seeding disabled in production")

    get_db()

    for _ in range(count):
        # Simulate Firebase Auth UID
        uid = uuid.uuid4().hex

        # Step 1: Create base user
        user = generate_base_user(uid)
        create_user(uid, user)

        # Step 2: Optionally complete onboarding
        if fake.random.random() < onboarding_ratio:
            onboarding_data = generate_onboarding_data()
            update_user(uid, onboarding_data)

    print(f"✅ Seeded {count} users (onboarding ratio={onboarding_ratio})")


if __name__ == "__main__":
    seed_users(count=5, onboarding_ratio=0.8)
