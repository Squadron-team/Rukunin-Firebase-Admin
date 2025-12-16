from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Literal


class UserSchema(BaseModel):
    # -----------------
    # Identity
    # -----------------
    name: str = Field(min_length=1)
    email: str
    role: Literal["resident", "admin", "treasurer", "secretary", "ketua_rt", "ketua_rw"]

    # -----------------
    # Audit fields
    # -----------------
    createdAt: datetime
    updatedAt: datetime

    # -----------------
    # Demographic / profile
    # -----------------
    nik: Optional[str] = None
    birthPlace: Optional[str] = None
    birthdate: Optional[datetime] = None
    gender: Optional[str] = None
    address: Optional[str] = None

    rt: Optional[str] = None
    rw: Optional[str] = None
    kelurahan: Optional[str] = None
    kecamatan: Optional[str] = None

    religion: Optional[str] = None
    maritalStatus: Optional[str] = None
    occupation: Optional[str] = None
    education: Optional[str] = None

    kkNumber: Optional[str] = None
    headOfFamily: Optional[str] = None
    relationToHead: Optional[str] = None

    onboardingCompleted: bool = False
