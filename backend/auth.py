"""
Authentication module - JWT with bcrypt password hashing.
In-memory user store (suitable for single-instance deployments).
"""

import os
import re
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from jose import jwt, JWTError
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

SECRET_KEY: str = os.getenv("SECRET_KEY") or secrets.token_urlsafe(48)
if SECRET_KEY in ("dev-secret-key-change-in-production",):
    SECRET_KEY = secrets.token_urlsafe(48)
    logger.warning("Insecure SECRET_KEY detected — generated a random one for this session.")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

bearer_scheme = HTTPBearer()

# { username -> {"hashed_password": str, "created_at": str} }
_users: dict[str, dict] = {}


# ── Pydantic models ──────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        v = v.strip().lower()
        if not (3 <= len(v) <= 50):
            raise ValueError("Username must be 3–50 characters")
        if not re.fullmatch(r"[a-z0-9_\-]+", v):
            raise ValueError("Username may only contain letters, digits, _ and -")
        return v

    @field_validator("password")
    @classmethod
    def password_strong(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Helpers ──────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(username: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {"sub": username, "exp": expire, "iat": datetime.utcnow()}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# ── FastAPI dependency ────────────────────────────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub", "")
        if not username:
            raise JWTError("missing sub")
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if username not in _users:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return {"username": username}


# ── Registration / Login logic (called from main.py) ─────────────────────────

def register_user(data: UserCreate) -> Token:
    if data.username in _users:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken")
    _users[data.username] = {
        "hashed_password": hash_password(data.password),
        "created_at": datetime.utcnow().isoformat(),
    }
    logger.info("New user registered: %s", data.username)
    token = create_access_token(data.username)
    return Token(access_token=token)


def login_user(data: UserCreate) -> Token:
    user = _users.get(data.username)
    if not user or not verify_password(data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(data.username)
    return Token(access_token=token)
