"""Password hashing. Uses only the Python standard library, so there is nothing to
install and nothing that can fail to build on the hosting platform."""

from __future__ import annotations

import hashlib
import hmac
import os

ITERATIONS = 200_000


def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 with a fresh random salt. Stored as 'salt$hash' in hex."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split("$", 1)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                     bytes.fromhex(salt_hex), ITERATIONS)
        return hmac.compare_digest(digest.hex(), digest_hex)   # constant-time
    except Exception:
        return False
