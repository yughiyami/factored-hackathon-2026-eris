"""Mock trusted identity service: OTP challenge -> HMAC-signed (HS256) session token.

In production this is the bank's IdP (e.g. OTP over the registered phone). Here the OTP is a fixed
test code from settings so demos and the eval are reproducible. Tokens carry the customer_id and
expire after `ttl_minutes`; expiry is checked against an injectable clock so tests can fast-forward.
"""
from __future__ import annotations

import hmac
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import jwt

AUDIENCE = "iris-bot"
ISSUER = "iris-mock-idp"


class AuthError(Exception):
    pass


class SessionExpired(AuthError):
    pass


class InvalidSession(AuthError):
    pass


@dataclass(frozen=True)
class Session:
    customer_id: str
    session_id: str
    issued_at: float
    expires_at: float


class IdentityService:
    def __init__(self, secret: str, test_otp: str, ttl_minutes: int = 15,
                 clock: Callable[[], float] = time.time):
        if len(secret) < 16:
            raise ValueError("session secret too short")
        self._secret = secret
        self._otp = test_otp
        self.ttl_seconds = ttl_minutes * 60
        self.clock = clock
        self._challenges: dict[str, str] = {}  # challenge_id -> customer_id

    def start_challenge(self, customer_id: str) -> str:
        """Send an OTP to the customer's registered device (simulated). Returns a challenge id.

        The same response is returned whether or not the customer exists (no account enumeration).
        """
        challenge_id = uuid.uuid4().hex
        self._challenges[challenge_id] = customer_id
        return challenge_id

    def verify_otp(self, challenge_id: str, otp: str, customer_exists: bool) -> str:
        customer_id = self._challenges.get(challenge_id)
        if customer_id is None:
            raise InvalidSession("unknown challenge")
        if not customer_exists or not hmac.compare_digest(otp.strip(), self._otp):
            raise AuthError("otp rejected")
        del self._challenges[challenge_id]
        return self.issue(customer_id)

    def issue(self, customer_id: str) -> str:
        now = self.clock()
        claims = {"sub": customer_id, "sid": uuid.uuid4().hex, "iat": int(now),
                  "exp": int(now + self.ttl_seconds), "aud": AUDIENCE, "iss": ISSUER}
        return jwt.encode(claims, self._secret, algorithm="HS256")

    def validate(self, token: str | None) -> Session:
        if not token:
            raise InvalidSession("no session")
        try:
            claims = jwt.decode(token, self._secret, algorithms=["HS256"], audience=AUDIENCE,
                                issuer=ISSUER, options={"verify_exp": False, "verify_iat": False})
        except jwt.PyJWTError as exc:
            raise InvalidSession(str(exc)) from exc
        if self.clock() >= claims["exp"]:
            raise SessionExpired("session expired")
        return Session(claims["sub"], claims["sid"], claims["iat"], claims["exp"])
