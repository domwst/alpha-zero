"""Single administrator; persistent revocable sessions with synchronizer CSRF tokens."""

import hashlib
import hmac
import secrets
import os
import sqlite3
import threading
import time


class Auth:
    def __init__(self, password_hash=None, sessions_path=None):
        self.password_hash = password_hash
        self.credential = hashlib.sha256((password_hash or "").encode()).hexdigest()
        if sessions_path is not None:
            fd = os.open(sessions_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                os.fchmod(fd, 0o600)
            finally:
                os.close(fd)
        self.sessions = sqlite3.connect(
            str(sessions_path) if sessions_path is not None else ":memory:",
            check_same_thread=False,
        )
        with self.sessions:
            self.sessions.execute("""CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY, csrf TEXT NOT NULL,
                credential TEXT NOT NULL
            )""")
            if "expires" in {
                row[1] for row in self.sessions.execute("PRAGMA table_info(sessions)")
            }:
                # Preserve active logins, without reviving already-expired sessions.
                self.sessions.execute(
                    "DELETE FROM sessions WHERE expires<=?", (time.time(),)
                )
                self.sessions.execute("ALTER TABLE sessions DROP COLUMN expires")
            self.sessions.execute(
                "DELETE FROM sessions WHERE credential<>?",
                (self.credential,),
            )
        self.lock = threading.Lock()
        self.failures = []

    @staticmethod
    def hash_password(password):
        if len(password) < 12:
            raise ValueError("Use at least 12 characters")
        salt = secrets.token_hex(16)
        digest = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1
        ).hex()
        return salt + ":" + digest

    def login(self, password):
        with self.lock:
            now = time.time()
            self.failures = [t for t in self.failures if now - t < 60]
            if len(self.failures) >= 10:
                raise PermissionError("Please wait before trying to log in again")
            if (
                not self.password_hash
                or not isinstance(password, str)
                or len(password) > 1024
            ):
                raise PermissionError("Invalid credentials")
            salt, expected = self.password_hash.split(":")
            actual = hashlib.scrypt(
                password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1
            ).hex()
            if not hmac.compare_digest(actual, expected):
                self.failures.append(now)
                raise PermissionError("Invalid credentials")
            session = {"csrf": secrets.token_urlsafe(32)}
            token = secrets.token_urlsafe(32)
            with self.sessions:
                self.sessions.execute(
                    "INSERT INTO sessions(token_hash,csrf,credential) VALUES (?,?,?)",
                    (
                        self.token_hash(token),
                        session["csrf"],
                        self.credential,
                    ),
                )
            return token, session.copy()

    @staticmethod
    def token_hash(token):
        return hashlib.sha256(token.encode()).hexdigest()

    def get(self, token):
        if not isinstance(token, str) or not 1 <= len(token) <= 128:
            return None
        with self.lock:
            row = self.sessions.execute(
                "SELECT csrf FROM sessions WHERE token_hash=? AND credential=?",
                (self.token_hash(token), self.credential),
            ).fetchone()
            return {"csrf": row[0]} if row else None

    def require(self, token, csrf):
        session = self.get(token)
        if not session or not csrf or not hmac.compare_digest(session["csrf"], csrf):
            raise PermissionError("Administrator session and CSRF token required")
        return session

    def logout(self, token):
        with self.lock, self.sessions:
            self.sessions.execute(
                "DELETE FROM sessions WHERE token_hash=?", (self.token_hash(token),)
            )

    def close(self):
        with self.lock:
            self.sessions.close()
