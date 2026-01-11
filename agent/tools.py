from __future__ import annotations
import re


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(email.strip()))


def mock_lead_capture(name: str, email: str, platform: str) -> None:
    # This simulates a backend API call.
    print(f"Lead captured successfully: {name}, {email}, {platform}")
