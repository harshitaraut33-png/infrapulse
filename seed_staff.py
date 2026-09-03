"""
Create the three staff accounts — one per category.

Run once after setting up:      python seed_staff.py

Passwords come from environment variables if set, otherwise a default is used and
printed. Change them before the app is public.
"""

import os

from app import db, security

STAFF = [
    ("structural@infrapulse.local", "Structural Team", "Structural"),
    ("functional@infrapulse.local", "Functional Team", "Functional"),
    ("performance@infrapulse.local", "Performance Team", "Performance"),
]

def main() -> None:
    db.init_db()
    password = os.environ.get("STAFF_PASSWORD", "staff1234")

    for email, name, category in STAFF:
        if db.get_user_by_email(email):
            print(f"exists   {email}  ({category})")
            continue
        db.create_user(email, name, security.hash_password(password),
                       role="staff", category=category)
        print(f"created  {email}  ({category})")

    print(f"\nStaff password: {password}")
    print("Each staff account sees only its own category's queue.")


if __name__ == "__main__":
    main()
