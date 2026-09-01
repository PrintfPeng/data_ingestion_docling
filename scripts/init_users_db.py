"""
scripts/init_users_db.py

Bootstrap the multi-tenant DB and (optionally) create the initial admin user.

Usage (inside container):
  docker exec ingestion-backend python scripts/init_users_db.py \
      --admin-user admin --admin-password 'somestrongpw'

If the admin user already exists, this is a no-op.
Safe to re-run — schema is IF NOT EXISTS; user creation is skip-if-exists.
"""
from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.db import init_db, DB_PATH  # noqa: E402
from backend.services.users import (  # noqa: E402
    create_user,
    get_user_by_username,
    UserError,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("init_users_db")


def main() -> None:
    p = argparse.ArgumentParser(description="Initialise the users DB and (optionally) create the first admin.")
    p.add_argument("--admin-user", default=os.getenv("INIT_ADMIN_USER", "admin"))
    p.add_argument(
        "--admin-password",
        default=os.getenv("INIT_ADMIN_PASSWORD"),
        help="Admin password. Skip flag to leave admin creation to a later step.",
    )
    p.add_argument("--admin-email", default=os.getenv("INIT_ADMIN_EMAIL", ""))
    p.add_argument("--skip-admin", action="store_true", help="Only run schema init, don't create admin.")
    args = p.parse_args()

    logger.info(f"Initialising DB at {DB_PATH}")
    init_db()
    logger.info("Schema OK")

    if args.skip_admin:
        logger.info("Skipping admin creation (--skip-admin)")
        return

    if not args.admin_password:
        logger.warning("No --admin-password (and no INIT_ADMIN_PASSWORD env). Skipping admin bootstrap.")
        return

    existing = get_user_by_username(args.admin_user)
    if existing:
        logger.info(f"Admin '{args.admin_user}' already exists (id={existing['id']}). Skipping.")
        return

    try:
        u = create_user(
            username=args.admin_user,
            password=args.admin_password,
            email=args.admin_email or None,
            is_admin=True,
        )
        logger.info(f"Admin created: id={u['id']} username={u['username']} is_admin=1")
    except UserError as e:
        logger.error(f"Admin creation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
