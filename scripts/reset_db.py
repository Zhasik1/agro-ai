"""CLI tool to wipe the MalChain SQLite database and FAISS indexes.

Usage::

    python scripts/reset_db.py          # prompts for confirmation
    python scripts/reset_db.py --yes    # skip prompt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _delete_sqlite(database_url: str) -> None:
    """Delete the SQLite file referenced by *database_url*.

    Args:
        database_url: SQLAlchemy database URL (e.g. ``sqlite:///./malchain.db``).
    """
    if not database_url.startswith("sqlite:///"):
        print(f"Skipped (not a SQLite URL): {database_url}")
        return
    raw_path = database_url[len("sqlite:///"):]
    db_path = Path(raw_path).resolve()
    if db_path.exists():
        db_path.unlink()
        print(f"Deleted: {db_path}")
    else:
        print(f"Not found (skipped): {db_path}")


def _wipe_vector_dbs(vector_db_path: Path) -> None:
    """Wipe all per-species FAISS subdirectories and recreate them empty.

    Args:
        vector_db_path: Root directory that contains per-species sub-dirs.
    """
    vec_root = Path(vector_db_path).resolve()
    if not vec_root.exists():
        print(f"Vector DB path not found (skipped): {vec_root}")
        return
    for species_dir in sorted(vec_root.iterdir()):
        if species_dir.is_dir():
            shutil.rmtree(species_dir)
            species_dir.mkdir()
            print(f"Wiped and recreated: {species_dir}")


def main() -> None:
    """Entry point — parse args, confirm, then reset."""
    parser = argparse.ArgumentParser(
        description="Reset MalChain SQLite DB and FAISS indexes"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt",
    )
    args = parser.parse_args()

    if not args.yes:
        answer = input("Are you sure? [y/N]: ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    from app.config import Settings

    settings = Settings()
    _delete_sqlite(settings.DATABASE_URL)
    _wipe_vector_dbs(settings.VECTOR_DB_PATH)
    print("Reset complete.")


if __name__ == "__main__":
    main()
