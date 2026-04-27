"""Insert demo owners (no animals — those should be registered with real photos)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database.models import Owner  # noqa: E402
from app.database.session import SessionLocal, init_db  # noqa: E402

DEMO_OWNERS: list[dict[str, str]] = [
    {
        "iin": "880101300123",
        "full_name": "Айдар Серікұлы",
        "phone": "+77001112233",
        "region": "Солтүстік Қазақстан облысы",
    },
    {
        "iin": "920215400456",
        "full_name": "Айгерім Бекқызы",
        "phone": "+77014445566",
        "region": "Ақмола облысы",
    },
    {
        "iin": "750303500789",
        "full_name": "Болат Қалиұлы",
        "phone": "+77027778899",
        "region": "Қостанай облысы",
    },
]


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        added = 0
        for entry in DEMO_OWNERS:
            existing = session.get(Owner, entry["iin"])
            if existing is None:
                session.add(Owner(**entry))
                added += 1
        session.commit()
        print(f"✅ Демо иелер: қосылды {added}, барлығы {session.query(Owner).count()}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
