"""Statistics endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import Animal, Species
from app.database.session import get_db
from app.schemas.animal import RecentRegistration, SpeciesCount, StatsResponse

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/stats", response_model=StatsResponse, summary="Статистика / Statistics")
def stats(db: Session = Depends(get_db)) -> StatsResponse:
    """Return per-species totals and the 10 most-recent registrations."""
    counts: dict[Species, int] = dict.fromkeys(Species, 0)
    rows = db.execute(select(Animal.species, func.count(Animal.id)).group_by(Animal.species)).all()
    for species_value, count in rows:
        counts[species_value] = int(count)

    recent_rows = (
        db.execute(select(Animal).order_by(Animal.registered_at.desc()).limit(10)).scalars().all()
    )

    return StatsResponse(
        total=sum(counts.values()),
        per_species=[SpeciesCount(species=s, count=c) for s, c in counts.items()],
        recent=[RecentRegistration.model_validate(a) for a in recent_rows],
    )
