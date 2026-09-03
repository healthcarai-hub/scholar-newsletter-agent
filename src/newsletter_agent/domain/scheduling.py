from __future__ import annotations

from datetime import date, datetime, timedelta


def publication_date(cutoff: datetime) -> date:
    """Return the reader-facing Friday date for a Thursday evening run."""
    return (cutoff + timedelta(days=1)).date()
