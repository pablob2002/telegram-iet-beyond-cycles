"""Data access for the pseudonymized Telegram message tables.

The analysis notebooks only need three fields per message: the timestamp,
the (pseudonymous) sender and the sub-channel (Telegram topic). These are
released in ``data/processed/<community>_messages.csv``; see data/README.md.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
COMMUNITIES = ("dana", "vagarenfe")


def load_messages(community: str) -> pd.DataFrame:
    """Return the message table of one community.

    Parameters
    ----------
    community : {"dana", "vagarenfe"}
        ``"dana"`` is @voluntariosdanavalencia, ``"vagarenfe"`` is @vagarenfe.

    Returns
    -------
    DataFrame with columns ``date`` (tz-aware, UTC), ``sender_id``
    (integer pseudonym; NaN for messages without a sender) and
    ``topic_name`` (sub-channel), in the original export order.
    """
    if community not in COMMUNITIES:
        raise ValueError(f"community must be one of {COMMUNITIES}, got {community!r}")
    path = PROCESSED / f"{community}_messages.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with scripts/pseudonymize.py (see data/README.md)."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
    return df
