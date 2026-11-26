# Data

## Message tables (`processed/`)

> **Not yet included in the repository.** Release is pending approval by the DRI project;
> until then the tables are available from the authors on request. The notebooks expect them
> in `data/processed/` with the format below.

| File | Community | Messages | Senders | Sub-channels | Export window (UTC) |
|---|---|---|---|---|---|
| `dana_messages.csv` | @voluntariosdanavalencia | 18,064 | 2,539 | 18 | 2024-10-30 → 2025-11-10 |
| `vagarenfe_messages.csv` | @vagarenfe | 92,813 | 2,016 | 12 | 2019-02-08 → 2026-04-28 |

Columns:

- `date`: message timestamp, ISO 8601, UTC, 1 s resolution.
- `sender_id`: integer pseudonym (≥ 100000), consistent within a community. Empty for
  messages without a sender (posts by the group itself or anonymous admins; 9,961 in @vagarenfe).
- `topic_name`: Telegram topic (sub-channel) of the message.

Rows keep the order of the original export. Pseudonyms in `results/cnhpp/` (`user_id`) are the
same as `sender_id` here.

## Provenance

Messages were collected from the public Telegram API with
[tgkit](https://github.com/DataPolitik/tgkit) and [tgmon](https://github.com/DataPolitik/tgmon).
All other fields of the export (message text, user names and handles, message ids, replies,
reactions, views, forwards, media) are removed: none of them is needed for the analyses.

## Pseudonymization is not anonymization

Both communities are public. Exact timestamps could in principle be matched against the public
group history to recover who sent a message. The data are released for research on collective
temporal dynamics only; do not attempt to re-identify users.

## Rebuilding from the raw exports

Place the raw exports (`voluntariosdanavalencia_old.json`, `vagarenfe.json`) in `data/raw/` and run

```bash
python scripts/pseudonymize.py
```

The raw-id → pseudonym tables are written to `data/private/`, which is git-ignored and must
never be published. Re-running reuses them, so pseudonyms stay stable.

## Terms of use

To be confirmed with the DRI project before release. Result tables in `results/` (per-user
model parameters and statistics, pseudonymous ids) are included.
