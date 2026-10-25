"""Fit each selected method on weeks 11-30 and write submissions/<method>.csv for the weeks 31-42 pool.

Parameters are the validation winners from artifacts/validation*.json.
"""

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from sweettv_rec import recommenders as rec
from sweettv_rec.config import K, settings
from sweettv_rec.data import load
from sweettv_rec.snapshot import Snapshot, make_snapshot

logger = logging.getLogger("submit")

METHODS: dict[str, Callable[[Snapshot], np.ndarray]] = {
    "popularity": rec.popularity_scores,
    "habit_schedule": lambda s: rec.habit_schedule_scores(s, prior=100.0),
    "ppr": lambda s: rec.ppr_scores(s, alpha=0.5, weighting="count"),
    "als": lambda s: rec.als_scores(s, factors=128, regularization=5.0, alpha=0.5),
    "fusion": lambda s: rec.rank_fusion_scores(
        {
            "habit_schedule": rec.top_n_ranks(rec.habit_schedule_scores(s, prior=100.0), s),
            "ppr": rec.top_n_ranks(rec.ppr_scores(s, alpha=0.5, weighting="count"), s),
            "als": rec.top_n_ranks(rec.als_scores(s, factors=128, regularization=5.0, alpha=0.5), s),
        },
        {"habit_schedule": 2.0, "ppr": 1.0, "als": 0.5},
        k=20.0,
    ),
}


def write_submission(recs: dict[int, list[int]], template: pd.DataFrame, snap: Snapshot, name: str) -> None:
    pool = {int(p) for p in snap.pool}
    rows = []
    for u in template["user_id"]:
        items = recs[int(u)]
        if len(items) != K or len(set(items)) != K or not set(items) <= pool:
            raise ValueError(f"bad list for user {u}: {items}")
        rows.append(" ".join(str(i) for i in items))
    out = template[["user_id"]].assign(tv_show_id=rows)
    path = settings.submissions_dir / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    logger.info("wrote %s (%s rows)", path, len(out))


def main() -> None:
    ua = load("user_airings")
    airings = load("airings_dedup")
    template = pd.read_csv(settings.raw_dir / "submission.csv", dtype={"user_id": np.uint64})
    snap = make_snapshot(ua, airings, (11, 30), (31, 42), with_truth=False)
    logger.info("users %s, items %s, pool %s", len(snap.users), len(snap.items), len(snap.pool))
    users = [int(u) for u in template["user_id"]]
    for name, fn in METHODS.items():
        write_submission(rec.to_top5(fn(snap), snap, users), template, snap, name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
