"""Rank fusion of habit x schedule, PPR and ALS on the hold-out (history 11-20, target 21-30).

Writes artifacts/validation_fusion.json. Weights are selected on this hold-out, so its number is optimistic.
"""

import itertools
import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from sweettv_rec import recommenders as rec
from sweettv_rec.config import settings
from sweettv_rec.data import load
from sweettv_rec.snapshot import make_snapshot
from sweettv_rec.target import map_at_k

logger = logging.getLogger("fusion")

COMPONENTS = {
    "habit_schedule": lambda s: rec.habit_schedule_scores(s, prior=100.0),
    "ppr": lambda s: rec.ppr_scores(s, alpha=0.5, weighting="count"),
    "als": lambda s: rec.als_scores(s, factors=128, regularization=5.0, alpha=0.5),
}


def main() -> None:
    snap = make_snapshot(load("user_airings"), load("airings_dedup"), (11, 20), (21, 30))
    sub = pd.read_csv(settings.raw_dir / "submission.csv", dtype={"user_id": np.uint64})
    test_users = {int(u) for u in sub["user_id"]}
    truth_test = {u: v for u, v in snap.truth.items() if u in test_users}
    ranks = {name: rec.top_n_ranks(fn(snap), snap) for name, fn in COMPONENTS.items()}
    rows: list[dict[str, Any]] = []
    for w_hs, w_ppr, w_als, k in itertools.product((0, 0.5, 1, 2), (0, 0.5, 1, 2), (0, 0.5, 1), (5.0, 20.0, 60.0)):
        if w_hs + w_ppr + w_als == 0:
            continue
        weights = {"habit_schedule": w_hs, "ppr": w_ppr, "als": w_als}
        recs = rec.to_top5(rec.rank_fusion_scores(ranks, weights, k), snap)
        rows.append(
            {
                "weights": weights,
                "k": k,
                "map5_all": map_at_k(snap.truth, recs),
                "map5_test_users": map_at_k(truth_test, recs),
            }
        )
    rows.sort(key=lambda r: -r["map5_all"])
    for r in rows[:8]:
        logger.info("%s k=%s all=%.4f test=%.4f", r["weights"], r["k"], r["map5_all"], r["map5_test_users"])
    (settings.artifacts_dir / "validation_fusion.json").write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
