"""Hold-out comparison: history weeks 11-20, target weeks 21-30. Writes artifacts/validation.json."""

import itertools
import json
import logging
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from sweettv_rec import recommenders as rec
from sweettv_rec.config import settings
from sweettv_rec.data import load
from sweettv_rec.snapshot import Snapshot, make_snapshot
from sweettv_rec.target import map_at_k

logger = logging.getLogger("validate")


def evaluate(
    name: str,
    params: dict,
    fn: Callable[[Snapshot], np.ndarray],
    snap: Snapshot,
    test_users: set,
) -> dict:
    t0 = time.perf_counter()
    recs = rec.to_top5(fn(snap), snap)
    seconds = time.perf_counter() - t0
    truth_test = {u: v for u, v in snap.truth.items() if u in test_users}
    row = {
        "method": name,
        "params": params,
        "map5_all": map_at_k(snap.truth, recs),
        "map5_test_users": map_at_k(truth_test, recs),
        "seconds": round(seconds, 1),
    }
    logger.info(
        "%s %s all=%.4f test=%.4f (%.1fs)",
        name,
        params,
        row["map5_all"],
        row["map5_test_users"],
        seconds,
    )
    return row


def main() -> None:
    ua = load("user_airings")
    airings = load("airings_dedup")
    test_users = {
        int(u) for u in pd.read_csv(settings.raw_dir / "submission.csv", dtype={"user_id": np.uint64})["user_id"]
    }
    snap = make_snapshot(ua, airings, (11, 20), (21, 30))
    logger.info(
        "users %s, items %s, pool %s, truth users %s",
        len(snap.users),
        len(snap.items),
        len(snap.pool),
        len(snap.truth),
    )

    rows = [
        evaluate("popularity", {}, rec.popularity_scores, snap, test_users),
        evaluate("habit", {}, rec.habit_scores, snap, test_users),
    ]
    for prior in (0.0, 3.0, 10.0, 30.0):
        rows.append(
            evaluate(
                "habit_schedule",
                {"prior": prior},
                lambda s, p=prior: rec.habit_schedule_scores(s, p),
                snap,
                test_users,
            )
        )
    for alpha, weighting in itertools.product((0.15, 0.3, 0.5, 0.7, 0.85), ("count", "log", "binary")):
        rows.append(
            evaluate(
                "ppr",
                {"alpha": alpha, "weighting": weighting},
                lambda s, a=alpha, w=weighting: rec.ppr_scores(s, a, weighting=w),
                snap,
                test_users,
            )
        )
    for factors, reg, alpha in itertools.product((32, 64, 128), (1.0, 10.0, 50.0), (1.0, 5.0, 20.0)):
        rows.append(
            evaluate(
                "als",
                {"factors": factors, "regularization": reg, "alpha": alpha},
                lambda s, f=factors, r=reg, a=alpha: rec.als_scores(s, f, r, a),
                snap,
                test_users,
            )
        )

    out = settings.artifacts_dir / "validation.json"
    out.write_text(
        json.dumps(
            {
                "snapshot": {
                    "history": [11, 20],
                    "target": [21, 30],
                    "users": len(snap.users),
                    "items": len(snap.items),
                    "pool": len(snap.pool),
                    "truth_users": len(snap.truth),
                    "truth_test_users": sum(u in test_users for u in snap.truth),
                },
                "rows": rows,
            },
            indent=1,
        )
    )
    logger.info("wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
