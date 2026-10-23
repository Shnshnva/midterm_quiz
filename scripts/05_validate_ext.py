"""Extended grids where 04_validate.py hit a grid edge (ALS alpha=1, habit_schedule prior=30).

Appends rows to artifacts/validation_ext.json.
"""

import itertools
import json
import logging
from importlib import import_module

import numpy as np
import pandas as pd

from sweettv_rec import recommenders as rec
from sweettv_rec.config import settings
from sweettv_rec.data import load
from sweettv_rec.snapshot import make_snapshot

evaluate = import_module("04_validate").evaluate
logger = logging.getLogger("validate_ext")


def main() -> None:
    ua = load("user_airings")
    airings = load("airings_dedup")
    sub = pd.read_csv(settings.raw_dir / "submission.csv", dtype={"user_id": np.uint64})
    test_users = {int(u) for u in sub["user_id"]}
    snap = make_snapshot(ua, airings, (11, 20), (21, 30))
    rows = []
    for prior in (50.0, 100.0, 300.0):
        rows.append(
            evaluate(
                "habit_schedule",
                {"prior": prior},
                lambda s, p=prior: rec.habit_schedule_scores(s, p),
                snap,
                test_users,
            )
        )
    for factors, reg, alpha in itertools.product((64, 128, 256), (5.0, 10.0, 20.0), (0.1, 0.25, 0.5, 1.0)):
        rows.append(
            evaluate(
                "als",
                {"factors": factors, "regularization": reg, "alpha": alpha},
                lambda s, f=factors, r=reg, a=alpha: rec.als_scores(s, f, r, a),
                snap,
                test_users,
            )
        )
    out = settings.artifacts_dir / "validation_ext.json"
    out.write_text(json.dumps({"rows": rows}, indent=1))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
