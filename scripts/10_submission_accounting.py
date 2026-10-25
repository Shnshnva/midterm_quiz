"""Accounting for the final submission (history 11-30, pool 31-42). Writes artifacts/submission_accounting.json."""

import json
import logging

import numpy as np
import pandas as pd

from sweettv_rec.config import settings
from sweettv_rec.data import load
from sweettv_rec.snapshot import make_snapshot
from sweettv_rec.target import frequency

ua, airings = load("user_airings"), load("airings_dedup")
snap = make_snapshot(ua, airings, (11, 30), (31, 42), with_truth=False)
template = pd.read_csv(settings.raw_dir / "submission.csv", dtype={"user_id": np.uint64})
test = [int(u) for u in template["user_id"]]
progs = frequency(snap.history).groupby("user_id").size()
progs.index = progs.index.astype("uint64").map(int)
n = np.array([int(progs.get(u, 0)) for u in test])
final = pd.read_csv(settings.submissions_dir / "ranker_v1.csv", dtype=str)
lists = final["tv_show_id"].str.split()
hist = frequency(snap.history)
seen = set(zip(hist["user_id"].astype("uint64").map(int), hist["tv_show_id"], strict=True))
repeat = np.mean([(int(u), int(i)) in seen for u, items in zip(final["user_id"], lists, strict=True) for i in items])
out = {
    "history_users": len(snap.users),
    "items": len(snap.items),
    "pool_weeks_31_42": len(snap.pool),
    "test_users": len(test),
    "test_users_no_full_watch_history": int((n == 0).sum()),
    "test_users_1_2_programmes": int(((n >= 1) & (n <= 2)).sum()),
    "test_users_lt5_programmes": int((n < 5).sum()),
    "final_list_repeat_share": float(repeat),
    "final_distinct_programmes_recommended": int(lists.explode().nunique()),
}
(settings.artifacts_dir / "submission_accounting.json").write_text(json.dumps(out, indent=1))
logging.basicConfig(level=logging.INFO)
logging.getLogger("accounting").info("%s", out)
