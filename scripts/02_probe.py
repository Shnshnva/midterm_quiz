"""Sanity probes before the join: clock alignment, coverage of test users, schedule overlaps.

Writes artifacts/probe.json.
"""

import json
import logging

import numpy as np
import pandas as pd

from sweettv_rec.config import settings
from sweettv_rec.data import load, week_of

logger = logging.getLogger("probe")


def near_start_share(views: pd.DataFrame, airings: pd.DataFrame, offset: int, tol: int = 120) -> float:
    """Share of view starts within `tol` s of an airing start on the same channel, schedule shifted by `offset`."""
    hits = 0
    for channel_id, v in views.groupby("channel_id"):
        starts = np.sort(airings.loc[airings["channel_id"] == channel_id, "start"].to_numpy() + offset)
        if len(starts) == 0:
            continue
        t = v["start"].to_numpy()
        i = np.clip(np.searchsorted(starts, t), 1, len(starts) - 1)
        gap = np.minimum(np.abs(t - starts[i - 1]), np.abs(starts[i] - t))
        hits += int((gap <= tol).sum())
    return hits / len(views)


def main() -> None:
    views = load("views")
    airings = load("airings")
    result: dict[str, object] = {}

    result["views_time_range"] = [int(views["start"].min()), int(views["stop"].max())]
    result["views_weeks"] = sorted({int(w) for w in np.unique(week_of(views["start"]))})
    result["schedule_weeks_by_file"] = {
        f: [int(week_of(g["start"]).min()), int(week_of(g["start"]).max())] for f, g in airings.groupby("source_file")
    }
    result["duraton_equals_stop_minus_start"] = float((views["stop"] - views["start"] == views["duration"]).mean())
    result["view_duration_quantiles"] = {
        str(q): float(views["duration"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
    }

    sample = views.sample(300_000, random_state=0)
    result["clock_offset_probe"] = {str(h): near_start_share(sample, airings, h * 3600) for h in range(-3, 4)}

    sub = pd.read_csv(settings.raw_dir / "submission.csv", dtype={"user_id": np.uint64})
    users = set(views["user_id"].unique())
    result["submission_rows"] = len(sub)
    result["submission_unique_users"] = int(sub["user_id"].nunique())
    result["submission_users_in_log"] = int(sub["user_id"].isin(users).sum())
    last_week = views[week_of(views["start"]) >= 29]["user_id"].unique()
    result["submission_users_active_weeks_29_30"] = int(sub["user_id"].isin(set(last_week)).sum())

    a = airings.sort_values(["channel_id", "start"])
    same_channel = a["channel_id"].to_numpy()[1:] == a["channel_id"].to_numpy()[:-1]
    overlap = a["start"].to_numpy()[1:] < a["end"].to_numpy()[:-1]
    result["schedule_consecutive_overlaps"] = int((same_channel & overlap).sum())
    dup = a.duplicated(["channel_id", "start"]).sum()
    result["schedule_duplicate_channel_start"] = int(dup)
    result["schedule_zero_duration"] = int((a["duration"] <= 0).sum())
    result["view_channels_missing_from_schedule"] = int(
        (~views["channel_id"].isin(set(airings["channel_id"].unique()))).sum()
    )

    out = settings.artifacts_dir / "probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
