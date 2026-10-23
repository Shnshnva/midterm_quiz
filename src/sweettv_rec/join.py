"""View log x schedule join.

Rule: a view matches an airing on the same channel when the two intervals overlap. Overlapping views of one
user on one channel are merged first, so a second counted twice can never push the watched share above 1.
The watched share of an airing is covered seconds / airing duration, as on the competition Data page.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def merge_views(views: pd.DataFrame) -> pd.DataFrame:
    """Union of each user's view intervals per channel."""
    v = views.sort_values(["user_id", "channel_id", "start"], kind="stable")
    user = v["user_id"].to_numpy()
    channel = v["channel_id"].to_numpy()
    start = v["start"].to_numpy()
    stop = v["stop"].to_numpy()

    new_group = np.r_[True, (user[1:] != user[:-1]) | (channel[1:] != channel[:-1])]
    group_id = np.cumsum(new_group)
    # Running max of stop inside each (user, channel) group: offset groups so cummax never crosses them.
    offset = (group_id - 1).astype(np.int64) * (int(stop.max()) + 1)
    run_max = np.maximum.accumulate(stop + offset) - offset
    prev_max = np.r_[np.iinfo(np.int64).min, run_max[:-1]]
    new_segment = new_group | (start > prev_max)

    merged = pd.DataFrame(
        {
            "user_id": user[new_segment],
            "channel_id": channel[new_segment],
            "start": start[new_segment],
            "stop": np.maximum.reduceat(stop, np.flatnonzero(new_segment)),
        }
    )
    logger.info("merged %s views into %s segments", len(v), len(merged))
    return merged


def join_airings(segments: pd.DataFrame, airings: pd.DataFrame) -> pd.DataFrame:
    """Covered seconds per (user, airing). `airings` must carry an `airing_id` column."""
    parts = []
    for channel_id, seg in segments.groupby("channel_id", sort=False):
        a = airings[airings["channel_id"] == channel_id].sort_values("start", kind="stable")
        if a.empty:
            continue
        a_start = a["start"].to_numpy()
        a_end = a["end"].to_numpy()
        s_start = seg["start"].to_numpy()
        s_stop = seg["stop"].to_numpy()
        # The first candidate airing is the first whose running-max end passes the view start; ends are almost
        # sorted (13 overlaps in the whole grid), so the running max only adds a few non-overlapping candidates
        # that the overlap filter below removes.
        lo = np.searchsorted(np.maximum.accumulate(a_end), s_start, side="right")
        hi = np.searchsorted(a_start, s_stop, side="left")
        n = np.maximum(hi - lo, 0)
        if n.sum() == 0:
            continue
        seg_idx = np.repeat(np.arange(len(seg)), n)
        air_idx = np.repeat(lo, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        covered = np.minimum(s_stop[seg_idx], a_end[air_idx]) - np.maximum(s_start[seg_idx], a_start[air_idx])
        keep = covered > 0
        parts.append(
            pd.DataFrame(
                {
                    "user_id": seg["user_id"].to_numpy()[seg_idx[keep]],
                    "airing_id": a["airing_id"].to_numpy()[air_idx[keep]],
                    "covered": covered[keep],
                }
            )
        )
    pairs = pd.concat(parts, ignore_index=True)
    out = pairs.groupby(["user_id", "airing_id"], sort=False)["covered"].sum().reset_index()
    cols = ["airing_id", "channel_id", "start", "duration", "tv_show_id"]
    out = out.merge(airings[cols], on="airing_id", how="left")
    out["frac"] = np.where(out["duration"] > 0, out["covered"] / out["duration"].clip(lower=1), 0.0)
    logger.info("joined %s (user, airing) pairs", len(out))
    return out
