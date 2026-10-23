"""Join the view log to labelled airings; write data/cache/user_airings.parquet and artifacts/join_stats.json."""

import json
import logging
import time

import numpy as np
import pandas as pd

from sweettv_rec.config import settings
from sweettv_rec.data import load, week_of, week_start
from sweettv_rec.join import join_airings, merge_views
from sweettv_rec.target import frequency, top_k

logger = logging.getLogger("join")


def main() -> None:
    t0 = time.perf_counter()
    views = load("views")
    airings = load("airings")
    airings["airing_id"] = np.arange(len(airings))
    # Files overlap by a few hours at week boundaries; one airing per (channel, start, programme) is kept.
    airings = airings.drop_duplicates(["channel_id", "start", "tv_show_id"]).reset_index(drop=True)
    airings.to_parquet(settings.cache_dir / "airings_dedup.parquet", index=False)
    labelled = airings[(airings["tv_show_id"] != 0) & (airings["duration"] > 0)]

    segments = merge_views(views)
    ua = join_airings(segments, labelled)
    ua["week"] = week_of(ua["start"])
    ua.to_parquet(settings.cache_dir / "user_airings.parquet", index=False)

    stats: dict[str, object] = {
        "views": len(views),
        "segments": len(segments),
        "airings_total": len(airings),
        "airings_labelled": len(labelled),
        "labelled_share": len(labelled) / len(airings),
        "user_airing_pairs": len(ua),
        "pairs_frac_ge_0_8": int((ua["frac"] >= 0.8).sum()),
        "frac_quantiles": {str(q): float(ua["frac"].quantile(q)) for q in (0.25, 0.5, 0.75, 0.9)},
        "users_with_any_labelled_pair": int(ua["user_id"].nunique()),
    }
    sub_users = set(pd.read_csv(settings.raw_dir / "submission.csv", dtype={"user_id": np.uint64})["user_id"])
    for name, lo, hi in (
        ("weeks_11_20", 11, 20),
        ("weeks_21_30", 21, 30),
        ("weeks_11_30", 11, 30),
    ):
        win = ua[(ua["start"] >= week_start(lo)) & (ua["start"] < week_start(hi + 1))]
        freq = frequency(win)
        strict = top_k(freq)
        stats[name] = {
            "users_with_full_watch": int(freq["user_id"].nunique()),
            "users_strict_top5": len(strict),
            "test_users_strict_top5": sum(u in sub_users for u in strict),
            "programmes_full_watched": int(freq["tv_show_id"].nunique()),
        }
    stats["runtime_s"] = round(time.perf_counter() - t0, 1)
    out = settings.artifacts_dir / "join_stats.json"
    out.write_text(json.dumps(stats, indent=1))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
