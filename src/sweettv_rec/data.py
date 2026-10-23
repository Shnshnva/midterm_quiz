"""Parse the raw competition CSV files once into typed parquet files.

Times are stored as integer seconds since WEEK11_START, so week = 11 + t // SECONDS_PER_WEEK.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sweettv_rec.config import SECONDS_PER_WEEK, WEEK11_START, settings

logger = logging.getLogger(__name__)

SCHEDULE_FILES = (
    "export_arh_11-20-final.csv",
    "export_arh_21-30-final.csv",
    "export_arh_31-42-final.csv",
)
SCHEDULE_META_COLUMNS = [
    "tv_show_title",
    "tv_show_category",
    "tv_show_genre_1",
    "tv_show_genre_2",
    "tv_show_genre_3",
    "year_of_production",
    "director",
    "actors",
]


def to_seconds(ts: pd.Series) -> np.ndarray:
    return ((ts - WEEK11_START) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)


def week_of(seconds: np.ndarray | pd.Series) -> np.ndarray:
    return 11 + np.asarray(seconds) // SECONDS_PER_WEEK


def week_start(week: int) -> int:
    """First second of `week` on the same clock as the parsed tables."""
    return (week - 11) * SECONDS_PER_WEEK


def parse_views(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        raw_dir / "dataset11-30.csv",
        dtype={"user_id": np.uint64, "vsetv_id": np.int32, "duraton": np.int64},
    )
    start = to_seconds(pd.to_datetime(df["start_time"], format="%Y-%m-%d %H:%M:%S"))
    stop = to_seconds(pd.to_datetime(df["stop_time"], format="%Y-%m-%d %H:%M:%S"))
    views = pd.DataFrame(
        {
            "user_id": df["user_id"].to_numpy(),
            "channel_id": df["vsetv_id"].to_numpy(),
            "start": start,
            "stop": stop,
            "duration": df["duraton"].to_numpy(),
        }
    )
    logger.info("views: %s rows, %s users", len(views), views["user_id"].nunique())
    return views


def parse_schedule(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (airings, programmes). Airings keep tv_show_id = 0 rows; callers decide what to do with them."""
    parts = []
    for name in SCHEDULE_FILES:
        df = pd.read_csv(
            raw_dir / name,
            dtype={"tv_show_id": np.int64, "channel_id": np.int32},
            keep_default_na=False,
        )
        df["source_file"] = name
        parts.append(df)
        logger.info(
            "%s: %s rows, %s labelled",
            name,
            len(df),
            int((df["tv_show_id"] != 0).sum()),
        )
    sched = pd.concat(parts, ignore_index=True)
    start = to_seconds(pd.to_datetime(sched["start_time"], format="%d.%m.%Y %H:%M:%S"))
    airings = pd.DataFrame(
        {
            "channel_id": sched["channel_id"].to_numpy(),
            "start": start,
            "end": start + sched["duration"].to_numpy(dtype=np.int64),
            "duration": sched["duration"].to_numpy(dtype=np.int64),
            "tv_show_id": sched["tv_show_id"].to_numpy(),
            "source_file": sched["source_file"].to_numpy(),
        }
    )
    labelled = sched[sched["tv_show_id"] != 0]
    programmes = labelled.groupby("tv_show_id")[SCHEDULE_META_COLUMNS].agg(lambda s: s.mode().iat[0]).reset_index()
    return airings, programmes


def build_cache() -> None:
    cache = settings.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    parse_views(settings.raw_dir).to_parquet(cache / "views.parquet", index=False)
    airings, programmes = parse_schedule(settings.raw_dir)
    airings.to_parquet(cache / "airings.parquet", index=False)
    programmes.to_parquet(cache / "programmes.parquet", index=False)
    logger.info("cache written to %s", cache)


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(settings.cache_dir / f"{name}.parquet")
