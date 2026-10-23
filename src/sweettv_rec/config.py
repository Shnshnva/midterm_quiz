"""Paths and global constants. Paths can be overridden with SWEETTV_* environment variables."""

from pathlib import Path

import pandas as pd
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SWEETTV_")

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    submissions_dir: Path = Path("submissions")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


settings = Settings()

# Week 11 of 2020 starts on Monday 2020-03-09, the first day of the view log.
WEEK11_START = pd.Timestamp("2020-03-09")
SECONDS_PER_WEEK = 7 * 24 * 3600

# Organisers' rule from the competition Data page: an airing counts when the user
# watched at least 80% of its duration; the target is the 5 most frequent programmes.
SCREEN_TIME_THRESHOLD = 0.8
K = 5
