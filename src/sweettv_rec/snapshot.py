"""A time snapshot: history before a cutoff week, a target window after it, and the shared candidate pool.

Validation: history weeks 11-20, target weeks 21-30. Submission: history weeks 11-30, target weeks 31-42, where
the target-window schedule is used only to know which programme airs when, never who watched it.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from sweettv_rec.data import week_start
from sweettv_rec.target import frequency, top_k


@dataclass
class Snapshot:
    history: pd.DataFrame  # user_airings rows before the cutoff
    window_airings: pd.DataFrame  # labelled schedule rows inside the target window
    history_airings: pd.DataFrame  # labelled schedule rows inside the history window
    truth: dict[int, list[int]]  # empty for the submission snapshot
    users: np.ndarray  # row order of the interaction matrix
    items: np.ndarray  # column order of the interaction matrix (programmes seen in history or in the pool)
    pool: np.ndarray = field(init=False)  # programmes airing in the target window
    matrix: sparse.csr_matrix = field(init=False)  # users x items, fully watched airings in history

    def __post_init__(self) -> None:
        self.pool = np.unique(self.window_airings["tv_show_id"].to_numpy())
        freq = frequency(self.history)
        u = pd.Index(self.users).get_indexer(freq["user_id"])
        i = pd.Index(self.items).get_indexer(freq["tv_show_id"])
        keep = (u >= 0) & (i >= 0)
        self.matrix = sparse.csr_matrix(
            (freq["freq"].to_numpy(dtype=np.float32)[keep], (u[keep], i[keep])),
            shape=(len(self.users), len(self.items)),
        )

    @property
    def pool_mask(self) -> np.ndarray:
        return np.isin(self.items, self.pool)


def make_snapshot(
    user_airings: pd.DataFrame,
    airings: pd.DataFrame,
    history_weeks: tuple[int, int],
    target_weeks: tuple[int, int],
    with_truth: bool = True,
) -> Snapshot:
    h_lo, h_hi = week_start(history_weeks[0]), week_start(history_weeks[1] + 1)
    t_lo, t_hi = week_start(target_weeks[0]), week_start(target_weeks[1] + 1)
    labelled = airings[(airings["tv_show_id"] != 0) & (airings["duration"] > 0)]
    history = user_airings[(user_airings["start"] >= h_lo) & (user_airings["start"] < h_hi)]
    window_airings = labelled[(labelled["start"] >= t_lo) & (labelled["start"] < t_hi)]
    history_airings = labelled[(labelled["start"] >= h_lo) & (labelled["start"] < h_hi)]
    truth: dict[int, list[int]] = {}
    if with_truth:
        future = user_airings[(user_airings["start"] >= t_lo) & (user_airings["start"] < t_hi)]
        truth = top_k(frequency(future))
    users = np.unique(np.r_[history["user_id"].to_numpy(), np.fromiter(truth, dtype=np.uint64)])
    items = np.unique(np.r_[history["tv_show_id"].to_numpy(), window_airings["tv_show_id"].to_numpy()])
    return Snapshot(history, window_airings, history_airings, truth, users, items)
