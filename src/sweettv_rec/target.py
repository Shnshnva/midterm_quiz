"""Target construction and MAP@K, following the competition Data page.

Target: drop tv_show_id = 0, keep airings with watched share >= 0.8, count them per (user, programme), take
strictly the 5 most frequent programmes per user. The page does not say how ties are broken; here ties go to
more covered seconds, then to the smaller id.
"""

import numpy as np
import pandas as pd

from sweettv_rec.config import SCREEN_TIME_THRESHOLD, K


def frequency(user_airings: pd.DataFrame, threshold: float = SCREEN_TIME_THRESHOLD) -> pd.DataFrame:
    """Per (user, programme): number of airings watched for at least `threshold`, and their covered seconds."""
    q = user_airings[(user_airings["tv_show_id"] != 0) & (user_airings["frac"] >= threshold)]
    return (
        q.groupby(["user_id", "tv_show_id"]).agg(freq=("airing_id", "size"), seconds=("covered", "sum")).reset_index()
    )


def top_k(freq: pd.DataFrame, k: int = K, strict: bool = True) -> dict[int, list[int]]:
    """user -> programmes in target order. With `strict`, only users with at least k programmes are kept."""
    f = freq.sort_values(
        ["user_id", "freq", "seconds", "tv_show_id"],
        ascending=[True, False, False, True],
    )
    lists = f.groupby("user_id", sort=False)["tv_show_id"].apply(list)
    users = lists.index.to_numpy(dtype=np.uint64)
    return {int(u): items[:k] for u, items in zip(users, lists, strict=True) if not strict or len(items) >= k}


def average_precision(actual: list[int], predicted: list[int], k: int = K) -> float:
    """AP@k with denominator min(k, |actual|); duplicates in `predicted` score once."""
    if not actual:
        return 0.0
    relevant = set(actual)
    hits, score, seen = 0, 0.0, set()
    for rank, item in enumerate(predicted[:k], start=1):
        if item in relevant and item not in seen:
            hits += 1
            score += hits / rank
        seen.add(item)
    return score / min(k, len(relevant))


def map_at_k(truth: dict[int, list[int]], recs: dict[int, list[int]], k: int = K) -> float:
    """Mean AP@k over the users of `truth`; a user missing from `recs` scores 0."""
    if not truth:
        raise ValueError("empty truth")
    return float(np.mean([average_precision(items, recs.get(u, []), k) for u, items in truth.items()]))
