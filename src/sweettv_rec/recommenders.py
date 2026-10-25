"""Score functions. Each returns a dense users x items float32 matrix aligned with `Snapshot.users` / `.items`.

Scores from different methods are not comparable; `to_top5` only uses their order within one user's row.
"""

import logging

import numpy as np
import pandas as pd
from scipy import sparse

from sweettv_rec.config import K
from sweettv_rec.snapshot import Snapshot

logger = logging.getLogger(__name__)


def window_airing_counts(snap: Snapshot) -> np.ndarray:
    counts = snap.window_airings["tv_show_id"].value_counts()
    return counts.reindex(snap.items, fill_value=0).to_numpy(dtype=np.float32)


def history_airing_counts(snap: Snapshot) -> np.ndarray:
    counts = snap.history_airings["tv_show_id"].value_counts()
    return counts.reindex(snap.items, fill_value=0).to_numpy(dtype=np.float32)


def popularity_ranking(snap: Snapshot) -> np.ndarray:
    """Pool items by number of distinct users with a fully watched airing in history; ties by window airings."""
    users_per_item = np.asarray((snap.matrix > 0).sum(axis=0)).ravel()
    air = window_airing_counts(snap)
    order = np.lexsort((-air, -users_per_item))
    return order[snap.pool_mask[order]]


def popularity_scores(snap: Snapshot) -> np.ndarray:
    users_per_item = np.asarray((snap.matrix > 0).sum(axis=0)).ravel().astype(np.float32)
    return np.broadcast_to(users_per_item, (len(snap.users), len(snap.items))).copy()


def habit_scores(snap: Snapshot) -> np.ndarray:
    """The user's own history frequency: 'watch again what you watched most'."""
    return snap.matrix.toarray()


def habit_schedule_scores(snap: Snapshot, prior: float = 0.0) -> np.ndarray:
    """Expected fully watched airings in the window: watched share of history airings x airings in the window."""
    rate = snap.matrix.multiply(1.0 / (history_airing_counts(snap) + prior).clip(min=1.0)).tocsr()
    return rate.multiply(window_airing_counts(snap)).toarray().astype(np.float32)


def _row_normalise(m: sparse.csr_matrix) -> sparse.csr_matrix:
    s = np.asarray(m.sum(axis=1)).ravel()
    return sparse.diags(np.where(s > 0, 1.0 / np.maximum(s, 1e-12), 0.0)) @ m


def ppr_scores(snap: Snapshot, alpha: float = 0.5, iterations: int = 30, weighting: str = "count") -> np.ndarray:
    """Personalised PageRank on the programme co-view graph, restarting at the user's history.

    Item -> item transitions are the two-step walk programme -> viewer -> programme over the bipartite graph,
    P = D_i^-1 R^T D_u^-1 R. With restart probability `alpha`, x = alpha s + (1 - alpha) x P, s = R_u / |R_u|.
    """
    r = snap.matrix.copy().astype(np.float64)
    if weighting == "log":
        r.data = np.log1p(r.data)
    elif weighting == "binary":
        r.data = np.ones_like(r.data)
    item_to_user = _row_normalise(r.T.tocsr())
    user_to_item = _row_normalise(r)
    p = (item_to_user @ user_to_item).toarray().astype(np.float32)  # items x items, rows sum to 1 or 0
    s = _row_normalise(r).toarray().astype(np.float32)
    x = s.copy()
    for _ in range(iterations):
        x = alpha * s + (1 - alpha) * (x @ p)
    return x


def als_scores(
    snap: Snapshot,
    factors: int = 64,
    regularization: float = 10.0,
    alpha: float = 2.0,
    iterations: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Implicit ALS (Hu, Koren, Volinsky 2008) on confidence 1 + alpha * log(1 + count)."""
    from implicit.cpu.als import AlternatingLeastSquares

    c = snap.matrix.copy().astype(np.float32)
    c.data = alpha * np.log1p(c.data)
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=seed,
        calculate_training_loss=False,
    )
    model.fit(c, show_progress=False)
    return (np.asarray(model.user_factors) @ np.asarray(model.item_factors).T).astype(np.float32)


def to_top5(scores: np.ndarray, snap: Snapshot, users: list[int] | None = None) -> dict[int, list[int]]:
    """Five pool items per user by score; positions without a positive score are filled from popularity."""
    fallback = popularity_ranking(snap)
    mask = snap.pool_mask
    index = pd.Index(snap.users)
    targets = list(snap.truth) if users is None else users
    rows = index.get_indexer(np.array(targets, dtype=np.uint64))
    out: dict[int, list[int]] = {}
    masked = np.where(mask, scores, -np.inf)
    for u, row in zip(targets, rows, strict=True):
        chosen: list[int] = []
        if row >= 0:
            s = masked[row]
            top = np.argpartition(-s, K)[:K]
            top = top[np.argsort(-s[top], kind="stable")]
            chosen = [int(j) for j in top if s[j] > 0]
        for j in fallback:
            if len(chosen) == K:
                break
            if j not in chosen:
                chosen.append(int(j))
        out[u] = [int(snap.items[j]) for j in chosen]
    return out


def top_n_ranks(scores: np.ndarray, snap: Snapshot, n: int = 50) -> np.ndarray:
    """0-based rank of each pool item among the user's top-n positive scores; inf outside the top-n."""
    masked = np.where(snap.pool_mask, scores, -np.inf)
    top = np.argpartition(-masked, n, axis=1)[:, :n]
    top_scores = np.take_along_axis(masked, top, axis=1)
    order = np.argsort(-top_scores, axis=1, kind="stable")
    top = np.take_along_axis(top, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)
    ranks = np.full(scores.shape, np.inf, dtype=np.float32)
    rows = np.repeat(np.arange(len(scores)), n)
    rank_values = np.where(top_scores > 0, np.tile(np.arange(n), (len(scores), 1)), np.inf).ravel()
    ranks[rows, top.ravel()] = rank_values
    return ranks


def rank_fusion_scores(ranks: dict[str, np.ndarray], weights: dict[str, float], k: float = 10.0) -> np.ndarray:
    """Weighted reciprocal rank fusion: sum over methods of w / (k + rank + 1)."""
    out = np.zeros(next(iter(ranks.values())).shape, dtype=np.float32)
    for name, w in weights.items():
        if w > 0:
            out += w / (k + ranks[name] + 1.0)
    return out
