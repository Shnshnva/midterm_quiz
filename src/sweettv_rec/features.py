"""Candidate generation and features for the LightGBM re-ranker.

Candidates are the union of each base method's top-N pool items per user plus the most popular pool items.
Every feature is computed from the snapshot's history window and the target-window schedule only.
"""

import logging

import numpy as np
import pandas as pd
from scipy import sparse

from sweettv_rec import recommenders as rec
from sweettv_rec.config import SCREEN_TIME_THRESHOLD
from sweettv_rec.data import week_start
from sweettv_rec.snapshot import Snapshot

logger = logging.getLogger(__name__)

TOP_N = 80
TOP_POPULAR = 30
SECONDS_PER_DAY = 24 * 3600


def base_scores(snap: Snapshot) -> dict[str, np.ndarray]:
    return {
        "habit": rec.habit_scores(snap),
        "habit_schedule": rec.habit_schedule_scores(snap, prior=100.0),
        "ppr": rec.ppr_scores(snap, alpha=0.5, weighting="count"),
        "als": rec.als_scores(snap, factors=128, regularization=5.0, alpha=0.5),
    }


def candidates(snap: Snapshot, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """(user row, item column) pairs: top-N of every method with a positive score, plus top popular items."""
    mask = snap.pool_mask
    n_users = len(snap.users)
    parts = []
    for s in scores.values():
        masked = np.where(mask, s, -np.inf)
        top = np.argpartition(-masked, TOP_N, axis=1)[:, :TOP_N]
        rows = np.repeat(np.arange(n_users), TOP_N)
        cols = top.ravel()
        keep = masked[rows, cols] > 0
        parts.append(np.c_[rows[keep], cols[keep]])
    popular = rec.popularity_ranking(snap)[:TOP_POPULAR]
    parts.append(np.c_[np.repeat(np.arange(n_users), len(popular)), np.tile(popular, n_users)])
    pairs = np.unique(np.vstack(parts), axis=0)
    return pd.DataFrame({"u": pairs[:, 0], "i": pairs[:, 1]})


def _pair_lookup(cand: pd.DataFrame, table: pd.DataFrame, snap: Snapshot, column: str) -> np.ndarray:
    """Look up a (user_id, tv_show_id) aggregate for each candidate pair; 0 when absent."""
    key = pd.MultiIndex.from_arrays([snap.users[cand["u"].to_numpy()], snap.items[cand["i"].to_numpy()]])
    series = table.set_index(["user_id", "tv_show_id"])[column]
    return series.reindex(key).fillna(0).to_numpy(dtype=np.float32)


def _slot_ids(channel: np.ndarray, start: np.ndarray, slot_index: pd.Index) -> np.ndarray:
    """Index of (channel, weekday, hour) for each timestamp; -1 for slots not in `slot_index`."""
    day = start // SECONDS_PER_DAY
    keys = channel.astype(np.int64) * 1000 + (day % 7) * 24 + (start % SECONDS_PER_DAY) // 3600
    return slot_index.get_indexer(keys)


def slot_habit(snap: Snapshot, views: pd.DataFrame, cutoff: int, weeks_back: int | None) -> np.ndarray:
    """users x items: the user's weekly seconds in each (channel, weekday, hour) slot, summed over the slots of
    the programme's target-window airings and divided by the window length in weeks.

    Uses all viewing, including unlabelled airtime, so it scores programmes the user has never watched.
    """
    lo = 0 if weeks_back is None else cutoff - weeks_back * 7 * SECONDS_PER_DAY
    v = views[(views["start"] < cutoff) & (views["start"] >= lo)]
    weeks = (cutoff - max(lo, int(v["start"].min()))) / (7 * SECONDS_PER_DAY)
    w = snap.window_airings
    window_weeks = (int(w["end"].max()) - int(w["start"].min())) / (7 * SECONDS_PER_DAY)
    keys = np.unique(
        np.r_[
            v["channel_id"].to_numpy().astype(np.int64) * 1000
            + (v["start"].to_numpy() // SECONDS_PER_DAY % 7) * 24
            + (v["start"].to_numpy() % SECONDS_PER_DAY) // 3600,
        ]
    )
    slot_index = pd.Index(keys)
    user_rows = pd.Index(snap.users).get_indexer(v["user_id"])
    v_slots = _slot_ids(v["channel_id"].to_numpy(), v["start"].to_numpy(), slot_index)
    keep = (user_rows >= 0) & (v_slots >= 0)
    user_slot = sparse.csr_matrix(
        (v["duration"].to_numpy(dtype=np.float32)[keep] / weeks, (user_rows[keep], v_slots[keep])),
        shape=(len(snap.users), len(slot_index)),
    )
    item_cols = pd.Index(snap.items).get_indexer(w["tv_show_id"])
    w_slots = _slot_ids(w["channel_id"].to_numpy(), w["start"].to_numpy(), slot_index)
    keep = (item_cols >= 0) & (w_slots >= 0)
    slot_item = sparse.csr_matrix(
        (np.full(int(keep.sum()), 1.0 / window_weeks, dtype=np.float32), (w_slots[keep], item_cols[keep])),
        shape=(len(slot_index), len(snap.items)),
    )
    return (user_slot @ slot_item).toarray()


def build_features(snap: Snapshot, views: pd.DataFrame, cutoff_week: int) -> pd.DataFrame:
    cutoff = week_start(cutoff_week)
    scores = base_scores(snap)
    cand = candidates(snap, scores)
    u, i = cand["u"].to_numpy(), cand["i"].to_numpy()
    feats = cand.copy()

    for name, s in scores.items():
        feats[name] = s[u, i]
        ranks = rec.top_n_ranks(s, snap, n=TOP_N)
        feats[f"{name}_rank"] = np.where(np.isinf(ranks[u, i]), TOP_N, ranks[u, i])

    for name, back in (("slot_habit", None), ("slot_habit_4w", 4)):
        sh = slot_habit(snap, views, cutoff, back)
        feats[name] = sh[u, i]
        feats[f"{name}_per_airing"] = feats[name] / np.maximum(rec.window_airing_counts(snap)[i], 1.0)

    feats["popularity"] = np.asarray((snap.matrix > 0).sum(axis=0)).ravel()[i]
    feats["window_airings"] = rec.window_airing_counts(snap)[i]
    feats["history_airings"] = rec.history_airing_counts(snap)[i]

    hist = snap.history
    full = hist[hist["frac"] >= SCREEN_TIME_THRESHOLD]
    recent = full[full["start"] >= cutoff - 14 * SECONDS_PER_DAY]
    agg_full = full.groupby(["user_id", "tv_show_id"]).agg(last=("start", "max")).reset_index()
    agg_recent = recent.groupby(["user_id", "tv_show_id"]).size().rename("n").reset_index()
    agg_any = hist.groupby(["user_id", "tv_show_id"]).agg(sec=("covered", "sum"), n=("airing_id", "size")).reset_index()
    last = _pair_lookup(cand, agg_full, snap, "last")
    feats["weeks_since_full_watch"] = np.where(last > 0, (cutoff - last) / (7 * SECONDS_PER_DAY), 99.0)
    feats["full_watches_last_2w"] = _pair_lookup(cand, agg_recent, snap, "n")
    feats["partial_seconds"] = _pair_lookup(cand, agg_any, snap, "sec")
    feats["airings_touched"] = _pair_lookup(cand, agg_any, snap, "n")

    user_total = np.asarray(snap.matrix.sum(axis=1)).ravel()
    feats["user_full_watches"] = user_total[u]
    feats["user_programmes"] = np.diff(snap.matrix.indptr)[u]
    feats["habit_share"] = feats["habit"] / np.maximum(feats["user_full_watches"], 1.0)

    # Channel affinity over all airtime, including unlabelled (tv_show_id = 0) programmes.
    v = views[views["start"] < cutoff]
    seconds = v.groupby(["user_id", "channel_id"])["duration"].sum()
    share = (seconds / seconds.groupby(level=0).transform("sum")).rename("share").reset_index()
    main_channel = snap.window_airings.groupby("tv_show_id")["channel_id"].agg(lambda s: s.value_counts().index[0])
    item_channel = main_channel.reindex(snap.items).to_numpy()
    key = pd.MultiIndex.from_arrays([snap.users[u], item_channel[i]])
    feats["channel_share"] = (
        share.set_index(["user_id", "channel_id"])["share"].reindex(key).fillna(0).to_numpy(dtype=np.float32)
    )
    logger.info("cutoff week %s: %s candidates for %s users", cutoff_week, len(feats), feats["u"].nunique())
    return feats


def add_labels(feats: pd.DataFrame, snap: Snapshot) -> pd.DataFrame:
    """Relevance 5..1 for the user's target programmes #1..#5, 0 otherwise. Users without a strict top-5 dropped."""
    rel = {(int(u), int(item)): 5 - pos for u, items in snap.truth.items() for pos, item in enumerate(items)}
    user_ids = snap.users[feats["u"].to_numpy()]
    item_ids = snap.items[feats["i"].to_numpy()]
    feats = feats.assign(label=[rel.get((int(a), int(b)), 0) for a, b in zip(user_ids, item_ids, strict=True)])
    has_truth = np.isin(user_ids, np.fromiter(snap.truth, dtype=np.uint64))
    return feats[has_truth].reset_index(drop=True)
