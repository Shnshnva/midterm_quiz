"""Report material on the hold-out (history 11-20, target 21-30):

- per-method MAP@5 with paired bootstrap 95% CIs of differences (artifacts/report_methods.json)
- two example users with every method's top-5, their history and their actual top-5 (artifacts/report_users.md)
- raw rows for a hand check of the join and of one AP@5 (artifacts/hand_check.md)
"""

import json
import logging
from importlib import import_module

import numpy as np
import pandas as pd

from sweettv_rec import recommenders as rec
from sweettv_rec.config import settings
from sweettv_rec.data import load, week_of
from sweettv_rec.features import build_features
from sweettv_rec.snapshot import make_snapshot
from sweettv_rec.target import average_precision, frequency

logger = logging.getLogger("report")
ranker = import_module("08_ranker")

BOOTSTRAP = 2000
SEED = 0


def bootstrap_diff(a: np.ndarray, b: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(a), size=(BOOTSTRAP, len(a)))
    d = (a[idx] - b[idx]).mean(axis=1)
    return [float(np.mean(a - b)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def main() -> None:
    ua, airings, views = load("user_airings"), load("airings_dedup"), load("views")
    programmes = load("programmes").set_index("tv_show_id")
    template = pd.read_csv(settings.raw_dir / "submission.csv", dtype={"user_id": np.uint64})
    test_set = {int(u) for u in template["user_id"]}
    snap = make_snapshot(ua, airings, (11, 20), (21, 30))

    scores = {
        "popularity": rec.popularity_scores(snap),
        "habit_schedule": rec.habit_schedule_scores(snap, prior=100.0),
        "ppr": rec.ppr_scores(snap, alpha=0.5, weighting="count"),
        "als": rec.als_scores(snap, factors=128, regularization=5.0, alpha=0.5),
    }
    ranks = {k: rec.top_n_ranks(scores[k], snap) for k in ("habit_schedule", "ppr", "als")}
    scores["fusion"] = rec.rank_fusion_scores(ranks, {"habit_schedule": 2.0, "ppr": 1.0, "als": 0.5}, k=20.0)
    models = ranker.train([ranker.labelled_frame(ua, airings, views, c, e) for c, e in [(14, 20), (16, 20)]])
    scores["ranker"] = ranker.predict_scores(models, build_features(snap, views, 21), snap)
    recs = {name: rec.to_top5(s, snap) for name, s in scores.items()}

    users = list(snap.truth)
    ap = {name: np.array([average_precision(snap.truth[u], r[u]) for u in users]) for name, r in recs.items()}
    is_test = np.array([u in test_set for u in users])
    methods = {
        name: {
            "map5_all": float(a.mean()),
            "map5_test_users": float(a[is_test].mean()),
            "users_ap_zero_share": float((a == 0).mean()),
        }
        for name, a in ap.items()
    }
    diffs = {
        "ranker_minus_ppr": bootstrap_diff(ap["ranker"], ap["ppr"]),
        "ranker_minus_fusion": bootstrap_diff(ap["ranker"], ap["fusion"]),
        "ppr_minus_als": bootstrap_diff(ap["ppr"], ap["als"]),
        "als_minus_popularity": bootstrap_diff(ap["als"], ap["popularity"]),
        "habit_schedule_minus_ppr": bootstrap_diff(ap["habit_schedule"], ap["ppr"]),
    }

    # Share of target programmes already fully watched in history: how much of the task is repetition.
    hist_freq = frequency(snap.history)
    seen = set(zip(hist_freq["user_id"].astype("uint64").map(int), hist_freq["tv_show_id"], strict=True))
    repeat_share = float(np.mean([(u, i) in seen for u in users for i in snap.truth[u]]))
    rec_repeat_share = {name: float(np.mean([(u, i) in seen for u in users for i in r[u]])) for name, r in recs.items()}
    # MAP@5 by history size: number of fully watched programmes in weeks 11-20.
    n_prog = hist_freq.groupby("user_id").size()
    n_prog.index = n_prog.index.astype("uint64").map(int)
    sizes = np.array([int(n_prog.get(u, 0)) for u in users])
    buckets = {"0": (0, 0), "1-2": (1, 2), "3-10": (3, 10), "11-30": (11, 30), ">30": (31, 10**9)}
    segments = {
        b: {"users": int(((sizes >= lo) & (sizes <= hi)).sum())}
        | {name: float(a[(sizes >= lo) & (sizes <= hi)].mean()) for name, a in ap.items()}
        for b, (lo, hi) in buckets.items()
        if ((sizes >= lo) & (sizes <= hi)).any()
    }
    (settings.artifacts_dir / "report_methods.json").write_text(
        json.dumps(
            {
                "truth_users": len(users),
                "truth_test_users": int(is_test.sum()),
                "methods": methods,
                "paired_bootstrap_mean_ci95": diffs,
                "target_repeat_share": repeat_share,
                "recommended_repeat_share": rec_repeat_share,
                "map5_by_history_programmes": segments,
                "bootstrap": {"resamples": BOOTSTRAP, "seed": SEED},
            },
            indent=1,
        )
    )

    # Two users from the test set: heavy = 95th percentile of history full watches; light = 1-2 programmes.
    per_user = hist_freq.groupby("user_id").agg(watches=("freq", "sum"), programmes=("tv_show_id", "size"))
    per_user.index = per_user.index.astype("uint64").map(int)
    cand = per_user[per_user.index.isin([u for u in users if u in test_set])]
    target_watches = cand["watches"].quantile(0.95)
    heavy = int((cand["watches"] - target_watches).abs().idxmin())
    light = int(cand[cand["programmes"] <= 2].index.min())

    def label(pid: int) -> str:
        row = programmes.loc[pid]
        return f"{pid} | {str(row['tv_show_title'])[:45]} | {row['tv_show_category']}"

    lines = ["# Example users (hold-out: history weeks 11-20, actual = target weeks 21-30)\n"]
    for tag, u in (("heavy", heavy), ("light", light)):
        hf = hist_freq[hist_freq["user_id"].astype("uint64").map(int) == u].sort_values("freq", ascending=False)
        lines.append(
            f"## {tag} user {u}: {int(cand.loc[u, 'watches'])} full watches of "
            f"{int(cand.loc[u, 'programmes'])} programmes in weeks 11-20\n"
        )
        lines.append("History top-5 (programme | title | category | full watches):\n")
        lines += [f"- {label(int(r.tv_show_id))} | {int(r.freq)}" for r in hf.head(5).itertuples()]
        lines.append("\nActual top-5 in weeks 21-30:\n")
        lines += [f"{k}. {label(p)}" for k, p in enumerate(snap.truth[u], start=1)]
        for name, r in recs.items():
            hits = [("HIT" if p in snap.truth[u] else "") for p in r[u]]
            lines.append(f"\n**{name}** (AP@5 = {average_precision(snap.truth[u], r[u]):.4f}):\n")
            lines += [f"{k}. {label(p)} {h}" for k, (p, h) in enumerate(zip(r[u], hits, strict=True), start=1)]
        lines.append("")
    (settings.artifacts_dir / "report_users.md").write_text("\n".join(lines))

    # Hand check: one fully watched airing of the light user and the ranker's AP@5 for that user.
    ua_u = ua[(ua["user_id"].astype("uint64").map(int) == light) & (ua["frac"] >= 0.8) & (week_of(ua["start"]) <= 20)]
    a = ua_u.sort_values("start").iloc[0]
    air = airings[airings["airing_id"] == a["airing_id"]].iloc[0]
    v = views[
        (views["user_id"].astype("uint64").map(int) == light)
        & (views["channel_id"] == air["channel_id"])
        & (views["stop"] > air["start"])
        & (views["start"] < air["end"])
    ]
    t0 = pd.Timestamp("2020-03-09")
    check = [
        "# Hand-check material (to be recomputed by hand by the student)\n",
        f"## Join: user {light}, channel {int(air['channel_id'])}, programme {int(air['tv_show_id'])}\n",
        f"Schedule row: start {t0 + pd.Timedelta(seconds=int(air['start']))}, duration {int(air['duration'])} s, "
        f"end {t0 + pd.Timedelta(seconds=int(air['end']))}",
        "\nView-log rows overlapping it (start, stop, duraton):\n",
    ]
    check += [
        f"- {t0 + pd.Timedelta(seconds=int(r.start))}  {t0 + pd.Timedelta(seconds=int(r.stop))}  {int(r.duration)}"
        for r in v.itertuples()
    ]
    check.append(f"\nCode output: covered {int(a['covered'])} s, watched share {a['frac']:.4f}\n")
    # A partially covered airing of the heavy user, reached by several views.
    ua_h = ua[(ua["user_id"].astype("uint64").map(int) == heavy) & (ua["frac"] > 0.5) & (ua["frac"] < 0.95)]
    for a in ua_h.sort_values("start").itertuples():
        air = airings[airings["airing_id"] == a.airing_id].iloc[0]
        v = views[
            (views["user_id"].astype("uint64").map(int) == heavy)
            & (views["channel_id"] == air["channel_id"])
            & (views["stop"] > air["start"])
            & (views["start"] < air["end"])
        ]
        if len(v) >= 2:
            break
    check += [
        f"\n## Join, partial coverage: user {heavy}, channel {int(air['channel_id'])}, "
        f"programme {int(air['tv_show_id'])}\n",
        f"Schedule row: start {t0 + pd.Timedelta(seconds=int(air['start']))}, duration {int(air['duration'])} s, "
        f"end {t0 + pd.Timedelta(seconds=int(air['end']))}",
        "\nView-log rows overlapping it (start, stop, duraton):\n",
    ]
    check += [
        f"- {t0 + pd.Timedelta(seconds=int(r.start))}  {t0 + pd.Timedelta(seconds=int(r.stop))}  {int(r.duration)}"
        for r in v.itertuples()
    ]
    check.append(f"\nCode output: covered {int(a.covered)} s, watched share {a.frac:.4f}\n")
    truth, pred = snap.truth[light], recs["ranker"][light]
    check.append(f"## AP@5 of the ranker for user {light}\n\nactual top-5: {truth}\n\nranker top-5: {pred}\n")
    check.append(f"Code output: AP@5 = {average_precision(truth, pred):.6f}\n")
    (settings.artifacts_dir / "hand_check.md").write_text("\n".join(check))
    summary = {k: round(v["map5_all"], 4) for k, v in methods.items()}
    logger.info("heavy %s, light %s; methods %s", heavy, light, summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
