"""LightGBM lambdarank re-ranker over the candidates of habit x schedule, habit, PPR, ALS and popularity.

`validate`: train on snapshots whose labels end by week 20 (cutoffs 14, 16), score the hold-out 11-20 -> 21-30.
`submit`:   train on cutoffs 14, 16, 21, 24 (labels end by week 30), predict history 11-30 -> pool 31-42.
Every snapshot's history starts at week 11 and its target window ends at week 20 or 30.
"""

import argparse
import json
import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from sweettv_rec import recommenders as rec
from sweettv_rec.config import settings
from sweettv_rec.data import load
from sweettv_rec.features import add_labels, build_features
from sweettv_rec.snapshot import Snapshot, make_snapshot
from sweettv_rec.target import map_at_k

logger = logging.getLogger("ranker")

PARAMS = {
    "objective": "lambdarank",
    "metric": "map",
    "eval_at": [5],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 8,
}
NUM_ROUNDS = 300
SEEDS = (0, 1, 2)
NOT_FEATURES = ("u", "i", "label", "cutoff")


def labelled_frame(ua: pd.DataFrame, airings: pd.DataFrame, views: pd.DataFrame, cutoff: int, end: int) -> pd.DataFrame:
    snap = make_snapshot(ua, airings, (11, cutoff - 1), (cutoff, end))
    return add_labels(build_features(snap, views, cutoff), snap).assign(cutoff=cutoff)


def train(frames: list[pd.DataFrame]) -> list[lgb.Booster]:
    data = pd.concat(frames, ignore_index=True).sort_values(["cutoff", "u"], kind="stable")
    features = [c for c in data.columns if c not in NOT_FEATURES]
    groups = data.groupby(["cutoff", "u"], sort=False).size().to_numpy()
    logger.info("training rows %s, groups %s, positives %s", len(data), len(groups), int((data["label"] > 0).sum()))
    models = []
    for seed in SEEDS:
        dataset = lgb.Dataset(data[features], label=data["label"], group=groups)
        models.append(lgb.train({**PARAMS, "seed": seed}, dataset, num_boost_round=NUM_ROUNDS))
    return models


def predict_scores(models: list[lgb.Booster], feats: pd.DataFrame, snap: Snapshot) -> np.ndarray:
    """Dense users x items matrix, positive only for candidates, ordered by the mean seed prediction."""
    features = [c for c in feats.columns if c not in NOT_FEATURES]
    pred = np.mean([m.predict(feats[features]) for m in models], axis=0)
    scores = np.zeros((len(snap.users), len(snap.items)), dtype=np.float32)
    scores[feats["u"].to_numpy(), feats["i"].to_numpy()] = pred - pred.min() + 1.0
    return scores


def main(mode: str) -> None:
    t0 = time.perf_counter()
    ua, airings, views = load("user_airings"), load("airings_dedup"), load("views")
    template = pd.read_csv(settings.raw_dir / "submission.csv", dtype={"user_id": np.uint64})
    test_users = [int(u) for u in template["user_id"]]
    windows = [(14, 20), (16, 20)]
    if mode == "submit":
        windows += [(21, 30), (24, 30)]
    models = train([labelled_frame(ua, airings, views, c, e) for c, e in windows])

    importance = pd.Series(models[0].feature_importance("gain"), index=models[0].feature_name()).sort_values(
        ascending=False
    )
    logger.info("top features by gain: %s", importance.head(10).round(0).to_dict())

    if mode == "validate":
        snap = make_snapshot(ua, airings, (11, 20), (21, 30))
        feats = build_features(snap, views, 21)
        recs = rec.to_top5(predict_scores(models, feats, snap), snap)
        test_set = set(test_users)
        result = {
            "map5_all": map_at_k(snap.truth, recs),
            "map5_test_users": map_at_k({u: v for u, v in snap.truth.items() if u in test_set}, recs),
            "candidate_recall": float(
                np.mean(
                    [
                        len(set(t) & set(snap.items[feats.loc[feats["u"] == r, "i"]])) / 5
                        for u, t in list(snap.truth.items())[:500]
                        for r in [int(pd.Index(snap.users).get_indexer([np.uint64(u)])[0])]
                    ]
                )
            ),
            "windows": windows,
            "params": PARAMS,
            "rounds": NUM_ROUNDS,
            "seeds": list(SEEDS),
            "importance_gain": importance.round(1).to_dict(),
            "runtime_s": round(time.perf_counter() - t0, 1),
        }
        logger.info(
            "hold-out MAP@5 all=%.4f test=%.4f, candidate recall@~(first 500 users)=%.3f",
            result["map5_all"],
            result["map5_test_users"],
            result["candidate_recall"],
        )
        (settings.artifacts_dir / "validation_ranker.json").write_text(json.dumps(result, indent=1))
    else:
        snap = make_snapshot(ua, airings, (11, 30), (31, 42), with_truth=False)
        feats = build_features(snap, views, 31)
        recs = rec.to_top5(predict_scores(models, feats, snap), snap, test_users)
        from importlib import import_module

        import_module("06_make_submissions").write_submission(recs, template, snap, "ranker")
    logger.info("done in %.0f s", time.perf_counter() - t0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["validate", "submit"])
    main(parser.parse_args().mode)
