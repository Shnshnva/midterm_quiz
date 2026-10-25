# sweet.tv TV programme recommender — LLM4Rec final exam (AVITO)

For each of the 1,260 users of the Kaggle community competition
[sweet.tv — TV program recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender), predict the
five programmes they will watch most often (airings watched for at least 80%) in weeks 31–42.

**Final submission:** `submissions/ranker_v1.csv` — LightGBM lambdarank re-ranker over habit, habit × schedule,
Personalized PageRank, implicit ALS and popularity candidates. Kaggle private 0.43773 / public 0.44867.
Hold-out MAP@5 (history weeks 11–20 → target 21–30): 0.3537. Report: `report/`.

| Method | Hold-out MAP@5 | Kaggle private | File |
|---|---|---|---|
| Popularity (baseline) | 0.2138 | 0.29969 | `submissions/popularity.csv` |
| Personalized PageRank | 0.3232 | 0.40882 | `submissions/ppr.csv` |
| Implicit ALS | 0.3029 | 0.38450 | `submissions/als.csv` |
| Rank fusion | 0.3340 | 0.41555 | `submissions/fusion.csv` |
| **Re-ranker (final)** | **0.3537** | **0.43773** | `submissions/ranker_v1.csv` |

## Reproduce

Requires [uv](https://docs.astral.sh/uv/), the Kaggle CLI with the competition rules accepted, and on macOS
`brew install libomp` for LightGBM. The data is not in the repository.

```
uv sync
kaggle competitions download -c sweettv-tv-program-recommender -p data/raw && (cd data/raw && unzip *.zip)
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=scripts
uv run python scripts/01_build_cache.py          # CSV -> parquet, 24 s
uv run python scripts/02_probe.py                # clock alignment, coverage -> artifacts/probe.json
uv run python scripts/03_join.py                 # view x schedule join, 4 s -> artifacts/join_stats.json
uv run python scripts/04_validate.py             # hold-out grids for all methods -> artifacts/validation.json
uv run python scripts/05_validate_ext.py         # extended ALS / habit x schedule grids
uv run python scripts/06_make_submissions.py     # popularity, habit_schedule, ppr, als, fusion submissions
uv run python scripts/07_validate_fusion.py      # rank-fusion weights on the hold-out
uv run python scripts/08_ranker.py validate      # re-ranker hold-out score, 49 s
uv run python scripts/08_ranker.py submit        # writes submissions/ranker.csv (== ranker_v1.csv), 91 s
uv run python scripts/09_report_data.py          # method table, bootstrap CIs, example users, hand-check rows
uv run python scripts/10_submission_accounting.py
uv run pytest -q
```

Seeds: ALS `random_state=0`, LightGBM seeds 0–2, bootstrap seed 0. Environment: Apple M5 (10 cores, 24 GB),
Python 3.12.15, versions pinned in `uv.lock`.

## Main choices

- A view matches every airing on the same channel whose interval overlaps it; overlapping views of a user on a
  channel are merged first. An airing is watched when covered seconds / duration ≥ 0.8 (competition Data page).
- `tv_show_id = 0` airings are never items; they feed channel-share and slot-habit features.
- Candidate pool: programmes airing in the target window; repeats allowed; empty places and users without history
  get the popularity ranking.
