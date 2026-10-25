# Re-ranker experiments (hold-out: history 11-20, target 21-30; MAP@5 over 4,030 users / 1,150 test users)

One change at a time against the previous kept version. Numbers copied from the run logs in the AI session.

| # | Change | MAP@5 all | MAP@5 test users | Kept |
|---|---|---|---|---|
| R0 | Rank fusion (habit x schedule 2, PPR 1, ALS 0.5, k=20), no learning | 0.3340 | 0.3706 | baseline |
| R1 | LightGBM lambdarank, candidates top-30/method + 10 popular, cutoffs 14,16 | 0.3481 | 0.3842 | yes |
| R2 | Candidates top-80/method + 30 popular (candidate recall 0.62 -> 0.715) | 0.3483 | 0.3848 | yes |
| R3 | + slot-habit features (channel x weekday x hour, all airtime incl. id 0) | **0.3537** | 0.3909 | **yes = final v1** |
| R4 | Training cutoffs 13,14,15,16,17 instead of 14,16 | 0.3525 | 0.3895 | no |
| R5 | Blend lambdarank + binary objective (standardised outputs) | 0.3532 | 0.3915 | no |
| R6 | Recency features (4-week counts, habit with 4-week half-life) | 0.3527 | 0.3907 | no |
| R7 | Extra PPR features alpha 0.3 and 0.85 | 0.3532 | 0.3906 | no |
| R8 | lr 0.03, 600 rounds, 31 leaves, min_child 100 | 0.3524 | 0.3908 | no |
| R9 | Active-day exposure rate features | 0.3524 | 0.3909 | no |

Run-to-run spread is not measured beyond R3 being reproduced exactly (0.3537) after reverting R4-R9.
Kaggle: R0 private 0.41555 / public 0.42654; R3 (final, submissions/ranker_v1.csv) private 0.43773 / public 0.44867.
