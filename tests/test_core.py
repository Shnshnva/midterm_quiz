import pandas as pd
import pytest

from sweettv_rec.join import join_airings, merge_views
from sweettv_rec.target import average_precision, frequency, map_at_k, top_k


def test_ap_matches_lecture_slide() -> None:
    # Midterm Quiz slide 5: relevant {A, B, C}; X = A B D C E -> 0.917, Y = D A E B C -> 0.533.
    assert average_precision([1, 2, 3], [1, 2, 4, 3, 5]) == pytest.approx((1 + 1 + 0.75) / 3)
    assert average_precision([1, 2, 3], [4, 1, 5, 2, 3]) == pytest.approx((0.5 + 0.5 + 0.6) / 3)


def test_ap_counts_duplicates_once_and_missing_user_is_zero() -> None:
    assert average_precision([1, 2, 3, 4, 5], [1, 1, 1, 1, 1]) == pytest.approx(1 / 5)
    assert map_at_k({7: [1, 2, 3, 4, 5], 8: [1, 2, 3, 4, 5]}, {7: [1, 2, 3, 4, 5]}) == pytest.approx(0.5)


def test_merge_and_join_by_hand() -> None:
    # User 1 watches channel 10 in two overlapping views [0, 70) and [50, 100): union is [0, 100).
    views = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "channel_id": [10, 10, 20],
            "start": [0, 50, 0],
            "stop": [70, 100, 30],
        }
    )
    airings = pd.DataFrame(
        {
            "airing_id": [0, 1, 2],
            "channel_id": [10, 10, 20],
            "start": [0, 90, 0],
            "end": [90, 200, 100],
            "duration": [90, 110, 100],
            "tv_show_id": [5, 6, 7],
        }
    )
    out = join_airings(merge_views(views), airings).set_index("airing_id")
    assert out.loc[0, "covered"] == 90  # not 70 + 40 = 110: merged views are not double-counted
    assert out.loc[0, "frac"] == pytest.approx(1.0)
    assert out.loc[1, "covered"] == 10
    assert out.loc[2, "frac"] == pytest.approx(0.3)


def test_target_threshold_and_strict_top5() -> None:
    rows = [(1, i, 5 + i % 6, 1.0 if i < 12 else 0.5, 100) for i in range(14)]
    ua = pd.DataFrame(rows, columns=["user_id", "airing_id", "tv_show_id", "frac", "covered"])
    freq = frequency(ua)
    assert freq["freq"].sum() == 12  # two airings below 0.8 are dropped
    assert top_k(freq)[1] == [
        5,
        6,
        7,
        8,
        9,
    ]  # all six tie at 2 airings and 200 s, so ties go to the smaller id
    assert top_k(frequency(ua[ua["tv_show_id"] < 8]), strict=True) == {}
