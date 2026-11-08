import math

import pandas as pd
import pytest

from ecsa.prices.returns import abnormal_returns

# Mon 2024-01-08 .. Fri 2024-01-19, skipping the MLK holiday (Mon 2024-01-15).
DATES = pd.DatetimeIndex(["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
                          "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19"])
CLOSE = pd.DataFrame({"X": [100, 110, 121, 100, 90, 99, 90, 80, 88],
                      "SPY": [400, 404, 408, 400, 396, 400, 404, 400, 396]}, index=DATES, dtype=float)


def et(s):
    return pd.Timestamp(s, tz="America/New_York")


@pytest.mark.parametrize("event, start", [
    ("2024-01-09 16:30", "2024-01-09"),  # after close: that day's close
    ("2024-01-09 15:50", "2024-01-09"),  # 8-K accepted minutes before an after-close release
    ("2024-01-09 08:00", "2024-01-08"),  # before open: prior close
    ("2024-01-09 11:00", "2024-01-08"),  # lagged mid-session filing: prior close
    ("2024-01-15 07:00", "2024-01-12"),  # holiday: last close before it
])
def test_start_price_by_timing(event, start):
    i = DATES.get_loc(pd.Timestamp(start))
    k1 = CLOSE.iloc[i + 1] / CLOSE.iloc[i]
    assert abnormal_returns(CLOSE, "X", et(event))["ar_1d"] == pytest.approx(k1["X"] - k1["SPY"])


def test_hand_computed_after_close():
    # Start 2024-01-09: 1d ends 01-10 (X 110 -> 121, SPY 404 -> 408); 3d ends 01-12 (X -> 90, SPY -> 396).
    ar = abnormal_returns(CLOSE, "X", et("2024-01-09 16:30"))
    assert ar["ar_1d"] == pytest.approx(0.10 - 4 / 404)
    assert ar["ar_3d"] == pytest.approx(-20 / 110 - (-8 / 404))


def test_horizon_past_data_is_nan():
    assert math.isnan(abnormal_returns(CLOSE, "X", et("2024-01-17 16:30"))["ar_5d"])
