"""Daily prices (yfinance, cached) and post-call abnormal returns vs SPY."""

import pandas as pd
import yfinance as yf

from ecsa import RAW

# Releases at or after this time count as after the close. Not 16:00: some after-close 8-Ks are
# accepted minutes early (MSFT 2011-01-27 15:57), while lagged next-day filings land earlier
# (INTC 2013-07-18 15:38, a day after the release).
AFTER_CLOSE = pd.Timedelta(hours=15, minutes=45)


def load_prices(tickers: list[str], start: str = "2004-12-01", refresh: bool = False) -> pd.DataFrame:
    """Long table: ticker, date, adj_close, volume. Includes SPY. Downloaded once, then cached
    because Yahoo throttles heavy use (HTTP 429)."""
    path = RAW / "yf_prices.parquet"
    want = sorted({*tickers, "SPY"})
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        if set(want) <= set(df["ticker"]):
            return df[df["ticker"].isin(want)].reset_index(drop=True)
    raw = yf.download(want, start=start, auto_adjust=True, progress=False, group_by="column")
    df = (raw[["Close", "Volume"]].stack(level=1, future_stack=True).reset_index()
          .rename(columns={"Date": "date", "Ticker": "ticker", "Close": "adj_close", "Volume": "volume"})
          .dropna(subset=["adj_close"]))
    df = df[["ticker", "date", "adj_close", "volume"]].sort_values(["ticker", "date"])
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df.reset_index(drop=True)


def start_index(dates: pd.DatetimeIndex, event_et: pd.Timestamp) -> int | None:
    """Position of the start close: same-day close for events from 15:45 ET, prior close for
    events before the open or during market hours."""
    cutoff = event_et.tz_localize(None) if event_et.tzinfo else event_et
    i = dates.searchsorted(cutoff - AFTER_CLOSE, side="right") - 1
    return int(i) if i >= 0 else None


def abnormal_returns(close: pd.DataFrame, ticker: str, event_et: pd.Timestamp,
                     horizons=(1, 3, 5)) -> dict[str, float]:
    """AR(t->t+k) = P[t+k]/P[t] - SPY[t+k]/SPY[t]; `close` is wide (date index, ticker columns)."""
    px = close[[ticker, "SPY"]].dropna()
    i = start_index(px.index, event_et)
    out = {}
    for k in horizons:
        ok = i is not None and i + k < len(px)
        out[f"ar_{k}d"] = (px[ticker].iloc[i + k] / px[ticker].iloc[i]
                           - px["SPY"].iloc[i + k] / px["SPY"].iloc[i]) if ok else float("nan")
    return out
