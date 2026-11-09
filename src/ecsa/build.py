"""Phase 1 build: calls, sentences and prices tables in data/processed/.

Run: uv run python -m ecsa.build
"""

import pandas as pd

from ecsa import PROCESSED, TICKERS
from ecsa.ingest.edgar import load_earnings_releases
from ecsa.ingest.hf import load_transcripts
from ecsa.parse.transcript import propagate_roles, to_sentences
from ecsa.prices.returns import AFTER_CLOSE, load_prices

ET = "America/New_York"


def timing_label(t: pd.Timestamp) -> str:
    since_midnight = pd.Timedelta(hours=t.hour, minutes=t.minute)
    if since_midnight >= AFTER_CLOSE:
        return "after_close"
    return "before_open" if since_midnight < pd.Timedelta(hours=9, minutes=30) else "during_market"


def match_releases(calls: pd.DataFrame, releases: pd.DataFrame) -> pd.DataFrame:
    """Event time per call: the nearest Item 2.02 8-K from 7 days before to 1 day after the
    transcript date (transcript dates are sometimes publication dates), else the transcript time."""
    out = []
    for c in calls.itertuples():
        r = releases[releases.ticker.eq(c.ticker)]
        gap = r.accepted_et.dt.tz_localize(None).dt.normalize() - c.transcript_dt.normalize()
        r = r[(gap >= pd.Timedelta(days=-7)) & (gap <= pd.Timedelta(days=1))]
        if len(r):
            out.append((r.loc[gap[r.index].abs().idxmin(), "accepted_et"], "edgar_8k"))
        else:
            out.append((c.transcript_dt.tz_localize(ET), "transcript"))
    return pd.DataFrame(out, columns=["call_datetime", "timing_source"], index=calls.index)


def main() -> None:
    tickers = list(TICKERS)
    tr = load_transcripts(tickers)
    tr = tr[tr.structured_content.map(len) > 0]
    tr["call_id"] = tr.symbol + "_" + tr.year.astype(str) + "Q" + tr.quarter.astype(str)
    tr = tr.sort_values("date").drop_duplicates("call_id", keep="last").reset_index(drop=True)

    calls = pd.DataFrame({
        "call_id": tr.call_id, "ticker": tr.symbol, "company": tr.company_name,
        "sector": tr.symbol.map(TICKERS),
        "fiscal_quarter": tr.year.astype(str) + "Q" + tr.quarter.astype(str),
        "transcript_dt": pd.to_datetime(tr.date), "source": "hf:kurry/sp500_earnings_transcripts",
    })
    calls = calls.join(match_releases(calls, load_earnings_releases(tickers)))
    calls["timing"] = calls.call_datetime.map(timing_label)

    sentences = pd.concat(
        [to_sentences(c, list(t)) for c, t in zip(tr.call_id, tr.structured_content)],
        ignore_index=True)
    sentences["speaker_role"] = propagate_roles(
        sentences, calls.set_index("call_id").call_datetime.dt.tz_localize(None))

    prices = load_prices(tickers)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    calls.to_parquet(PROCESSED / "calls.parquet", index=False)
    prices.to_parquet(PROCESSED / "prices.parquet", index=False)
    # One file per ticker: the app loads only the selected ticker, and no file nears GitHub's 100 MB cap.
    (PROCESSED / "sentences").mkdir(exist_ok=True)
    for t, g in sentences.groupby(sentences.call_id.str.split("_").str[0]):
        g.to_parquet(PROCESSED / "sentences" / f"{t}.parquet", index=False)

    report(calls, sentences, prices)


def report(calls: pd.DataFrame, sentences: pd.DataFrame, prices: pd.DataFrame) -> None:
    edgar = calls.timing_source.eq("edgar_8k")

    def after_close(t: pd.Series) -> pd.Series:
        return t.dt.hour * 60 + t.dt.minute >= 16 * 60

    agree = after_close(calls.call_datetime[edgar]) == after_close(calls.transcript_dt[edgar])
    per_call = sentences.groupby("call_id")

    def share(col: str, value: str) -> str:
        return f"{per_call[col].apply(lambda s: s.eq(value).any()).mean():.1%}"

    print(f"tickers {calls.ticker.nunique()}, calls {len(calls)}, sentences {len(sentences):,}, "
          f"price rows {len(prices):,}")
    print(f"timing from EDGAR 8-K: {edgar.mean():.1%} (gate: >= 90%)")
    print(f"transcript time agrees with EDGAR on before/after close: {agree.mean():.1%}")
    print("timing:", calls.timing.value_counts().to_dict())
    print(f"calls with Q&A {share('section', 'qa')}, CEO {share('speaker_role', 'ceo')}, "
          f"CFO {share('speaker_role', 'cfo')}, analyst {share('speaker_role', 'analyst')}")
    print("calls per ticker:", calls.ticker.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
