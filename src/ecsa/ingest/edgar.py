"""Earnings release timestamps from SEC EDGAR 8-K filings with Item 2.02."""

import time

import pandas as pd
import requests

from ecsa import RAW

# SEC asks every client to identify itself; max 10 requests/second.
HEADERS = {"User-Agent": "ecsa research enochzhang898@gmail.com"}
BASE = "https://data.sec.gov/submissions/"

# Predecessor registrants whose earnings 8-Ks predate the current CIK.
EXTRA_CIKS = {
    "GOOGL": [1288776],  # Google Inc., before the 2015 Alphabet reorganization
    "AVGO": [1441634, 1649338],  # Avago Technologies, Broadcom Ltd.
}


def _get(url: str) -> dict:
    time.sleep(0.15)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _ciks(ticker: str) -> list[int]:
    path = RAW / "sec_company_tickers.json"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
    by_ticker = {v["ticker"]: v["cik_str"] for v in pd.read_json(path).T.to_dict("records")}
    return [by_ticker[ticker], *EXTRA_CIKS.get(ticker, [])]


def _filings(cik: int) -> pd.DataFrame:
    d = _get(f"{BASE}CIK{cik:010d}.json")
    pages = [d["filings"]["recent"]] + [_get(BASE + f["name"]) for f in d["filings"]["files"]]
    return pd.concat([pd.DataFrame(p) for p in pages], ignore_index=True)


def load_earnings_releases(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """One row per Item 2.02 8-K: ticker, accession, accepted_et (America/New_York)."""
    path = RAW / "edgar_earnings_8k.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        if set(tickers) <= set(df["ticker"]):
            return df[df["ticker"].isin(tickers)].reset_index(drop=True)
    frames = []
    for t in tickers:
        for cik in _ciks(t):
            f = _filings(cik)
            f = f[f["form"].eq("8-K") & f["items"].str.contains("2.02", regex=False)]
            frames.append(pd.DataFrame({"ticker": t, "accession": f["accessionNumber"],
                                        "accepted_et": f["acceptanceDateTime"]}))
    df = pd.concat(frames, ignore_index=True)
    # acceptanceDateTime is true UTC; filingDate rolls to the next day after 17:30 ET, so never use it.
    df["accepted_et"] = pd.to_datetime(df["accepted_et"], utc=True).dt.tz_convert("America/New_York")
    df.to_parquet(path)
    return df
