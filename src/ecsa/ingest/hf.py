"""Historical transcripts from kurry/sp500_earnings_transcripts (MIT, 2005-2025)."""

import subprocess

import duckdb
import pandas as pd

from ecsa import RAW

URL = (
    "https://huggingface.co/datasets/kurry/sp500_earnings_transcripts"
    "/resolve/main/parquet_files/part-0.parquet"
)


def load_transcripts(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """Rows for `tickers`, cached to data/raw/hf_transcripts.parquet."""
    path = RAW / "hf_transcripts.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        if set(tickers) <= set(df["symbol"]):
            return df[df["symbol"].isin(tickers)].reset_index(drop=True)
    # Remote DuckDB reads of this 1.8 GB file hang; download once (resumable) and filter locally.
    full = RAW / "hf_sp500_transcripts_full.parquet"
    RAW.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-fL", "-C", "-", "-o", str(full), URL], check=True)
    df = duckdb.execute(
        "SELECT * FROM read_parquet(?) WHERE list_contains(?, symbol)", [str(full), tickers]
    ).df()
    df.to_parquet(path)
    return df
