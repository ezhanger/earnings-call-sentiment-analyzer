from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Ticker -> GICS sector. GOOGL, not GOOG: the dataset has fuller history under GOOGL.
TICKERS = {
    "NVDA": "Information Technology",
    "AAPL": "Information Technology",
    "AMD": "Information Technology",
    "MSFT": "Information Technology",
    "AVGO": "Information Technology",
    "ORCL": "Information Technology",
    "CRM": "Information Technology",
    "ADBE": "Information Technology",
    "INTC": "Information Technology",
    "PLTR": "Information Technology",
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "NFLX": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
}
