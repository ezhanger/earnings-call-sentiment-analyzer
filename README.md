# earnings-call-sentiment-analyzer
Scores the tone of quarterly earnings calls with FinBERT and tests whether it predicts the stock's abnormal return over the next 1 to 5 trading days. See [docs/design-doc.pdf](docs/design-doc.pdf).

## Repository layout

```
src/ecsa/
  ingest/     # loaders: HF transcripts, API Ninjas, SEC EDGAR, yfinance
  parse/      # sentence split, section + speaker tagging
  prices/     # daily prices, trading calendar, abnormal returns
  score/      # batched FinBERT scoring + hand-label evaluation
  features/   # call-level tone features and controls
  model/      # baselines, LightGBM, walk-forward backtest, SHAP
app/          # Streamlit app (pages/ holds each screen)
notebooks/    # Colab scoring notebook
data/raw/     # raw downloads, not committed
data/processed/ # small Parquet tables the app reads
results/      # metrics, SHAP values, charts
tests/
docs/         # design doc
```

## Running the pipeline

```
uv sync
uv run python -m ecsa.build   # Phase 1: calls, sentences, prices -> data/processed/
uv run pytest
```

The first build downloads the 1.8 GB transcript file once into `data/raw/` (git-ignored), pulls
earnings-release timestamps from SEC EDGAR, and caches daily prices from yfinance. The ticker list
lives in `src/ecsa/__init__.py`.
