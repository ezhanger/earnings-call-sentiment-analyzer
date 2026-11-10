# Expectations MVP (temporary)

A throwaway demo of what the analyzer expects after each earnings call, why, and what else in the
market could change the outcome. It will be replaced by the Phase 4 app.

```
uv sync --group mvp
uv run streamlit run app/mvp/app.py
```

`data/processed/mvp/` is committed, so the app runs without rebuilding. To rebuild it from the
Phase 1 tables (about a minute): `uv run python app/mvp/build.py`.

**Stand-ins:** tone uses a small finance word list instead of FinBERT; the expected move is a
walk-forward ridge regression (each year predicted by a model fit only on earlier years) instead of
LightGBM; there is no earnings-surprise control yet. Out of sample it gets the 1-day direction
right on about 55% of 3,131 calls (correlation 0.14), with analyst-question tone the strongest
input, likely because it reflects the beat or miss.
