"""Temporary MVP: builds the tables the expectations demo reads.

Stand-ins until the real pipeline exists:
- Tone comes from a small finance word list, not FinBERT (Phase 2).
- The expected move comes from a walk-forward ridge regression, not LightGBM (Phase 3).
- There is no earnings-surprise (SUE) control yet.

Run: uv run python app/mvp/build.py   (about a minute; writes data/processed/mvp/)
"""

import re

import numpy as np
import pandas as pd

from ecsa import PROCESSED, TICKERS
from ecsa.prices.returns import start_index

OUT = PROCESSED / "mvp"

# Compact finance word lists in the spirit of Loughran-McDonald (2011).
POSITIVE = {
    "strong",
    "stronger",
    "strongest",
    "record",
    "records",
    "growth",
    "grew",
    "grow",
    "growing",
    "accelerate",
    "accelerated",
    "accelerating",
    "improve",
    "improved",
    "improvement",
    "improvements",
    "improving",
    "exceed",
    "exceeded",
    "exceeding",
    "beat",
    "beats",
    "outperform",
    "outperformed",
    "momentum",
    "robust",
    "solid",
    "healthy",
    "expand",
    "expanded",
    "expanding",
    "expansion",
    "gain",
    "gains",
    "gained",
    "profitable",
    "profitability",
    "opportunity",
    "opportunities",
    "pleased",
    "confident",
    "confidence",
    "excited",
    "exciting",
    "success",
    "successful",
    "successfully",
    "benefit",
    "benefited",
    "benefits",
    "favorable",
    "upside",
    "raise",
    "raised",
    "raising",
    "increase",
    "increased",
    "increasing",
    "higher",
    "achieve",
    "achieved",
    "achievement",
    "innovative",
    "innovation",
    "leadership",
    "efficient",
    "efficiency",
    "margin",
    "resilient",
    "resilience",
    "tailwind",
    "tailwinds",
    "encouraging",
    "encouraged",
    "optimistic",
    "progress",
    "outstanding",
    "excellent",
    "exceptional",
    "great",
    "good",
    "better",
    "best",
    "positive",
    "positively",
    "boost",
    "boosted",
}
NEGATIVE = {
    "weak",
    "weaker",
    "weakness",
    "weakening",
    "decline",
    "declined",
    "declines",
    "declining",
    "decrease",
    "decreased",
    "decreasing",
    "loss",
    "losses",
    "lost",
    "lower",
    "lowered",
    "lowering",
    "miss",
    "missed",
    "shortfall",
    "challenge",
    "challenges",
    "challenging",
    "difficult",
    "difficulty",
    "headwind",
    "headwinds",
    "pressure",
    "pressures",
    "pressured",
    "slowdown",
    "slow",
    "slower",
    "slowing",
    "soft",
    "softer",
    "softness",
    "impairment",
    "impairments",
    "restructuring",
    "layoffs",
    "charge",
    "charges",
    "writedown",
    "write-down",
    "down",
    "drop",
    "dropped",
    "fell",
    "fall",
    "falling",
    "disappointing",
    "disappointed",
    "concern",
    "concerns",
    "concerned",
    "risk",
    "risks",
    "adverse",
    "adversely",
    "negative",
    "negatively",
    "deteriorate",
    "deteriorated",
    "deterioration",
    "volatile",
    "volatility",
    "litigation",
    "recall",
    "delay",
    "delays",
    "delayed",
    "cut",
    "cuts",
    "reduce",
    "reduced",
    "reducing",
    "reduction",
    "unfavorable",
    "worse",
    "worst",
    "downturn",
    "recession",
    "inflation",
    "shortage",
    "shortages",
    "disruption",
    "disruptions",
    "problem",
    "problems",
}
UNCERTAIN = {
    "uncertain",
    "uncertainty",
    "uncertainties",
    "may",
    "might",
    "could",
    "possibly",
    "perhaps",
    "approximately",
    "appears",
    "appear",
    "depend",
    "depends",
    "depending",
    "unclear",
    "unpredictable",
    "fluctuate",
    "fluctuations",
    "assume",
    "assumption",
    "assumptions",
    "believe",
    "roughly",
    "tentative",
    "variable",
    "visibility",
}
NEGATORS = {"not", "no", "never", "without", "neither", "nor"}
WORD = re.compile(r"[a-z][a-z\-']*")

# Broad market episodes that can swamp a single company's call.
MACRO = [
    ("2007-08-01", "2009-03-31", "Global financial crisis"),
    ("2010-04-23", "2010-07-02", "Euro debt crisis / flash crash"),
    ("2011-07-22", "2011-10-04", "US downgrade and euro crisis selloff"),
    ("2015-08-10", "2016-02-29", "China devaluation and oil collapse"),
    ("2018-10-01", "2018-12-31", "Q4 2018 rate-hike selloff"),
    ("2020-02-20", "2020-06-30", "COVID crash and rebound"),
    ("2022-01-03", "2022-10-31", "Fed hiking cycle bear market"),
    ("2023-03-08", "2023-05-31", "Regional bank failures"),
    ("2025-04-02", "2025-05-31", "Tariff shock"),
]

FEATURES = {
    "tone_prepared": "Tone of prepared remarks",
    "tone_qa": "Tone of Q&A answers",
    "tone_forward": "Tone of forward-looking statements",
    "tone_change": "Tone change vs last quarter",
    "uncertainty": "Uncertainty language",
    "analyst_tone": "Tone of analyst questions",
    "pre_runup_20d": "Stock move vs SPY, 20 days before",
    "spy_20d": "S&P 500 move, 20 days before",
    "spy_vol_20d": "Market volatility (20-day)",
    "sector_5d": "Sector peers, 5 days before",
}


def score_sentence(text: str) -> tuple[int, int, int, int]:
    """(positive, negative, uncertain, word count); a negator up to 3 words back flips polarity."""
    words = WORD.findall(text.lower())
    pos = neg = unc = 0
    for i, w in enumerate(words):
        flipped = any(p in NEGATORS for p in words[max(0, i - 3) : i])
        if w in POSITIVE:
            neg, pos = (neg + 1, pos) if flipped else (neg, pos + 1)
        elif w in NEGATIVE:
            pos, neg = (pos + 1, neg) if flipped else (pos, neg + 1)
        if w in UNCERTAIN:
            unc += 1
    return pos, neg, unc, len(words)


def net(df: pd.DataFrame) -> float:
    p, n = df.pos.sum(), df.neg.sum()
    return (p - n) / (p + n) if p + n else 0.0


def call_tone(s: pd.DataFrame) -> pd.DataFrame:
    """Call-level tone features plus the 5 most positive and negative management sentences."""
    rows, quotes = [], []
    mgmt = s[s.speaker_role.isin(["ceo", "cfo", "other_exec"])]
    for cid, g in mgmt.groupby("call_id"):
        allg = s[s.call_id.eq(cid)]
        rows.append(
            {
                "call_id": cid,
                "tone": net(g),
                "tone_prepared": net(g[g.section.eq("prepared")]),
                "tone_qa": net(g[g.section.eq("qa")]),
                "tone_forward": net(g[g.is_forward_looking]),
                "tone_ceo": net(g[g.speaker_role.eq("ceo")]),
                "tone_cfo": net(g[g.speaker_role.eq("cfo")]),
                "analyst_tone": net(allg[allg.speaker_role.eq("analyst")]),
                "uncertainty": 100 * g.unc.sum() / max(g.words.sum(), 1),
                "n_sentences": len(allg),
            }
        )
        g = g[g.words.between(8, 80)].assign(score=lambda d: (d.pos - d.neg) / np.sqrt(d.words))
        for side, part in (
            ("positive", g.nlargest(5, "score")),
            ("negative", g.nsmallest(5, "score")),
        ):
            part = part[part.score > 0] if side == "positive" else part[part.score < 0]
            quotes += [
                {
                    "call_id": cid,
                    "side": side,
                    "text": r.text,
                    "speaker_role": r.speaker_role,
                    "section": r.section,
                    "score": r.score,
                }
                for r in part.itertuples()
            ]
    return pd.DataFrame(rows), pd.DataFrame(quotes)


def market_context(calls: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Returns around each call: realized abnormal moves and the market backdrop before it."""
    rets = close.pct_change()
    out = []
    for c in calls.itertuples():
        px = close[[c.ticker, "SPY"]].dropna()
        i = start_index(px.index, c.call_datetime)
        if i is None or i < 21:
            out.append({"call_id": c.call_id})
            continue
        d0 = px.index[i]

        def r(col, a, b, px=px):
            return px[col].iloc[b] / px[col].iloc[a] - 1

        row = {
            "call_id": c.call_id,
            "start_date": d0,
            "pre_runup_20d": r(c.ticker, i - 20, i) - r("SPY", i - 20, i),
            "spy_20d": r("SPY", i - 20, i),
            "spy_vol_20d": rets["SPY"].loc[:d0].iloc[-20:].std() * np.sqrt(252),
        }
        for k in (1, 3, 5):
            ok = i + k < len(px)
            row[f"ar_{k}d"] = r(c.ticker, i, i + k) - r("SPY", i, i + k) if ok else np.nan
            row[f"spy_{k}d"] = r("SPY", i, i + k) if ok else np.nan
        peers = [t for t, sec in TICKERS.items() if sec == c.sector and t != c.ticker]
        j = close.index.get_loc(d0)
        row["sector_5d"] = (close[peers].iloc[j] / close[peers].iloc[j - 5] - 1).mean()
        row["sector_after_1d"] = (
            close[peers].iloc[min(j + 1, len(close) - 1)] / close[peers].iloc[j] - 1
        ).mean()
        row["macro_event"] = next(
            (name for a, b, name in MACRO if pd.Timestamp(a) <= d0 <= pd.Timestamp(b)), ""
        )
        out.append(row)
    return pd.DataFrame(out)


def walk_forward(df: pd.DataFrame, target: str = "ar_1d", alpha: float = 50.0) -> pd.DataFrame:
    """Ridge on z-scored features; each year is predicted by a model fit on all earlier years.
    Returns predictions and per-feature contributions (coef x z), which the app shows as reasons."""
    cols = list(FEATURES)
    df = df.copy()
    df["year"] = df.start_date.dt.year
    df["pred_ar_1d"] = np.nan
    for c in cols:
        df[f"contrib_{c}"] = np.nan
    for y in sorted(df.year.dropna().unique()):
        train = df[(df.year < y)].dropna(subset=cols + [target])
        test = df.year.eq(y)
        if len(train) < 200:
            continue
        mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
        X = ((train[cols] - mu) / sd).to_numpy()
        yv = (
            train[target].clip(-0.15, 0.15).to_numpy()
        )  # trim outliers so a few calls don't dominate
        b0 = yv.mean()
        beta = np.linalg.solve(X.T @ X + alpha * np.eye(len(cols)), X.T @ (yv - b0))
        Z = ((df.loc[test, cols] - mu) / sd).fillna(0).to_numpy()
        contrib = Z * beta
        df.loc[test, [f"contrib_{c}" for c in cols]] = contrib
        df.loc[test, "pred_ar_1d"] = b0 + contrib.sum(axis=1)
        df.loc[test, "train_n"] = len(train)
        df.loc[test, "base_ar_1d"] = b0
    return df


def main() -> None:
    calls = pd.read_parquet(PROCESSED / "calls.parquet")
    prices = pd.read_parquet(PROCESSED / "prices.parquet")
    close = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()

    tones, quotes = [], []
    for t in sorted(calls.ticker.unique()):
        s = pd.read_parquet(PROCESSED / "sentences" / f"{t}.parquet")
        scored = pd.DataFrame(
            [score_sentence(x) for x in s.text],
            columns=["pos", "neg", "unc", "words"],
            index=s.index,
        )
        a, b = call_tone(s.join(scored))
        tones.append(a)
        quotes.append(b)
        print(t, len(a))
    tone = pd.concat(tones, ignore_index=True)

    df = calls.merge(tone, on="call_id", how="left").merge(
        market_context(calls, close), on="call_id", how="left"
    )
    df = df.sort_values(["ticker", "call_datetime"])
    df["tone_change"] = df.tone - df.groupby("ticker").tone.shift(1)
    df = walk_forward(df)

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "expectations.parquet", index=False)
    pd.concat(quotes, ignore_index=True).to_parquet(OUT / "quotes.parquet", index=False)

    ok = df.dropna(subset=["pred_ar_1d", "ar_1d"])
    hit = (np.sign(ok.pred_ar_1d) == np.sign(ok.ar_1d)).mean()
    print(
        f"calls {len(df)}, with out-of-sample expectation {len(ok)}, direction hit rate {hit:.1%}, "
        f"corr {ok.pred_ar_1d.corr(ok.ar_1d):.3f}"
    )


if __name__ == "__main__":
    main()
