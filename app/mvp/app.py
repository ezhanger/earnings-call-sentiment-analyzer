"""Temporary MVP: what the analyzer expects after each earnings call, why, and what else is going on.

Run: uv run streamlit run app/mvp/app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ecsa import PROCESSED

MVP = PROCESSED / "mvp"
FEATURES = {
    "tone_prepared": "Tone of prepared remarks",
    "tone_qa": "Tone of management's Q&A answers",
    "tone_forward": "Tone of forward-looking statements",
    "tone_change": "Tone change vs last quarter",
    "uncertainty": "Uncertainty language",
    "analyst_tone": "Tone of analyst questions",
    "pre_runup_20d": "Stock vs S&P 500, 20 days before",
    "spy_20d": "S&P 500 trend, 20 days before",
    "spy_vol_20d": "Market volatility (20-day)",
    "sector_5d": "Sector peers, 5 days before",
}
TONE = {"tone_prepared", "tone_qa", "tone_forward", "tone_change", "uncertainty", "analyst_tone"}
UP, DOWN, GREY = "#1a9850", "#d73027", "#8c8c8c"

st.set_page_config(page_title="Earnings call expectations (MVP)", layout="wide")


@st.cache_data
def load():
    df = pd.read_parquet(MVP / "expectations.parquet")
    df["label"] = df.fiscal_quarter + " · " + df.call_datetime.dt.strftime("%Y-%m-%d")
    preds = df.pred_ar_1d.dropna()
    df["strength_pct"] = (df.pred_ar_1d - preds.median()).abs().rank(pct=True)
    prices = pd.read_parquet(PROCESSED / "prices.parquet")
    close = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return df, close, pd.read_parquet(MVP / "quotes.parquet")


def pct(x, signed=True):
    return "n/a" if pd.isna(x) else f"{x:+.1%}" if signed else f"{x:.1%}"


def direction(pred, med):
    """Expected moves are tiny; call it Up/Down relative to the typical prediction."""
    if pd.isna(pred):
        return "No expectation", GREY
    return ("Up", UP) if pred >= med else ("Down", DOWN)


df, close, quotes = load()
med = df.pred_ar_1d.median()

st.title("Earnings call expectations · MVP")
st.caption(
    "Temporary demo. **Expectation** = the 1-day move vs the S&P 500 that the call's tone and the "
    "market backdrop point to, from a model trained only on earlier years. Stand-ins: tone comes "
    "from a finance word list (FinBERT comes in Phase 2), the model is a ridge regression "
    "(LightGBM in Phase 3), and earnings surprise is not controlled for yet."
)

with st.sidebar:
    ticker = st.selectbox(
        "Company", sorted(df.ticker.unique()), index=sorted(df.ticker.unique()).index("NVDA")
    )
    tdf = df[df.ticker.eq(ticker)].sort_values("call_datetime", ascending=False)
    label = st.selectbox("Earnings call", tdf.label)
    st.markdown(
        f"**{tdf.company.dropna().iloc[0] if tdf.company.notna().any() else ticker}**  \n"
        f"{tdf.sector.iloc[0]} · {len(tdf)} calls"
    )

c = tdf[tdf.label.eq(label)].iloc[0]
tab_call, tab_hist, tab_eval = st.tabs(["This call", "Company history", "How reliable is it?"])

# ---------------------------------------------------------------- this call
with tab_call:
    word, color = direction(c.pred_ar_1d, med)
    m = st.columns(5)
    m[0].markdown(
        f"<div style='font-size:0.9rem;color:{GREY}'>Expectation</div>"
        f"<div style='font-size:2rem;font-weight:700;color:{color}'>{word}</div>",
        unsafe_allow_html=True,
    )
    m[1].metric(
        "Expected 1-day move vs S&P",
        pct(c.pred_ar_1d),
        help="Model output. Typical expected moves are within ±1% because tone explains "
        "only a small part of post-call moves.",
    )
    strength = (
        "n/a"
        if pd.isna(c.strength_pct)
        else "Strong"
        if c.strength_pct >= 0.8
        else "Moderate"
        if c.strength_pct >= 0.4
        else "Weak"
    )
    m[2].metric(
        "Signal strength",
        strength,
        None if pd.isna(c.strength_pct) else f"{c.strength_pct:.0%} percentile",
        delta_color="off",
        delta_arrow="off",
        help="How far this expectation sits from the typical one, across all calls. "
        "Up/Down is relative to the typical expectation.",
    )
    m[3].metric("Actual 1-day move vs S&P", pct(c.ar_1d))
    hit = (
        not pd.isna(c.pred_ar_1d)
        and not pd.isna(c.ar_1d)
        and (c.pred_ar_1d >= med) == (c.ar_1d >= 0)
    )
    m[4].metric(
        "Did it play out?",
        "n/a" if pd.isna(c.pred_ar_1d) or pd.isna(c.ar_1d) else "Yes" if hit else "No",
        f"3d {pct(c.ar_3d)} · 5d {pct(c.ar_5d)}",
        delta_color="off",
        delta_arrow="off",
    )
    if pd.isna(c.pred_ar_1d):
        st.info(
            "No expectation for this call: the model needs at least 200 earlier calls to "
            "train on, so the first few years have none."
        )

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Reasoning")
        contrib = pd.Series({FEATURES[f]: c[f"contrib_{f}"] for f in FEATURES}).dropna()
        if len(contrib):
            contrib = contrib.reindex(contrib.abs().sort_values().index)
            fig = go.Figure(
                go.Bar(
                    x=contrib * 1e4,
                    y=contrib.index,
                    orientation="h",
                    marker_color=[UP if v > 0 else DOWN for v in contrib],
                    hovertemplate="%{y}: %{x:+.0f} bp<extra></extra>",
                )
            )
            fig.update_layout(
                height=340,
                margin={"l": 0, "r": 10, "t": 10, "b": 30},
                xaxis_title="Push on expected move (basis points; 100 bp = 1%)",
            )
            st.plotly_chart(fig, width="stretch")

            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:3]
            raw = {FEATURES[f]: f for f in FEATURES}
            lines = []
            for name, v in top.items():
                f = raw[name]
                val = c[f]
                shown = (
                    f"{val:+.2f}"
                    if f in TONE and f != "uncertainty"
                    else f"{val:.2f} per 100 words"
                    if f == "uncertainty"
                    else pct(val, signed=f != "spy_vol_20d")
                )
                lines.append(
                    f"- **{name}** was {shown}, which "
                    f"{'lifts' if v > 0 else 'lowers'} the expectation by "
                    f"{abs(v) * 1e4:.0f} bp."
                )
            st.markdown("\n".join(lines))
            st.caption(
                f"Starting point is the average post-call move in training years "
                f"({pct(c.base_ar_1d)}); each bar is how far this call's value is from "
                f"normal times what that feature has been worth historically. Model "
                f"trained on {int(c.train_n):,} earlier calls."
            )

        t = st.columns(3)
        t[0].metric(
            "Management tone",
            f"{c.tone:+.2f}",
            help="Net tone, -1 to +1, of CEO, CFO "
            "and other executive sentences: (positive - negative) / (positive + negative).",
        )
        t[1].metric("CEO / CFO tone", f"{c.tone_ceo:+.2f} / {c.tone_cfo:+.2f}")
        t[2].metric(
            "Change vs last quarter", "n/a" if pd.isna(c.tone_change) else f"{c.tone_change:+.2f}"
        )

        q = quotes[quotes.call_id.eq(c.call_id)]
        for side, title in (
            ("positive", "Most upbeat management lines"),
            ("negative", "Most downbeat management lines"),
        ):
            with st.expander(title, expanded=side == "positive"):
                part = q[q.side.eq(side)]
                if part.empty:
                    st.write("None found.")
                for r in part.itertuples():
                    st.markdown(
                        f"> {r.text}  \n<small>{r.speaker_role.upper()} · "
                        f"{'Q&A' if r.section == 'qa' else 'prepared remarks'}</small>",
                        unsafe_allow_html=True,
                    )

    with right:
        st.subheader("Market context")
        vol_med = df.spy_vol_20d.median()
        k = st.columns(2)
        k[0].metric("S&P 500, 20 days before", pct(c.spy_20d))
        k[1].metric(
            "Market volatility",
            pct(c.spy_vol_20d, signed=False),
            f"typical {vol_med:.0%}",
            delta_color="off",
            delta_arrow="off",
        )
        k = st.columns(2)
        k[0].metric(f"{c.sector} peers, 5 days before", pct(c.sector_5d))
        k[1].metric("Stock vs S&P, 20 days before", pct(c.pre_runup_20d))
        k = st.columns(2)
        k[0].metric(
            "S&P 500, day after",
            pct(c.spy_1d),
            help="Already removed from the abnormal move, but a big market day adds noise.",
        )
        k[1].metric(
            "Sector peers, day after",
            pct(c.sector_after_1d),
            help="Peers often move with the reporting company (read-through).",
        )

        flags = []
        timing = {
            "after_close": "after the close",
            "before_open": "before the open",
            "during_market": "during market hours",
        }.get(c.timing, c.timing)
        flags.append(
            f"Results came out **{timing}** ({c.call_datetime:%b %d, %Y %H:%M} ET), "
            f"so the move is measured from the close on {c.start_date:%b %d}."
        )
        if c.macro_event:
            flags.append(
                f"**Macro backdrop: {c.macro_event}.** Broad market swings can drown "
                f"out what the company said."
            )
        if c.spy_vol_20d > 1.5 * vol_med:
            flags.append(
                "**Market volatility is high**, so single-stock moves are noisier and "
                "any tone signal is weaker."
            )
        if abs(c.pre_runup_20d) > 0.08:
            flags.append(
                f"The stock had moved **{pct(c.pre_runup_20d)} vs the market** in the "
                f"month before, so expectations were "
                f"{'already high: good news may be priced in' if c.pre_runup_20d > 0 else 'already low: a decent call can spark a relief rally'}."
            )
        if abs(c.sector_after_1d) > 0.015:
            flags.append(
                f"Sector peers moved **{pct(c.sector_after_1d)}** the next day, which "
                f"points to sector news (or read-through from this report)."
            )
        if abs(c.spy_1d) > 0.02:
            flags.append(f"The S&P 500 moved **{pct(c.spy_1d)}** the next day, a big market day.")
        flags.append(
            "**Not yet included: earnings surprise.** Whether EPS beat or missed "
            "consensus usually drives the first day more than tone does. Analyst tone "
            "partly stands in for it today."
        )
        st.markdown("\n".join(f"- {f}" for f in flags))

        # Price around the call, stock vs S&P, both rebased to 100 at the start close.
        d0 = c.start_date
        win = close.loc[d0 - pd.Timedelta(days=45) : d0 + pd.Timedelta(days=20), [ticker, "SPY"]]
        if len(win) and d0 in win.index:
            reb = win / win.loc[d0] * 100
            fig = go.Figure()
            fig.add_scatter(x=reb.index, y=reb[ticker], name=ticker, line_color="#2c7fb8")
            fig.add_scatter(x=reb.index, y=reb.SPY, name="S&P 500", line_color=GREY)
            fig.add_vline(x=d0, line_dash="dot", line_color=color)
            fig.update_layout(
                height=260,
                margin={"l": 0, "r": 0, "t": 30, "b": 0},
                title={"text": "Price around the call (start close = 100)", "font_size": 13},
                legend={"orientation": "h", "y": -0.15},
            )
            st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- company history
with tab_hist:
    h = tdf.sort_values("call_datetime")
    px = close[ticker].dropna()
    fig = go.Figure()
    fig.add_scatter(x=px.index, y=px, line_color="#9ecae1", name="Price", hoverinfo="skip")
    mk = h.dropna(subset=["start_date"])
    mk = mk.assign(price=px.reindex(mk.start_date).to_numpy())
    cols = [GREY if pd.isna(p) else UP if p >= med else DOWN for p in mk.pred_ar_1d]
    fig.add_scatter(
        x=mk.start_date,
        y=mk.price,
        mode="markers",
        name="Earnings call",
        marker={
            "color": cols,
            "size": 6 + 120 * mk.ar_1d.abs().fillna(0).clip(upper=0.15),
            "line": {"width": 1, "color": "white"},
        },
        customdata=np.stack(
            [mk.fiscal_quarter, mk.pred_ar_1d * 100, mk.ar_1d * 100, mk.tone], axis=1
        ),
        hovertemplate="%{customdata[0]}<br>Expected %{customdata[1]:+.2f}%<br>"
        "Actual %{customdata[2]:+.2f}%<br>Tone %{customdata[3]:+.2f}<extra></extra>",
    )
    fig.update_layout(
        height=420,
        yaxis_type="log",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        showlegend=False,
        yaxis_title="Adjusted price (log)",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Markers: green = expected up, red = expected down, grey = no expectation yet. "
        "Size = actual 1-day move vs the S&P 500."
    )
    ok = h.dropna(subset=["pred_ar_1d", "ar_1d"])
    if len(ok):
        st.write(
            f"For {ticker}, the expected direction matched the actual 1-day move on "
            f"**{((ok.pred_ar_1d >= med) == (ok.ar_1d >= 0)).mean():.0%}** of "
            f"{len(ok)} calls."
        )
    table = h.sort_values("call_datetime", ascending=False)[
        [
            "fiscal_quarter",
            "call_datetime",
            "tone",
            "tone_change",
            "pred_ar_1d",
            "ar_1d",
            "ar_5d",
            "spy_1d",
            "macro_event",
        ]
    ].rename(
        columns={
            "fiscal_quarter": "Quarter",
            "call_datetime": "Released",
            "tone": "Tone",
            "tone_change": "Tone Δ",
            "pred_ar_1d": "Expected 1d",
            "ar_1d": "Actual 1d",
            "ar_5d": "Actual 5d",
            "spy_1d": "S&P 1d",
            "macro_event": "Macro backdrop",
        }
    )
    st.dataframe(
        table.style.format(
            {
                "Tone": "{:+.2f}",
                "Tone Δ": "{:+.2f}",
                "Expected 1d": "{:+.2%}",
                "Actual 1d": "{:+.2%}",
                "Actual 5d": "{:+.2%}",
                "S&P 1d": "{:+.2%}",
                "Released": lambda d: f"{d:%Y-%m-%d %H:%M}",
            },
            na_rep="–",
        ),
        width="stretch",
        hide_index=True,
    )

# ---------------------------------------------------------------- reliability
with tab_eval:
    ok = df.dropna(subset=["pred_ar_1d", "ar_1d"]).copy()
    ok["hit"] = (ok.pred_ar_1d >= med) == (ok.ar_1d >= 0)
    e = st.columns(3)
    e[0].metric(
        "Direction hit rate",
        f"{ok.hit.mean():.1%}",
        "coin flip = 50%",
        delta_color="off",
        delta_arrow="off",
    )
    e[1].metric("Correlation, expected vs actual", f"{ok.pred_ar_1d.corr(ok.ar_1d):.2f}")
    e[2].metric("Calls with an out-of-sample expectation", f"{len(ok):,}")

    a, b = st.columns(2)
    ok["decile"] = pd.qcut(ok.pred_ar_1d, 10, labels=False) + 1
    dec = ok.groupby("decile").ar_1d.mean()
    fig = go.Figure(
        go.Bar(x=dec.index, y=dec * 100, marker_color=[UP if v > 0 else DOWN for v in dec])
    )
    fig.update_layout(
        height=320,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        title={"text": "Average actual move by expectation decile", "font_size": 13},
        xaxis_title="Expectation decile (1 = most bearish)",
        yaxis_title="Actual 1-day move vs S&P (%)",
    )
    a.plotly_chart(fig, width="stretch")

    yr = ok.groupby("year").hit.mean()
    fig = go.Figure(go.Bar(x=yr.index.astype(int), y=yr * 100, marker_color="#2c7fb8"))
    fig.add_hline(y=50, line_dash="dot", line_color=GREY)
    fig.update_layout(
        height=320,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        title={"text": "Direction hit rate by year", "font_size": 13},
        yaxis_title="%",
        yaxis_range=[30, 75],
    )
    b.plotly_chart(fig, width="stretch")

    calm = ok[ok.macro_event.eq("") & (ok.spy_vol_20d <= ok.spy_vol_20d.median() * 1.5)]
    rough = ok.drop(calm.index)
    st.markdown(
        f"**Market context matters:** in calm markets the hit rate is **{calm.hit.mean():.1%}** "
        f"({len(calm):,} calls); during macro events or high volatility it is "
        f"**{rough.hit.mean():.1%}** ({len(rough):,} calls)."
    )
    st.caption(
        "Each year's expectations come from a model fit only on earlier years, so none of "
        "these are in-sample. This is a word-list stand-in; treat it as a baseline for "
        "FinBERT + LightGBM to beat, not a trading signal."
    )
