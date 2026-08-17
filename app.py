"""QuantBoost research terminal (Streamlit).

    streamlit run app.py

This file is presentation only. Every calculation lives in src/quantboost/, so the
app, the notebooks and scripts/run_pipeline.py all run the identical pipeline.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from quantboost import tsa
from quantboost.backtest import (backtest, compare_models, cost_sensitivity,
                                equity_curve, performance, regime_breakdown,
                                run_walk_forward)
from quantboost.config import Config, DEFAULT_UNIVERSE
from quantboost.data import HAS_YF, audit, load_prices
from quantboost.features import build_panel, feature_columns
from quantboost.labeling import triple_barrier
from quantboost.models import HAS_CAT, HAS_XGB
from quantboost.portfolio import factor_attribution, kelly_fraction
from quantboost.risk import var_table
from quantboost.tuning import HAS_OPTUNA, rank_ic, tune_xgb

# palette shared with style.css so the charts belong to the page
PAPER, INK, INK_2, RULE = "#F8F3EC", "#2B2320", "#6B5F58", "#DDD3C8"
ACCENT, UP, DOWN = "#8C2F26", "#1F6F63", "#B4453B"

st.set_page_config(page_title="QuantBoost Research Terminal", page_icon="§",
                   layout="wide", initial_sidebar_state="expanded")


def inject_css(name: str = "style.css") -> None:
    p = ROOT / name
    if p.exists():
        st.markdown(f"<style>{p.read_text()}</style>", unsafe_allow_html=True)


inject_css()


# ------------------------------------------------------------------ charts
def base_fig(height: int = 340) -> go.Figure:
    f = go.Figure()
    f.update_layout(
        height=height, paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font=dict(family="system-ui, -apple-system, sans-serif", size=12, color=INK),
        margin=dict(l=8, r=8, t=28, b=8), hovermode="x unified",
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor=RULE, ticks="outside", tickcolor=RULE),
        yaxis=dict(gridcolor=RULE, zerolinecolor=RULE, tickformat=",.2f"))
    return f


def fig_lines(curves: dict, height: int = 400, ytitle: str = "") -> go.Figure:
    f = base_fig(height)
    colors = [ACCENT, INK, UP, INK_2, DOWN]
    for i, (name, s) in enumerate(curves.items()):
        f.add_trace(go.Scatter(x=s.index, y=s.to_numpy(), name=name, mode="lines",
                               line=dict(width=2.2 if i == 0 else 1.3,
                                         color=colors[i % len(colors)])))
    if ytitle:
        f.update_yaxes(title=ytitle)
    return f


def fig_bars(s: pd.Series, title: str = "", horizontal: bool = False) -> go.Figure:
    f = base_fig(360 if horizontal else 280)
    cols = [UP if v >= 0 else DOWN for v in s.to_numpy()]
    if horizontal:
        f.add_trace(go.Bar(x=s.to_numpy(), y=[str(i) for i in s.index],
                           orientation="h", marker_color=cols))
        f.update_layout(yaxis=dict(autorange="reversed"))
    else:
        f.add_trace(go.Bar(x=[str(i) for i in s.index], y=s.to_numpy(), marker_color=cols))
    if title:
        f.update_layout(title=dict(text=title, font=dict(size=13, color=INK_2)))
    return f


def fig_drawdown(eq: pd.Series) -> go.Figure:
    dd = (eq / eq.cummax() - 1) * 100
    f = base_fig(230)
    f.add_trace(go.Scatter(x=dd.index, y=dd.to_numpy(), fill="tozeroy", name="Drawdown",
                           line=dict(color=DOWN, width=1),
                           fillcolor="rgba(180,69,59,0.16)"))
    f.update_yaxes(title="%", tickformat=",.0f")
    return f


# ------------------------------------------------------------------ cached work
@st.cache_data(show_spinner=False, ttl=3600)
def cached_prices(tickers: tuple, start: str, end: str, offline: bool):
    px = load_prices(list(tickers), start, end, offline, cache=False)
    return px, px.attrs.get("source", "unknown")


@st.cache_data(show_spinner=False)
def cached_panel(prices: pd.DataFrame, horizon: int):
    return build_panel(prices, horizon)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<span class="eyebrow">Research desk</span>', unsafe_allow_html=True)
    st.markdown("## Specification")
    tickers = st.multiselect("Universe", DEFAULT_UNIVERSE, default=DEFAULT_UNIVERSE)
    c1, c2 = st.columns(2)
    start = c1.text_input("From", "2012-01-01")
    end = c2.text_input("To", str(pd.Timestamp.today().date()))
    offline = st.toggle("Simulated data", value=not HAS_YF,
                        help="Regime-switching GBM panel. Doubles as a null test: "
                             "signal on simulated noise means the pipeline leaks.")

    st.markdown("### Signal")
    horizon = st.select_slider("Forecast horizon (days)", [1, 5, 21], value=5)
    quantile = st.slider("Long / short tail", 0.05, 0.5, 0.2, 0.05)
    rebalance = st.slider("Rebalance every (days)", 1, 21, 5)

    st.markdown("### Validation")
    n_folds = st.slider("Walk-forward folds", 3, 8, 5)
    embargo = st.slider("Embargo (days)", 0, 20, 5)
    n_trials = st.slider("Optuna trials", 0, 120, 25 if (HAS_OPTUNA and HAS_XGB) else 0,
                         disabled=not (HAS_OPTUNA and HAS_XGB))

    st.markdown("### Execution & risk")
    spread = st.slider("Half-spread (bps)", 0.0, 25.0, 5.0, 0.5)
    impact = st.slider("Impact coefficient (bps)", 0.0, 30.0, 8.0, 1.0)
    vol_target = st.slider("Vol target (ann.)", 0.04, 0.25, 0.10, 0.01)
    seed = int(st.number_input("Seed", 0, 9999, 7))

    run = st.button("Run research pipeline", use_container_width=True)
    st.markdown('<p class="footnote">Imputation, scaling, purging and tuning all '
                'happen inside the training folds. Test blocks are scored once.</p>',
                unsafe_allow_html=True)

BT = dict(quantile=quantile, half_spread_bps=spread, impact_bps=impact,
          vol_target=vol_target, rebalance=rebalance)

# ------------------------------------------------------------------ masthead
engines = ", ".join([n for n, ok in (("XGBoost", HAS_XGB), ("CatBoost", HAS_CAT),
                                     ("AdaBoost", True)) if ok])
st.markdown(f"""
<div class="masthead">
  <div>
    <span class="eyebrow">Quantitative research &middot; systematic equity</span>
    <h1>QuantBoost</h1>
    <p class="standfirst">Regime-aware gradient-boosted forecasts of vol-normalised
    forward returns, converted into a cost- and risk-budgeted long/short book.</p>
  </div>
  <dl class="masthead-meta">
    <div><dt>Edition</dt><dd>{pd.Timestamp.today().strftime('%d %b %Y')}</dd></div>
    <div><dt>Horizon</dt><dd>{horizon}d</dd></div>
    <div><dt>Engines</dt><dd>{engines}</dd></div>
    <div><dt>Validation</dt><dd>Purged WF</dd></div>
  </dl>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ run
if run:
    if len(tickers) < 8:
        st.error("Pick at least 8 names: the strategy is cross-sectional.")
        st.stop()
    prog = st.progress(0.0, text="Loading prices")
    prices, source = cached_prices(tuple(tickers), start, end, offline)
    prog.progress(0.15, text="Building the feature panel")
    panel = cached_panel(prices, horizon)
    feats = [c for c in feature_columns() if c in panel.columns]
    prog.progress(0.3, text=f"Tuning, {n_trials} trials" if n_trials else "Tuning skipped")
    params, study = tune_xgb(panel, feats, n_trials, horizon, embargo, seed)
    prog.progress(0.45, text="Purged walk-forward training")
    oof, imp, folds = run_walk_forward(panel, feats, params, n_folds, horizon, embargo,
                                      seed, verbose=False)
    prog.progress(1.0, text="Done")
    prog.empty()
    st.session_state["res"] = dict(prices=prices, source=source, panel=panel, oof=oof,
                                   imp=imp, folds=folds, params=params, feats=feats,
                                   trials=max(n_trials, 1))

res = st.session_state.get("res")
tabs = st.tabs(["Overview", "Data & features", "Time series", "Models",
                "Backtest", "Risk", "Methodology"])

# ------------------------------------------------------------------ empty state
if res is None:
    with tabs[0]:
        st.markdown("## Nothing priced yet")
        st.markdown(
            "Set the universe, horizon and cost assumptions on the left, then run the "
            "pipeline. It loads prices, builds the feature panel, tunes XGBoost on rank "
            "IC, trains all three boosters under purged walk-forward validation, and "
            "reports net-of-cost performance next to a Deflated Sharpe Ratio.")
        a, b, c = st.columns(3)
        a.markdown("**1. Panel**\n\nDaily bars, cross-sectionally standardised features, "
                   "vol-scaled labels, regime tags.")
        b.markdown("**2. Validation**\n\nExpanding train window, purged label overlap, "
                   "embargo, one scoring pass.")
        c.markdown("**3. Book**\n\nRank long/short, vol targeting, spread plus "
                   "square-root impact charged on turnover.")
        missing = [n for n, ok in (("xgboost", HAS_XGB), ("catboost", HAS_CAT),
                                   ("optuna", HAS_OPTUNA), ("yfinance", HAS_YF)) if not ok]
        if missing:
            st.warning("Reduced mode, not installed: " + ", ".join(missing))
    st.stop()

panel, oof = res["panel"], res["oof"]
models = [c for c in oof.columns
          if c not in ("date", "ticker", "target", "fwd_ret", "fold")]
default_model = models.index("Ensemble") if "Ensemble" in models else 0

# ------------------------------------------------------------------ overview
with tabs[0]:
    pick = st.selectbox("Book driven by", models, index=default_model)
    ic = rank_ic(oof.rename(columns={pick: "pred"})[["date", "pred", "target"]])
    bt = backtest(oof, panel, pick, **BT)
    perf = performance(bt, ic, res["trials"])

    k = st.columns(5)
    k[0].metric("Sharpe, net", f"{perf['sharpe_net']:.2f}",
                f"gross {perf['sharpe_gross']:.2f}")
    k[1].metric("Ann. return", f"{perf['ann_return']:.1%}")
    k[2].metric("Max drawdown", f"{perf['max_drawdown']:.1%}")
    k[3].metric("Mean rank IC", f"{perf['mean_ic']:.4f}", f"t = {perf['ic_t_stat']:.1f}")
    k[4].metric("Deflated Sharpe", f"{perf['deflated_sharpe_prob']:.2f}",
                help="Probability the Sharpe survives the trial count, non-normal "
                     "returns and the sample length. Below 0.95, you searched.")

    bench = panel.groupby("date")["ret1"].mean().reindex(bt.index).fillna(0)
    st.plotly_chart(fig_lines({pick: equity_curve(bt),
                              "Equal-weight universe": (1 + bench).cumprod()},
                             ytitle="Cumulative net growth of 1.00"),
                    use_container_width=True)
    st.plotly_chart(fig_drawdown(equity_curve(bt)), use_container_width=True)

    st.markdown("## Scorecard")
    st.dataframe(pd.Series(perf, name="value").to_frame().style.format("{:.4f}"),
                 use_container_width=True)
    st.markdown(f'<p class="footnote">Source: {res["source"]}. Out-of-sample, net of '
                f'{spread:.1f}bp half-spread plus square-root impact, rebalanced every '
                f'{rebalance} days, scaled toward a {vol_target:.0%} vol target.</p>',
                unsafe_allow_html=True)

# ------------------------------------------------------------------ data
with tabs[1]:
    st.markdown("## The panel")
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(panel):,}")
    b.metric("Names", panel["ticker"].nunique())
    c.metric("Features", len(res["feats"]))
    d.metric("Trading days", panel["date"].nunique())

    tick = st.selectbox("Inspect", sorted(panel["ticker"].unique()))
    one = panel[panel["ticker"] == tick].set_index("date").sort_index()

    f = base_fig(300)
    f.add_trace(go.Scatter(x=one.index, y=one["close"], name="Close",
                           line=dict(color=INK, width=1.4)))
    f.add_trace(go.Scatter(x=one.index, y=one["vol20"] * math.sqrt(252) * 100,
                           name="Realised vol, ann. %", yaxis="y2",
                           line=dict(color=ACCENT, width=1.2)))
    f.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="vol %"))
    st.plotly_chart(f, use_container_width=True)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### Triple-barrier labels")
        tb = triple_barrier(one["close"], one["vol20"].fillna(0.01))
        dist = tb["label"].value_counts().sort_index()
        dist.index = [{-1.0: "stop-loss", 0.0: "time barrier", 1.0: "profit-take"}[i]
                      for i in dist.index]
        st.plotly_chart(fig_bars(dist, "Label distribution"), use_container_width=True)
        st.markdown(f'<p class="footnote">Median holding period '
                    f'{np.nanmedian(tb["bars_held"]):.0f} bars. Barriers sit at 2σ / 1σ '
                    f'of conditional vol, so labels adapt to regime instead of using a '
                    f'fixed percentage.</p>', unsafe_allow_html=True)
    with right:
        st.markdown("### Data quality")
        st.dataframe(audit(res["prices"]).head(12), use_container_width=True)

    st.markdown("### Cross-sectional snapshot, latest date")
    last = panel[panel["date"] == panel["date"].max()]
    cols = ["ticker", "mom21_cs", "mom252_cs", "rsi14_cs", "bb_z_cs", "vol20_cs",
            "vol_z_cs", "fdiff_cs", "hurst126_cs", "regime"]
    st.dataframe(last[[c for c in cols if c in last.columns]]
                 .set_index("ticker").round(3), use_container_width=True)

# ------------------------------------------------------------------ time series
with tabs[2]:
    st.markdown("## Time series diagnostics")
    tick2 = st.selectbox("Series", sorted(panel["ticker"].unique()), key="tsa_ticker")
    one = panel[panel["ticker"] == tick2].set_index("date").sort_index()
    r = one["ret1"].dropna()

    st.dataframe(tsa.stationarity_report(
        {"log price": np.log(one["close"]), "returns": r,
         "fractionally differenced (d=0.4)": one["fdiff"]}).round(4),
        use_container_width=True)

    left, right = st.columns(2)
    with left:
        ac = tsa.autocorrelation_report(r)
        st.plotly_chart(fig_bars(ac["acf"].round(4), "ACF of returns"),
                        use_container_width=True)
        if "acf_squared" in ac:
            st.plotly_chart(fig_bars(ac["acf_squared"].round(4),
                                     "ACF of squared returns: volatility clustering"),
                            use_container_width=True)
    with right:
        st.markdown("### Baselines you must beat")
        st.dataframe(tsa.baseline_forecasts(r, horizon).round(4), use_container_width=True)
        cond, meta = tsa.fit_garch(r)
        st.plotly_chart(fig_lines({f"{meta['model']} conditional vol":
                                   cond * math.sqrt(252) * 100,
                                   "20d realised": one["vol20"] * math.sqrt(252) * 100},
                                  height=280, ytitle="ann. vol %"),
                        use_container_width=True)
        st.markdown(f'<p class="footnote">Volatility persistence '
                    f'α+β = {meta["persistence"]}. Close to 1 means shocks decay '
                    f'slowly, which is exactly why vol-scaling the label matters.</p>',
                    unsafe_allow_html=True)

    st.markdown("### Structural breaks")
    breaks = tsa.cusum_breaks(r, threshold=5.0)
    st.markdown(f"CUSUM flags **{len(breaks)}** shifts, roughly one every "
                f"**{len(r) // max(len(breaks), 1)}** trading days. That sets the "
                f"retraining cadence: refit at least that often.")

# ------------------------------------------------------------------ models
with tabs[3]:
    st.markdown("## Model comparison")
    lb = compare_models(oof, panel, models, res["trials"], **BT)
    st.dataframe(lb.style.format("{:.4f}").background_gradient(
        subset=[c for c in ("sharpe_net", "mean_ic") if c in lb.columns], cmap="BuGn"),
        use_container_width=True)
    st.markdown('<p class="footnote">AdaBoost is in here on purpose. Its loss is far '
                'more sensitive to the fat tails of returns, so when it loses to the '
                'modern boosters you can explain why rather than just cite a score.</p>',
                unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        if res["imp"]:
            which = st.selectbox("Importance from", list(res["imp"].keys()))
            st.plotly_chart(fig_bars(res["imp"][which].head(18)[::-1],
                                     f"{which} importance, averaged across folds",
                                     horizontal=True), use_container_width=True)
    with right:
        st.markdown("### Tuned parameters")
        st.dataframe(pd.Series(res["params"], name="value").to_frame(),
                     use_container_width=True)
        st.markdown("### Folds")
        st.dataframe(res["folds"], use_container_width=True)

    st.markdown("### Information coefficient through time")
    ic = rank_ic(oof.rename(columns={models[default_model]: "pred"})
                 [["date", "pred", "target"]])
    f = base_fig(300)
    f.add_trace(go.Bar(x=ic.index, y=ic.to_numpy(), marker_color=RULE, name="daily IC"))
    f.add_trace(go.Scatter(x=ic.index, y=ic.rolling(63).mean().to_numpy(),
                           name="63d mean", line=dict(color=ACCENT, width=1.8)))
    st.plotly_chart(f, use_container_width=True)

# ------------------------------------------------------------------ backtest
with tabs[4]:
    st.markdown("## Backtest anatomy")
    pick2 = st.selectbox("Model", models, index=default_model, key="bt_model")
    bt = backtest(oof, panel, pick2, **BT)
    st.plotly_chart(fig_lines({"Net of cost": equity_curve(bt),
                              "Gross": equity_curve(bt, "gross")},
                             ytitle="Cumulative growth of 1.00"),
                    use_container_width=True)

    a, b, c, d = st.columns(4)
    a.metric("Cost drag, ann.", f"{bt['cost'].mean() * 252:.2%}")
    b.metric("Ann. turnover", f"{bt['turnover'].mean() * 252:.1f}x")
    c.metric("Avg leverage", f"{bt['leverage'].mean():.2f}x")
    d.metric("Avg gross exposure", f"{bt['gross_exposure'].mean():.2f}")

    left, right = st.columns(2)
    with left:
        st.markdown("### Cost sensitivity")
        st.dataframe(cost_sensitivity(oof, panel, pick2, **BT).style.format("{:.4f}"),
                     use_container_width=True)
        st.markdown('<p class="footnote">A strategy that only works at 1x costs is not '
                    'a strategy. Read this table before the equity curve.</p>',
                    unsafe_allow_html=True)
    with right:
        st.markdown("### Factor attribution")
        mkt = panel.groupby("date")["ret1"].mean()
        facs = pd.DataFrame({"market": mkt,
                             "momentum": panel.groupby("date").apply(
                                 lambda g: np.average(g["ret1"].fillna(0),
                                                      weights=(g["mom252_cs"].rank()
                                                               - g["mom252_cs"].rank().mean()
                                                               ).abs() + 1e-9))})
        st.dataframe(factor_attribution(bt["net"], facs).round(4),
                     use_container_width=True)
        st.markdown('<p class="footnote">If alpha loses significance once market and '
                    'momentum are in the regression, the signal is repackaged beta.</p>',
                    unsafe_allow_html=True)

    st.markdown("### Monthly net returns")
    idx = bt.index.to_series()
    mon = bt["net"].groupby([idx.dt.year, idx.dt.month]).sum().unstack()
    mon.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in mon.columns]
    st.dataframe(mon.style.format("{:.2%}", na_rep="").background_gradient(
        cmap="RdYlGn", vmin=-0.06, vmax=0.06), use_container_width=True)

# ------------------------------------------------------------------ risk
with tabs[5]:
    st.markdown("## Risk and capital")
    bt = backtest(oof, panel, models[default_model], **BT)
    r = bt["net"].dropna()

    a, b, c, d = st.columns(4)
    a.metric("Daily vol", f"{r.std():.2%}")
    b.metric("Skew", f"{r.skew():.2f}")
    c.metric("Excess kurtosis", f"{r.kurt():.2f}")
    d.metric("Half-Kelly leverage", f"{kelly_fraction(r.mean(), r.var()):.2f}x",
             help="Fractional Kelly on realised moments, before the vol cap. Full "
                  "Kelly assumes you know the moments exactly. You do not.")

    st.markdown("### VaR, expected shortfall and the tests that judge them")
    vt = var_table(r)
    st.dataframe(vt.T, use_container_width=True)
    st.markdown('<p class="footnote">Kupiec tests the number of breaches, '
                'Christoffersen tests whether they cluster. LR above 3.84 rejects at '
                '95%. Clustering, not count, is what kills a risk model.</p>',
                unsafe_allow_html=True)

    v95 = float(vt.loc["95%", "historical_var"])
    f = base_fig(300)
    f.add_trace(go.Scatter(x=r.index, y=r.to_numpy() * 100, name="Daily P&L, %",
                           mode="lines", line=dict(color=INK_2, width=0.9)))
    br = r[r < -v95]
    f.add_trace(go.Scatter(x=br.index, y=br.to_numpy() * 100, name="VaR breach",
                           mode="markers", marker=dict(color=DOWN, size=6)))
    f.add_hline(y=-v95 * 100, line=dict(color=ACCENT, width=1, dash="dot"))
    st.plotly_chart(f, use_container_width=True)

    st.markdown("### Performance by volatility regime")
    st.dataframe(regime_breakdown(bt, panel).style.format(
        {"mean": "{:.4%}", "std": "{:.4%}", "sharpe": "{:.2f}"}), use_container_width=True)

# ------------------------------------------------------------------ methodology
with tabs[6]:
    st.markdown("## Methodology")
    st.markdown("""
**Target.** Forward log return over the horizon divided by conditional volatility,
clipped at ±6σ. Vol scaling makes the label roughly homoskedastic across names and
regimes, which is what allows one cross-sectional model to serve the whole universe.

**Features.** Momentum at four look-backs including 12-1, short-term reversal, RSI,
MACD histogram, Bollinger z-score, stochastic, ATR, Parkinson vol, realised and
downside vol, vol-of-vol, 60-day skew and kurtosis, volume shock, Amihud illiquidity,
Roll spread, fractionally differenced log price (d = 0.4, so memory survives),
rolling entropy, Hurst exponent, calendar flags and a volatility-regime tag from an
expanding quantile. Every continuous feature is standardised cross-sectionally per
date, converting a level-forecasting problem into relative value.

**Validation.** Expanding-window walk-forward. Training rows within horizon plus
embargo days of a test block are purged, because overlapping label windows leak even
when nothing obviously looks forward. Median imputation and standardisation are
fitted on training rows only.

**Tuning.** Optuna TPE with median pruning, maximising mean rank IC divided by its
cross-fold dispersion. Optimising RMSE would reward fitting the fat tails and ignore
ordering, and ordering is what a portfolio trades.

**Portfolio.** Scores are smoothed, ranked per date, demeaned, truncated to the
tails, normalised to unit gross, held between rebalance dates, then scaled by a
trailing-vol leverage estimate toward the vol target. Trades pay a half-spread plus a
square-root impact term on realised turnover, in the spirit of Almgren-Chriss.

**Honesty checks.** Deflated Sharpe adjusts for trial count, non-normality and sample
length. Cost sensitivity runs to 5x. Factor attribution separates alpha from beta and
momentum. The simulated-data toggle is a null test: signal on regime-switching GBM
noise means the pipeline leaks.
""")
    st.markdown("### Known limitations")
    st.markdown("""
- Daily bars only: no microstructure, no intraday fills, no queue position.
- The universe is fixed as of today, so survivorship bias is not fully removed.
  Point-in-time membership is the next build.
- Borrow cost is a flat annual charge on short exposure, not a per-name rate.
- Capacity is untested: the impact model is calibrated by assumption, not by fills.
""")
    st.markdown('<p class="footnote">QuantBoost research terminal. Educational '
                'research code, not investment advice.</p>', unsafe_allow_html=True)
