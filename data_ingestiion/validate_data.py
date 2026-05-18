
import os, re, math
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

try:
    import textstat
except Exception:
    textstat = None


IN_SEC = "sec10k.csv"
IN_PX  = "stock_price.csv"
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

TARGET_TICKERS = {"AAPL", "GOOGL", "GOOG", "MSFT", "NVDA"}
DATE_START, DATE_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31")

# --------- Helpers ---------
def read_csv_safely(path):
    for enc in [None, "utf-8", "utf-8-sig", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Could not read {path}")

def ensure_col(df, candidates, rename_to=None):
    for c in candidates:
        if c in df.columns:
            if rename_to and c != rename_to:
                df.rename(columns={c: rename_to}, inplace=True)
            return rename_to or c
    return None

def word_count(text):
    if not isinstance(text, str): return 0
    return sum(1 for t in text.split() if t.strip())

def safe_polarity(text):
    if TextBlob is None or not isinstance(text, str): return np.nan
    try: return float(TextBlob(text).sentiment.polarity)
    except Exception: return np.nan

def safe_flesch(text):
    if textstat is None or not isinstance(text, str): return np.nan
    try: return float(textstat.flesch_reading_ease(text))
    except Exception: return np.nan

def pct(n, d): 
    return (100.0 * n / d) if d else 0.0

# --------- Load Data ---------
sec = read_csv_safely(IN_SEC)
px  = read_csv_safely(IN_PX)

sec.columns = [c.strip().lower() for c in sec.columns]
px.columns  = [c.strip().lower() for c in px.columns]

# Detect/standardize key columns
ticker_col = ensure_col(sec, ["ticker","symbol","company_ticker"], "ticker") or "ticker"
if "ticker" not in sec.columns: sec["ticker"] = np.nan

date_col_sec = ensure_col(sec, ["filing_date","file_date","report_date","date"], "filing_date") or "filing_date"
sec["filing_date"] = pd.to_datetime(sec.get("filing_date"), errors="coerce")

mdna_col = None
for c in sec.columns:
    lc = c.lower()
    if ("md&a" in lc) or ("mdna" in lc) or ("management" in lc and "analysis" in lc) or (lc=="mdna_text"):
        mdna_col = c
        break
if mdna_col is None:
    for c in sec.columns:
        if "md" in c and "text" in c: mdna_col = c; break

risk_col = None
for c in sec.columns:
    lc = c.lower()
    if "risk" in lc and ("factor" in lc or "factors" in lc):
        risk_col = c; break
if risk_col is None:
    for c in sec.columns:
        if "risk" in c and "text" in c: risk_col = c; break

# Price columns
ensure_col(px, ["date","trade_date"], "date")
px["date"] = pd.to_datetime(px["date"], errors="coerce")
ensure_col(px, ["ticker","symbol"], "ticker")
ensure_col(px, ["close","adj close","adj_close","adjclose"], "close")

# Filter tickers + dates
sec["ticker"] = sec["ticker"].astype(str).str.upper()
px["ticker"]  = px["ticker"].astype(str).str.upper()

sec = sec[sec["ticker"].isin(TARGET_TICKERS)]
px  = px[px["ticker"].isin(TARGET_TICKERS)]

sec = sec[(sec["filing_date"] >= DATE_START) & (sec["filing_date"] <= DATE_END)]
px  = px[(px["date"] >= DATE_START) & (px["date"] <= DATE_END)]

# =========================================================
# Task 2.2 – Describing Data  (Data Description Report)
# =========================================================

# Summaries for filings
desc_filings = pd.DataFrame({
    "n_rows": [len(sec)],
    "n_companies": [sec["ticker"].nunique()],
    "period_min": [sec["filing_date"].min()],
    "period_max": [sec["filing_date"].max()],
    "has_mdna_col": [mdna_col in sec.columns],
    "has_risk_col": [risk_col in sec.columns]
})

# Basic completeness
comp = pd.DataFrame({
    "column": sec.columns,
    "missing": sec.isna().sum().values,
    "pct_missing": (sec.isna().mean()*100).round(2).values
}).sort_values("pct_missing", ascending=False)

# Word-count snapshots (no heavy NLP yet)
if mdna_col in sec.columns: sec["mdna_wc"] = sec[mdna_col].apply(word_count)
if risk_col in sec.columns: sec["risk_wc"] = sec[risk_col].apply(word_count)

by_ticker_wc = sec.groupby("ticker").agg(
    n_filings=("filing_date","count"),
    mdna_wc_avg=("mdna_wc","mean"),
    risk_wc_avg=("risk_wc","mean"),
    mdna_wc_median=("mdna_wc","median"),
    risk_wc_median=("risk_wc","median")
).reset_index()

# Stock description
px = px.sort_values(["ticker","date"])
px["return_1d"] = px.groupby("ticker")["close"].pct_change()

desc_prices = px.groupby("ticker").agg(
    n_days=("date","count"),
    start=("date","min"),
    end=("date","max"),
    mean_price=("close","mean"),
    std_price=("close","std"),
    mean_ret=("return_1d","mean"),
    std_ret=("return_1d","std")
).reset_index()

# Save Task 2.2 outputs
desc_filings.to_csv(f"{OUTDIR}/t22_filings_overview.csv", index=False)
comp.to_csv(f"{OUTDIR}/t22_filings_missingness.csv", index=False)
by_ticker_wc.to_csv(f"{OUTDIR}/t22_wordcounts_by_ticker.csv", index=False)
desc_prices.to_csv(f"{OUTDIR}/t22_prices_overview.csv", index=False)

# =========================================================
# Task 2.3 – Exploring Data  (Data Exploration Report)
# =========================================================

# --- Text exploration: readability + sentiment (lightweight) ---
if mdna_col in sec.columns:
    sec["mdna_polarity"] = sec[mdna_col].apply(safe_polarity)
    sec["mdna_flesch"]   = sec[mdna_col].apply(safe_flesch)
if risk_col in sec.columns:
    sec["risk_polarity"] = sec[risk_col].apply(safe_polarity)
    sec["risk_flesch"]   = sec[risk_col].apply(safe_flesch)

text_metrics = sec.groupby("ticker").agg(
    n_filings=("filing_date","count"),
    mdna_wc_avg=("mdna_wc","mean"),
    risk_wc_avg=("risk_wc","mean"),
    mdna_polarity_avg=("mdna_polarity","mean"),
    risk_polarity_avg=("risk_polarity","mean"),
    mdna_flesch_avg=("mdna_flesch","mean"),
    risk_flesch_avg=("risk_flesch","mean"),
).reset_index()
text_metrics.to_csv(f"{OUTDIR}/t23_text_metrics_by_ticker.csv", index=False)

# --- Stock exploration: simple visuals & stats ---
for t in sorted(px["ticker"].unique()):
    df = px[px["ticker"]==t]
    plt.figure()
    plt.plot(df["date"], df["close"])
    plt.title(f"{t} Close Price (2023–2025)")
    plt.xlabel("Date"); plt.ylabel("Close")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/t23_{t}_close_trend.png"); plt.close()

# --- Event-study: align filing_date to next trading day; compute ±3-day returns ---
events = []
for _, r in sec.dropna(subset=["filing_date","ticker"]).iterrows():
    t = r["ticker"]
    d = r["filing_date"]
    pr = px[px["ticker"]==t].reset_index(drop=True)
    if pr.empty: 
        continue
    idx0 = int(pr["date"].searchsorted(d, side="left"))  # first trading day on/after filing
    if idx0 >= len(pr): 
        continue
    def cumret(df, i0, i1):
        if i0<0 or i1>=len(df) or i1<i0: return np.nan
        a, b = df.iloc[i0]["close"], df.iloc[i1]["close"]
        if pd.isna(a) or pd.isna(b) or a==0: return np.nan
        return (b/a) - 1.0
    pre  = cumret(pr, idx0-3, idx0-1)
    post = cumret(pr, idx0, min(idx0+3, len(pr)-1))
    events.append({
        "ticker": t,
        "filing_date": d,
        "event_trading_date": pr.iloc[idx0]["date"],
        "pre_(-3,-1)_ret": pre,
        "post_(0,+3)_ret": post,
        "mdna_word_count": r.get("mdna_wc", np.nan),
        "risk_word_count": r.get("risk_wc", np.nan),
        "mdna_polarity": r.get("mdna_polarity", np.nan),
        "risk_polarity": r.get("risk_polarity", np.nan),
    })

events_df = pd.DataFrame(events).sort_values(["ticker","filing_date"])
events_df.to_csv(f"{OUTDIR}/t23_event_level_returns.csv", index=False)

event_summary = events_df.groupby("ticker").agg(
    n_filings=("filing_date","count"),
    avg_pre_ret=("pre_(-3,-1)_ret","mean"),
    avg_post_ret=("post_(0,+3)_ret","mean"),
    avg_mdna_polarity=("mdna_polarity","mean"),
    avg_risk_polarity=("risk_polarity","mean"),
    median_mdna_wc=("mdna_word_count","median"),
    median_risk_wc=("risk_word_count","median"),
).reset_index()
event_summary.to_csv(f"{OUTDIR}/t23_event_summary_by_ticker.csv", index=False)

# Optional: correlations (tone vs. returns/vol)
corr_rows = []
if not events_df.empty:
    for t in events_df["ticker"].unique():
        sub = events_df[events_df["ticker"]==t]
        r1 = sub[["mdna_polarity","post_(0,+3)_ret"]].corr().iloc[0,1]
        corr_rows.append({"ticker": t, "corr_mdna_to_post3": r1})
corrs = pd.DataFrame(corr_rows)
corrs.to_csv(f"{OUTDIR}/t23_correlations.csv", index=False)

# =========================================================
# Task 2.4 – Verifying Data Quality  (Data Quality Report)
# =========================================================

# --- 10-K quality flags ---
q_filings = pd.DataFrame()
if mdna_col in sec.columns and risk_col in sec.columns:
    q_filings = pd.DataFrame({
        "ticker": sec["ticker"],
        "filing_date": sec["filing_date"],
        "mdna_wc": sec["mdna_wc"],
        "risk_wc": sec["risk_wc"],
        "flag_mdna_too_short": sec["mdna_wc"].fillna(0) < 500,   # tweak threshold as needed
        "flag_risk_too_short": sec["risk_wc"].fillna(0) < 200,
        "flag_missing_mdna": sec[mdna_col].isna() | (sec[mdna_col].astype(str).str.len()==0),
        "flag_missing_risk": sec[risk_col].isna() | (sec[risk_col].astype(str).str.len()==0),
    })
    q_filings.to_csv(f"{OUTDIR}/t24_filings_quality_flags.csv", index=False)

# --- Stock continuity & outliers ---
px = px.sort_values(["ticker","date"])
px["date_diff_days"] = px.groupby("ticker")["date"].diff().dt.days
px["flag_gap_gt3d"] = px["date_diff_days"] > 3
px["flag_big_move"] = px["return_1d"].abs() > 0.25  # large move heuristic

q_prices = px.groupby("ticker").agg(
    n_days=("date","count"),
    gaps_gt3d=("flag_gap_gt3d","sum"),
    big_moves=("flag_big_move","sum"),
    min_date=("date","min"),
    max_date=("date","max")
).reset_index()
q_prices.to_csv(f"{OUTDIR}/t24_prices_quality_summary.csv", index=False)

# --- Integration checks ---
# Confirm each filing mapped to an event trading day
integration_ok = pd.DataFrame({
    "ticker": events_df["ticker"],
    "filing_date": events_df["filing_date"],
    "event_trading_date": events_df["event_trading_date"],
    "mapped": ~events_df["event_trading_date"].isna()
})
integration_ok.to_csv(f"{OUTDIR}/t24_integration_checks.csv", index=False)


with open(f"{OUTDIR}/phase2_report_summary.md","w",encoding="utf-8") as f:
    f.write("# Phase 2 – Data Understanding: Auto Summary\n\n")
    f.write("## Task 2.2 – Describing Data\n")
    f.write(desc_filings.to_markdown(index=False)+"\n\n")
    f.write(by_ticker_wc.round(2).to_markdown(index=False)+"\n\n")
    f.write(desc_prices.round(4).to_markdown(index=False)+"\n\n")
    f.write("## Task 2.3 – Exploring Data\n")
    if not text_metrics.empty: f.write(text_metrics.round(4).to_markdown(index=False)+"\n\n")
    if not event_summary.empty: f.write(event_summary.round(4).to_markdown(index=False)+"\n\n")
    if not corrs.empty: f.write(corrs.round(4).to_markdown(index=False)+"\n\n")
    f.write("## Task 2.4 – Verifying Data Quality\n")
    if not q_filings.empty: f.write(q_filings.head(20).to_markdown(index=False)+"\n\n")
    f.write(q_prices.to_markdown(index=False)+"\n")
print(f"Done. Artifacts saved to: {OUTDIR}/")
