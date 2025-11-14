#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modeling Pipeline v3a — Sentiment → Returns (stable)
Fixes vs v3:
- Robust sentiment extraction via label mapping (no order assumptions)
- Clean entity reconciliation (no duplicated try blocks)
- Per-entity merge_asof is stable and sorted
- Drop NaN/±inf before modeling; consistent TimeSeries CV indexing
- Always reports window comparison table
"""

import os, re, json
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut, TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr

# ---------------- Config / Env ----------------
pd.options.mode.copy_on_write = True
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")

class CFG:
    event_windows: Tuple[int, ...] = (15, 30, 60, 90)
    uncertainty_cutoff: float = 0.35
    max_rows_preview: int = 12

cfg = CFG()

# ---------------- DB helpers ----------------
def get_engine():
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, pool_pre_ping=True)

def ensure_company_ref_table(engine):
    import requests
    try:
        with engine.connect() as conn:
            conn.execute(text(f'SELECT 1 FROM "{DB_SCHEMA}"."company_ref" LIMIT 1;'))
        print("🔎 company_ref exists — refreshing with latest SEC list...")
    except Exception:
        print("ℹ️ company_ref not found — creating...")
    data = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": "zihanshao1996@gmail.com"},
        timeout=60
    ).json()
    df = pd.DataFrame.from_dict(data, orient="index").rename(
        columns={"cik_str": "cik", "title": "company_name"}
    )
    df["cik"] = df["cik"].astype(int).astype(str).str.zfill(10)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df[["cik", "ticker", "company_name"]]
    df.to_sql("company_ref", engine, schema=DB_SCHEMA, if_exists="replace", index=False)
    print(f"✅ company_ref refreshed with {len(df)} rows.")

# ---------------- Canonical tickers ----------------
ALIASES = {"BRK.B": "BRK-B", "BF.B": "BF-B"}
def to_canonical_ticker(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip().upper().replace(".", "-")
    t = re.sub("-{2,}", "-", t)
    return ALIASES.get(t, t)

# ---------------- Sentiment ----------------
class SentimentEngine:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        mdl = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert", use_safetensors=True
        )
        self.pipe = pipeline(
            "text-classification",
            model=mdl,
            tokenizer=tok,
            truncation=True,
            top_k=None,
        )
        print("✅ FinBERT loaded with safetensors.")

    @staticmethod
    def _extract_triplet(res_item: List[dict]) -> Tuple[float, float, float]:
        """
        Map by label, not index. FinBERT labels: POSITIVE/NEGATIVE/NEUTRAL
        """
        lbls = {d["label"].lower(): float(d["score"]) for d in res_item if isinstance(d, dict)}
        p_pos = lbls.get("positive", 0.0)
        p_neg = lbls.get("negative", 0.0)
        p_neu = lbls.get("neutral", 0.0)
        return p_pos, p_neu, p_neg

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        texts = df[text_col].astype(str).tolist()
        raw = self.pipe(texts)  # returns List[List[dict]] or List[dict] depending on model
        # Normalize to List[List[dict]]
        packed = [r if isinstance(r, list) else [r] for r in raw]
        triplets = [self._extract_triplet(r) for r in packed]
        out = df.copy()
        out[["p_pos", "p_neu", "p_neg"]] = pd.DataFrame(triplets, index=out.index)
        out["polarity"] = out["p_pos"] - out["p_neg"]
        # a simple confidence proxy: 1 - neutral
        out["conf_w"] = 1.0 - out["p_neu"].clip(lower=0.0, upper=1.0)
        return out

# ---------------- Loading ----------------
def to_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)

def load_frames(engine):
    filings = pd.read_sql(f"""
        SELECT 
            COALESCE(NULLIF(ltrim(m.cik,'0'),''), cr.cik) AS cik,
            COALESCE(UPPER(m.ticker), cr.ticker) AS ticker,
            COALESCE(m.company_name, cr.company_name) AS company_name,
            (m.filing_date AT TIME ZONE 'UTC') AS filing_date,
            m.chunk_index AS paragraph_id,
            m.content AS text
        FROM "{DB_SCHEMA}"."mdna_sections" m
        LEFT JOIN "{DB_SCHEMA}"."company_ref" cr 
          ON ltrim(m.cik,'0') = ltrim(cr.cik,'0') OR UPPER(m.ticker)=cr.ticker;
    """, engine)

    prices = pd.read_sql(f"""
        SELECT 
            (sp."Date" AT TIME ZONE 'UTC') AS date,
            sp."Open" AS open, sp."Close" AS close, sp."Volume" AS volume,
            UPPER(sp.ticker) AS ticker,
            cr.cik
        FROM "{DB_SCHEMA}"."stock_prices" sp
        LEFT JOIN "{DB_SCHEMA}"."company_ref" cr ON UPPER(sp.ticker)=cr.ticker;
    """, engine)

    benchmark = pd.read_sql(f"""
        SELECT ("Date" AT TIME ZONE 'UTC') AS date, close
        FROM "{DB_SCHEMA}"."sp500_index" ORDER BY "Date";
    """, engine)

    filings["filing_date"] = to_naive(filings["filing_date"])
    prices["date"] = to_naive(prices["date"])
    benchmark["date"] = to_naive(benchmark["date"])

    filings["ticker"] = filings["ticker"].map(to_canonical_ticker)
    prices["ticker"] = prices["ticker"].map(to_canonical_ticker)

    filings["cik"] = filings["cik"].astype(str).str.strip().str.lstrip("0").replace({"": np.nan})
    prices["cik"] = prices["cik"].astype(str).str.strip().str.lstrip("0").replace({"": np.nan})

    filings["entity"] = np.where(filings["cik"].notna(), filings["cik"], filings["ticker"])
    prices["entity"] = np.where(prices["cik"].notna(), prices["cik"], prices["ticker"])

    print("Distinct tickers from SQL:", filings["ticker"].nunique())
    return filings, prices, benchmark

# ---------------- Aggregation ----------------
def aggregate_to_filing(engine, filings_scored: pd.DataFrame) -> pd.DataFrame:
    d = filings_scored.copy()

    # Weighted means (confidence weight); keep ≥1 paragraph per filing
    keys = ["entity", "ticker", "company_name", "filing_date"]
    g = d.groupby(keys, dropna=False)
    out = g.apply(
        lambda x: pd.Series({
            "pos_mean": np.average(x["p_pos"], weights=x["conf_w"].clip(lower=1e-6)),
            "neg_mean": np.average(x["p_neg"], weights=x["conf_w"].clip(lower=1e-6)),
            "para_count": int(len(x)),
        })
    ).reset_index()

    out["polarity"] = out["pos_mean"] - out["neg_mean"]
    out = out[out["para_count"] >= 1].copy()

    # Tone change / shock
    out = out.sort_values(["entity", "filing_date"], kind="mergesort")
    out["tone_change"] = out.groupby("entity", sort=False)["polarity"].diff()
    q_hi, q_lo = out["tone_change"].quantile([0.95, 0.05])
    out["tone_shock"] = ((out["tone_change"] >= q_hi) | (out["tone_change"] <= q_lo)).astype(int)

    print(f"✅ After relaxed filtering — entities: {out['entity'].nunique()} | rows: {len(out)}")

    # Reconcile entities using company_ref, merging on ticker only (avoid name collisions)
    with engine.connect() as conn:
        ref = pd.read_sql(f'''
            SELECT cik AS ref_entity, ticker
            FROM "{DB_SCHEMA}"."company_ref"
            WHERE UPPER(ticker) IN (
                SELECT DISTINCT UPPER(ticker) FROM "{DB_SCHEMA}"."mdna_sections"
            )
        ''', conn)
    ref["ref_entity"] = ref["ref_entity"].astype(str)
    out = out.merge(ref[["ticker", "ref_entity"]], on="ticker", how="left")
    out["entity"] = out["entity"].fillna(out["ref_entity"]).astype(str)
    out.drop(columns=["ref_entity"], inplace=True, errors="ignore")

    print(f"✅ After entity reconciliation — entities: {out['entity'].nunique()} | rows: {len(out)}")
    return out

# ---------------- Modeling helpers ----------------
def compute_price_features(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    p = (
        prices.dropna(subset=["entity", "date"])
        .drop_duplicates(subset=["entity", "date"])
        .sort_values(["entity", "date"], kind="mergesort")
        .copy()
    )
    p["close"] = pd.to_numeric(p["close"], errors="coerce")
    p = p.dropna(subset=["close"])
    p["log_close"] = np.log(p["close"])
    p["ret_d"] = p.groupby("entity", sort=False)["log_close"].diff()
    p["vol_pre30"] = (
        p.groupby("entity", sort=False)["ret_d"]
        .rolling(window=30, min_periods=10).std()
        .reset_index(level=0, drop=True)
    )
    p[f"ret_fwd_{window}d"] = (
        p.groupby("entity", sort=False)["log_close"].shift(-window) - p["log_close"]
    )
    p.drop(columns=["log_close"], inplace=True)
    return p

def merge_asof_safe(filing_sent: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    left = (
        filing_sent.rename(columns={"filing_date": "f_date"})
        .dropna(subset=["entity", "f_date"])
        .drop_duplicates(subset=["entity", "f_date"])
        .sort_values(["entity", "f_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    right = (
        prices.dropna(subset=["entity", "date"])
        .drop_duplicates(subset=["entity", "date"])
        .sort_values(["entity", "date"], kind="mergesort")
        .reset_index(drop=True)
    )
    chunks = []
    for ent, lf in left.groupby("entity", sort=False):
        rf = right[right["entity"] == ent]
        if rf.empty:
            continue
        arr = rf["date"].to_numpy()
        idx = np.searchsorted(arr, lf["f_date"].to_numpy(), side="left")
        idx = np.clip(idx, 0, len(rf) - 1)
        take = rf.iloc[idx].reset_index(drop=True)
        chunks.append(pd.concat([lf.reset_index(drop=True), take], axis=1))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

def fit_models(df: pd.DataFrame, target_col: str, mkt_col: str) -> Dict[str, float]:
    """
    Fit OLS and Ridge models, with Leave-One-Group-Out (by entity)
    and time-series cross-validation.

    Automatically handles missing/inf data and entity column alignment.
    """
    # Clean and validate input
    mm = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[target_col, "polarity", "vol_pre30", mkt_col, "tone_change"]
    ).copy()

    # Handle duplicated entity columns (from merge)
    entity_cols = [c for c in mm.columns if c.startswith("entity")]
    if "entity" not in mm.columns and entity_cols:
        mm["entity"] = mm[entity_cols[0]]
    elif "entity_x" in mm.columns and "entity_y" in mm.columns:
        mm["entity"] = mm["entity_x"].fillna(mm["entity_y"])
        mm.drop(columns=["entity_x", "entity_y"], inplace=True, errors="ignore")

    if mm.empty:
        return {"ols_adj_r2": np.nan, "logo_cv_r2_mean": np.nan, "timecv_r2_mean": np.nan, "n_obs": 0}

    # Interaction terms
    mm["polarity_x_vol"] = mm["polarity"] * mm["vol_pre30"]
    mm["polarity_x_mkt"] = mm["polarity"] * mm[mkt_col]

    predictors = [
        "polarity",
        "tone_change",
        "tone_shock",
        "vol_pre30",
        mkt_col,
        "polarity_x_vol",
        "polarity_x_mkt",
    ]

    X_raw = mm[predictors]
    y = mm[target_col]

    # ----- OLS -----
    X_ols = sm.add_constant(X_raw, has_constant="add")
    ols = sm.OLS(y, X_ols).fit(cov_type="HC3")

    # ----- Ridge (in-sample) -----
    ridge_pipe = Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=42))]
    )
    ridge_pipe.fit(X_raw, y)
    rho, _ = spearmanr(y, ridge_pipe.predict(X_raw), nan_policy="omit")

    # ----- Leave-One-Group-Out CV -----
    logo = LeaveOneGroupOut()

    # Ensure all have consistent lengths
    groups = mm["entity"].astype(str).values.ravel()
    X_np = np.asarray(X_raw)
    y_np = np.asarray(y).ravel()

    if not (len(groups) == len(X_np) == len(y_np)):
        print(f"⚠️ Length mismatch before LOGO: len(X)={len(X_np)}, len(y)={len(y_np)}, len(groups)={len(groups)}")
        min_len = min(len(X_np), len(y_np), len(groups))
        X_np, y_np, groups = X_np[:min_len], y_np[:min_len], groups[:min_len]

    r2s_logo = []
    for tr, te in logo.split(X_np, y_np, groups):
        if len(te) < 2:
            continue
        Xt = sm.add_constant(pd.DataFrame(X_np[tr], columns=X_raw.columns), has_constant="add")
        Xv = sm.add_constant(pd.DataFrame(X_np[te], columns=X_raw.columns), has_constant="add")
        mod = sm.OLS(y_np[tr], Xt).fit(cov_type="HC3")
        pred = mod.predict(Xv)
        r2s_logo.append(r2_score(y_np[te], pred))
    logo_mean = float(np.nanmean(r2s_logo)) if len(r2s_logo) else np.nan

    # ----- TimeSeries CV -----
    sort_col = "f_date" if "f_date" in mm.columns else "filing_date"
    mm_ord = mm.sort_values([sort_col], kind="mergesort")
    X_ord = mm_ord[predictors].to_numpy()
    y_ord = mm_ord[target_col].to_numpy()
    r2s_time = []

    tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(mm_ord) // 20)))
    for tr, te in tscv.split(X_ord):
        Xt = sm.add_constant(pd.DataFrame(X_ord[tr], columns=X_raw.columns), has_constant="add")
        Xv = sm.add_constant(pd.DataFrame(X_ord[te], columns=X_raw.columns), has_constant="add")
        mod = sm.OLS(y_ord[tr], Xt).fit(cov_type="HC3")
        pred = mod.predict(Xv)
        r2s_time.append(r2_score(y_ord[te], pred))
    time_mean = float(np.nanmean(r2s_time)) if len(r2s_time) else np.nan

    # ----- Output summary -----
    return {
        "ols_adj_r2": float(ols.rsquared_adj),
        "ols_params": {k: float(v) for k, v in ols.params.items()},
        "ols_pvalues": {k: float(v) for k, v in ols.pvalues.items()},
        "ridge_rho_in": float(rho) if np.isfinite(rho) else np.nan,
        "logo_cv_r2_mean": logo_mean,
        "timecv_r2_mean": time_mean,
        "n_obs": int(len(mm)),
    }

# ---------------- Runner per window ----------------
def run_for_window(filing_sent: pd.DataFrame, prices: pd.DataFrame, benchmark: pd.DataFrame, window: int) -> Dict:
    p = compute_price_features(prices, window)
    merged = merge_asof_safe(filing_sent, p)
    if merged.empty:
        return {"window": window, "ols_adj_r2": np.nan, "logo_cv_r2_mean": np.nan, "timecv_r2_mean": np.nan, "n_obs": 0}

    # add market forward
    b = benchmark.sort_values("date").copy()
    b["close"] = pd.to_numeric(b["close"], errors="coerce")
    b = b.dropna(subset=["close"])
    b[f"mkt_ret_fwd_{window}d"] = np.log(b["close"].shift(-window) / b["close"])
    merged = pd.merge_asof(
        merged.sort_values("f_date"),
        b[["date", f"mkt_ret_fwd_{window}d"]].sort_values("date"),
        left_on="f_date", right_on="date",
        direction="nearest", allow_exact_matches=True
    ).rename(columns={"f_date": "filing_date"})

    target = f"ret_fwd_{window}d"
    mkt_col = f"mkt_ret_fwd_{window}d"
    res = fit_models(merged, target, mkt_col)
    res["window"] = window

    # attach tiny preview (safe columns)
    keep = ["entity", "ticker", "company_name", "filing_date", "polarity", "tone_change", "vol_pre30", target, mkt_col]
    prev_cols = [c for c in keep if c in merged.columns]
    if prev_cols:
        prev = merged.sort_values("filing_date").head(cfg.max_rows_preview)[prev_cols]
        res["preview"] = prev
    return res

# ---------------- Main ----------------
def main():
    engine = get_engine()
    ensure_company_ref_table(engine)
    filings, prices, benchmark = load_frames(engine)

    print("⚙️ Running sentiment...")
    se = SentimentEngine()
    filings = se.score_dataframe(filings)

    filing_sent = aggregate_to_filing(engine, filings)
    filing_sent = filing_sent.sort_values(["entity", "filing_date"]).drop_duplicates(["entity", "filing_date"])

    print(f"After aggregation — entities: {filing_sent['entity'].nunique()} | rows: {len(filing_sent)}")
    print(f"Entities in prices: {prices['entity'].nunique()}")
    missing = set(prices["entity"].astype(str)) - set(filing_sent["entity"].astype(str))
    print(f"ℹ️ Missing in filings: {len(missing)}")

    # Run windows
    results = [run_for_window(filing_sent, prices, benchmark, w) for w in cfg.event_windows]

    # Summarize
    df = pd.DataFrame(results)
    print("\nWindow Summary:")
    cols = [c for c in ["window","ols_adj_r2","logo_cv_r2_mean","timecv_r2_mean","n_obs","ridge_rho_in"] if c in df.columns]
    if cols:
        print(df[cols].fillna("–").to_string(index=False))
    else:
        print("(no results)")

    # Best window by Adj R² (if available)
    if "ols_adj_r2" in df.columns and df["ols_adj_r2"].notna().any():
        best_idx = df["ols_adj_r2"].astype(float).idxmax()
        best = results[int(best_idx)]
        w = best.get("window")
        print(f"\nBest window: {w}d")
        if "ols_params" in best:
            print("OLS coefficients:")
            for k, v in best["ols_params"].items():
                pv = best["ols_pvalues"].get(k, np.nan)
                print(f"  {k:>16s}  β={v:+.4f}  p={pv:.4g}")
        if "preview" in best:
            print("\nSample rows:")
            print(best["preview"].to_string(index=False))

    # Optional: write a tiny JSON summary for quick reuse
    try:
        with open("model_results_summary_v3a.json", "w") as f:
            json.dump({"windows": comp.to_dict(orient="records")}, f, indent=2, default=str)
        print("\n🧾 Saved window summary → model_results_summary_v3a.json")
    except Exception as e:
        print(f"\n⚠️ Could not save JSON summary: {e}")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
