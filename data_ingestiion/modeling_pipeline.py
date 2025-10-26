#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4.0 — Modeling (DB version)
- Reads mdna_sections / stock_prices / sp500_index from PostgreSQL
- Builds company_ref (CIK↔ticker↔name) automatically from SEC
- Sentiment: FinBERT (preferred) → VADER fallback
- Features: filing-level polarity, pre-vol, 30D forward log returns (prices + benchmark)
- Models: OLS (HC3), QuantReg (0.25/0.5/0.75), Robust Group CV by entity (boolean masks)
- Anomalies: |z(resid)| > z_thresh OR sign(polarity) ≠ sign(return)
- Outputs: model_results (DB), model_results_summary.json, model_scatter.html

Env vars (.env or shell):
  DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, DB_SCHEMA(optional)
"""

import os
import json
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Stats / ML
import statsmodels.api as sm
from sklearn.metrics import r2_score

# Visualization
import plotly.express as px

# ======================= Config =======================
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")

@dataclass
class CFG:
    event_window: int = 30
    tokenizer_max_len: int = 512
    cv_folds: int = 5
    z_thresh: float = 2.0
    min_paragraphs: int = 5
    uncertainty_cutoff: float = 0.35
    out_html: str = "model_scatter.html"
    random_state: int = 42

cfg = CFG()

def get_engine():
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
        raise RuntimeError("Missing DB env vars. Set DB_USER, DB_PASSWORD, DB_HOST, DB_NAME.")
    try:
        int(DB_PORT)
    except Exception as e:
        raise RuntimeError(f"DB_PORT must be an integer (got {DB_PORT!r})") from e
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, pool_pre_ping=True)

# ================= Company mapping (CIK↔Ticker) =================
def ensure_company_ref_table(engine):
    try:
        _ = pd.read_sql(f"SELECT 1 FROM {DB_SCHEMA}.company_ref LIMIT 1;", engine)
        print("🔎 company_ref exists — refreshing with latest SEC list...")
    except Exception:
        print("ℹ️ company_ref not found — creating...")

    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers={"User-Agent": "zihanshao1996@gmail.com"}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.rename(columns={"cik_str": "cik", "title": "company_name"})
    df["cik"] = df["cik"].astype(int).astype(str).str.zfill(10)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df[["cik", "ticker", "company_name"]]
    df.to_sql("company_ref", engine, schema=DB_SCHEMA, if_exists="replace", index=False)
    print(f"✅ company_ref refreshed with {len(df)} rows.")

# ================= Sentiment (FinBERT → VADER) =================
class SentimentEngine:
    def __init__(self, cfg: CFG):
        self.cfg = cfg
        self.backend = None
        self.pipe = None
        self.vader = None

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.pipe = pipeline("text-classification", model=mdl, tokenizer=tok,
                                 truncation=True, return_all_scores=True)
            self.backend = "finbert"
            print("✅ Loaded FinBERT.")
        except Exception as e:
            print(f"⚠️ FinBERT unavailable ({e}). Trying VADER fallback...")

        if self.backend is None:
            try:
                import nltk
                from nltk.sentiment import SentimentIntensityAnalyzer
                try:
                    nltk.data.find("sentiment/vader_lexicon.zip")
                except LookupError:
                    nltk.download("vader_lexicon")
                self.vader = SentimentIntensityAnalyzer()
                self.backend = "vader"
                print("✅ Using VADER sentiment fallback.")
            except Exception as e:
                raise RuntimeError("No sentiment backend available. Install transformers+torch or nltk.") from e

    def score_paragraph(self, text: str) -> Tuple[float, float, float, float]:
        t = (text or "")[: self.cfg.tokenizer_max_len]
        if self.backend == "finbert":
            res = self.pipe(t)[0]
            scores = {d["label"].lower(): float(d["score"]) for d in res}
            p_pos, p_neu, p_neg = scores.get("positive", 0.0), scores.get("neutral", 0.0), scores.get("negative", 0.0)
            uncertainty = 1.0 - max(p_pos, p_neu, p_neg)
            return p_pos, p_neu, p_neg, uncertainty
        else:
            s = self.vader.polarity_scores(t)
            comp = s["compound"]
            p_pos = max(0.0, comp) / 2.0 + 0.25 if comp > 0 else 0.25
            p_neg = max(0.0, -comp) / 2.0 + 0.25 if comp < 0 else 0.25
            p_neu = 1.0 - min(0.98, p_pos + p_neg)
            uncertainty = 1.0 - max(p_pos, p_neu, p_neg)
            return float(p_pos), float(p_neu), float(p_neg), float(uncertainty)

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        ps, ns, zs, us = [], [], [], []
        for _, row in df.iterrows():
            p_pos, p_neu, p_neg, unc = self.score_paragraph(str(row[text_col]))
            ps.append(p_pos); ns.append(p_neu); zs.append(p_neg); us.append(unc)
        out = df.copy()
        out["p_pos"], out["p_neu"], out["p_neg"], out["uncertainty"] = ps, ns, zs, us
        return out

# ================= Feature engineering =================
def to_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)

def compute_price_features(prices: pd.DataFrame, event_window: int) -> pd.DataFrame:
    p = prices.copy()
    p = p[p["entity"].notna() & p["date"].notna()].copy()
    p = p.drop_duplicates(subset=["entity", "date"]).sort_values(["entity", "date"])
    p["close"] = pd.to_numeric(p["close"], errors="coerce")

    p["log_close"] = np.log(p["close"])
    p["ret_d"] = p.groupby("entity", group_keys=False)["log_close"].diff()

    p["vol_pre30"] = (
        p.groupby("entity", group_keys=False)["ret_d"]
         .rolling(window=30, min_periods=10)
         .std()
         .reset_index(level=0, drop=True)
    )

    p["ret_fwd_w"] = (
        p.groupby("entity", group_keys=False)["log_close"].shift(-event_window) - p["log_close"]
    )
    p.drop(columns=["log_close"], inplace=True)
    return p

def aggregate_to_filing(paras: pd.DataFrame, backend: str, uncertainty_cutoff: float) -> pd.DataFrame:
    d = paras.copy()
    if backend == "finbert":
        d = d[d["uncertainty"] <= uncertainty_cutoff]
    g = d.groupby(["entity", "filing_date"])
    out = g.agg(
        pos_mean=("p_pos", "mean"),
        neg_mean=("p_neg", "mean"),
        neu_mean=("p_neu", "mean"),
        para_count=("paragraph_id", "count"),
        uncertainty_mean=("uncertainty", "mean"),
        cik=("cik", "first"),
        ticker=("ticker", "first"),
        company_name=("company_name", "first"),
    ).reset_index()
    out["polarity"] = out["pos_mean"] - out["neg_mean"]
    return out

# --------- robust group K-fold that returns boolean masks ----------
def group_kfold_masks(groups: np.ndarray, n_splits: int, random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups).reshape(-1)  # 1-D
    N = len(groups)
    valid = ~pd.isna(groups)
    uniq = np.unique(groups[valid])

    if uniq.size < 2 or n_splits < 2:
        return []

    n_splits = min(n_splits, uniq.size)
    rng = np.random.RandomState(random_state)
    order = np.arange(uniq.size)
    rng.shuffle(order)
    uniq_shuf = uniq[order]

    buckets = [[] for _ in range(n_splits)]
    for i, g in enumerate(uniq_shuf):
        buckets[i % n_splits].append(g)

    masks = []
    for i in range(n_splits):
        test_groups = set(buckets[i])
        if not test_groups:
            continue
        test_mask = np.isin(groups, list(test_groups))
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        if test_mask.shape[0] != N or train_mask.shape[0] != N:
            continue
        masks.append((train_mask, test_mask))
    return masks

# --------- coalesce duplicate-named columns to a single Series ----------
def pick_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Coalesce possibly-duplicated columns named `col` into a single Series.
    If multiple, take first non-null across them; if none, return empty Series with that name.
    """
    block = df.loc[:, df.columns == col]
    if block.shape[1] == 0:
        return pd.Series(index=df.index, dtype=object, name=col)
    if block.shape[1] == 1:
        s = block.iloc[:, 0]
        s.name = col
        return s
    s = block.ffill(axis=1).bfill(axis=1).iloc[:, 0]
    s.name = col
    return s

# --------- coalesce duplicate-named columns in a DataFrame ----------
def coalesce_dup_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    For each name in `cols`, if duplicates exist, take first non-null across duplicates
    and drop extras. Finally drop any other lingering duplicate column names.
    """
    out = df.copy()
    for c in cols:
        same = out.loc[:, out.columns == c]
        if same.shape[1] > 1:
            out[c] = same.ffill(axis=1).bfill(axis=1).iloc[:, 0]
            # keep first occurrence of c, drop the rest
            first_idx = np.where(out.columns == c)[0][0]
            keep_mask = ~((out.columns == c) & (np.arange(len(out.columns)) > first_idx))
            out = out.loc[:, keep_mask]
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    return out

# ========================= Main =========================
def main():
    engine = get_engine()
    ensure_company_ref_table(engine)

    print("✅ Loading data with mapping...")
    filings = pd.read_sql(
        f"""
        SELECT
          m.cik,
          m.company_name,
          cr.ticker,
          (m.filing_date AT TIME ZONE 'UTC') AS filing_date,
          m.chunk_index AS paragraph_id,
          m.content AS text
        FROM {DB_SCHEMA}.mdna_sections m
        LEFT JOIN {DB_SCHEMA}.company_ref cr
          ON ltrim(m.cik,'0') = ltrim(cr.cik,'0');
        """,
        engine,
    )

    prices = pd.read_sql(
        f"""
        SELECT
          (sp."Date" AT TIME ZONE 'UTC') AS date,
          sp."Open"  AS open,
          sp."Close" AS close,
          sp."Volume" AS volume,
          UPPER(sp.ticker) AS ticker,
          cr.cik
        FROM {DB_SCHEMA}.stock_prices sp
        LEFT JOIN {DB_SCHEMA}.company_ref cr
          ON UPPER(sp.ticker) = UPPER(cr.ticker);
        """,
        engine,
    )

    benchmark = pd.read_sql(
        f"""SELECT ("Date" AT TIME ZONE 'UTC') AS date, close
            FROM {DB_SCHEMA}.sp500_index
            ORDER BY "Date";""",
        engine,
    )

    filings["filing_date"] = to_naive(filings["filing_date"])
    prices["date"]         = to_naive(prices["date"])
    benchmark["date"]      = to_naive(benchmark["date"])

    filings["cik"]    = filings["cik"].astype(str).str.lstrip("0")
    filings["ticker"] = filings["ticker"].astype(str).str.upper().replace({"NAN": np.nan, "NA": np.nan, "": np.nan})

    prices.columns     = [c.lower() for c in prices.columns]
    prices["cik"]      = prices["cik"].astype(str).str.lstrip("0")
    prices["ticker"]   = prices["ticker"].astype(str).str.upper()

    ciks_in_prices = set(prices["cik"].dropna().unique())
    tix_in_prices  = set(prices["ticker"].dropna().unique())
    before = len(filings)
    filings = filings[filings["cik"].isin(ciks_in_prices) | filings["ticker"].isin(tix_in_prices)].copy()
    after = len(filings)
    print(f"ℹ️ Filtered filings to those with price coverage: {before} -> {after}")
    if after == 0:
        raise RuntimeError("No overlapping firms between mdna_sections and stock_prices.")

    filings["entity"] = filings["ticker"]
    filings.loc[filings["entity"].isna(), "entity"] = filings["cik"]

    print("⚙️ Running sentiment...")
    se = SentimentEngine(cfg)
    filings = se.score_dataframe(filings, text_col="text")
    filings["polarity"] = filings["p_pos"] - filings["p_neg"]

    filing_sent = aggregate_to_filing(filings, se.backend, cfg.uncertainty_cutoff)

    keep = filing_sent["para_count"] >= cfg.min_paragraphs
    if not keep.any():
        print(f"⚠️ No filings met paragraph threshold ({cfg.min_paragraphs}). "
              f"Relaxing to >=2 and skipping uncertainty filter.")
        filing_sent = aggregate_to_filing(filings.assign(uncertainty=0.0), se.backend, 1.0)
        filing_sent = filing_sent[filing_sent["para_count"] >= 2]
    else:
        filing_sent = filing_sent[keep].copy()

    if filing_sent.empty:
        raise RuntimeError("Filings empty after aggregation — increase coverage or lower min_paragraphs.")

    price_ciks = set(prices["cik"].dropna().astype(str).str.lstrip("0").unique())
    price_tix  = set(prices["ticker"].dropna().astype(str).str.upper().unique())

    filing_sent["cik"]    = filing_sent["cik"].astype(str).str.lstrip("0")
    filing_sent["ticker"] = filing_sent["ticker"].astype(str).str.upper()

    filing_sent["entity"] = np.where(
        filing_sent["cik"].isin(price_ciks),
        filing_sent["cik"],
        np.where(filing_sent["ticker"].isin(price_tix), filing_sent["ticker"], np.nan)
    )

    before_fs = len(filing_sent)
    filing_sent = filing_sent[filing_sent["entity"].notna()].copy()
    after_fs = len(filing_sent)
    print(f"ℹ️ Filings aligned to price keys: {before_fs} -> {after_fs}")
    if after_fs == 0:
        raise RuntimeError("No overlap between filings and prices by CIK/ticker.")

    prices["entity"] = prices["cik"]
    mask_missing_cik = prices["entity"].isna() | (prices["entity"] == "")
    prices.loc[mask_missing_cik, "entity"] = prices.loc[mask_missing_cik, "ticker"]

    prices = compute_price_features(prices, cfg.event_window)
    benchmark = benchmark.sort_values("date").copy()
    benchmark["mkt_ret_fwd_w"] = np.log(benchmark["close"].shift(-cfg.event_window) / benchmark["close"])

    print("🔗 Merging filings to prices by ENTITY...")
    merged_list = []
    for ent, f_df in filing_sent.groupby("entity"):
        p_df = prices[prices["entity"] == ent].sort_values("date").copy()
        if p_df.empty:
            continue
        idx = np.searchsorted(p_df["date"].values, f_df["filing_date"].values, side="left")
        idx = np.clip(idx, 0, len(p_df) - 1)
        take = p_df.iloc[idx].reset_index(drop=True)
        merged = pd.concat([f_df.reset_index(drop=True), take.reset_index(drop=True)], axis=1)
        merged_list.append(merged)

    merged = pd.concat(merged_list, ignore_index=True) if merged_list else pd.DataFrame()
    if merged.empty:
        raise RuntimeError("No merged rows — verify same firms exist in mdna_sections and stock_prices.")

    b_dates = benchmark["date"].values
    i_b = np.searchsorted(b_dates, merged["filing_date"].values, side="left")
    i_b = np.clip(i_b, 0, len(b_dates) - 1)
    merged["mkt_ret_fwd_w"] = benchmark.iloc[i_b]["mkt_ret_fwd_w"].values

    if "sector" not in merged.columns:
        merged["sector"] = "Unknown"
    if "size_mcap" not in merged.columns:
        merged["size_mcap"] = np.nan

    merged = merged.dropna(subset=["ret_fwd_w", "vol_pre30", "polarity"]).copy()

    # ================= Modeling =================
    print("📈 Running OLS / QuantReg / CV...")

    # ---------- Build a single modeling frame M (coalescing dup columns) ----------
    X_base = merged[["polarity", "vol_pre30", "mkt_ret_fwd_w"]].copy()
    secD = pd.get_dummies(merged["sector"], prefix="sec", drop_first=True)
    if not secD.empty:
        X_base = pd.concat([X_base, secD], axis=1)

    y_series = pick_series(merged, "ret_fwd_w")
    grp_series = pick_series(merged, "entity").astype(str)

    M = pd.concat(
        [
            X_base.reset_index(drop=True),
            y_series.reset_index(drop=True).rename("y"),
            grp_series.reset_index(drop=True).rename("group"),
        ],
        axis=1,
    ).dropna().reset_index(drop=True)

    # OLS on all data
    X_full = sm.add_constant(M.drop(columns=["y", "group"]))
    y_full = M["y"]
    ols = sm.OLS(y_full, X_full).fit(cov_type="HC3")

    # QuantReg on all data
    qres = {}
    for q in (0.25, 0.50, 0.75):
        qr = sm.QuantReg(y_full, X_full).fit(q=q)
        qres[q] = {"params": qr.params.to_dict(), "pvalues": qr.pvalues.to_dict()}

    # ---------- Robust Group CV using boolean masks on M ----------
    groups = M["group"].to_numpy().reshape(-1)
    uniq, counts = np.unique(groups, return_counts=True)
    keep_groups = set(uniq[counts >= 2])
    M_cv = M.loc[np.isin(groups, list(keep_groups))].reset_index(drop=True) if len(keep_groups) >= 2 else M.copy()

    # Base features (without y/group)
    Z_all = M_cv.drop(columns=["y", "group"]).copy()
    y_all = M_cv["y"].copy()
    groups_cv = M_cv["group"].to_numpy().reshape(-1)

    n_groups = np.unique(groups_cv).size
    cv_folds_used = min(cfg.cv_folds, n_groups)

    def add_const_and_align(df: pd.DataFrame, cols_ref: List[str] | None = None) -> pd.DataFrame:
        df = df.copy()
        if "const" not in df.columns:
            df.insert(0, "const", 1.0)
        if cols_ref is not None:
            df = df.reindex(columns=cols_ref, fill_value=0.0)
        return df

    r2s = []
    if cv_folds_used >= 2:
        masks = group_kfold_masks(groups_cv, n_splits=cv_folds_used, random_state=cfg.random_state)
        if not masks:
            print(f"ℹ️ Not enough groups for CV after filtering (groups={n_groups}). Skipping CV.")
        else:
            for train_mask, test_mask in masks:
                Z_tr = Z_all.loc[train_mask]
                Z_te = Z_all.loc[test_mask]
                y_tr = y_all.loc[train_mask]
                y_te = y_all.loc[test_mask]

                X_tr = add_const_and_align(Z_tr)
                train_cols = list(X_tr.columns)
                X_te = add_const_and_align(Z_te, cols_ref=train_cols)

                m_cv = sm.OLS(y_tr, X_tr).fit(cov_type="HC3")
                preds = m_cv.predict(X_te)
                r2s.append(r2_score(y_te, preds))
    else:
        print(f"ℹ️ Not enough groups for CV (groups={n_groups}). Skipping CV.")

    # anomalies on full-sample OLS
    resid = ols.resid
    resid_z = (resid - resid.mean()) / (resid.std(ddof=0) + 1e-9)
    merged = merged.reset_index(drop=True)
    merged["resid_z"] = resid_z.reindex(index=merged.index, fill_value=np.nan).to_numpy()
    merged["red_flag"] = (np.abs(merged["resid_z"]) > cfg.z_thresh) | (
        np.sign(merged["polarity"]) != np.sign(merged["ret_fwd_w"])
    )

    # ================= Persist + Visualize =================
    print("💾 Writing results to DB and saving chart...")

    # Coalesce duplicate-named columns before writing
    merged = coalesce_dup_columns(
        merged,
        cols=["ticker", "cik", "entity", "filing_date", "date", "company_name"]
    )

    merged.to_sql("model_results", engine, schema=DB_SCHEMA, if_exists="replace", index=False)

    cv_mean = float(np.mean(r2s)) if len(r2s) else float("nan")
    summary = {
        "ols_params": ols.params.to_dict(),
        "ols_pvalues": ols.pvalues.to_dict(),
        "ols_r2_adj": float(ols.rsquared_adj),
        "cv_r2_mean": cv_mean,
        "cv_r2_all": [float(x) for x in r2s],
        "quantile": qres,
        "backend": "FinBERT" if hasattr(SentimentEngine, "backend") and getattr(SentimentEngine, "backend", "") == "finbert" else "VADER",
        "entity_note": "ENTITY = CIK if present in prices else ticker",
        "min_paragraphs": cfg.min_paragraphs,
        "n_entities": int(np.unique(groups).size),
        "cv_folds_used": int(cv_folds_used),
    }
    with open("model_results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig = px.scatter(
        merged,
        x="polarity",
        y="ret_fwd_w",
        color=merged["red_flag"].map({True: "Red Flag", False: "Aligned"}),
        hover_data=["entity", "ticker", "cik", "filing_date", "company_name"],
        title="Narrative Polarity vs 30-Day Log Return",
    )
    fig.write_html(cfg.out_html)

    print("=== OLS (key numbers) ===")
    print(f"Adj. R²: {summary['ols_r2_adj']:.4f}")
    if np.isfinite(cv_mean):
        print(f"CV R² mean ({cv_folds_used} folds): {summary['cv_r2_mean']:.4f}")
    else:
        print("CV R² mean: (skipped; not enough groups)")
    beta_pol = summary["ols_params"].get("polarity", float('nan'))
    p_pol = summary["ols_pvalues"].get("polarity", float('nan'))
    print(f"β_polarity: {beta_pol:.4f} (p={p_pol:.4g})")
    print(f"✅ Done. Chart saved to: {cfg.out_html}")

if __name__ == "__main__":
    main()
