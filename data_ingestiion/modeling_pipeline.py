#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4.0 — Modeling (Full Coverage Version)
- Keeps all tickers (no paragraph or uncertainty pruning)
- Prints model summary (no DB write)
- Outputs HTML scatter plot
"""

import os, re, numpy as np, pandas as pd, requests
from dataclasses import dataclass
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
import plotly.express as px

# ======================= Config =======================
pd.options.mode.copy_on_write = True
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
    z_thresh: float = 2.0
    out_html: str = "model_scatter.html"

cfg = CFG()

# ================= DB Engine =================
def get_engine():
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
        raise RuntimeError("Missing DB env vars.")
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, pool_pre_ping=True)

# ================= Company Ref =================
def ensure_company_ref_table(engine):
    try:
        with engine.connect() as conn:
            conn.execute(text(f'SELECT 1 FROM "{DB_SCHEMA}"."company_ref" LIMIT 1;'))
        print("🔎 company_ref exists — refreshing with latest SEC list...")
    except Exception:
        print("ℹ️ company_ref not found — creating...")
    data = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": "zihanshao1996@gmail.com"}, timeout=60
    ).json()
    df = pd.DataFrame.from_dict(data, orient="index")\
        .rename(columns={"cik_str": "cik", "title": "company_name"})
    df["cik"] = df["cik"].astype(int).astype(str).str.zfill(10)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df[["cik", "ticker", "company_name"]]
    df.to_sql("company_ref", engine, schema=DB_SCHEMA, if_exists="replace", index=False)
    print(f"✅ company_ref refreshed with {len(df)} rows.")

# ================= Helpers =================
ALIASES = {"BRK.B": "BRK-B", "BF.B": "BF-B"}
def to_canonical_ticker(s):
    if not s: return None
    t = str(s).strip().upper().replace(".", "-")
    return ALIASES.get(re.sub("-{2,}", "-", t), re.sub("-{2,}", "-", t))

def to_naive(s): return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)

def ensure_single_entity_col(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    candidates = [c for c in df.columns if c.startswith("entity")]
    base = None
    for c in candidates:
        col = df[c]
        if isinstance(col, pd.DataFrame): col = col.iloc[:, 0]
        base = col if base is None else base.combine_first(col)
    df["entity"] = base.astype(str)
    for c in candidates:
        if c != "entity": df.drop(columns=c, inplace=True, errors="ignore")
    return df

def compute_price_features(prices, event_window):
    p = prices.dropna(subset=["entity","date"]).drop_duplicates(["entity","date"])\
        .sort_values(["entity","date"]).copy()
    for c in ("open","close","volume"):
        if c in p: p[c] = pd.to_numeric(p[c], errors="coerce")
    p["log_close"] = np.log(p["close"])
    p["ret_d"] = p.groupby("entity")["log_close"].diff()
    p["vol_pre30"] = p.groupby("entity")["ret_d"].rolling(30,min_periods=10).std().reset_index(level=0,drop=True)
    p["ret_fwd_w"] = p.groupby("entity")["log_close"].shift(-event_window)-p["log_close"]
    p.drop(columns=["log_close"], inplace=True)
    return p

def merge_asof_safe(filing_sent, prices):
    left = filing_sent.rename(columns={"filing_date":"f_date"}).dropna(subset=["entity","f_date"]).copy()
    left["f_date"] = pd.to_datetime(left["f_date"], errors="coerce")
    left = left.dropna(subset=["f_date"]).sort_values(["entity","f_date"])
    right = prices.dropna(subset=["entity","date"]).copy()
    right["date"] = pd.to_datetime(right["date"], errors="coerce")
    right = right.sort_values(["entity","date"])
    try:
        merged = pd.merge_asof(left,right,left_on="f_date",right_on="date",by="entity",
                               direction="nearest",allow_exact_matches=True)
    except ValueError as e:
        print(f"⚠️ merge_asof failed ({e}) — trying fallback...")
        chunks=[]
        for ent,lf in left.groupby("entity"):
            rf=right[right["entity"]==ent]
            if rf.empty: continue
            idx=np.searchsorted(rf["date"],lf["f_date"],side="left")
            idx=np.clip(idx,0,len(rf)-1)
            chunks.append(pd.concat([lf.reset_index(drop=True),rf.iloc[idx].reset_index(drop=True)],axis=1))
        merged=pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame()
    return merged

# ================= Sentiment =================
class SentimentEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")

        try:
            # 🔒 Prefer safetensors to bypass torch>=2.6 restriction
            mdl = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert",
                use_safetensors=True
            )
            print("✅ FinBERT loaded with safetensors.")
        except Exception as e:
            print(f"⚠️ safetensors load failed, trying fallback: {e}")
            mdl = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert",
                trust_remote_code=True
            )

        self.pipe = pipeline(
            "text-classification",
            model=mdl,
            tokenizer=tok,
            truncation=True,
            top_k=None
        )
        self.backend = "finbert"
        print("Loaded FinBERT.")

    def _extract_scores(self, res):
        """Extract p_pos, p_neu, p_neg from FinBERT output"""
        items = res[0] if isinstance(res, list) and res and isinstance(res[0], list) else res
        scores = {d["label"].lower(): float(d["score"]) for d in items if isinstance(d, dict)}
        return scores.get("positive", 0), scores.get("neutral", 0), scores.get("negative", 0)

    def score_paragraph(self, text: str):
        """Score a single paragraph."""
        res = self.pipe(text[: self.cfg.tokenizer_max_len])
        p_pos, p_neu, p_neg = self._extract_scores(res)
        uncertainty = 1.0 - max(p_pos, p_neu, p_neg)
        return p_pos, p_neu, p_neg, uncertainty

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Apply FinBERT to a whole DataFrame."""
        results = [self.score_paragraph(str(x)) for x in df[text_col]]
        out = df.copy()
        out[["p_pos", "p_neu", "p_neg", "uncertainty"]] = pd.DataFrame(results, index=df.index)
        return out



# ================= Main =================
def main():
    engine = get_engine()
    ensure_company_ref_table(engine)

    print("✅ Loading data with mapping...")
    filings = pd.read_sql(f"""
        SELECT DISTINCT ON (m.cik, UPPER(m.ticker), m.filing_date, m.chunk_index)
            COALESCE(NULLIF(ltrim(m.cik,'0'),''), cr.cik) AS cik,
            COALESCE(UPPER(m.ticker), cr.ticker)         AS ticker,
            COALESCE(m.company_name, cr.company_name)    AS company_name,
            (m.filing_date AT TIME ZONE 'UTC')           AS filing_date,
            m.chunk_index AS paragraph_id,
            m.content AS text
        FROM "{DB_SCHEMA}"."mdna_sections" m
        LEFT JOIN "{DB_SCHEMA}"."company_ref" cr 
          ON ltrim(m.cik,'0') = ltrim(cr.cik,'0')
          OR UPPER(m.ticker) = cr.ticker;
    """, engine)
    print("Distinct tickers from SQL:", filings["ticker"].nunique())

    prices = pd.read_sql(f"""
        SELECT (sp."Date" AT TIME ZONE 'UTC') AS date,
               sp."Open" AS open, sp."Close" AS close, sp."Volume" AS volume,
               UPPER(sp.ticker) AS ticker, cr.cik
        FROM "{DB_SCHEMA}"."stock_prices" sp
        LEFT JOIN "{DB_SCHEMA}"."company_ref" cr ON UPPER(sp.ticker)=cr.ticker;
    """, engine)
    benchmark = pd.read_sql(f"""
        SELECT ("Date" AT TIME ZONE 'UTC') AS date, close
        FROM "{DB_SCHEMA}"."sp500_index" ORDER BY "Date";
    """, engine)

    filings["filing_date"]=to_naive(filings["filing_date"])
    prices["date"]=to_naive(prices["date"])
    benchmark["date"]=to_naive(benchmark["date"])

    filings["ticker"]=filings["ticker"].map(to_canonical_ticker)
    prices["ticker"]=prices["ticker"].map(to_canonical_ticker)
    filings["cik"]=filings["cik"].astype(str).str.strip().str.lstrip("0").replace({"":np.nan})
    prices["cik"]=prices["cik"].astype(str).str.strip().str.lstrip("0").replace({"":np.nan})
    filings["entity"]=np.where(filings["cik"].notna(),filings["cik"],filings["ticker"])
    prices["entity"]=np.where(prices["cik"].notna(),prices["cik"],prices["ticker"])

    # Sentiment
    print("⚙️ Running sentiment...")
    se=SentimentEngine(cfg)
    filings=se.score_dataframe(filings,"text")
    filings["polarity"]=filings["p_pos"]-filings["p_neg"]

    # Aggregation — keep all tickers
    def aggregate_to_filing(df):
        g = df.groupby(["entity","ticker","company_name","filing_date"], dropna=False)
        out = g.agg(
            pos_mean=("p_pos","mean"),
            neg_mean=("p_neg","mean"),
            para_count=("paragraph_id","count"),
            unc_median=("uncertainty","median")
        ).reset_index()
        out["polarity"]=out["pos_mean"]-out["neg_mean"]
        return out[out["para_count"]>=1]  # keep all filings

    filing_sent=aggregate_to_filing(filings)
    print("After aggregation — entities:",filing_sent["entity"].nunique(),
          "| tickers:",filing_sent["ticker"].nunique(),"| rows:",len(filing_sent))

    print("🧪 Pre-merge diagnostics:")
    print("  filing_sent shape:",filing_sent.shape)
    print("  prices shape    :",prices.shape)
    print("  filings entities:",filing_sent["entity"].nunique())
    print("  prices  entities:",prices["entity"].nunique())

    prices=compute_price_features(prices,cfg.event_window)
    benchmark=benchmark.sort_values("date").assign(
        close=lambda x: pd.to_numeric(x["close"], errors="coerce"),
        mkt_ret_fwd_w=lambda x: np.log(x["close"].shift(-cfg.event_window)/x["close"])
    ).dropna(subset=["close"])

    merged=merge_asof_safe(filing_sent,prices)
    merged=pd.merge_asof(
        merged.sort_values("f_date").reset_index(drop=True),
        benchmark[["date","mkt_ret_fwd_w"]].sort_values("date"),
        left_on="f_date", right_on="date", direction="nearest",
        allow_exact_matches=True
    ).rename(columns={"f_date":"filing_date"})
    merged=ensure_single_entity_col(merged)

    # Modeling
    for c in ("ret_fwd_w","vol_pre30","polarity","mkt_ret_fwd_w"):
        if c in merged: merged[c]=pd.to_numeric(merged[c], errors="coerce")
    core_ok=merged[["ret_fwd_w","vol_pre30","polarity"]].notna().all(axis=1)
    merged_model=ensure_single_entity_col(merged.loc[core_ok].copy())

    entity_series = merged_model["entity"]
    if isinstance(entity_series, pd.DataFrame):
        entity_series = entity_series.iloc[:, 0]

    n_entities = int(float(pd.Series(entity_series.astype(str)).nunique()))
    print(f"✅ Entity column flattened; {n_entities} entities for modeling")

    print("📈 Running OLS / QuantReg / LOGO CV...")
    X_raw=merged_model[["polarity","vol_pre30","mkt_ret_fwd_w"]]
    y=merged_model["ret_fwd_w"]
    X=sm.add_constant(X_raw, has_constant="add")
    ols=sm.OLS(y,X).fit(cov_type="HC3")
    qres={q:sm.QuantReg(y,X).fit(q=q) for q in (0.25,0.5,0.75)}
    ridge_pipe=Pipeline([("scaler",StandardScaler()),("ridge",Ridge(alpha=1.0))])
    ridge_pipe.fit(X_raw,y)
    rho,_=spearmanr(y,ridge_pipe.predict(X_raw),nan_policy="omit")

    logo=LeaveOneGroupOut(); r2s_logo=[]
    if n_entities>=2:
        groups = merged_model["entity"]
        if isinstance(groups, pd.DataFrame):
            groups = groups.iloc[:, 0]
        groups = pd.Series(groups).astype(str).reset_index(drop=True)
        for tr, te in logo.split(X_raw, y, groups):
            if len(te)<2: continue
            X_tr,X_te=sm.add_constant(X_raw.iloc[tr]),sm.add_constant(X_raw.iloc[te])
            X_te=X_te.reindex(columns=X_tr.columns, fill_value=0)
            model=sm.OLS(y.iloc[tr],X_tr).fit(cov_type="HC3")
            pred=model.predict(X_te)
            score=r2_score(y.iloc[te],pred)
            if np.isfinite(score): r2s_logo.append(score)

    resid_z=(ols.resid-ols.resid.mean())/(ols.resid.std(ddof=0)+1e-9)
    merged_model["resid_z"]=resid_z
    merged_model["red_flag"]=(np.abs(resid_z)>cfg.z_thresh)|(np.sign(merged_model["polarity"])!=np.sign(merged_model["ret_fwd_w"]))

    print("🧾 Model summary (printed only, not saved):")
    summary={
        "ols_r2_adj":float(ols.rsquared_adj),
        "ols_params":ols.params.to_dict(),
        "ols_pvalues":ols.pvalues.to_dict(),
        "quantile":{str(q):{"params":qres[q].params.to_dict()} for q in qres},
        "logo_cv_r2_mean":float(np.mean(r2s_logo)) if r2s_logo else np.nan,
        "spearman_rho_full":float(rho),
        "n_entities":n_entities
    }
    for k,v in summary.items(): print(f"{k}: {v}")

    print("\n📊 Top 10 merged_model rows (sample):")
    print(merged_model.head(10)[["entity","ticker","company_name","filing_date","polarity","ret_fwd_w","vol_pre30"]])

    print(f"\nAdj.R²={summary['ols_r2_adj']:.4f} | Entities={n_entities} | ✅ Done → {cfg.out_html}")


    dfp=merged_model.dropna(subset=["polarity","ret_fwd_w"]).copy()
    dfp["flag_label"]=dfp["red_flag"].map({True:"Red Flag",False:"Aligned"})

    dfp = dfp.loc[:, ~dfp.columns.duplicated()].copy()
    dfp.columns = pd.io.parsers._parser.ParserBase({'names': dfp.columns})._maybe_dedup_names(dfp.columns) if hasattr(pd.io.parsers, "_parser") else pd.Index(
    [f"{c}_{i}" if c in dfp.columns[:i] else c for i, c in enumerate(dfp.columns)])
    px.scatter(
        dfp, x="polarity", y="ret_fwd_w", color="flag_label",
        hover_data=["entity","ticker","company_name","filing_date"],
        title="Narrative Polarity vs 30-Day Log Return"
    ).write_html(cfg.out_html)
    print(f"\nAdj.R²={summary['ols_r2_adj']:.4f} | Entities={n_entities} | ✅ Done → {cfg.out_html}")

if __name__=="__main__":
    main()
