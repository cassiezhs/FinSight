# fetch_sec.py
import re
import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "zihanshao1996@gmail.com"}  # single, consistent UA

# ---------------- Text helpers ---------------- #

def _normalize_text(html: str) -> tuple[str, str]:
    """Return (orig, lower) with whitespace collapsed so indices align."""
    soup = BeautifulSoup(html, "html.parser")
    orig = soup.get_text(" ", strip=True)
    orig = re.sub(r"\s+", " ", orig).strip()
    lower = orig.lower()
    return orig, lower

def _compile_label_pattern(label: str) -> re.Pattern:
    """
    Turn 'item 1a' into a regex that tolerates NBSP and punctuation.
    Example: r'\bitem[\s\xa0]*1a\b[.\-–—: ]*'
    """
    parts = label.split()
    joined = r"[\s\xa0]*".join(map(re.escape, parts))
    return re.compile(rf"\b{joined}\b[.\-–—: ]*", re.IGNORECASE)

def _find_best_section(
    orig: str,
    lower: str,
    start_label: str,
    end_labels: list[str],
    prefer_after_label: str | None = "part i",
    min_chars: int = 800,
) -> str:
    """Find all start matches, cut at earliest end, prefer after 'Part I', choose the longest plausible span."""
    start_pat = _compile_label_pattern(start_label)
    end_pats = [_compile_label_pattern(lab) for lab in end_labels]

    prefer_after = 0
    if prefer_after_label:
        m_pref = _compile_label_pattern(prefer_after_label).search(lower)
        if m_pref:
            prefer_after = m_pref.start()

    candidates = []
    for m in start_pat.finditer(lower):
        s0 = m.start()
        e0 = len(lower)
        for ep in end_pats:
            em = ep.search(lower, m.end())
            if em:
                e0 = min(e0, em.start())
        span_len = max(0, e0 - s0)
        score = (s0 >= prefer_after, span_len)  # prefer after 'Part I', then longer
        candidates.append((score, s0, e0))

    if not candidates:
        return ""

    # first pass: prefer-after + longest
    candidates.sort(key=lambda t: (t[0][0], t[0][1]), reverse=True)
    _, s0, e0 = candidates[0]

    # if tiny (likely TOC), pick absolute longest span
    if (e0 - s0) < min_chars:
        candidates.sort(key=lambda t: t[2] - t[1], reverse=True)
        _, s0, e0 = candidates[0]

    if e0 <= s0:
        e0 = min(s0 + 20000, len(orig))  # safety

    return orig[s0:e0].strip()

def _fallback_heading_grab(orig: str, lower: str, heading_words: list[str], end_labels: list[str]) -> str:
    """
    If 'Item 1A' is formatted oddly, fall back to a heading keyword like 'Risk Factors'.
    Grab from the first good heading after 'Part I' to the next end label.
    """
    # Find 'part i' to skip TOC
    prefer_after = 0
    m_pref = _compile_label_pattern("part i").search(lower)
    if m_pref:
        prefer_after = m_pref.start()

    # Find candidate indices for heading terms
    head_idxs = []
    for w in heading_words:
        for m in re.finditer(rf"\b{re.escape(w)}\b", lower, re.IGNORECASE):
            if m.start() >= prefer_after:
                head_idxs.append(m.start())
    if not head_idxs:
        return ""

    s0 = min(head_idxs)  # the earliest plausible heading after 'Part I'

    end_pats = [_compile_label_pattern(lab) for lab in end_labels]
    e0 = len(lower)
    for ep in end_pats:
        em = ep.search(lower, s0)
        if em:
            e0 = min(e0, em.start())

    if e0 <= s0:
        e0 = min(s0 + 20000, len(orig))

    return orig[s0:e0].strip()

# ---------------- SEC helpers ---------------- #

def get_cik(ticker: str) -> str | None:
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=UA, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    for item in data.values():
        if item["ticker"].lower() == ticker.lower():
            return str(item["cik_str"]).zfill(10)
    return None

def get_10k_meta_for_year(cik: str, year: int) -> tuple[str | None, str | None]:
    """Return (index.json URL, filingDate) for the company's 10-K filed in a given calendar year."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=UA, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    forms = data["filings"]["recent"]["form"]
    dates = data["filings"]["recent"]["filingDate"]
    accs = [a.replace("-", "") for a in data["filings"]["recent"]["accessionNumber"]]

    for i, f in enumerate(forms):
        if f == "10-K" and dates[i].startswith(str(year)):
            acc_no = accs[i]
            idx = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/index.json"
            return idx, dates[i]
    return None, None

def get_10k_html_url(doc_index_url: str) -> str | None:
    resp = requests.get(doc_index_url, headers=UA, timeout=60)
    if resp.status_code != 200:
        return None
    data = resp.json()
    files = data.get("directory", {}).get("item", []) or []
    if not files:
        return None

    items = []
    for f in files:
        name = f.get("name", "")
        lname = name.lower()
        try:
            size = int(f.get("size", 0))
        except Exception:
            size = 0
        items.append({"name": name, "lname": lname, "size": size})

    # Prefer issuer-slug primary doc like nvda-YYYYMMDD.htm
    slug = [
        it for it in items
        if it["lname"].endswith(".htm")
        and not it["lname"].endswith("_htm.xml")
        and re.search(r"-20\d{6}\.htm$", it["lname"])
    ]
    if slug:
        best = sorted(slug, key=lambda it: it["size"], reverse=True)[0]
        return doc_index_url.replace("index.json", best["name"])

    # Heuristic: largest .htm that's not obviously an exhibit
    EXCLUDE = ("exhibit", "consent", "policy", "subsidiar", "description", "plan")
    cand = [
        it for it in items
        if it["lname"].endswith(".htm")
        and not it["lname"].endswith("_htm.xml")
        and not any(x in it["lname"] for x in EXCLUDE)
        and not re.fullmatch(r"r\d+\.htm", it["lname"])
    ]
    if cand:
        best = sorted(cand, key=lambda it: it["size"], reverse=True)[0]
        return doc_index_url.replace("index.json", best["name"])

    # Last resort
    for it in items:
        if it["lname"].endswith(".htm") and not it["lname"].endswith("_htm.xml"):
            return doc_index_url.replace("index.json", it["name"])

    return None

# ---------------- Section extractors ---------------- #

def extract_risk_from_main_html(html_url: str) -> str:
    """
    Extract 'Item 1A' (Risk Factors). Strategy:
    1) Normalize text
    2) Use label-driven span with TOC avoidance
    3) If empty/too small, fall back to heading-based grab around 'risk factors'
    """
    r = requests.get(html_url, headers=UA, timeout=60)
    r.raise_for_status()
    orig, lower = _normalize_text(r.text)

    section = _find_best_section(
        orig, lower,
        start_label="item 1a",
        end_labels=["item 1b", "item 2", "part ii", "signatures"],
        prefer_after_label="part i",
        min_chars=800,
    )

    if not section or len(section) < 400:
        # Fallback: look for a heading like "Risk Factors"
        section2 = _fallback_heading_grab(
            orig, lower,
            heading_words=["risk factors", "risk factor"],  # tolerant heading search
            end_labels=["item 1b", "item 2", "part ii", "signatures"],
        )
        if len(section2) > len(section):
            section = section2

    return section or "Risk Factors section not found"

def extract_mdna_from_main_html(html_url: str) -> str:
    """
    Extract 'Item 7' (MD&A) generically (no Apple-specific anchors).
    """
    r = requests.get(html_url, headers=UA, timeout=60)
    r.raise_for_status()
    orig, lower = _normalize_text(r.text)

    section = _find_best_section(
        orig, lower,
        start_label="item 7",
        end_labels=["item 7a", "item 8", "signatures", "part ii"],
        prefer_after_label=None,
        min_chars=1500,
    )
    return section or "MD&A section not found"

# ---------------- Demo/main ---------------- #

if __name__ == "__main__":
    for tkr in ["NVDA"]:
        cik = get_cik(tkr)
        for yr in [2023, 2024, 2025]:
            idx_url, filing_date = get_10k_meta_for_year(cik, yr)
            if not idx_url:
                print(f"❌ No 10-K for {tkr} {yr}")
                continue
            html_url = get_10k_html_url(idx_url)
            if not html_url:
                print(f"❌ No HTML for {tkr} {yr}")
                continue

            risk = extract_risk_from_main_html(html_url)
            mdna = extract_mdna_from_main_html(html_url)

            print(f"\n=== {tkr} {yr} (filed {filing_date}) ===")
            print("Risk length:", len(risk))
            print("MD&A length:", len(mdna))
            print("Risk preview:", risk[:400])
            print("MD&A preview:", mdna[:400])
