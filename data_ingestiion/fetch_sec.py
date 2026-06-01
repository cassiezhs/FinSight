# fetch_sec.py
import re
import requests
from bs4 import BeautifulSoup

try:
    from .config import TICKERS, settings
except ImportError:
    from config import TICKERS, settings

UA = {"User-Agent": settings.sec_user_agent}

SEC_8K_ITEM_DESCRIPTIONS = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety - Reporting of Shutdowns and Patterns of Violations",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure or Appointment of Directors or Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.04": "Temporary Suspension of Trading Under Employee Benefit Plans",
    "5.05": "Amendments to Code of Ethics",
    "5.06": "Change in Shell Company Status",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "5.08": "Shareholder Director Nominations",
    "6.01": "ABS Informational and Computational Material",
    "6.02": "Change of Servicer or Trustee",
    "6.03": "Change in Credit Enhancement or External Support",
    "6.04": "Failure to Make Required Distribution",
    "6.05": "Securities Act Updating Disclosure",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}

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
    Example: r'\\bitem[\\s\\xa0]*1a\\b[.\\-–—: ]*'
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
    head_idxs = set()
    for w in heading_words:
        for m in re.finditer(rf"\b{re.escape(w)}\b", lower, re.IGNORECASE):
            if m.start() >= prefer_after:
                head_idxs.add(m.start())
    if not head_idxs:
        return ""

    end_pats = [_compile_label_pattern(lab) for lab in end_labels]
    candidates = []
    for s0 in sorted(head_idxs):
        e0 = len(lower)
        for ep in end_pats:
            em = ep.search(lower, s0)
            if em:
                e0 = min(e0, em.start())
        if e0 <= s0:
            e0 = min(s0 + 20000, len(orig))
        candidates.append((e0 - s0, s0, e0))

    _, s0, e0 = max(candidates)

    return orig[s0:e0].strip()

def describe_8k_items(items: str) -> str:
    codes = [item.strip() for item in re.split(r"[,;]", items or "") if item.strip()]
    descriptions = [
        f"{code}: {SEC_8K_ITEM_DESCRIPTIONS.get(code, 'Unmapped 8-K item')}"
        for code in codes
    ]
    return "; ".join(descriptions)

def _truncate_text(text: str, max_chars: int = 1400) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0].rstrip()
    return f"{cut}..."

def _extract_event_text_from_html(html: str, item_codes: str, max_chars: int = 1400) -> str:
    orig, lower = _normalize_text(html)
    codes = [item.strip() for item in re.split(r"[,;]", item_codes or "") if item.strip()]
    sections = []

    for code in codes:
        label = f"item {code}"
        section = _find_best_section(
            orig,
            lower,
            start_label=label,
            end_labels=[
                "item 1.01", "item 1.02", "item 1.03", "item 2.01", "item 2.02",
                "item 2.03", "item 2.04", "item 2.05", "item 2.06", "item 3.01",
                "item 3.02", "item 3.03", "item 4.01", "item 4.02", "item 5.01",
                "item 5.02", "item 5.03", "item 5.04", "item 5.05", "item 5.06",
                "item 5.07", "item 5.08", "item 7.01", "item 8.01", "item 9.01",
                "signatures",
            ],
            prefer_after_label=None,
            min_chars=80,
        )
        if section and len(section) > 80:
            sections.append(section)

    if sections:
        return _truncate_text(" ".join(sections), max_chars)

    return _truncate_text(orig, max_chars)

def _get_filing_package_html_urls(index_url: str, primary_document: str) -> list[tuple[str, str]]:
    resp = requests.get(index_url, headers=UA, timeout=60)
    if resp.status_code != 200:
        return []

    data = resp.json()
    files = data.get("directory", {}).get("item", []) or []
    ranked = []
    for file_info in files:
        name = file_info.get("name", "")
        lname = name.lower()
        doc_type = file_info.get("type", "")
        if not lname.endswith((".htm", ".html")) or lname.endswith("_htm.xml"):
            continue

        rank = 99
        if name == primary_document:
            rank = 0
        elif "ex-99" in lname or "ex99" in lname or doc_type.upper().startswith("EX-99"):
            rank = 1
        elif "ex-10" in lname or doc_type.upper().startswith("EX-10"):
            rank = 2
        elif lname.startswith("ex"):
            rank = 3

        if rank < 99:
            ranked.append((rank, name, index_url.replace("index.json", name)))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [(name, url) for _, name, url in ranked[:3]]

def extract_8k_detail_preview(
    filing_url: str,
    filing_index_url: str,
    primary_document: str,
    item_codes: str,
    max_chars: int = 1400,
) -> tuple[str, str]:
    """Extract a concise non-AI preview from the primary 8-K and key exhibits."""
    urls = _get_filing_package_html_urls(filing_index_url, primary_document)
    if not urls and filing_url:
        urls = [(primary_document or "primary filing", filing_url)]

    snippets = []
    sources = []
    for name, url in urls:
        try:
            resp = requests.get(url, headers=UA, timeout=60)
            if resp.status_code != 200:
                continue
            snippet = _extract_event_text_from_html(resp.text, item_codes, max_chars=700)
            if snippet and snippet not in snippets:
                snippets.append(snippet)
                sources.append(name)
        except Exception:
            continue

    return _truncate_text(" ".join(snippets), max_chars), ", ".join(sources)

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

def get_filing_meta_for_year(cik: str, year: int, form_types: tuple[str, ...]) -> tuple[str | None, str | None, str | None]:
    """Return (index.json URL, filingDate, formType) for the first matching form filed in a year."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=UA, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    forms = data["filings"]["recent"]["form"]
    dates = data["filings"]["recent"]["filingDate"]
    accs = [a.replace("-", "") for a in data["filings"]["recent"]["accessionNumber"]]

    wanted = set(form_types)
    for i, f in enumerate(forms):
        if f in wanted and dates[i].startswith(str(year)):
            acc_no = accs[i]
            idx = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/index.json"
            return idx, dates[i], f
    return None, None, None


def get_10k_meta_for_year(cik: str, year: int) -> tuple[str | None, str | None]:
    """Return (index.json URL, filingDate) for the company's 10-K filed in a given calendar year."""
    idx, filing_date, _ = get_filing_meta_for_year(cik, year, ("10-K",))
    return idx, filing_date

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

def get_8k_meta(cik: str, ticker: str, start_year: int, end_year: int) -> list[dict]:
    """Return recent 8-K/8-K-A metadata rows filed within the selected calendar years."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=UA, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_documents = recent.get("primaryDocument", [])
    primary_descriptions = recent.get("primaryDocDescription", [])
    items = recent.get("items", [])

    rows = []
    for i, form in enumerate(forms):
        if form not in {"8-K", "8-K/A"}:
            continue

        filing_date = dates[i]
        filing_year = int(filing_date[:4])
        if filing_year < start_year or filing_year > end_year:
            continue

        accession_number = accession_numbers[i]
        accession_no_dash = accession_number.replace("-", "")
        primary_document = primary_documents[i] if i < len(primary_documents) else ""
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession_no_dash}/{primary_document}"
            if primary_document else
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dash}/"
        )
        filing_index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dash}/index.json"
        item_codes = items[i] if i < len(items) else ""

        rows.append({
            "cik": cik,
            "ticker": ticker.upper(),
            "filing_date": filing_date,
            "accession_number": accession_number,
            "form_type": form,
            "items": item_codes,
            "item_descriptions": describe_8k_items(item_codes),
            "primary_document": primary_document,
            "primary_doc_description": primary_descriptions[i] if i < len(primary_descriptions) else "",
            "filing_url": filing_url,
            "filing_index_url": filing_index_url,
        })

    return rows

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

def extract_mdna_from_main_html(html_url: str, form_type: str = "10-K") -> str:
    """
    Extract MD&A generically (no Apple-specific anchors).
    10-K MD&A is Item 7; 10-Q MD&A is Item 2.
    """
    r = requests.get(html_url, headers=UA, timeout=60)
    r.raise_for_status()
    orig, lower = _normalize_text(r.text)

    if form_type == "10-Q":
        section = _fallback_heading_grab(
            orig, lower,
            heading_words=[
                "management’s discussion and analysis",
                "management's discussion and analysis",
                "management discussion and analysis",
            ],
            end_labels=["item 3", "item 4", "signatures"],
        )
        if not section or len(section) < 800:
            section = _find_best_section(
                orig, lower,
                start_label="item 2",
                end_labels=["item 3", "item 4", "signatures"],
                prefer_after_label="part i",
                min_chars=1500,
            )
    else:
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
    for tkr in TICKERS:
        cik = get_cik(tkr)
        for yr in range(settings.start_year, settings.end_year + 1):
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
