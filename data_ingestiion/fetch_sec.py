import requests
from bs4 import BeautifulSoup
import re
import psycopg2
import yfinance as yf


UA = {"User-Agent": "zihanshao1996@gmail.com"}

# --- put near top of fetch_sec.py ---
import re
import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "zihanshao1996@gmail.com"}  # keep one consistent UA/email

def _normalize_text(html: str):
    soup = BeautifulSoup(html, "html.parser")
    orig = soup.get_text(" ", strip=True)
    # collapse whitespace so lower() remains byte-aligned
    orig = re.sub(r"\s+", " ", orig).strip()
    lower = orig.lower()
    return orig, lower

def _compile_label_pattern(label: str) -> re.Pattern:
    """
    Turn 'item 1a' into a regex that tolerates non-breaking spaces & punctuation:
    \bitem[\s\xa0]*1a\b[.\-–:]*
    """
    # allow optional spaces/NBSP between parts (e.g., 'Item 1A')
    parts = label.split()
    joined = r"[\s\xa0]*".join(map(re.escape, parts))
    return re.compile(rf"\b{joined}\b[\.\-–: ]*", re.IGNORECASE)

def _find_best_section(orig: str, lower: str, start_label: str, end_labels: list[str],
                       prefer_after_label: str | None = "part i",
                       min_chars: int = 800) -> str:
    """
    Find ALL occurrences of `start_label`, end each at earliest of `end_labels`,
    then choose the longest span (prefer starts after 'Part I').
    """
    start_pat = _compile_label_pattern(start_label)
    end_pats  = [_compile_label_pattern(lab) for lab in end_labels]

    # Optional: discourage TOC by preferring matches after 'Part I'
    prefer_after = 0
    if prefer_after_label:
        m_pref = _compile_label_pattern(prefer_after_label).search(lower)
        if m_pref:
            prefer_after = m_pref.start()

    candidates = []
    for m in start_pat.finditer(lower):
        s0 = m.start()
        # find earliest end AFTER this start
        e0 = len(lower)
        for ep in end_pats:
            em = ep.search(lower, m.end())
            if em:
                e0 = min(e0, em.start())
        span_len = max(0, e0 - s0)
        # score: prefer after 'Part I' and longer text
        score = (s0 >= prefer_after, span_len)
        candidates.append((score, s0, e0))

    if not candidates:
        return ""

    # choose the best by score (bool then length), then by length if tied
    candidates.sort(key=lambda t: (t[0][0], t[0][1]), reverse=True)
    _, s0, e0 = candidates[0]

    # Fallback if span is tiny (still TOC): pick the longest overall
    if (e0 - s0) < min_chars:
        candidates.sort(key=lambda t: t[2] - t[1], reverse=True)
        _, s0, e0 = candidates[0]

    # safety fallback
    if e0 <= s0:
        e0 = min(s0 + 20000, len(orig))

    return orig[s0:e0].strip()

def extract_risk_from_main_html(html_url: str) -> str:
    """
    Extract Item 1A (Risk Factors) by choosing the longest plausible section,
    preferably after 'Part I'.
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
    return section or "Risk Factors section not found"

def extract_mdna_from_main_html(html_url: str) -> str:
    """
    Extract Item 7 (MD&A) by choosing the longest plausible section,
    preferably after 'Part II' (some filings structure this way).
    """
    r = requests.get(html_url, headers=UA, timeout=60)
    r.raise_for_status()
    orig, lower = _normalize_text(r.text)

    section = _find_best_section(
        orig, lower,
        start_label="item 7",
        end_labels=["item 7a", "item 8", "part ii", "signatures"],
        prefer_after_label=None,  # leave None if you don't want to bias by 'Part II'
        min_chars=1500,           # MD&A tends to be long
    )
    return section or "MD&A section not found"

def _get_main_text(html_url: str) -> tuple[str, str]:
    """Fetch HTML and return (orig_text, lower_text) with mild whitespace cleanup."""
    r = requests.get(html_url, headers=UA, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # Use spaces, not \n, to keep regex ranges predictable across PDFs/HTMLs
    orig = soup.get_text(" ", strip=True)
    # normalize spaces
    orig = re.sub(r"\s+", " ", orig).strip()
    lower = orig.lower()
    return orig, lower

def _extract_section(orig: str, lower: str, start_label: str, end_labels: list[str]) -> str:
    """
    Extract text between a start label (e.g., 'item 1a') and earliest matching end label
    (e.g., 'item 1b', 'item 2', 'part ii', 'signatures').
    Tolerant to punctuation after labels.
    """
    # start: \bitem\s+1a\b[\.\-–:]*
    s_pat = re.compile(rf"\b{re.escape(start_label)}\b[\.\-–: ]*", re.IGNORECASE)
    s = s_pat.search(lower)
    if not s:
        return ""  # start not found

    end_pos = len(lower)
    for lab in end_labels:
        e_pat = re.compile(rf"\b{re.escape(lab)}\b[\.\-–: ]*", re.IGNORECASE)
        m = e_pat.search(lower, s.end())
        if m:
            end_pos = min(end_pos, m.start())

    # Map back to original-case substring using same indices
    start_i = s.start()
    end_i = end_pos if end_pos > start_i else min(start_i + 20000, len(orig))  # 20k-char fallback
    return orig[start_i:end_i].strip()

def get_cik(ticker):
    url = f"https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "zihansha19960@gmail.com"}
    resp = requests.get(url, headers=headers)
    data = resp.json()
    
    for item in data.values():
        if item['ticker'].lower() == ticker.lower():
            cik = str(item['cik_str']).zfill(10)
            return cik
    return None


# def get_latest_10k_url(cik):
#     url = f"https://data.sec.gov/submissions/CIK{cik}.json"
#     headers = {"User-Agent": "zihansha19960@gmail.com"}
#     resp = requests.get(url, headers=headers)
#     data = resp.json()
    
#     for filing in data['filings']['recent']['form']:
#         if filing == '10-K':
#             index = data['filings']['recent']['form'].index(filing)
#             acc_no = data['filings']['recent']['accessionNumber'][index].replace("-", "")
#             doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/index.json"
#             return doc_url
#     return None

def get_10k_meta_for_year(cik: str, year: int) -> tuple[str | None, str | None]:
    """
    Return (index.json URL, filingDate) for the company's 10-K filed in a given calendar year.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=UA, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    forms = data["filings"]["recent"]["form"]
    dates = data["filings"]["recent"]["filingDate"]
    accs  = [a.replace("-", "") for a in data["filings"]["recent"]["accessionNumber"]]

    for i, f in enumerate(forms):
        if f == "10-K" and dates[i].startswith(str(year)):
            acc_no = accs[i]
            idx = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/index.json"
            return idx, dates[i]  # e.g., '2024-02-01'
    return None, None

cik = get_cik("NVDA")
# print(get_latest_10k_url(cik))

# def get_10k_html_url(doc_index_url):
#     headers = {"User-Agent": "zihanshao1996@gmail.com"}
#     resp = requests.get(doc_index_url, headers=headers)
    
#     if resp.status_code != 200:
#         print("Failed to fetch index:", doc_index_url)
#         return None

#     data = resp.json()
#     files = data.get('directory', {}).get('item', [])

#     print("📂 Available files in the filing:")
#     for f in files:
#         print("-", f['name'])

#     # 优先寻找 aapl-YYYYMMDD.htm 格式
#     for f in files:
#         if f['name'].startswith("aapl-") and f['name'].endswith(".htm"):
#             html_url = doc_index_url.replace('index.json', f['name'])
#             print("Found primary HTML:", html_url)
#             return html_url

#     # fallback: 找第一个 .htm 文件
#     for f in files:
#         if f['name'].endswith(".htm"):
#             html_url = doc_index_url.replace('index.json', f['name'])
#             print("Fallback HTML:", html_url)
#             return html_url

#     print("No HTML file found.")
#     return None

def extract_mdna_from_main_html(html_url):
    headers = {"User-Agent": "zihanshao1996@gmail.com"}
    html = requests.get(html_url, headers=headers).text
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator='\n', strip=True)

    print("\n🔍 Searching for 'Item 7'...")

    # Find all potential 'Item 7' matches
    item7_matches = list(re.finditer(r'Item\s*7[\.\-–]?\s+(Management.*?Operations)?', text, re.IGNORECASE))

    # Use page headers to anchor Item 7A and Item 8 more reliably
    item7a_match = re.search(r'Apple Inc\.\s*\|\s*2024 Form 10-K\s*\|\s*\d+\s*[\r\n]+Item\s*7A[\.\-–]?', text, re.IGNORECASE)
    item8_match = re.search(r'Apple Inc\.\s*\|\s*2024 Form 10-K\s*\|\s*\d+\s*[\r\n]+Item\s*8[\.\-–]?', text, re.IGNORECASE)

    if not item7_matches:
        print("❌ No 'Item 7' match found.")
        return "MD&A section not found"

    #print(f"✅ Found {len(item7_matches)} 'Item 7' matches.")
    for i, m in enumerate(item7_matches):
        print(f"  Match {i+1} at char index: {m.start()}")

    # Select best starting point (3rd match usually skips the TOC and summary)
    if len(item7_matches) >= 3:
        start_index = item7_matches[2].start()
    else:
        start_index = item7_matches[-1].start()

    # Choose end index based on presence of 7A or 8
    if item7a_match:
        end_index = item7a_match.end()
        if item8_match and item8_match.start() > end_index:
            end_index = item8_match.start()
    elif item8_match:
        end_index = item8_match.start()
    else:
        end_index = start_index + 20000  # Fallback slice

    if end_index <= start_index:
        print("⚠️ End index before start index. Using fallback slice of 5000 chars.")
        return text[start_index:start_index + 5000].strip()

    extracted = text[start_index:end_index].strip()
    print(f"\n📏 Extracted length: {len(extracted)} characters\n")
    return extracted or "Empty content extracted"


def get_10k_url_for_year(cik, year):
    """
    Return the index.json URL for the company's 10-K filed in a given year.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": "zihanshao19960@gmail.com"}
    resp = requests.get(url, headers=headers)
    data = resp.json()

    forms = data['filings']['recent']['form']
    dates = data['filings']['recent']['filingDate']
    accs  = [a.replace("-", "") for a in data['filings']['recent']['accessionNumber']]

    for i, f in enumerate(forms):
        if f == '10-K' and dates[i].startswith(str(year)):
            acc_no = accs[i]
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/index.json"
    return None

def get_10k_html_url(doc_index_url):
    headers = {"User-Agent": "zihanshao1996@gmail.com"}
    resp = requests.get(doc_index_url, headers=headers, timeout=30)
    if resp.status_code != 200:
        print("Failed to fetch index:", doc_index_url)
        return None

    data = resp.json()
    files = data.get('directory', {}).get('item', [])
    if not files:
        print("No files in index.json.")
        return None

    # Build normalized list
    items = []
    for f in files:
        name = f.get('name', '')
        lname = name.lower()
        try:
            size = int(f.get('size', 0))
        except Exception:
            size = 0
        items.append({'name': name, 'lname': lname, 'size': size})

    # 1) Strong preference: slug like nvda-YYYYMMDD.htm (or aapl-..., msft-..., etc.)
    import re
    slug = [it for it in items
            if it['lname'].endswith('.htm')
            and not it['lname'].endswith('_htm.xml')
            and re.search(r'-20\d{6}\.htm$', it['lname'])]
    if slug:
        best = sorted(slug, key=lambda it: it['size'], reverse=True)[0]
        html_url = doc_index_url.replace('index.json', best['name'])
        print("Primary slug HTML:", html_url)
        return html_url

    # 2) Heuristic: large .htm that isn't obviously an exhibit/policy/consent/subsidiary or R#.htm
    EXCLUDE = ('exhibit', 'consent', 'policy', 'subsidiar', 'description', 'plan')
    cand = [it for it in items
            if it['lname'].endswith('.htm')
            and not it['lname'].endswith('_htm.xml')
            and not any(x in it['lname'] for x in EXCLUDE)
            and not re.fullmatch(r'r\d+\.htm', it['lname'])]
    if cand:
        best = sorted(cand, key=lambda it: it['size'], reverse=True)[0]
        html_url = doc_index_url.replace('index.json', best['name'])
        print("Primary (heuristic) HTML:", html_url)
        return html_url

    # 3) Last resort: first .htm (non-xml)
    for it in items:
        if it['lname'].endswith('.htm') and not it['lname'].endswith('_htm.xml'):
            html_url = doc_index_url.replace('index.json', it['name'])
            print("Fallback HTML:", html_url)
            return html_url

    print("No HTML file found.")
    return None


def extract_risk_from_main_html(html_url: str) -> str:
    """Risk Factors ≈ Item 1A … up to Item 1B or Item 2 (or Part II)"""
    orig, lower = _get_main_text(html_url)
    return _extract_section(
        orig, lower,
        start_label="item 1a",
        end_labels=["item 1b", "item 2", "part ii", "signatures"]
    ) or "Risk Factors section not found"
# ---------------- Test: Pull 2023 and 2025 AAPL Filings ---------------- #

if __name__ == "__main__":
    for tkr in ["NVDA"]:
        cik = get_cik(tkr)
        for yr in [2023, 2024, 2025]:
            idx_url, filing_date = get_10k_meta_for_year(cik, yr)
            if not idx_url:
                print(f"❌ No 10-K for {tkr} {yr}"); continue
            html_url = get_10k_html_url(idx_url)
            if not html_url:
                print(f"❌ No HTML for {tkr} {yr}"); continue

            risk = extract_risk_from_main_html(html_url)
            mdna = extract_mdna_from_main_html(html_url)

            print(f"\n=== {tkr} {yr} (filed {filing_date}) ===")
            print("Risk length:", len(risk))
            print("MD&A length:", len(mdna))
            # sanity: print first 400 chars
            print("Risk preview:", risk[:400])
            print("MD&A preview:", mdna[:400])


# mdna_text = extract_mdna_from_main_html("https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm")
# print(len(mdna_text))
# print(mdna_text)