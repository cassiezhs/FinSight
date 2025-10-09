import requests
from bs4 import BeautifulSoup
import re
import psycopg2
import yfinance as yf

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


def get_latest_10k_url(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": "zihansha19960@gmail.com"}
    resp = requests.get(url, headers=headers)
    data = resp.json()
    
    for filing in data['filings']['recent']['form']:
        if filing == '10-K':
            index = data['filings']['recent']['form'].index(filing)
            acc_no = data['filings']['recent']['accessionNumber'][index].replace("-", "")
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/index.json"
            return doc_url
    return None


cik = get_cik("NVDA")
print(get_latest_10k_url(cik))

def get_10k_html_url(doc_index_url):
    headers = {"User-Agent": "zihanshao1996@gmail.com"}
    resp = requests.get(doc_index_url, headers=headers)
    
    if resp.status_code != 200:
        print("Failed to fetch index:", doc_index_url)
        return None

    data = resp.json()
    files = data.get('directory', {}).get('item', [])

    print("📂 Available files in the filing:")
    for f in files:
        print("-", f['name'])

    # 优先寻找 aapl-YYYYMMDD.htm 格式
    for f in files:
        if f['name'].startswith("aapl-") and f['name'].endswith(".htm"):
            html_url = doc_index_url.replace('index.json', f['name'])
            print("Found primary HTML:", html_url)
            return html_url

    # fallback: 找第一个 .htm 文件
    for f in files:
        if f['name'].endswith(".htm"):
            html_url = doc_index_url.replace('index.json', f['name'])
            print("Fallback HTML:", html_url)
            return html_url

    print("No HTML file found.")
    return None

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

# ---------------- Test: Pull 2023 and 2025 AAPL Filings ---------------- #

if __name__ == "__main__":
    nvda_cik = get_cik("NVDA")
    for year in [2023,2024,2025]:
        print(f"\n=== Fetching NVDA {year} 10-K ===")
        idx_url = get_10k_url_for_year(nvda_cik, year)
        if idx_url:
            html_url = get_10k_html_url(idx_url)
            print(html_url)
            if html_url:
                mdna_text = extract_mdna_from_main_html(html_url)
                print(mdna_text)
                print(f"✅ Extracted {len(mdna_text)} chars for {year}")
            else:
                print(f"⚠️ No HTML found for {year}")
        else:
            print(f"⚠️ No 10-K found for {year}")
# mdna_text = extract_mdna_from_main_html("https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm")
# print(len(mdna_text))
# print(mdna_text)