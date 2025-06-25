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

print(get_cik("AAPL"))  

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


cik = get_cik("AAPL")
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


def extract_mdna_section(html_url):
    headers = {"User-Agent": "zihanshao1996@gmail.com"}
    html = requests.get(html_url, headers=headers).text
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)

    # 多种形式的 ITEM 7
    mdna_start = re.search(r'(Item\s*7[^A-Za-z]*\.?\s*(Management.*?Analysis.*?Operations)?)', text, re.IGNORECASE)
    mdna_end = re.search(r'(Item\s*7A|Item\s*8)', text, re.IGNORECASE)

    if mdna_start and mdna_end:
        start = mdna_start.start()
        end = mdna_end.start()
        return text[start:end]
    
    # fallback: just return the text chunk around Item 7
    elif mdna_start:
        start = mdna_start.start()
        return text[start:start + 5000]  # 返回5000字符以内

    return "MD&A section not found"

# 示例运行
url = get_latest_10k_url(get_cik("AAPL"))
html_url = get_10k_html_url(url)
mdna = extract_mdna_section(html_url)

def extract_mdna_from_main_html(html_url):
    headers = {"User-Agent": "zihanshao1996@gmail.com"}
    html = requests.get(html_url, headers=headers).text
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator='\n', strip=True)

    # Debug print
    print("\n Searching for 'Item 7'...")

    item7_matches = list(re.finditer(r'Item\s*7[\.\-–]?\s+(Management.*?Operations)?', text, re.IGNORECASE))
    item7a_or_8_match = re.search(r'Item\s*7A[\.\-–]?|Item\s*8[\.\-–]?', text, re.IGNORECASE)

    if not item7_matches:
        print("No 'Item 7' match found.")
        return "MD&A section not found"

    print(f"Found {len(item7_matches)} 'Item 7' matches.")
    for i, m in enumerate(item7_matches):
        print(f"  Match {i+1} at char index: {m.start()}")

    if item7a_or_8_match:
        print(f"Found 'Item 7A/8' at index: {item7a_or_8_match.start()}")
    else:
        print("No 'Item 7A or 8' found; using fallback length.")

    # Use second match if exists, else first
    
    if len(item7_matches) >= 3:
        start_index = item7_matches[2].start()
    elif len(item7_matches) >= 1:
        start_index = item7_matches[-1].start()
    else:
        return "No 'Item 7' section found."

    end_index = item7a_or_8_match.start() if item7a_or_8_match else start_index + 6000

    if end_index <= start_index:
        print("End index before start index. Using fallback slice of 5000 chars.")
        return text[start_index:start_index + 5000].strip()

    extracted = text[start_index:end_index].strip()
    print(f"\n Extracted length: {len(extracted)} characters\n")
    return extracted or "Empty content extracted"

mdna_text = extract_mdna_from_main_html("https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm")
print(mdna_text[:3000]) 
