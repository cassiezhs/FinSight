# FinSight
# 📈 Financial Data Pipeline and Insight Dashboard

This dashboard combinexs **stock market performance** with **company financial narrative** (extracted from 10-K filings) to help analysts, investors, and researchers understand how management commentary aligns with actual stock trends.

---

## 🚀 Features

- 📉 Interactive stock price visualization (historical data from Yahoo Finance)
- 🧾 Extracted MD&A section from the latest 10-K filings (via SEC EDGAR)
- 🔍 Compare narrative and market movement side-by-side
- 🗃️ Data stored in PostgreSQL for scalability
- 🌐 Web app built with Python Dash

---

## 📦 Tech Stack

- **Backend**: Python, PostgreSQL, SQLAlchemy
- **Frontend**: Dash, Plotly
- **Data Sources**:
  - Yahoo Finance API (via `yfinance`)
  - SEC EDGAR API for 10-K filings
- **Other Tools**: `psycopg2`, `BeautifulSoup`, `dotenv`, `pandas`

---

## 📂 Project Structure

