import os
import pandas as pd
from sqlalchemy import create_engine
import dash
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from dotenv import load_dotenv
import dash_bootstrap_components as dbc
from openai import OpenAI

# Load DB credentials
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_engine():
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)

def get_available_tickers(engine):
    query = "SELECT DISTINCT ticker FROM stock_prices"
    return pd.read_sql(query, engine)['ticker'].tolist()

def get_stock_data(engine, ticker, start_date, end_date):
    """Load time-series stock data."""
    query = """
        SELECT "Date", "Open", "Close", "Volume"
        FROM stock_prices
        WHERE ticker = %s 
        AND "Date" >= %s::date 
        AND "Date" <= %s::date
        ORDER BY "Date";
    """
    return pd.read_sql(query, engine, params=(ticker, start_date, end_date))

def load_mdna(ticker, engine):
    query = """
        SELECT filing_date, chunk_index, content
        FROM mdna_sections
        WHERE ticker = %s
        ORDER BY filing_date DESC, chunk_index ASC
    """
    df = pd.read_sql(query, engine, params=(ticker,))
    
    if df.empty:
        return pd.DataFrame()

    df['full_content'] = df.groupby('filing_date')['content'].transform(lambda x: ' '.join(x))
    df = df.drop_duplicates(subset=['filing_date'])
    return df[['filing_date', 'full_content']]

def detect_sentiment(text):
    prompt = f"""Analyze the sentiment of the following MD&A section and respond with one word only: Positive, Neutral, or Negative.\n\n{text[:3000]}"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
    return "🔒 Sentiment (OpenAI API disabled in test mode cause I'm poor)"

def summarize_mdna(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst. Summarize the following MD&A in 3-4 bullet points."},
                {"role": "user", "content": text[:3000]}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error generating summary: {str(e)}"
    return "🔒 Summary (OpenAI API disabled in test mode cause I'm poor)"

def load_risk_sections(ticker, engine):
    query = """
        SELECT filing_date, chunk_index, content
        FROM risk_sections
        WHERE ticker = %s
        ORDER BY filing_date DESC, chunk_index ASC
    """
    df = pd.read_sql(query, engine, params=(ticker,))
    
    if df.empty:
        return pd.DataFrame()

    df['full_content'] = df.groupby('filing_date')['content'].transform(lambda x: ' '.join(x))
    df = df.drop_duplicates(subset=['filing_date'])
    return df[['filing_date', 'full_content']]

def summarize_risk(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst. Summarize the following risk factors in 3-4 bullet points."},
                {"role": "user", "content": text[:3000]}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error generating risk summary: {str(e)}"
    return "🔒 Risk summary (OpenAI API disabled in test mode cause I'm poor)"

# Initialize
engine = get_engine()
tickers = get_available_tickers(engine)
MODERN_BG = "#F3F4F6"
MODERN_BORDER = "#E5E7EB"

assets_path = os.path.join(os.path.dirname(__file__), "assets")
app = Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR], assets_folder=assets_path)
app.title = "📈 Stock Price Dashboard"

app.layout = html.Div(className="page-shell", children=[
    html.Div(className="hero-card", children=[
        html.Div(className="hero-meta", children=[
            html.Span("Market pulse", className="eyebrow"),
            html.H1("📈 Stock Price Viewer"),
            html.P("Smooth filtering, crisp charts, and quick MD&A insights in one modern board.")
        ]),
        html.Div(className="hero-highlight", children=[
            html.Div(className="pill good", children="Live"),
            html.Div(className="pill neutral", children="AI summary ready"),
        ])
    ]),

    html.Div(className="controls-card card", children=[
        html.Div(className="field", children=[
            html.Label("Select Ticker"),
            dcc.Dropdown(tickers, tickers[0], id='ticker-dropdown', className="control")
        ]),
        html.Div(className="field", children=[
            html.Label("Select Date Range"),
            dcc.DatePickerRange(
                id='date-range',
                start_date='2023-01-01',
                end_date='2024-01-01',
                className="control"
            )
        ])
    ]),

    html.Div(className="grid charts", children=[
        html.Div(className="card chart-card", children=[
            html.Div(className="card-head", children=[
                html.Div([
                    html.Span("Trend"),
                    html.H3("Open vs Close")
                ])
            ]),
            dcc.Graph(id='price-graph')
        ]),
        html.Div(className="card chart-card", children=[
            html.Div(className="card-head", children=[
                html.Div([
                    html.Span("Liquidity"),
                    html.H3("Volume Traded")
                ])
            ]),
            dcc.Graph(id='volume-graph')
        ])
    ]),

    html.Div(className="grid info", children=[
        html.Div(className="stack", children=[
            html.Div(className="card info-card", children=[
                html.Div(className="card-head", children=[
                    html.Span("Filing stream"),
                    html.H3("MD&A - Management Discussion")
                ]),
                html.Div(id='mdna-text-box')
            ]),
            html.Div(className="card info-card", children=[
                html.Div(className="card-head", children=[
                    html.Span("Risk factors"),
                    html.H3("Risk Sections")
                ]),
                html.Div(id='risk-text-box')
            ])
        ]),
        html.Div(className="stack", children=[
            html.Div(id='sentiment-tag', className="sentiment-tag"),
            html.Div(className="card summary-card", children=[
                html.Div(className="card-head", children=[
                    html.Span("AI digest"),
                    html.H3("Summary of MD&A")
                ]),
                html.Div(id='summary-box')
            ]),
            html.Div(className="card summary-card", children=[
                html.Div(className="card-head", children=[
                    html.Span("AI digest"),
                    html.H3("Summary of Risk Sections")
                ]),
                html.Div(id='risk-summary-box')
            ])
        ])
    ])
])

@app.callback(
    Output('price-graph', 'figure'),
    Output('volume-graph', 'figure'),
    Input('ticker-dropdown', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_graphs(ticker, start_date, end_date):
    df = get_stock_data(engine, ticker, start_date, end_date)
    df['Date'] = pd.to_datetime(df['Date'])
    print(df.head())
    print(df.shape)

    fig_price = px.line(
    df, x='Date', y=['Open', 'Close'], 
    title=f"{ticker} Open/Close Prices",
    color_discrete_map={
        'Open': '#87ec1f',
        'Close': '#fe6233'
    }
)
    fig_price.update_layout(title_font=dict(size=20, color='#fe6233'))

    fig_volume = px.bar(
    df, x='Date', y='Volume', 
    title=f"{ticker} Volume",
    color_discrete_sequence=['#ffe11b']
)
    fig_volume.update_layout(title_font=dict(size=20, color='#87ec1f'))


    return fig_price, fig_volume


@app.callback(
    Output('mdna-text-box', 'children'),
    Output('summary-box', 'children'),
    Output('sentiment-tag', 'children'),
    Output('risk-text-box', 'children'),
    Output('risk-summary-box', 'children'),
    Input('ticker-dropdown', 'value')
)
def update_mdna(ticker):
    mdna_df = load_mdna(ticker, engine)
    risk_df = load_risk_sections(ticker, engine)

    mdna_text = "⚠️ No MD&A data found."
    mdna_summary = ""
    mdna_sentiment = "N/A"

    if mdna_df is not None and not mdna_df.empty:
        mdna_text = mdna_df.iloc[0]['full_content'][:5000]
        mdna_summary = summarize_mdna(mdna_text)
        mdna_sentiment = detect_sentiment(mdna_text)

    risk_text = "⚠️ No risk sections found."
    risk_summary = ""
    risk_sentiment = "N/A"

    if risk_df is not None and not risk_df.empty:
        risk_text = risk_df.iloc[0]['full_content'][:5000]
        risk_summary = summarize_risk(risk_text)
        risk_sentiment = detect_sentiment(risk_text)

    sentiment_badge = f"🧠 Sentiment — MD&A: {mdna_sentiment} | Risk: {risk_sentiment}"
    return mdna_text, mdna_summary, sentiment_badge, risk_text, risk_summary


if __name__ == "__main__":
    app.run(debug=True)
