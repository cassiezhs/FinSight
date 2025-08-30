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
    query = f"""
        SELECT date, open, close, volume
        FROM stock_prices
        WHERE ticker = %s AND date BETWEEN %s AND %s
        ORDER BY date
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

# Initialize
engine = get_engine()
tickers = get_available_tickers(engine)

app = dash.Dash(external_stylesheets=[dbc.themes.ZEPHYR])
app.title = "📈 Stock Price Dashboard"

color_mode_switch =  html.Span(
    [
        dbc.Label(className="fa fa-moon", html_for="switch"),
        dbc.Switch( id="switch", value=True, className="d-inline-block ms-1", persistence=True),
        dbc.Label(className="fa fa-sun", html_for="switch"),
    ]
)

app = Dash(__name__, external_stylesheets=[], assets_folder="app")

app.layout = html.Div([
    html.Div(className='container', children=[
        html.H1("📈 Stock Price Viewer"),

        html.Label("Select Ticker"),
        dcc.Dropdown(tickers, tickers[0], id='ticker-dropdown'),

        html.Label("Select Date Range"),
        dcc.DatePickerRange(
            id='date-range',
            start_date='2023-01-01',
            end_date='2024-01-01'
        ),

        html.Div(className="card", children=[
            dcc.Graph(id='price-graph')
        ]),

        html.Div(className="card", children=[
            html.H3("MD&A - Management Discussion"),
            html.Div(id='mdna-text-box')
        ]),

        html.Div(id='sentiment-tag', style={
        'marginTop': '1rem',
        'padding': '1rem',
        'borderRadius': '8px',
        'backgroundColor': '#f2f2f2',
        'fontWeight': 'bold'
        }),


        html.H3("📌 Summary of MD&A", style={"marginTop": "2rem"}),
        html.Div(id='summary-box', style={
            'whiteSpace': 'pre-wrap',
            'border': '1px dashed #87ec1f',
            'padding': '12px',
            'backgroundColor': '#f7fff4',
            'color': '#1a3311',
            'borderRadius': '8px',
            'marginBottom': '40px'
        }),

        html.Div(className="card", children=[
            dcc.Graph(id='volume-graph')
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

    fig_price = px.line(
    df, x='date', y=['open', 'close'], 
    title=f"{ticker} Open/Close Prices",
    color_discrete_map={
        'open': '#87ec1f',
        'close': '#fe6233'
    }
)
    fig_price.update_layout(title_font=dict(size=20, color='#fe6233'))

    fig_volume = px.bar(
    df, x='date', y='volume', 
    title=f"{ticker} Volume",
    color_discrete_sequence=['#ffe11b']
)
    fig_volume.update_layout(title_font=dict(size=20, color='#87ec1f'))


    return fig_price, fig_volume


@app.callback(
    Output('mdna-text-box', 'children'),
    Output('summary-box', 'children'),
    Output('sentiment-tag', 'children'),
    Input('ticker-dropdown', 'value')
)
def update_mdna(ticker):
    df = load_mdna(ticker, engine)
    if df is None or df.empty:
        return "⚠️ No MD&A data found.", "", ""

    latest_mdna = df.iloc[0]['full_content']
    summary = summarize_mdna(latest_mdna)
    sentiment = detect_sentiment(latest_mdna)

    sentiment_badge = f"🧠 Sentiment: {sentiment}"
    return latest_mdna[:5000], summary, sentiment_badge


if __name__ == "__main__":
    app.run(debug=True)


