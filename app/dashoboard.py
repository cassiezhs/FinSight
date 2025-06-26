import os
import pandas as pd
from sqlalchemy import create_engine
import dash
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from dotenv import load_dotenv
import dash_bootstrap_components as dbc

# Load DB credentials
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

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

app.layout = html.Div([
    html.H1("Stock Price Viewer"),
    html.Label("Select Ticker"),
    dcc.Dropdown(tickers, tickers[0], id='ticker-dropdown'),

    html.Label("Select Date Range"),
    dcc.DatePickerRange(
        id='date-range',
        start_date='2023-01-01',
        end_date='2024-01-01'
    ),

    dcc.Graph(id='price-graph'),
    dcc.Graph(id='volume-graph')
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

    fig_price = px.line(df, x='date', y=['open', 'close'], title=f"{ticker} Open/Close Prices")
    fig_volume = px.bar(df, x='date', y='volume', title=f"{ticker} Volume")

    return fig_price, fig_volume

if __name__ == "__main__":
    app.run(debug=True)
