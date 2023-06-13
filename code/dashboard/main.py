from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException, Query
from fastapi.responses import HTMLResponse
import pickle
import yfinance as yf

from tensorflow.keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_absolute_error
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import numpy as np
import pandas as pd

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cl = "Close"

# code of all 50 companies
symbols = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "HDFC.NS",
    "TCS.NS",
    "ITC.NS",
    "LT.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "HINDUNILVR.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "BAJFINANCE.NS",
    "M&M.NS",
    "ASIANPAINT.NS",
    "HCLTECH.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "TATASTEEL.NS",
    "NTPC.NS",
    "ULTRACEMCO.NS",
    "ADANIENT.NS",
    "TATAMOTORS.NS",
    "POWERGRID.NS",
    "BAJAJFINSV.NS",
    "INDUSINDBK.NS",
    "HINDALCO.NS",
    "JSWSTEEL.NS",
    "NESTLEIND.NS",
    "TECHM.NS",
    "GRASIM.NS",
    "WIPRO.NS",
    "HDFCLIFE.NS",
    "ONGC.NS",
    "CIPLA.NS",
    "SBILIFE.NS",
    "DRREDDY.NS",
    "BRITANNIA.NS",
    "COALINDIA.NS",
    "ADANIPORTS.NS",
    "EICHERMOT.NS",
    "APOLLOHOSP.NS",
    "TATACONSUM.NS",
    "BAJAJ-AUTO.NS",
    "DIVISLAB.NS",
    "UPL.NS",
    "HEROMOTOCO.NS",
    "BPCL.NS",
]

scalers = {}
for k in symbols:
    scalers[k] = MinMaxScaler(feature_range=(0, 1))


@app.get("")
async def home():
    return {"Details": "Stock Price Prediction by IT Group 10"}


@app.post("/dataset/{code}", status_code=201)
async def download_latest_dataset(code: str):
    if code not in symbols:
        raise HTTPException(status_code=400, detail="Invalid company code")
    data = yf.Ticker(code).history(period="7y")
    datadict = open(f"dataset/{code}.pkl", "wb")
    pickle.dump(data, datadict)
    datadict.close()
    return {"Dataset downloaded successfully"}


@app.get("/dataset/{code}")
async def see_current_dataset(code: str):
    if code not in symbols:
        raise HTTPException(status_code=400, detail="Invalid company code")

    fl = open(f"dataset/{code}.pkl", "rb")
    data = pickle.load(fl)
    data = data[cl]
    return data.to_dict()


@app.get("/company/{code}")
def get_stock_chart(code: str):

    fl = open(f"dataset/{code}.pkl", "rb")
    stock_data = pickle.load(fl)

    # Create a Plotly figure with a line chart
    fig = go.Figure(data=go.Scatter(
        x=stock_data.index, y=stock_data["Close"]))

    # Customize the chart layout
    fig.update_layout(
        title="Stock Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
    )

    # Convert the Plotly figure to HTML
    chart_html = fig.to_html(full_html=False)

    # Create the complete HTML page
    html_content = f"""
    <html>
        <head>
            <title>Stock Price Prediction</title>
            <!-- Include Plotly.js library -->
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>{code}</h1>
            <!-- Render the chart div -->
            {chart_html}
                <input type="text" id="searchBox" placeholder="Enter number of days">
                <button onclick="searchAPI()">Search</button>
                <div id="results"></div>
                <script>
    function searchAPI() {{
      var searchTerm = document.getElementById('searchBox').value;
      var currentUrl = window.location.href;
      fetch(currentUrl+'/prediction?q=' + encodeURIComponent(searchTerm))
        .then(response => response.text())
        .then(html => {{
          document.getElementById('results').innerHTML = html;
        }})
        .catch(error => {{
          console.error('Error:', error);
        }});

}}
        function displayResults(data) {{
      var resultsContainer = document.getElementById('results');
      resultsContainer.innerHTML = data;
        }}
        </script>
        </body>
    </html>
    """

    return HTMLResponse(content=html_content, status_code=200)


@app.get("/company/{code}/prediction")
async def get_predictions_from_company(code: str, days: int = Query(default=5)):
    fl = open(f"../models/{code}.pkl", "rb")
    model = pickle.load(fl)
    fl = open(f"dataset/{code}.pkl", "rb")
    data = pickle.load(fl)
    vec = data["Close"].values[-50:]
    vec = vec.reshape((-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    vec = scaler.fit_transform(vec)
    vec = [i[0] for i in vec]
    input_data = np.array(vec)  # Convert current_days_data to a numpy array
    # Reshape to match the input shape of the LSTM model
    input_data = np.reshape(input_data, (1, input_data.shape[0], 1))
    # Make predictions for the future 7 days
    predicted_prices = []
    for _ in range(days):
        # Use the LSTM model to predict the next day's price
        predicted_price = model.predict(input_data)
        # Append the predicted price to the list
        predicted_prices.append(predicted_price[0, 0])
        vec.pop(0)
        vec.append(predicted_prices[-1])
        input_data = np.array(vec)
        input_data = np.reshape(input_data, (1, input_data.shape[0], 1))
    arr = np.array(predicted_prices)
    arr = arr.reshape((-1, 1))
    arr = scaler.inverse_transform(arr)
    arr = arr.reshape(-1)
    di = {i+1: float(arr[i]) for i in range(len(arr))}
    fig = go.Figure(data=go.Scatter(
        x=list(di.keys()), y=list(di.values())))
    # Customize the chart layout
    fig.update_layout(
        title="Stock Price Chart",
        xaxis_title="Days Ahead",
        yaxis_title="Price",
    )
    # Convert the Plotly figure to HTML
    chart_html = fig.to_html(full_html=False)

    return chart_html
