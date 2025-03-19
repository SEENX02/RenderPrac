from flask import Flask, request, jsonify
import requests
import pandas as pd

app = Flask(__name__)

API_KEY = "your_twelve_data_api_key"
BASE_URL = "https://api.twelvedata.com/time_series"

def fetch_stock_data(ticker, start_date, end_date, interval="1day"):
    """Fetch historical stock data from Twelve Data API."""
    params = {
        "symbol": ticker,
        "interval": interval,
        "start_date": start_date,
        "end_date": end_date,
        "apikey": API_KEY,
        "outputsize": 5000,
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if "values" in data:
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df = df.astype(float)
        return df.to_dict()
    else:
        return {"error": "Failed to fetch data", "details": data}

@app.route("/stock", methods=["GET"])
def get_stock_data():
    ticker = request.args.get("ticker")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    interval = request.args.get("interval", "1day")
    
    if not ticker or not start_date or not end_date:
        return jsonify({"error": "Missing required parameters"}), 400
    
    data = fetch_stock_data(ticker, start_date, end_date, interval)
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
