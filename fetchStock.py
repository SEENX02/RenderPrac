import requests
import pandas as pd

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
        return df
    else:
        print("Error fetching data:", data)
        return None

# Example usage
data = fetch_stock_data("AAPL", "2024-01-01", "2024-03-01")
if data is not None:
    print(data.head())
