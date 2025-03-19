import yfinance as yf
import requests

# Check if Render blocks requests
def check_render_access():
    response = requests.get("https://query1.finance.yahoo.com/")
    print(f"Status Code: {response.status_code}")
    return response.status_code

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance with a custom User-Agent."""
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    
    stock = yf.Ticker(ticker, session=session)
    df = stock.history(period="1d", start=start_date, end=end_date)
    return df

# Check if Render allows requests
status_code = check_render_access()
if status_code == 200:
    # Example usage
    data = fetch_stock_data("AAPL", "2024-01-01", "2024-03-01")
    print(data.head())
else:
    print("Render is blocking requests. Consider using an alternative API.")
