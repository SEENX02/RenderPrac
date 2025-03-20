import requests
import pandas as pd
import plotly.graph_objs as go
import webbrowser
import LSTMpredict

API_KEY = "your_twelve_data_api_key"
BASE_URL = "https://api.twelvedata.com/time_series"

def getCompanyDetail(companyName, startDate, endDate):
    params = {
        "symbol": companyName,
        "interval": "1day",
        "apikey": API_KEY,
        "start_date": startDate,
        "end_date": endDate,
        "outputsize": 5000
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if "values" not in data:
        print(f"Invalid company: {companyName}")
        return None
    
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.astype(float)
    return df

def getCandle(companyName, startDate, endDate, theme):
    df = getCompanyDetail(companyName, startDate, endDate)
    if df is None:
        return None
    
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=f'{companyName} Candlestick Chart'
    )])

    fig.update_layout(
        title=f'{companyName} Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template=theme
    )

    file_path = "C:/College Notes/Sem6/Project/stockMarketApp/templates/chart.html"
    fig.write_html(file_path)
    # Automatically open in Microsoft Edge
    edge_path = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
    webbrowser.register('edge', None, webbrowser.BackgroundBrowser(edge_path))
    webbrowser.get('edge').open(file_path)

def predict(companyName, startDate, endDate, theme):
    df = getCompanyDetail(companyName, startDate, endDate)
    if df is None:
        return None
    return LSTMpredict.predict(companyName, startDate, endDate, theme)
