import requests
import pandas as pd

def get_data(start, end, symbol):
    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={start}&period2={end}"
        "&interval=1d&events=history&includeAdjustedClose=true"
    )

    headers = { 'User-Agent': 'Chrome/112.0.0.0'}

    response = requests.get(url, verify=False, headers=headers)
    if response.status_code == 200:
        data = response.json()
        #parse data
        timestamps = data['chart']['result'][0]['timestamp']
        d_len = len(timestamps)
        dates = pd.to_datetime(timestamps, unit='s').normalize()
        open = data['chart']['result'][0]['indicators']['quote'][0]['open']
        high = data['chart']['result'][0]['indicators']['quote'][0]['high']
        low = data['chart']['result'][0]['indicators']['quote'][0]['low']
        #close = data['chart']['result'][0]['indicators']['quote'][0]['close']
        close = data['chart']['result'][0]['indicators']['adjclose'][0]['adjclose']
        tick_vol = [-1] * d_len
        shrout = [-1] * d_len
        vol = data['chart']['result'][0]['indicators']['quote'][0]['volume']
        #close = data['chart']['result'][0]['indicators']['quote'][0]['close']
        #convert it to dataframe
        df = pd.DataFrame({
            'Open':      open,
            'High':      high,
            'Low':       low,
            'Close':     close,
            'Volume':    vol,
            'Tickvol':   tick_vol,
            'SHROUT':    shrout
        },
            index=pd.to_datetime(dates)   # convert your date list into a DatetimeIndex
        )
        df.index.name = 'Date'
        return df

    else:
        print(f"Error fetching data: HTTP {response.status_code}")
        return None
