import os
import matplotlib.pyplot as plt
from tvDatafeed import TvDatafeed, Interval
import mplfinance as mpf

def get_creds(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            lines = f.readlines()
            email = lines[0].strip() if len(lines) > 0 else ''
            password = lines[1].strip() if len(lines) > 1 else ''
            return email, password
    return '', ''

def save_creds(path, email, password):
    with open(path, 'w') as file:
        file.write(f"{email}\n{password}\n")

def prompt_creds():
    print("Please enter your credentials for TradingView")
    email = input("Email (leave empty to continue without logging in): ").strip()
    password = ''
    if email:
        password = input("Password: ").strip()
    return email, password

def init_tv(email, password):
    if email and password:
        tv = TvDatafeed(username=email, password=password)
    else:
        tv = TvDatafeed()
    return tv

def fetch_and_plot_data(tv):
    symbol = input("Enter symbol (e.g. AAPL): ").strip().upper()
    exchange = input("Enter exchange (e.g. NASDAQ): ").strip().upper()
    interval_input = input("Enter time frame (1_minute, 5_minute, 1_hour, daily, etc.): ").strip().lower()
    
    interval_map = {
        '1_minute': Interval.in_1_minute,
        '3_minute': Interval.in_3_minute,
        '5_minute': Interval.in_5_minute,
        '15_minute': Interval.in_15_minute,
        '30_minute': Interval.in_30_minute,
        '45_minute': Interval.in_45_minute,
        '1_hour': Interval.in_1_hour,
        '2_hour': Interval.in_2_hour,
        '3_hour': Interval.in_3_hour,
        '4_hour': Interval.in_4_hour,
        'daily': Interval.in_daily,
        'weekly': Interval.in_weekly,
        'monthly': Interval.in_monthly
    }
    
    interval = interval_map.get(interval_input, Interval.in_1_hour)
    print(f"Chosen interval: {interval.name}")
    
    n_bars = 100

    try:
        data = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
        if data.empty:
            print("No data to show.")
            return
        print(f"Got {len(data)} candles for {symbol} from {exchange} on interval {interval.name}.")
        
        mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc)
        mpf.plot(data, type='candle', style=s, title=f'{symbol} Candlestick Chart', volume=False)
    except Exception as e:
        print(f"Error while downloading data: {e}")

def run_demo():
    credentials_path = os.path.join(os.path.dirname(__file__), '../../data/credentials.txt')
    email, password = get_creds(credentials_path)

    if not (email and password):
        email, password = prompt_creds()
        if email and password:
            save_creds(credentials_path, email, password)
            tv = init_tv(email, password)

    tv = init_tv(email, password)
    fetch_and_plot_data(tv)

if __name__ == "__main__":
    run_demo()

