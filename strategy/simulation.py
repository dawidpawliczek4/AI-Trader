import joblib
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import os

def full_data(symbol = "AMD", exchange = "NASDAQ"):
    interval = Interval.in_1_minute
    n_bars = 5000
    tv = TvDatafeed()

    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
    df = df.drop(columns=['symbol'])

    def compute_rsi(series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_macd(series, short=12, long=26, signal=9):
        ema_short = series.ewm(span=short, adjust=False).mean()
        ema_long = series.ewm(span=long, adjust=False).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line

    def compute_bollinger_bands(series, window=20, num_std=2):
        sma = series.rolling(window=window).mean()
        std_dev = series.rolling(window=window).std()
        upper_band = sma + (num_std * std_dev)
        lower_band = sma - (num_std * std_dev)
        return sma, upper_band, lower_band

    def compute_features(df):
        # Compute RSI
        df['RSI'] = compute_rsi(df['close'])
        df['RSI_Change'] = df['RSI'].diff()

        # Compute Bollinger Bands
        df['SMA_20'], df['Bollinger_Upper'], df['Bollinger_Lower'] = compute_bollinger_bands(df['close'])
        df['SMA_10'] = df['close'].rolling(10).mean()
        df['SMA_5'] = df['close'].rolling(5).mean()

        # Feature Engineering for SMA
        df['SMA_5_Distance'] = (df['SMA_5'] - df['close']) / df['close']
        df['SMA_10_Distance'] = (df['SMA_10'] - df['close']) / df['close']
        df['SMA_Crossover'] = (df['SMA_5'] - df['SMA_10']) / df['close']

        # Calculate Volume features
        df['Volume_MA'] = df['volume'].rolling(window=50).mean()
        df['Volume_Spike'] = df['volume'] / df['Volume_MA']
        df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()

        # Normalized MACD
        df['Normalized_MACD'] = df['MACD'] / (df['MACD'].ewm(span=9, adjust=False).mean())

        # Normalized Bollinger Bands
        df['Normalized_BB'] = (df['close'] - df['Bollinger_Lower']) / (df['Bollinger_Upper'] - df['Bollinger_Lower'])

        # Rolling volatility (standard deviation of returns)
        df['Volatility_10_Days'] = df['close'].pct_change().rolling(window=10).std() * 100
        df['Volatility_20_Days'] = df['close'].pct_change().rolling(window=20).std() * 100

        # Target Value
        df['Mean_10_Days_Ahead'] = df['close'].shift(-10).rolling(window=10, min_periods=1).mean()
        df['Target'] = (df['Mean_10_Days_Ahead'] - df['close']) / df['close'] * 100

        # Drop raw unbounded features
        df = df.drop(columns=['SMA_10', 'SMA_5'])

        return df


    df = compute_features(df)
    df = df.dropna()

    return df


def single_shot(data):
    """
        gets n bars from a given time and predicts the movements
    """

    wanted_features = ['SMA_Crossover', 'RSI' ,'Volatility_10_Days', 'Volatility_20_Days'] 
    data = data[wanted_features]

    # Select the last n time_steps (for LSTM input)
    time_steps = 5
    if len(data) < time_steps:
        raise ValueError("Not enough data to create the required time steps for LSTM input.")

    # Extract last `time_steps` rows and reshape for LSTM input
    lstm_data = data.iloc[-time_steps:].to_numpy().reshape(1, time_steps, -1)

    # Extract the last row for standard models
    standard_data = data.iloc[-1].to_numpy().reshape(1, -1)
    print(lstm_data)
    return standard_data, lstm_data


def analysis_predict(data):
    """
    Load models and calculate the average prediction based on the data.
    """
    standard_data, lstm_data = single_shot(data)

    models_dir = "../analisys/models"
    model_files = [os.path.join(models_dir, file) for file in os.listdir(models_dir) if file.endswith(".pkl")]

    predictions = []
    for file_path in model_files:
        model = joblib.load(file_path)
        if "lstm" in file_path.lower():
            lstm_prediction = model.predict(lstm_data)
            predictions.append(lstm_prediction.flatten()[0])
        else:
            predictions.append(model.predict(standard_data)[0])  

    # normalize to [0, 1]
    # print(predictions)
    pred = np.mean(predictions)
    pred = (pred + 3) / 6
    print(pred)
    return pred

def strategy_manual(data, state):
    '''
    manually open/close all trades
    '''
    balance, stock_worth = state
    pred = analysis_predict(data)
    if pred > 0.58:
        amount = 0.10 * balance
        return {'act': 'buy', 'amount': amount}
    elif pred < 0.515:
        amount = 0.5 * stock_worth
        return {'act': 'sell', 'amount': amount}
    else:
        return {'act': 'hold'}

def simulate_manual(full_data, strategy):
    '''
    gets a full data set and a strategy, buys and sells [amount] based on a strategy
    '''
    starting_bal = 100000
    balance = starting_bal
    shares = 0.0
    n_bars = len(full_data)
    curr_bar = 5
    buy_actions = 0
    sell_actions = 0

    while curr_bar < n_bars:
        shot_data = full_data.iloc[curr_bar-5 : curr_bar]
        shot_data = shot_data.drop(columns=['open', 'close', 'high', 'low'])
        current_price = full_data.iloc[curr_bar]['close']
        current_stock_worth = shares * current_price

        action = strategy(shot_data, (balance, current_stock_worth))

        if action['act'] == 'buy':
            amount = action.get('amount', 0)
            amount = min(amount, balance)
            if amount > 0 and current_price > 0:
                shares_bought = amount / current_price
                balance -= amount
                shares += shares_bought
                buy_actions += 1
        elif action['act'] == 'sell':
            amount = action.get('amount', 0)
            if amount > 0 and current_price > 0:
                shares_to_sell = amount / current_price
                shares_to_sell = min(shares_to_sell, shares)
                balance += shares_to_sell * current_price
                shares -= shares_to_sell
                sell_actions += 1

        curr_bar += 1

    # Sell all remaining shares at the last close price
    if shares > 0:
        final_price = full_data.iloc[-1]['close']
        balance += shares * final_price
        sell_actions += 1
        shares = 0

    print(f"Buys: {buy_actions}, sells: {sell_actions}")

    total_return = balance / starting_bal
    return total_return, buy_actions

def strategy_tpsl(data, state):
    """
    Strategy that opens trades with set stop loss and take profit.
    """
    balance, stock_worth = state
    pred = analysis_predict(data)
    # Buy condition with SL=1%, TP=2%
    if pred > 0.58:
        amount = 0.10 * balance
        return {
            'act': 'buy', 
            'amount': amount,
            'stop_loss': 0.01,
            'take_profit': 0.02
        }
    else:
        return {'act': 'hold'}

def simulate_tpsl(full_data, strategy):
    """
    Simulates trading with stop loss and take profit management.
    """
    starting_bal = 100000
    balance = starting_bal
    open_trades = []
    n_bars = len(full_data)
    curr_bar = 5
    buy_actions = 0
    total_trades_closed = 0
    tp_hits = 0
    sl_hits = 0

    while curr_bar < n_bars:
        current_bar_data = full_data.iloc[curr_bar]
        current_price = current_bar_data['close']
        current_high = current_bar_data['high']
        current_low = current_bar_data['low']

        for trade in open_trades.copy():
            tp_price = trade['take_profit']
            sl_price = trade['stop_loss']
            
            if current_high >= tp_price:
                tp_hits += 1
                balance += trade['shares'] * tp_price
                open_trades.remove(trade)
                total_trades_closed += 1
            elif current_low <= sl_price:
                sl_hits += 1
                balance += trade['shares'] * sl_price
                open_trades.remove(trade)
                total_trades_closed += 1

        shot_data = full_data.iloc[curr_bar-5:curr_bar].drop(
            columns=['open', 'close', 'high', 'low']
        )
        current_stock_worth = sum(t['shares'] * current_price for t in open_trades)
        action = strategy(shot_data, (balance, current_stock_worth))

        if action['act'] == 'buy':
            amount = min(action['amount'], balance)
            if amount > 0 and current_price > 0:
                shares = amount / current_price
                sl = current_price * (1 - action['stop_loss'])
                tp = current_price * (1 + action['take_profit'])
                
                open_trades.append({
                    'entry_price': current_price,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'shares': shares
                })
                balance -= amount
                buy_actions += 1

        curr_bar += 1

    # Close remaining positions at final price
    final_price = full_data.iloc[-1]['close']
    for trade in open_trades:
        balance += trade['shares'] * final_price
    total_trades_closed += len(open_trades)

    print(f"Total Trades Opened: {buy_actions}")
    print(f"Total Trades Closed: {total_trades_closed}")
    
    return balance / starting_bal, total_trades_closed, tp_hits, sl_hits

def strategy_trailingsl(data, state):
    """
    Strategy that opens trades with two TP targets and breakeven SL adjustment
    """
    balance, stock_worth = state
    pred = analysis_predict(data)
    
    if pred > 0.58:
        amount = 0.10 * balance
        return {
            'act': 'buy', 
            'amount': amount,
            'stop_loss': 0.01,
            'take_profit1': 0.01,
            'take_profit2': 0.02
        }
    else:
        return {'act': 'hold'}

def simulate_trailingsl(full_data, strategy):
    """
    Simulates trading with two TP targets and breakeven adjustment
    """
    starting_bal = 100000
    balance = starting_bal
    open_trades = []
    n_bars = len(full_data)
    curr_bar = 5
    buy_actions = 0
    total_trades_closed = 0
    tp_hits = 0
    sl_hits = 0

    while curr_bar < n_bars:
        current_bar_data = full_data.iloc[curr_bar]
        current_price = current_bar_data['close']
        current_high = current_bar_data['high']
        current_low = current_bar_data['low']

        for trade in open_trades.copy():
            if trade['stage'] == 1:
                tp = trade['take_profit1']
                sl = trade['stop_loss']
            else:
                tp = trade['take_profit2']
                sl = trade['breakeven_sl']

            if current_high >= tp:
                tp_hits += 1

                if trade['stage'] == 1:
                    # Close 50% position at TP1 and adjust SL to breakeven
                    close_shares = trade['shares'] * 0.5
                    balance += close_shares * tp
                    trade['shares'] -= close_shares
                    trade['stage'] = 2
                    trade['breakeven_sl'] = trade['entry_price']
                    total_trades_closed += 0.5
                else:
                    balance += trade['shares'] * tp
                    open_trades.remove(trade)
                    total_trades_closed += 1

            elif current_low <= sl:
                sl_hits += 1

                balance += trade['shares'] * sl
                open_trades.remove(trade)
                total_trades_closed += 1 if trade['stage'] == 1 else 0.5

        shot_data = full_data.iloc[curr_bar-5:curr_bar].drop(
            columns=['open', 'close', 'high', 'low']
        )
        current_stock_worth = sum(t['shares'] * current_price for t in open_trades)
        action = strategy(shot_data, (balance, current_stock_worth))

        if action['act'] == 'buy':
            amount = min(action['amount'], balance)
            if amount > 0 and current_price > 0:
                shares = amount / current_price
                initial_sl = current_price * (1 - action['stop_loss'])
                tp1 = current_price * (1 + action['take_profit1'])
                tp2 = current_price * (1 + action['take_profit2'])
                
                open_trades.append({
                    'entry_price': current_price,
                    'stop_loss': initial_sl,
                    'breakeven_sl': None,
                    'take_profit1': tp1,
                    'take_profit2': tp2,
                    'shares': shares,
                    'stage': 1
                })
                balance -= amount
                buy_actions += 1

        curr_bar += 1

    final_price = full_data.iloc[-1]['close']
    for trade in open_trades:
        balance += trade['shares'] * final_price
    total_trades_closed += len(open_trades)

    print(f"Total Trades Opened: {buy_actions}")
    print(f"Total Trades Closed: {round(total_trades_closed, 1)}")
    
    return balance / starting_bal, total_trades_closed, tp_hits, sl_hits

# Example usage
if __name__ == "__main__":
    exchange = "NASDAQ"
    simul_len = 2000

    df_amd = full_data('AMD', exchange)[0:simul_len]
    df_aapl = full_data('AAPL', exchange)[0:simul_len]

    man = strategy_manual
    tpsl = strategy_tpsl
    trailing = strategy_trailingsl

    res_man_amd = simulate_manual(df_amd, man)
    res_tpsl_amd = simulate_tpsl(df_amd, tpsl)
    res_trailing_amd = simulate_trailingsl(df_amd, trailing)

    res_man_aapl = simulate_manual(df_aapl, man)
    res_tpsl_aapl = simulate_tpsl(df_aapl, tpsl)
    res_trailing_aapl = simulate_trailingsl(df_aapl, trailing)

    print("\nTimestamps - AMD:")
    print(df_amd.index[0])
    print(df_amd.index[-1])

    print("\nTimestamps - APPLE:")
    print(df_aapl.index[0])
    print(df_aapl.index[-1])

    print("\nfor manual: (return, opened trades)")
    print("for other: (return, opened trades, tp hits, sl hits)\n")

    print("amd manual")
    print(res_man_amd)
    print("amd tpsl")
    print(res_tpsl_amd)
    print("amd trailing")
    print(res_trailing_amd)

    print("\napple manual")
    print(res_man_aapl)
    print("apple tpsl")
    print(res_tpsl_aapl)
    print("apple trailing")
    print(res_trailing_aapl)
