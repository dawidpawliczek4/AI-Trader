import joblib
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import os


def cook_dataset(symbol, exchange):
    """
    Fetch data and preprocess it to extract features for standard models and LSTM models.
    """
    interval = Interval.in_1_minute
    n_bars = 100
    tv = TvDatafeed()

    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
    df = df.drop(columns=['symbol'])

    def compute_bollinger_bands(series, window=20, num_std=2):
        sma = series.rolling(window=window).mean()
        std_dev = series.rolling(window=window).std()
        upper_band = sma + (num_std * std_dev)
        lower_band = sma - (num_std * std_dev)
        return sma, upper_band, lower_band

    def compute_features(df):
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
        df['Mean_5_Days_Ahead'] = df['close'].shift(-10).rolling(window=10, min_periods=1).mean()
        df['Target'] = (df['Mean_5_Days_Ahead'] - df['close']) / df['close'] * 100

        return df

    df = compute_features(df)
    df = df.dropna()

    wanted_features = ['Volatility_10_Days', 'Volatility_20_Days', 'Normalized_MACD', 'SMA_Crossover', 'SMA_10_Distance'] 
    data = df[wanted_features]

    # Select the last n time_steps (for LSTM input)
    time_steps = 5
    if len(data) < time_steps:
        raise ValueError("Not enough data to create the required time steps for LSTM input.")

    # Extract last `time_steps` rows and reshape for LSTM input
    lstm_data = data.iloc[-time_steps:].to_numpy().reshape(1, time_steps, -1)

    # Extract the last row for standard models
    standard_data = data.iloc[-1].to_numpy().reshape(1, -1)
    return standard_data, lstm_data


def analysis_predict(symbol, exchange):
    """
    Load models and calculate the average prediction based on the data.
    """
    standard_data, lstm_data = cook_dataset(symbol, exchange)

    models_dir = "analisys/models"
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
    pred = np.mean(predictions)
    pred = (pred + 3) / 6
    return pred


# Example usage
if __name__ == "__main__":
    symbol = "AMD"
    exchange = "NASDAQ"
    try:
        result = analysis_predict(symbol, exchange)
        print("Prediction result:", result)
    except Exception as e:
        print("Error:", e)
