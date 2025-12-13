import pandas as pd
import numpy as np

NUM_TICKERS = 1000
YEARS = 10  # ~2500 rows per ticker

print(f"Generating {NUM_TICKERS} test tickers and branches...")

for i in range(NUM_TICKERS):
    ticker = f"TEST{i:04d}"

    # Generate ticker OHLCV data
    dates = pd.date_range('2014-01-01', periods=YEARS*252, freq='B')
    price = 100 * np.cumprod(1 + np.random.randn(len(dates)) * 0.02)

    ticker_df = pd.DataFrame({
        'Date': dates,
        'Open': price * (1 + np.random.randn(len(dates)) * 0.005),
        'High': price * (1 + abs(np.random.randn(len(dates))) * 0.01),
        'Low': price * (1 - abs(np.random.randn(len(dates))) * 0.01),
        'Close': price,
        'Adj Close': price,
        'Volume': np.random.randint(100000, 10000000, len(dates))
    })
    ticker_df.to_parquet(f'tickers/{ticker}.parquet', index=False)

    # Generate branch trade log
    branch_df = pd.DataFrame({
        'Date': dates,
        'RSI': np.random.randint(20, 80, len(dates)),
        'Active': np.random.choice([0, 1], len(dates), p=[0.9, 0.1]),
        'Return_Pct': np.random.randn(len(dates)) * 2,
        'Signal_Type': 'RSI',
        'Ticker': ticker
    })
    branch_df.to_parquet(f'trade_logs/14D_RSI_{ticker}_LT30_daily_trade_log.parquet', index=False)

    if (i + 1) % 100 == 0:
        print(f"  Generated {i + 1}/{NUM_TICKERS} tickers...")

print("Done!")
