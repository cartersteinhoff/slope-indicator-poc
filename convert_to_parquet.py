import pandas as pd
from pathlib import Path

# Convert ticker CSVs
print("Converting ticker files...")
for csv in Path('tickers').glob('*.csv'):
    df = pd.read_csv(csv)
    df.to_parquet(csv.with_suffix('.parquet'), index=False)
    print(f"  Converted {csv.name}")

# Convert trade log CSVs
print("\nConverting trade log files...")
for csv in Path('trade_logs').glob('*.csv'):
    df = pd.read_csv(csv)
    df.to_parquet(csv.with_suffix('.parquet'), index=False)
    print(f"  Converted {csv.name}")

print("\nDone!")
