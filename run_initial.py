import pandas as pd
import yfinance as yf
import os

if os.path.exists("sp500.csv"):
    sp500 = pd.read_csv("sp500.csv", index_col=0)
else:
    sp500 = yf.Ticker("^GSPC").history(period="max")
    sp500.to_csv("sp500.csv")

sp500.index = pd.to_datetime(sp500.index)

print("sp500 head:\n", sp500.head().to_string())
print("\nsp500 dtypes:\n", sp500.dtypes)
print("\nrows:", sp500.shape[0])
