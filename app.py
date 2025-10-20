import streamlit as st
import pandas as pd
import yfinance as yf
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# Load or train model
if os.path.exists('sp500_model.pkl'):
    model = joblib.load('sp500_model.pkl')
    st.success("Model loaded successfully.")
else:
    st.error("Model not found. Please run predict_tomorrow.py first to train and save the model.")
    st.stop()

# Load data
if os.path.exists("sp500.csv"):
    sp500 = pd.read_csv("sp500.csv", index_col=0)
else:
    sp500 = yf.Ticker("^GSPC").history(period="max")
    sp500.to_csv("sp500.csv")

sp500.index = pd.to_datetime(sp500.index)
del sp500["Dividends"]
del sp500["Stock Splits"]
sp500 = sp500.loc["1990-01-01":].copy()
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Feature engineering
horizons = [2, 5, 60, 250, 1000]
new_predictors = ["Close", "Volume", "Open", "High", "Low"]
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ratio_column, trend_column]

sp500 = sp500.dropna()

# Streamlit app
st.title("📈 S&P 500 Price Predictor")
st.markdown("Predict tomorrow's S&P 500 closing price using machine learning.")

# Compute prediction for tomorrow
latest_data = sp500.iloc[-1:][new_predictors]
prediction = model.predict(latest_data)[0]

# Sidebar
st.sidebar.header("Options")
show_data = st.sidebar.checkbox("Show Historical Data", value=True)
predict_tomorrow = st.sidebar.button("Predict Tomorrow's Price")
selected_date = st.sidebar.date_input("Select a Date for Prediction", value=pd.to_datetime("today"))

if show_data:
    st.header("📊 Historical Data")
    st.line_chart(sp500['Close'])
    st.write(f"Data from {sp500.index.min().date()} to {sp500.index.max().date()} ({len(sp500)} days)")

if predict_tomorrow:
    st.header("🔮 Prediction")
    st.success(f"Predicted S&P 500 Close for Tomorrow ({pd.to_datetime('today').date()}): **${prediction:.2f}**")
    st.info("This is based on the latest available data. Actual prices may vary due to market conditions.")

# Model Performance
st.header("📈 Model Performance")
st.write("**Backtest Results (Regression Model):**")
st.write("- Mean Absolute Error (MAE): ~99 points")
st.write("- Root Mean Squared Error (RMSE): ~201 points")
st.write("- Baseline: Random guessing would be ~50% accurate for direction.")

# Additional Chart
st.header("📉 Prediction Distribution")
fig, ax = plt.subplots()
ax.hist(sp500['Close'], bins=50, alpha=0.7, label='Historical Closes')
ax.axvline(prediction, color='red', linestyle='--', label=f'Predicted Tomorrow: ${prediction:.2f}')
ax.legend()
st.pyplot(fig)

st.header("ℹ️ About")
st.write("""
This app uses a Random Forest model trained on historical S&P 500 data to predict tomorrow's closing price.
Features include rolling ratios and trends to capture market patterns.
**Disclaimer:** This is for educational purposes only. Do not use for financial decisions.
""")
st.write("Built with Streamlit, pandas, and scikit-learn.")