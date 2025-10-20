import streamlit as st
import pandas as pd
import yfinance as yf
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# Load models
if os.path.exists('sp500_models.pkl'):
    trained_models = joblib.load('sp500_models.pkl')
    st.success("Models loaded successfully.")
else:
    st.error("Models not found. Please run predict_tomorrow.py first.")
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
st.title("📈 MyStockPredictor")
st.markdown("Predict tomorrow's S&P 500 closing price using machine learning.")

# Sidebar for customization
st.sidebar.header("Customization")
selected_model = st.sidebar.selectbox("Select Model", list(trained_models.keys()))
predict_date = st.sidebar.date_input("Predict for Date", value=pd.to_datetime("today"))

# Get selected model
model = trained_models[selected_model]["Model"]
mae = trained_models[selected_model]["MAE"]
rmse = trained_models[selected_model]["RMSE"]

# Compute prediction for selected date (for now, only today is supported)
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
    st.success(f"Predicted S&P 500 Close for {predict_date.date()} using {selected_model}: **${prediction:.2f}**")
    st.info("This is based on the latest available data. Actual prices may vary due to market conditions.")

# Model Performance
st.header("📈 Model Performance")
st.write(f"**Selected Model: {selected_model}**")
st.write(f"- Mean Absolute Error (MAE): {mae:.2f} points")
st.write(f"- Root Mean Squared Error (RMSE): {rmse:.2f} points")
st.write("- Baseline: Random guessing would be ~50% accurate for direction.")
st.write("**Model Comparison:**")
for name, info in trained_models.items():
    st.write(f"- {name}: MAE {info['MAE']:.2f}, RMSE {info['RMSE']:.2f}")

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