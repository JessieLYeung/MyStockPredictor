"""
S&P 500 Price Predictor - Regression Model

This script trains a Random Forest Regressor to predict tomorrow's S&P 500 closing price
using historical data. It includes backtesting to evaluate performance and avoid overfitting.
"""

import pandas as pd
import yfinance as yf
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
from xgboost import XGBRegressor

# Step 1: Download S&P 500 prices using yfinance
# This avoids overfitting by using real historical data, not simulated
def load_data():
    """Load S&P 500 data from CSV or download via yfinance."""
    if os.path.exists("sp500.csv"):
        sp500 = pd.read_csv("sp500.csv", index_col=0)
    else:
        sp500 = yf.Ticker("^GSPC").history(period="max")
        sp500.to_csv("sp500.csv")
    return sp500

def preprocess_data(sp500):
    """Clean and prepare data for modeling."""
    sp500.index = pd.to_datetime(sp500.index)
    del sp500["Dividends"]
    del sp500["Stock Splits"]
    sp500 = sp500.loc["1990-01-01":].copy()
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.dropna()
    return sp500

def add_features(sp500):
    """Add rolling features for better predictions."""
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
    return sp500, new_predictors

def backtest(data, model, predictors, start=2500, step=250):
    """
    Perform expanding-window backtesting for regression.

    Args:
        data (pd.DataFrame): Historical data with features and target.
        model: Scikit-learn model to train.
        predictors (list): List of feature column names.
        start (int): Initial training window size.
        step (int): Step size for testing windows.

    Returns:
        pd.DataFrame: Combined actual and predicted values.
    """
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        model.fit(train[predictors], train["Tomorrow"])
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index, name="Predictions")
        combined = pd.concat([test["Tomorrow"], preds], axis=1)
        all_predictions.append(combined)
    return pd.concat(all_predictions)

def compare_models(data, predictors):
    """Compare multiple regression models."""
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_split=50, random_state=1),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=1),
        "Linear Regression": LinearRegression()
    }
    
    results = {}
    for name, model in models.items():
        predictions = backtest(data, model, predictors)
        mae = mean_absolute_error(predictions["Tomorrow"], predictions["Predictions"])
        rmse = np.sqrt(mean_squared_error(predictions["Tomorrow"], predictions["Predictions"]))
        results[name] = {"MAE": mae, "RMSE": rmse, "Model": model}
        print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Select best model (lowest MAE)
    best_name = min(results, key=lambda x: results[x]["MAE"])
    print(f"Best model: {best_name}")
    return results[best_name]["Model"]

if os.path.exists("sp500.csv"):
    sp500 = pd.read_csv("sp500.csv", index_col=0)
else:
    sp500 = yf.Ticker("^GSPC").history(period="max")
    sp500.to_csv("sp500.csv")

sp500.index = pd.to_datetime(sp500.index)

# Step 2: Clean up the data with pandas
# Remove irrelevant columns to focus on price data
del sp500["Dividends"]
del sp500["Stock Splits"]

# Restrict to 1990 onward for more recent data
sp500 = sp500.loc["1990-01-01":].copy()

# Create target: tomorrow's closing price
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500 = sp500.dropna()  # Drop the last row where Tomorrow is NaN

# For trends, we need direction
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Main execution
if __name__ == "__main__":
    sp500 = load_data()
    sp500 = preprocess_data(sp500)
    sp500, new_predictors = add_features(sp500)

    print("Comparing models...")
    best_model = compare_models(sp500, new_predictors)

    # Save the best model
    joblib.dump(best_model, 'sp500_model.pkl')
    print("Best model saved as sp500_model.pkl")

    # Predict tomorrow
    latest_data = sp500.iloc[-1:][new_predictors]
    tomorrow_pred = best_model.predict(latest_data)[0]
    print(f"Predicted tomorrow's S&P 500 close: {tomorrow_pred:.2f}")