import pandas as pd
import yfinance as yf
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Load data
if os.path.exists("sp500.csv"):
    sp500 = pd.read_csv("sp500.csv", index_col=0)
else:
    sp500 = yf.Ticker("^GSPC").history(period="max")
    sp500.to_csv("sp500.csv")

sp500.index = pd.to_datetime(sp500.index)

# Cleanup
del sp500["Dividends"]
del sp500["Stock Splits"]

# Create target
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Restrict to 1990
sp500 = sp500.loc["1990-01-01":].copy()

# Initial model
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]
predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
initial_precision = precision_score(test["Target"], preds)
print(f"Initial precision: {initial_precision}")

# Define predict function (first version)
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Define backtest function
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# Run backtest
predictions = backtest(sp500, model, predictors)
backtest_precision = precision_score(predictions["Target"], predictions["Predictions"])
print(f"Backtest precision: {backtest_precision}")
print(f"Predictions value counts:\n{predictions['Predictions'].value_counts()}")
print(f"Target value counts / total:\n{predictions['Target'].value_counts() / predictions.shape[0]}")

# Feature engineering
horizons = [2,5,60,250,1000]
new_predictors = []
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ratio_column, trend_column]

# Drop NA
sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])

# Updated model
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Updated predict function
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Run backtest with new predictors
predictions = backtest(sp500, model, new_predictors)
final_precision = precision_score(predictions["Target"], predictions["Predictions"])
print(f"Final precision: {final_precision}")
print(f"Final predictions value counts:\n{predictions['Predictions'].value_counts()}")
print(f"Final target value counts / total:\n{predictions['Target'].value_counts() / predictions.shape[0]}")
