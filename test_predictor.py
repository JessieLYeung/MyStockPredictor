"""
Basic tests for S&P 500 Predictor
"""

import pandas as pd
import numpy as np
from predict_tomorrow import backtest, load_data, preprocess_data
from sklearn.ensemble import RandomForestRegressor

def test_load_data():
    """Test data loading."""
    data = load_data()
    assert isinstance(data, pd.DataFrame), "Data should be a DataFrame"
    assert 'Close' in data.columns, "Close column missing"
    print("✓ test_load_data passed")

def test_preprocess_data():
    """Test data preprocessing."""
    data = load_data()
    processed = preprocess_data(data)
    assert 'Tomorrow' in processed.columns, "Tomorrow column missing"
    assert processed.index.is_monotonic_increasing, "Index should be sorted"
    print("✓ test_preprocess_data passed")

def test_backtest():
    """Test backtest function."""
    data = load_data()
    data = preprocess_data(data)
    model = RandomForestRegressor(n_estimators=10, random_state=1)  # Small for speed
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    results = backtest(data, model, predictors, start=100, step=50)  # Small for test
    assert 'Predictions' in results.columns, "Predictions column missing"
    assert len(results) > 0, "No predictions generated"
    print("✓ test_backtest passed")

if __name__ == "__main__":
    test_load_data()
    test_preprocess_data()
    test_backtest()
    print("All tests passed!")