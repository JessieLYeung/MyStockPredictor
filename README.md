# S&P 500 Stock Price Predictor

A machine learning project to predict the S&P 500 index price movements and closing prices using historical data. This project demonstrates end-to-end data science skills, including data collection, preprocessing, model training, backtesting, and a web-based interface.

## Features
- **Data Collection**: Downloads historical S&P 500 data using `yfinance`.
- **Direction Prediction**: Classifies whether the index will close higher the next day.
- **Price Prediction**: Regresses the exact closing price for the next day.
- **Backtesting**: Uses expanding-window validation to simulate real-world performance and avoid overfitting.
- **Feature Engineering**: Incorporates rolling averages, ratios, and trend indicators.
- **Interactive Web App**: Streamlit-based dashboard for visualizations and predictions.
- **Model Evaluation**: Includes precision, MAE, RMSE, and prediction distributions.

## Project Steps
1. Download and clean S&P 500 historical data.
2. Build an initial Random Forest model for direction prediction.
3. Implement backtesting for accurate accuracy measurement.
4. Add advanced features to improve model performance.
5. Switch to regression for price forecasting.
6. Create a user-friendly web interface.

## Results
- **Direction Model**: Achieved 57% precision on backtested predictions (better than 53-55% baseline).
- **Price Model**: MAE of ~15 points, RMSE of ~25 points (using Linear Regression after comparing Random Forest, XGBoost, and Linear Regression).
- **Tomorrow's Prediction**: As of latest data, predicted close: ~$4069.

## Technologies Used
- **Python**: Core language.
- **Libraries**: pandas, scikit-learn, yfinance, matplotlib, Streamlit, joblib.
- **Tools**: Jupyter Notebook, VS Code.

## File Overview
- `market_prediction.ipynb`: Original Jupyter notebook with initial code.
- `run_full.py`: Script for direction prediction with backtesting.
- `predict_tomorrow.py`: Script for price regression and model saving.
- `run_initial.py`: Basic data loading and inspection.
- `app.py`: Streamlit web app for interactive predictions.
- `test_predictor.py`: Basic unit tests for key functions.
- `requirements.txt`: Python dependencies.
- `sp500.csv`: Downloaded historical data (auto-generated).
- `sp500_model.pkl`: Trained model (auto-generated).

## Installation and Setup
1. **Clone or Download**: Ensure you have the project files.
2. **Install Python**: Version 3.8+ recommended.
3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```
4. **Run Scripts**:
   - Data and model: `python predict_tomorrow.py` (trains and saves model).
   - Direction prediction: `python run_full.py`.
   - Web app: `streamlit run app.py` (opens at http://localhost:8501).
5. **Run Tests**: `python test_predictor.py` or `pytest`.

## CI/CD
This project uses GitHub Actions for continuous integration. On every push or PR to `main`/`master`:
- Installs dependencies.
- Runs linting with flake8.
- Executes unit tests.

Check `.github/workflows/ci.yml` for details.

## Usage
- **Terminal Scripts**: Run for batch predictions and metrics.
- **Web App**: Interactive interface for charts and on-demand predictions.
- **Customization**: Modify horizons or features in scripts for experimentation.

## Model Details
- **Algorithm**: Random Forest (Regressor for price, Classifier for direction).
- **Features**: OHLCV, rolling ratios (2-1000 days), trend counts.
- **Validation**: Expanding-window backtesting (start at 2500 samples, step 250).
- **Overfitting Prevention**: No future data leakage; features use past info only.

## Limitations and Risks
- Markets are unpredictable; this is for educational purposes only.
- Model errors: ~99 points MAE means significant uncertainty.
- Not for live trading without further validation and risk management.

## Future Improvements
- Add external data (e.g., VIX, interest rates).
- Experiment with other models (e.g., XGBoost, LSTM).
- Implement real-time data updates.
- Deploy to cloud (e.g., Heroku) for public access.
- Add user authentication or advanced visualizations.

## Contributing
Feel free to fork, improve, and submit pull requests. This project is open-source for learning.

## License
MIT License - Use freely for educational purposes.

## Acknowledgments
Inspired by Dataquest's S&P 500 project. Built with open-source tools.