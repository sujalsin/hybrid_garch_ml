import pandas as pd
import numpy as np
from src.models.hybrid_model import HybridModel
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_stock_data(symbol: str = 'AAPL', period: str = '5y'):
    """Fetch stock data using yfinance."""
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    
    # Calculate daily returns
    data['Returns'] = data['Close'].pct_change()
    
    # Calculate realized volatility (21-day rolling standard deviation of returns)
    data['RealizedVol'] = data['Returns'].rolling(window=21).std() * np.sqrt(252)
    
    # Drop NaN values
    data = data.dropna()
    
    return data

def evaluate_predictions(y_true, y_pred):
    """Calculate and return evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

def plot_results(actual, predicted, title='Volatility Predictions'):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label='Actual', alpha=0.7)
    plt.plot(predicted.index, predicted, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Fetch data
    logger.info("Fetching stock data...")
    data = fetch_stock_data('AAPL', '5y')
    
    # Initialize hybrid model
    logger.info("Initializing hybrid model...")
    model = HybridModel(
        ml_model_type='lstm',
        garch_p=1,
        garch_q=1,
        ensemble_method='weighted',
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )
    
    # Split data into training and testing
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Fit model
    logger.info("Training hybrid model...")
    metrics = model.fit(train_data, validation_split=0.2)
    logger.info(f"Training metrics: {metrics}")
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(test_data)
    
    # Evaluate results
    eval_metrics = evaluate_predictions(
        test_data['RealizedVol'],
        predictions
    )
    logger.info(f"Test metrics: {eval_metrics}")
    
    # Plot results
    plot_results(
        test_data['RealizedVol'],
        pd.Series(predictions, index=test_data.index),
        'AAPL Volatility Predictions'
    )

if __name__ == '__main__':
    main()
