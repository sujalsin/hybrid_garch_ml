import pandas as pd
import numpy as np
import yfinance as yf
from src.models.hybrid_model import HybridModel

def load_test_data():
    # Download sample stock data
    ticker = "AAPL"
    data = yf.download(ticker, start="2020-01-01", end="2023-12-31")
    
    # Calculate daily returns and realized volatility
    data['Returns'] = data['Adj Close'].pct_change()
    data['RealizedVol'] = data['Returns'].rolling(window=21).std() * np.sqrt(252)
    
    # Drop NaN values
    data = data.dropna()
    return data

def test_hybrid_model():
    """
    Test the hybrid volatility prediction model.
    """
    print("\nLoading test data...")
    data = load_test_data()
    
    print("\nInitializing hybrid model...")
    model = HybridModel(
        ml_model_type='lstm',
        ensemble_method='weighted'
    )
    
    print("\nFitting hybrid model...")
    try:
        metrics = model.fit(data)
        print("\nTraining metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        print("\nGenerating predictions...")
        predictions = model.predict(data)
        
        # Account for lookback period
        lookback = 21
        valid_indices = ~np.isnan(predictions)
        actual_vals = data['RealizedVol'].values[lookback:][valid_indices]
        predicted_vals = predictions[valid_indices]
        
        print(f"\nNumber of valid predictions: {len(predicted_vals)}")
        print(f"Total data points: {len(data)}")
        print(f"Prediction coverage: {(len(predicted_vals)/len(data))*100:.2f}%")
        
        # Print sample predictions for the last 5 days
        print("\nSample predictions (last 5 days):")
        dates = data.index[lookback:][valid_indices][-5:]
        for date, pred, actual in zip(dates, predicted_vals[-5:], actual_vals[-5:]):
            error_pct = abs(pred - actual) / actual * 100
            print(f"{date.strftime('%Y-%m-%d')}: Predicted={pred:.4f}, Actual={actual:.4f}, Error={error_pct:.2f}%")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_hybrid_model()
