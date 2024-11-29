import yfinance as yf
import pandas as pd
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataLoader:
    """Class for loading and preprocessing stock market data."""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            symbols: List of stock symbols to download
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
    def download_data(self) -> pd.DataFrame:
        """
        Download historical stock data using yfinance.
        
        Returns:
            DataFrame containing the stock data
        """
        logger.info(f"Downloading data for {len(self.symbols)} symbols...")
        
        data_frames = []
        for symbol in self.symbols:
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(start=self.start_date, end=self.end_date)
                df['Symbol'] = symbol
                data_frames.append(df)
            except Exception as e:
                logger.error(f"Error downloading data for {symbol}: {str(e)}")
                
        if not data_frames:
            raise ValueError("No data was downloaded for any symbol")
            
        combined_data = pd.concat(data_frames)
        logger.info(f"Downloaded {len(combined_data)} rows of data")
        return combined_data
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the downloaded data.
        
        Args:
            df: Raw stock data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Calculate daily returns
        df['Returns'] = df.groupby('Symbol')['Close'].pct_change()
        
        # Calculate realized volatility (21-day rolling standard deviation of returns)
        df['RealizedVol'] = df.groupby('Symbol')['Returns'].transform(
            lambda x: x.rolling(window=21).std() * (252 ** 0.5)  # Annualized
        )
        
        # Handle missing values
        df = df.dropna()
        
        return df
    
    def get_clean_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download and preprocess data, split into training and validation sets.
        
        Returns:
            Tuple of (training_data, validation_data)
        """
        # Download and preprocess data
        raw_data = self.download_data()
        processed_data = self.preprocess_data(raw_data)
        
        # Split into training and validation (last 20% for validation)
        cutoff_date = pd.to_datetime(self.end_date) - timedelta(days=365)
        train_data = processed_data[processed_data.index <= cutoff_date]
        val_data = processed_data[processed_data.index > cutoff_date]
        
        return train_data, val_data

def main():
    """Example usage of the StockDataLoader class."""
    # Example usage with S&P 500 stocks (using a small subset for demonstration)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    start_date = '2010-01-01'
    
    loader = StockDataLoader(symbols=symbols, start_date=start_date)
    train_data, val_data = loader.get_clean_data()
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    
if __name__ == '__main__':
    main()
