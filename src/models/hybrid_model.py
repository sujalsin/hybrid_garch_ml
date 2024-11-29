import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from scipy.optimize import minimize

from .garch_model import GARCHModel
from .ml_model import MLModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridModel:
    """Hybrid model combining GARCH and ML predictions for volatility forecasting."""
    
    def __init__(self, ml_model_type: str = 'rf', garch_p: int = 1, garch_q: int = 1,
                 ensemble_method: str = 'weighted', **kwargs):
        """
        Initialize the hybrid model.
        
        Args:
            ml_model_type: Type of ML model to use
            garch_p: Order of GARCH term
            garch_q: Order of ARCH term
            ensemble_method: Method to combine predictions ('weighted' or 'stacking')
            **kwargs: Additional parameters for ML model
        """
        self.garch_model = GARCHModel(p=garch_p, q=garch_q)
        self.ml_model = MLModel(model_type=ml_model_type, **kwargs)
        self.ensemble_method = ensemble_method
        self.stacking_model = Ridge(alpha=1.0)  # For stacking ensemble method
        self.weights = None  # For weighted ensemble method
        
    def fit(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Fit the hybrid model to the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary of metrics
        """
        lookback = 21  # LSTM lookback period
        
        # First, fit the GARCH model
        logger.info("Fitting GARCH model...")
        self.garch_model.fit(data)
        logger.info("Successfully fitted GARCH model")
        
        # Get GARCH predictions for the entire dataset
        garch_predictions = self.garch_model.predict(data)
        
        # Add GARCH predictions as a feature for ML model
        data_with_garch = data.copy()
        data_with_garch['GARCH_Pred'] = garch_predictions
        
        # Fit ML model
        logger.info("Fitting ML model...")
        self.ml_model.fit(data_with_garch)
        
        # Get ML predictions
        ml_predictions = self.ml_model.predict(data_with_garch)
        
        # Adjust predictions to account for lookback period
        garch_predictions = garch_predictions[lookback:]
        ml_predictions = ml_predictions
        y = data['RealizedVol'].values[lookback:]
        
        # Combine predictions using the optimal weights
        train_size = int(len(y) * 0.8)
        
        # Split predictions
        garch_train_pred = garch_predictions[:train_size]
        garch_val_pred = garch_predictions[train_size:]
        ml_train_pred = ml_predictions[:train_size]
        ml_val_pred = ml_predictions[train_size:]
        
        # Split target values
        y_train = y[:train_size]
        y_val = y[train_size:]
        
        # Find optimal weights using training data
        def objective(weights):
            w1, w2 = weights
            hybrid_pred = w1 * garch_train_pred + w2 * ml_train_pred
            return mean_squared_error(y_train, hybrid_pred)
        
        # Optimize weights with constraints (sum to 1, non-negative)
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
            {'type': 'ineq', 'fun': lambda w: w}  # weights are non-negative
        )
        
        initial_weights = [0.5, 0.5]
        bounds = [(0, 1), (0, 1)]
        
        result = minimize(objective, initial_weights, method='SLSQP',
                        constraints=constraints, bounds=bounds)
        
        self.weights = result.x
        logger.info(f"Optimal weights: GARCH={self.weights[0]:.3f}, ML={self.weights[1]:.3f}")
        
        # Calculate final predictions and metrics
        hybrid_val_pred = self.weights[0] * garch_val_pred + self.weights[1] * ml_val_pred
        
        metrics = {
            'mse': mean_squared_error(y_val, hybrid_val_pred),
            'mae': mean_absolute_error(y_val, hybrid_val_pred),
            'r2': r2_score(y_val, hybrid_val_pred),
            'garch_weight': self.weights[0],
            'ml_weight': self.weights[1]
        }
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the hybrid model.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Array of predictions
        """
        lookback = 21  # LSTM lookback period
        
        # Get individual model predictions
        garch_pred = self.garch_model.predict(data)
        
        # Add GARCH predictions as a feature for ML model
        data_with_garch = data.copy()
        data_with_garch['GARCH_Pred'] = garch_pred
        ml_pred = self.ml_model.predict(data_with_garch)
        
        # Adjust predictions to account for lookback period
        garch_pred = garch_pred[lookback:]
        ml_pred = ml_pred
        
        # Initialize array for final predictions
        final_predictions = np.zeros_like(ml_pred)
        
        # Combine predictions using optimized weights
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        final_predictions = (
            self.weights[0] * garch_pred +
            self.weights[1] * ml_pred
        )
        
        return final_predictions
    
    def save_model(self, model_dir: str) -> None:
        """
        Save the hybrid model to disk.
        
        Args:
            model_dir: Directory to save the model
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model components
        joblib.dump(self.garch_model, os.path.join(model_dir, 'garch_model.pkl'))
        joblib.dump(self.ml_model, os.path.join(model_dir, 'ml_model.pkl'))
        
        # Save ensemble parameters
        if self.ensemble_method == 'weighted':
            joblib.dump(self.weights, os.path.join(model_dir, 'ensemble_weights.pkl'))
        elif self.ensemble_method == 'stacking':
            joblib.dump(self.stacking_model, os.path.join(model_dir, 'stacking_model.pkl'))
            
        # Save configuration
        config = {
            'ensemble_method': self.ensemble_method,
            'ml_model_type': self.ml_model.model_type,
            'garch_p': self.garch_model.p,
            'garch_q': self.garch_model.q
        }
        joblib.dump(config, os.path.join(model_dir, 'config.pkl'))
        
    @classmethod
    def load_model(cls, model_dir: str) -> 'HybridModel':
        """
        Load a saved hybrid model from disk.
        
        Args:
            model_dir: Directory containing the saved model
            
        Returns:
            Loaded HybridModel instance
        """
        # Load configuration
        config = joblib.load(os.path.join(model_dir, 'config.pkl'))
        
        # Create model instance
        model = cls(
            ml_model_type=config['ml_model_type'],
            garch_p=config['garch_p'],
            garch_q=config['garch_q'],
            ensemble_method=config['ensemble_method']
        )
        
        # Load model components
        model.garch_model = joblib.load(os.path.join(model_dir, 'garch_model.pkl'))
        model.ml_model = joblib.load(os.path.join(model_dir, 'ml_model.pkl'))
        
        # Load ensemble parameters
        if config['ensemble_method'] == 'weighted':
            model.weights = joblib.load(os.path.join(model_dir, 'ensemble_weights.pkl'))
        elif config['ensemble_method'] == 'stacking':
            model.stacking_model = joblib.load(os.path.join(model_dir, 'stacking_model.pkl'))
            
        return model

def main():
    """Example usage of the HybridModel class."""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(end='2023-01-01', periods=1000)
    data = pd.DataFrame({
        'Symbol': ['AAPL'] * 1000,
        'Close': np.random.normal(100, 10, 1000).cumsum(),
        'Volume': np.random.randint(1000000, 10000000, 1000),
        'RealizedVol': np.random.normal(0.2, 0.05, 1000)
    }, index=dates)
    
    # Initialize and fit hybrid model
    model = HybridModel(ml_model_type='rf', ensemble_method='weighted')
    metrics = model.fit(data)
    
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Make predictions
    predictions = model.predict(data)
    print("\nSample predictions:", predictions[:5])
    
if __name__ == '__main__':
    main()
