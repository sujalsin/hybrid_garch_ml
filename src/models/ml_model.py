import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import itertools
import copy
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLModel:
    """Machine Learning model for volatility prediction."""
    
    def __init__(self, model_type: str = 'rf', **kwargs):
        """
        Initialize the ML model.
        
        Args:
            model_type: Type of model ('rf' for Random Forest, 'gb' for Gradient Boosting,
                       'lstm' for LSTM Neural Network)
            **kwargs: Additional model parameters
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.target_scaler = None
        self.feature_scaler = None
        self.feature_columns = None
        self.kwargs = kwargs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for the ML model with improved NaN handling.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with features
        """
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df['Adj Close'].pct_change().fillna(0)
        df['LogReturns'] = np.log(df['Adj Close']/df['Adj Close'].shift(1)).fillna(0)
        
        # Multiple timeframe features
        for window in [5, 10, 21, 63]:  # 1 week, 2 weeks, 1 month, 3 months
            df[f'RollingMean_{window}'] = df['Returns'].rolling(window=window, min_periods=1).mean()
            df[f'RollingStd_{window}'] = df['Returns'].rolling(window=window, min_periods=1).std()
            df[f'RollingSkew_{window}'] = df['Returns'].rolling(window=window, min_periods=window).skew().fillna(0)
            df[f'RollingKurt_{window}'] = df['Returns'].rolling(window=window, min_periods=window).kurt().fillna(0)
        
        # Price range and momentum features
        df['HighLowRange'] = ((df['High'] - df['Low'])/df['Close']).fillna(0)
        df['DailyReturn'] = (df['Close']/df['Open'] - 1).fillna(0)
        df['GapUp'] = ((df['Open'] - df['Close'].shift(1))/df['Close'].shift(1)).fillna(0)
        
        # Volume-based features
        df['VolumeChange'] = df['Volume'].pct_change().fillna(0)
        for window in [5, 10, 21]:
            df[f'RollingVolume_{window}'] = df['Volume'].rolling(window=window, min_periods=1).mean()
            df[f'RelativeVolume_{window}'] = (df['Volume']/df[f'RollingVolume_{window}']).fillna(1)
        
        # Volatility features
        df['TrueRange'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        ).fillna(0)
        
        for window in [5, 10, 21]:
            df[f'ATR_{window}'] = df['TrueRange'].rolling(window=window, min_periods=1).mean()
        
        # Technical indicators with NaN handling
        df['RSI'] = self._calculate_rsi(df['Close']).fillna(50)  # Neutral RSI value
        macd, signal = self._calculate_macd(df['Close'])
        df['MACD'] = macd.fillna(0)
        df['Signal'] = signal.fillna(0)
        
        upper, lower = self._calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = upper.fillna(method='ffill')
        df['BB_Lower'] = lower.fillna(method='ffill')
        
        # Trend strength indicators
        df['ADX'] = self._calculate_adx(df).fillna(0)
        df['CCI'] = self._calculate_cci(df).fillna(0)
        
        # Momentum indicators
        for window in [5, 10, 21]:
            df[f'ROC_{window}'] = df['Close'].pct_change(window).fillna(0)
            df[f'MFI_{window}'] = self._calculate_mfi(df, window).fillna(50)  # Neutral MFI value
        
        # Forward fill any remaining NaN values
        df = df.fillna(method='ffill')
        # Backward fill any remaining NaN values at the beginning
        df = df.fillna(method='bfill')
        # Fill any remaining NaN values with 0
        df = df.fillna(0)
        
        # Ensure RealizedVol is present if it exists in the input data
        if 'RealizedVol' in data.columns:
            df['RealizedVol'] = data['RealizedVol']
        
        return df

    def _prepare_lstm_data(self, data: pd.DataFrame, lookback: int = 21) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model by creating sequences.
        
        Args:
            data: DataFrame with features
            lookback: Number of lookback periods
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Ensure all features are numeric
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Get feature names excluding the target
        feature_cols = [col for col in numeric_data.columns if col != 'RealizedVol']
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.feature_scaler = StandardScaler()
            scaled_features = self.feature_scaler.fit_transform(data[feature_cols])
            if 'RealizedVol' in data.columns:
                scaled_target = self.scaler.fit_transform(data[['RealizedVol']])
            else:
                scaled_target = np.zeros((len(data), 1))  # Default if no target available
        else:
            scaled_features = self.feature_scaler.transform(data[feature_cols])
            if 'RealizedVol' in data.columns:
                scaled_target = self.scaler.transform(data[['RealizedVol']])
            else:
                scaled_target = np.zeros((len(data), 1))
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_features)):
            X.append(scaled_features[i-lookback:i])
            y.append(scaled_target[i])
        
        return np.array(X), np.array(y)

    def fit(self, data: pd.DataFrame, target_col: str = 'RealizedVol') -> None:
        """
        Fit the ML model to the data.
        
        Args:
            data: Input DataFrame
            target_col: Name of target column
        """
        # Create features
        df = self._create_features(data)
        
        # Prepare data for LSTM
        X, y = self._prepare_lstm_data(df)
        
        # Split data
        train_size = int(len(X) * 0.8)
        self.X_train, self.X_val = X[:train_size], X[train_size:]
        self.y_train, self.y_val = y[:train_size], y[train_size:]
        
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(random_state=42, **self.kwargs)
            self.model.fit(X.reshape(X.shape[0], -1), y)
            
        elif self.model_type == 'gb':
            self.model = GradientBoostingRegressor(random_state=42, **self.kwargs)
            self.model.fit(X.reshape(X.shape[0], -1), y)
            
        elif self.model_type == 'lstm':
            # Initialize and train LSTM
            self._build_lstm_model(X.shape[2])
            self._train_lstm(self.X_train, self.y_train, self.X_val, self.y_val)
            
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained ML model.
        
        Args:
            data: DataFrame with features
            
        Returns:
            Array of predictions
        """
        # Create features
        df = self._create_features(data)
        
        # Prepare data for LSTM
        X, _ = self._prepare_lstm_data(df)
        
        if self.model_type == 'lstm':
            # Make predictions
            predictions = []
            self.model.eval()
            with torch.no_grad():
                for i in range(0, len(X), 32):
                    batch_X = torch.FloatTensor(X[i:i+32]).to(self.device)
                    batch_pred = self.model(batch_X)
                    predictions.append(batch_pred.cpu().numpy())
            
            predictions = np.concatenate(predictions)
            
            # Inverse transform predictions
            if self.scaler is not None:
                predictions = self.scaler.inverse_transform(predictions)
            
            return predictions.flatten()
        else:
            predictions = self.model.predict(X.reshape(X.shape[0], -1))
            
        return predictions.flatten()
    
    def _build_lstm_model(self, input_dim: int) -> None:
        """
        Build LSTM model architecture.
        
        Args:
            input_dim: Input dimension
        """
        lookback = 21  # Define lookback period
        
        class AsymmetricLoss(nn.Module):
            def __init__(self, alpha: float = 2.5):
                super().__init__()
                self.alpha = alpha
            
            def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                diff = pred - target
                # Enhanced loss function with focal loss component
                loss = torch.where(
                    diff > 0,
                    self.alpha * torch.pow(torch.abs(diff), 2.0) * torch.log1p(torch.abs(diff)),  # Higher penalty for large overestimations
                    0.5 * torch.square(diff)  # Standard MSE for underestimation
                )
                return torch.mean(loss)
        
        class LSTMModel(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                
                # Input batch normalization
                self.batch_norm = nn.BatchNorm1d(lookback)
                
                # Bidirectional LSTM with residual connections
                self.lstm_layers = nn.ModuleList([
                    nn.LSTM(
                        input_size=input_dim if i == 0 else hidden_dim * 2,
                        hidden_size=hidden_dim,
                        num_layers=1,
                        dropout=0,
                        batch_first=True,
                        bidirectional=True
                    ) for i in range(num_layers)
                ])
                
                # Multi-head self-attention
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim * 2,
                    num_heads=4,
                    dropout=dropout,
                    batch_first=True
                )
                self.layer_norm = nn.LayerNorm(hidden_dim * 2)
                
                lstm_out_dim = hidden_dim * 2  # *2 for bidirectional
                
                # Enhanced fully connected layers with residual connections
                self.fc_layers = nn.ModuleDict({
                    'fc1': nn.Sequential(
                        nn.Linear(lstm_out_dim, 64),
                        nn.BatchNorm1d(64),
                        nn.GELU(),
                        nn.Dropout(dropout)
                    ),
                    'fc2': nn.Sequential(
                        nn.Linear(64, 32),
                        nn.BatchNorm1d(32),
                        nn.GELU(),
                        nn.Dropout(dropout)
                    ),
                    'fc3': nn.Sequential(
                        nn.Linear(32, 16),
                        nn.BatchNorm1d(16),
                        nn.GELU(),
                        nn.Dropout(dropout)
                    ),
                    'output': nn.Sequential(
                        nn.Linear(16, 1),
                        nn.Softplus()
                    )
                })
                
                # Initialize weights
                self._init_weights()
            
            def _init_weights(self):
                # Initialize LSTM weights
                for lstm in self.lstm_layers:
                    for name, param in lstm.named_parameters():
                        if 'weight_ih' in name:
                            nn.init.xavier_uniform_(param)
                        elif 'weight_hh' in name:
                            nn.init.orthogonal_(param)
                        elif 'bias' in name:
                            nn.init.constant_(param, 0.0)
                
                # Initialize attention weights
                for param in self.attention.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
                
                # Initialize fully connected layers
                for layer in self.fc_layers.values():
                    for m in layer:
                        if isinstance(m, nn.Linear):
                            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                            if m.bias is not None:
                                nn.init.constant_(m.bias, 0)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Apply input batch normalization
                batch_size = x.size(0)
                x = self.batch_norm(x)
                
                # Process through LSTM layers with residual connections
                lstm_out = x
                for lstm_layer in self.lstm_layers:
                    new_out, _ = lstm_layer(lstm_out)
                    lstm_out = new_out + lstm_out if lstm_out.size() == new_out.size() else new_out
                
                # Apply self-attention
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                lstm_out = self.layer_norm(lstm_out + attn_out)  # Residual connection
                
                # Get final hidden state
                final_hidden = lstm_out[:, -1, :]
                
                # Process through FC layers with residual connections
                fc_out = final_hidden
                for i, (name, layer) in enumerate(self.fc_layers.items()):
                    if name != 'output':
                        new_out = layer(fc_out)
                        fc_out = new_out + fc_out if fc_out.size() == new_out.size() else new_out
                    else:
                        fc_out = layer(fc_out)
                
                return fc_out
        
        # Initialize model with hyperparameters
        hidden_dim = self.kwargs.get('hidden_dim', 128)  # Increased hidden dimension
        num_layers = self.kwargs.get('num_layers', 3)    # Increased number of layers
        dropout = self.kwargs.get('dropout', 0.3)
        
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # Use custom loss and AdamW optimizer with cosine annealing
        self.criterion = AsymmetricLoss(alpha=2.5)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.kwargs.get('learning_rate', 0.001),
            weight_decay=self.kwargs.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing scheduler with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Initial restart interval
            T_mult=2,  # Multiply interval by 2 after each restart
            eta_min=1e-6  # Minimum learning rate
        )
    
    def _train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Train LSTM model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        
        Returns:
            Dictionary of metrics
        """
        # Convert data to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # Initialize optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=self.kwargs.get('learning_rate', 0.001))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.kwargs.get('epochs', 100)):
            # Training phase
            self.model.train()
            train_loss = 0
            for i in range(0, len(X_train), self.kwargs.get('batch_size', 32)):
                batch_X = X_train[i:i+self.kwargs.get('batch_size', 32)]
                batch_y = y_train[i:i+self.kwargs.get('batch_size', 32)]
                
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= (len(X_train) // self.kwargs.get('batch_size', 32))
            train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for i in range(0, len(X_val), self.kwargs.get('batch_size', 32)):
                    batch_X = X_val[i:i+self.kwargs.get('batch_size', 32)]
                    batch_y = y_val[i:i+self.kwargs.get('batch_size', 32)]
                    
                    outputs = self.model(batch_X)
                    val_loss += self.criterion(outputs, batch_y).item()
                    
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(batch_y.cpu().numpy())
            
            val_loss /= (len(X_val) // self.kwargs.get('batch_size', 32))
            val_losses.append(val_loss)
            
            # Convert predictions and actuals to numpy arrays
            predictions = np.array(predictions).flatten()
            actuals = np.array(actuals).flatten()
            
            # Calculate additional metrics
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.kwargs.get('patience', 15):
                print(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Update learning rate
            scheduler.step()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                      f"MSE = {mse:.4f}, MAE = {mae:.4f}, R2 = {r2:.4f}")
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
    
    def optimize_hyperparameters(self, data: pd.DataFrame, target_col: str = 'RealizedVol',
                               n_trials: int = 100) -> Dict[str, float]:
        """
        Use grid search to optimize hyperparameters.
        
        Args:
            data: Input DataFrame
            target_col: Name of target column
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary of best parameters
        """
        if self.model_type == 'rf':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'gb':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7]
            }
        else:
            # For LSTM, we'll use default parameters
            return self.kwargs
            
        # Prepare data
        X = self._create_features(data)
        y = data[target_col].values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        best_score = float('inf')
        best_params = None
        
        # Grid search
        for params in itertools.product(*param_grid.values()):
            current_params = dict(zip(param_grid.keys(), params))
            
            if self.model_type == 'rf':
                model = RandomForestRegressor(**current_params, random_state=42)
            else:  # gb
                model = GradientBoostingRegressor(**current_params, random_state=42)
                
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = mean_squared_error(y_val, y_pred)
            
            if score < best_score:
                best_score = score
                best_params = current_params
                
        self.kwargs.update(best_params)
        return best_params
    
    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate RSI technical indicator with NaN handling."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series,
                       fast_period: int = 12,
                       slow_period: int = 26,
                       signal_period: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD technical indicator with NaN handling."""
        exp1 = prices.ewm(span=fast_period, adjust=False, min_periods=1).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False, min_periods=1).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False, min_periods=1).mean()
        return macd, signal

    def _calculate_bollinger_bands(self, prices: pd.Series,
                                 window: int = 20,
                                 num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands technical indicator with NaN handling."""
        rolling_mean = prices.rolling(window=window, min_periods=1).mean()
        rolling_std = prices.rolling(window=window, min_periods=1).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX) with NaN handling."""
        df = data.copy()
        df['TR'] = self._true_range(df)
        df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                            np.maximum(df['High'] - df['High'].shift(1), 0),
                            0)
        df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                            np.maximum(df['Low'].shift(1) - df['Low'], 0),
                            0)
        
        df['+DI'] = 100 * (df['+DM'].rolling(period, min_periods=1).mean() / df['TR'].rolling(period, min_periods=1).mean())
        df['-DI'] = 100 * (df['-DM'].rolling(period, min_periods=1).mean() / df['TR'].rolling(period, min_periods=1).mean())
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
        
        return df['DX'].rolling(period, min_periods=1).mean()

    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index (CCI) with NaN handling."""
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        tp_ma = tp.rolling(period, min_periods=1).mean()
        tp_md = abs(tp - tp_ma).rolling(period, min_periods=1).mean()
        return (tp - tp_ma) / (0.015 * tp_md.replace(0, 1))  # Avoid division by zero

    def _calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index (MFI) with NaN handling."""
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        rmf = tp * data['Volume']
        
        pos_flow = np.where(tp > tp.shift(1), rmf, 0)
        neg_flow = np.where(tp < tp.shift(1), rmf, 0)
        
        pos_mf = pd.Series(pos_flow).rolling(period, min_periods=1).sum()
        neg_mf = pd.Series(neg_flow).rolling(period, min_periods=1).sum()
        
        return 100 - (100 / (1 + pos_mf / neg_mf.replace(0, 1)))  # Avoid division by zero

    def _true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift(1))
        low_close = np.abs(data['Low'] - data['Close'].shift(1))
        return pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close}).max(axis=1)

def main():
    """Example usage of the MLModel class."""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(end='2023-01-01', periods=1000)
    data = pd.DataFrame({
        'Symbol': ['AAPL'] * 1000,
        'Close': np.random.normal(100, 10, 1000).cumsum(),
        'Volume': np.random.randint(1000000, 10000000, 1000),
        'RealizedVol': np.random.normal(0.2, 0.05, 1000),
        'High': np.random.normal(100, 10, 1000).cumsum(),
        'Low': np.random.normal(100, 10, 1000).cumsum(),
        'Open': np.random.normal(100, 10, 1000).cumsum(),
        'Adj Close': np.random.normal(100, 10, 1000).cumsum()
    }, index=dates)
    
    # Initialize and fit model
    model = MLModel(model_type='lstm')
    model.fit(data)
    
    # Make predictions
    predictions = model.predict(data)
    print("\nSample predictions:", predictions[:5])
    
if __name__ == '__main__':
    main()
