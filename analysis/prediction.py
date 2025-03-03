# prediction.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import joblib
import os
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class PricePredictionModel:
    """Base class for price prediction models."""
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or {}
        self.logger = logging.getLogger(f"prediction.{model_name}")
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Set model directory
        self.model_dir = self.config.get('model_dir', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for model training.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Tuple of features and target arrays
        """
        # Abstract method to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _preprocess_data")
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the prediction model.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with training metrics
        """
        # Abstract method to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement train")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate price predictions.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with predictions
        """
        # Abstract method to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement predict")
    
    def save_model(self) -> str:
        """
        Save the trained model to disk.
        
        Returns:
            Path to the saved model
        """
        # Abstract method to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement save_model")
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        # Abstract method to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement load_model")
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Abstract method to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement evaluate")


class LSTMPricePredictor(PricePredictionModel):
    """LSTM-based price prediction model."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__('LSTM', config)
        
        # Model parameters
        self.seq_length = self.config.get('sequence_length', 60)
        self.forecast_days = self.config.get('forecast_days', 5)
        self.features = self.config.get('features', ['close', 'volume', 'rsi_14', 'sma_20', 'ema_12'])
        
        # Build model
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the LSTM model architecture."""
        feature_count = len(self.features)
        
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, 
                      input_shape=(self.seq_length, feature_count)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=self.forecast_days))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        self.logger.info("LSTM model built successfully")
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences and target values for LSTM.
        
        Args:
            data: Scaled feature data
            
        Returns:
            Tuple of input sequences and target values
        """
        X, y = [], []
        
        # Ensure we have enough data given the sequence length and forecast days
        for i in range(len(data) - self.seq_length - self.forecast_days + 1):
            X.append(data[i:(i + self.seq_length)])
            y.append(data[i + self.seq_length:i + self.seq_length + self.forecast_days, 0])
        
        return np.array(X), np.array(y)
    
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data for LSTM model.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Tuple of input sequences, target values, and original data for inverse scaling
        """
        # Check if required features are present
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Extract relevant features
        data = df[self.features].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        return X, y, data
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2, 
             epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            df: DataFrame with price data
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training metrics
        """
        # Preprocess data
        X, y, _ = self._preprocess_data(df)
        
        if len(X) == 0:
            raise ValueError("Not enough data for training")
        
        # Calculate split index
        split_idx = int(len(X) * (1 - validation_split))
        
        # Split into training and validation sets
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Set up callbacks
        model_path = os.path.join(self.model_dir, f"{self.model_name}_best.h5")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
        ]
        
        # Train the model
        self.logger.info(f"Training LSTM model with {len(X_train)} samples")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Extract training metrics
        metrics = {
            'train_loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
            'epochs_completed': len(history.history['loss']),
            'early_stopped': len(history.history['loss']) < epochs
        }
        
        self.logger.info(f"Model training completed with validation loss: {metrics['val_loss']:.4f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate price predictions for the next forecast_days.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Check if we have enough data
        if len(df) < self.seq_length:
            raise ValueError(f"Not enough data for prediction. Need at least {self.seq_length} data points.")
        
        # Extract and scale features
        data = df[self.features].values
        scaled_data = self.scaler.transform(data)
        
        # Take the last sequence for prediction
        last_sequence = scaled_data[-self.seq_length:].reshape(1, self.seq_length, len(self.features))
        
        # Make prediction
        scaled_prediction = self.model.predict(last_sequence)
        
        # Prepare for inverse transform
        # The prediction is only for the target variable (close price)
        # We need to prepare a proper array for inverse_transform
        pred_full_features = np.zeros((scaled_prediction.shape[1], scaled_data.shape[1]))
        pred_full_features[:, 0] = scaled_prediction[0]  # Assuming the first feature is 'close'
        
        # Repeat the last values for other features
        for i in range(1, scaled_data.shape[1]):
            pred_full_features[:, i] = scaled_data[-1, i]
        
        # Inverse transform to get actual values
        prediction = self.scaler.inverse_transform(pred_full_features)
        
        # Extract the predicted close prices
        predicted_closes = prediction[:, 0]
        
        # Create a DataFrame with predictions
        last_date = df['timestamp'].iloc[-1]
        pred_dates = [last_date + timedelta(days=i+1) for i in range(self.forecast_days)]
        
        result = pd.DataFrame({
            'timestamp': pred_dates,
            'predicted_close': predicted_closes
        })
        
        return result
    
    def save_model(self) -> str:
        """
        Save the trained model to disk.
        
        Returns:
            Path to the saved model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model and scaler
        model_path = os.path.join(self.model_dir, f"{self.model_name}.h5")
        scaler_path = os.path.join(self.model_dir, f"{self.model_name}_scaler.joblib")
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save configuration
        config_path = os.path.join(self.model_dir, f"{self.model_name}_config.joblib")
        config = {
            'seq_length': self.seq_length,
            'forecast_days': self.forecast_days,
            'features': self.features
        }
        joblib.dump(config, config_path)
        
        self.logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model, or None to use default path
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Use default path if not provided
            if model_path is None:
                model_path = os.path.join(self.model_dir, f"{self.model_name}.h5")
            
            # Derive paths for scaler and config
            scaler_path = model_path.replace('.h5', '_scaler.joblib')
            config_path = model_path.replace('.h5', '_config.joblib')
            
            # Load model, scaler, and config
            self.model = load_model(model_path)
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            if os.path.exists(config_path):
                config = joblib.load(config_path)
                self.seq_length = config.get('seq_length', self.seq_length)
                self.forecast_days = config.get('forecast_days', self.forecast_days)
                self.features = config.get('features', self.features)
            
            self.logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if len(df) <= self.seq_length + self.forecast_days:
            raise ValueError(f"Not enough data for evaluation. Need more than {self.seq_length + self.forecast_days} data points.")
        
        # Create evaluation window
        test_windows = []
        actual_values = []
        
        for i in range(len(df) - self.seq_length - self.forecast_days + 1):
            # Window for input sequence
            window_data = df.iloc[i:i + self.seq_length + self.forecast_days]
            
            # Input data
            X_window = window_data.iloc[:self.seq_length][self.features].values
            X_window_scaled = self.scaler.transform(X_window)
            X_window_reshaped = X_window_scaled.reshape(1, self.seq_length, len(self.features))
            
            # Actual future values
            actual = window_data.iloc[self.seq_length:self.seq_length + self.forecast_days]['close'].values
            
            test_windows.append(X_window_reshaped)
            actual_values.append(actual)
        
        # Convert to arrays
        X_test = np.vstack(test_windows)
        y_actual = np.array(actual_values)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_test)
        
        # Inverse transform predictions
        y_pred = np.zeros_like(y_pred_scaled)
        for i in range(len(y_pred_scaled)):
            # Prepare for inverse transform
            pred_full_features = np.zeros((y_pred_scaled.shape[1], len(self.features)))
            pred_full_features[:, 0] = y_pred_scaled[i]
            
            # Use the last input values for other features
            for j in range(1, len(self.features)):
                pred_full_features[:, j] = X_test[i, -1, j]
            
            # Inverse transform
            prediction = self.scaler.inverse_transform(pred_full_features)
            y_pred[i] = prediction[:, 0]
        
        # Calculate metrics
        mse = mean_squared_error(y_actual.flatten(), y_pred.flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual.flatten(), y_pred.flatten())
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_actual.flatten() - y_pred.flatten()) / y_actual.flatten())) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }


class RandomForestPredictor(PricePredictionModel):
    """Random Forest-based price prediction model."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__('RandomForest', config)
        
        # Model parameters
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 10)
        self.min_samples_split = self.config.get('min_samples_split', 2)
        self.min_samples_leaf = self.config.get('min_samples_leaf', 1)
        self.forecast_days = self.config.get('forecast_days', 5)
        self.features = self.config.get('features', [
            'open', 'high', 'low', 'close', 'volume', 
            'sma_20', 'ema_12', 'rsi_14', 'macd', 'bb_upper', 'bb_lower'
        ])
        self.target_feature = self.config.get('target_feature', 'close')
        self.lookback_days = self.config.get('lookback_days', 30)
        
        # Initialize model
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the Random Forest model."""
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        self.logger.info("Random Forest model initialized")
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for the Random Forest model.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with engineered features
        """
        result = df.copy()
        
        # Add lag features
        for feature in self.features:
            if feature in result.columns:
                for lag in range(1, self.lookback_days + 1):
                    result[f"{feature}_lag_{lag}"] = result[feature].shift(lag)
        
        # Add day of week (if timestamp is available)
        if 'timestamp' in result.columns:
            result['day_of_week'] = pd.to_datetime(result['timestamp']).dt.dayofweek
        
        # Add rolling statistics
        for feature in ['close', 'volume']:
            if feature in result.columns:
                # Rolling mean
                result[f"{feature}_rolling_mean_7"] = result[feature].rolling(window=7).mean()
                result[f"{feature}_rolling_mean_14"] = result[feature].rolling(window=14).mean()
                
                # Rolling std
                result[f"{feature}_rolling_std_7"] = result[feature].rolling(window=7).std()
                result[f"{feature}_rolling_std_14"] = result[feature].rolling(window=14).std()
        
        # Drop NaN values resulting from lag and rolling features
        result = result.dropna()
        
        return result
    
    def _create_future_df(self, last_row: pd.Series) -> pd.DataFrame:
        """
        Create a DataFrame for future prediction based on the last known data.
        
        Args:
            last_row: Last row of the input DataFrame
            
        Returns:
            DataFrame prepared for future predictions
        """
        future_df = pd.DataFrame()
        
        # Initialize with lag values from the last row
        for feature in self.features:
            if feature in last_row.index:
                for lag in range(1, self.lookback_days + 1):
                    if lag == 1:
                        # For lag 1, use the actual value
                        future_df.loc[0, f"{feature}_lag_{lag}"] = last_row[feature]
                    else:
                        # For other lags, use the previous lag value from last row
                        lag_col = f"{feature}_lag_{lag-1}"
                        if lag_col in last_row.index:
                            future_df.loc[0, f"{feature}_lag_{lag}"] = last_row[lag_col]
        
        # Add other engineered features from the last row
        for col in last_row.index:
            if col.startswith(('day_of_week', 'rolling_mean', 'rolling_std')):
                future_df.loc[0, col] = last_row[col]
        
        return future_df
    
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for Random Forest model.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Tuple of features and target arrays
        """
        # Engineer features
        engineered_df = self._engineer_features(df)
        
        # Prepare for multiple days ahead forecasting
        X_list = []
        y_list = []
        
        # Create target for each forecast day
        for i in range(1, self.forecast_days + 1):
            # Target is the close price N days ahead
            engineered_df[f'target_{i}'] = engineered_df[self.target_feature].shift(-i)
        
        # Drop rows with NaN targets
        engineered_df = engineered_df.dropna()
        
        # Select features and targets
        feature_cols = [col for col in engineered_df.columns if col.startswith(tuple(f"{f}_lag" for f in self.features)) 
                       or col.startswith(('day_of_week', 'rolling_mean', 'rolling_std'))]
        
        X = engineered_df[feature_cols].values
        
        # Create separate target for each forecast day
        for i in range(1, self.forecast_days + 1):
            y = engineered_df[f'target_{i}'].values
            X_list.append(X)
            y_list.append(y)
        
        return X_list, y_list, feature_cols
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the Random Forest model.
        
        Args:
            df: DataFrame with price data
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training metrics
        """
        # Preprocess data
        X_list, y_list, feature_cols = self._preprocess_data(df)
        
        if not X_list or len(X_list[0]) == 0:
            raise ValueError("Not enough data for training")
        
        # Store feature columns for later use
        self.feature_cols = feature_cols
        
        # Train a model for each forecast day
        self.models = []
        metrics = []
        
        for day, (X, y) in enumerate(zip(X_list, y_list), 1):
            # Split into training and validation sets
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Create and train model for this forecast day
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42
            )
            
            self.logger.info(f"Training Random Forest model for day {day} with {len(X_train)} samples")
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, val_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, val_pred)
            
            self.logger.info(f"Day {day} model validation: RMSE={rmse:.4f}, R²={r2:.4f}")
            
            # Store model and metrics
            self.models.append(model)
            metrics.append({
                'day': day,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            })
        
        # Calculate average metrics
        avg_metrics = {
            'avg_mse': np.mean([m['mse'] for m in metrics]),
            'avg_rmse': np.mean([m['rmse'] for m in metrics]),
            'avg_r2': np.mean([m['r2'] for m in metrics]),
            'day_metrics': metrics
        }
        
        return avg_metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate price predictions for the next forecast_days.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with predictions
        """
        if not hasattr(self, 'models') or not self.models:
            raise ValueError("Models not trained or loaded")
        
        # Engineer features
        engineered_df = self._engineer_features(df)
        
        if len(engineered_df) == 0:
            raise ValueError("Not enough data after feature engineering")
        
        # Get the last row for prediction
        last_row = engineered_df.iloc[-1]
        
        predictions = []
        current_df = self._create_future_df(last_row)
        
        # Make predictions for each day
        for day, model in enumerate(self.models, 1):
            # Make prediction using the current features
            X = current_df[self.feature_cols].values.reshape(1, -1)
            pred = model.predict(X)[0]
            
            # Record prediction
            predictions.append(pred)
            
            # If we need to predict more days, update the features for the next day
            if day < len(self.models):
                # Shift lag features
                for feature in self.features:
                    for lag in range(self.lookback_days, 1, -1):
                        lag_col = f"{feature}_lag_{lag}"
                        prev_lag_col = f"{feature}_lag_{lag-1}"
                        if lag_col in current_df.columns and prev_lag_col in current_df.columns:
                            current_df.loc[0, lag_col] = current_df.loc[0, prev_lag_col]
                
                # Set the prediction as the new lag 1 value for the target feature
                current_df.loc[0, f"{self.target_feature}_lag_1"] = pred
        
        # Create a DataFrame with predictions
        last_date = df['timestamp'].iloc[-1]
        pred_dates = [last_date + timedelta(days=i+1) for i in range(self.forecast_days)]
        
        result = pd.DataFrame({
            'timestamp': pred_dates,
            'predicted_close': predictions
        })
        
        return result
    
    def save_model(self) -> str:
        """
        Save the trained model to disk.
        
        Returns:
            Path to the saved model
        """
        if not hasattr(self, 'models') or not self.models:
            raise ValueError("No models to save")
        
        # Create model directory
        model_dir = os.path.join(self.model_dir, self.model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save each day's model
        for day, model in enumerate(self.models, 1):
            model_path = os.path.join(model_dir, f"day_{day}.joblib")
            joblib.dump(model, model_path)
        
        # Save configuration and feature columns
        config_path = os.path.join(model_dir, "config.joblib")
        config = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'forecast_days': self.forecast_days,
            'features': self.features,
            'target_feature': self.target_feature,
            'lookback_days': self.lookback_days,
            'feature_cols': self.feature_cols
        }
        joblib.dump(config, config_path)
        
        self.logger.info(f"Models saved to {model_dir}")
        
        return model_dir
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model directory, or None to use default path
            
        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            # Use default path if not provided
            if model_path is None:
                model_path = os.path.join(self.model_dir, self.model_name)
            
            # Load configuration
            config_path = os.path.join(model_path, "config.joblib")
            if os.path.exists(config_path):
                config = joblib.load(config_path)
                self.n_estimators = config.get('n_estimators', self.n_estimators)
                self.max_depth = config.get('max_depth', self.max_depth)
                self.min_samples_split = config.get('min_samples_split', self.min_samples_split)
                self.min_samples_leaf = config.get('min_samples_leaf', self.min_samples_leaf)
                self.forecast_days = config.get('forecast_days', self.forecast_days)
                self.features = config.get('features', self.features)
                self.target_feature = config.get('target_feature', self.target_feature)
                self.lookback_days = config.get('lookback_days', self.lookback_days)
                self.feature_cols = config.get('feature_cols', [])
            
            # Load models for each day
            self.models = []
            for day in range(1, self.forecast_days + 1):
                model_file = os.path.join(model_path, f"day_{day}.joblib")
                if os.path.exists(model_file):
                    model = joblib.load(model_file)
                    self.models.append(model)
            
            if len(self.models) == 0:
                raise ValueError("No models found in directory")
            
            self.logger.info(f"Models loaded from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            return False
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not hasattr(self, 'models') or not self.models:
            raise ValueError("Models not trained or loaded")
        
        # Engineer features
        engineered_df = self._engineer_features(df)
        
        if len(engineered_df) < self.forecast_days:
            raise ValueError(f"Not enough data for evaluation. Need at least {self.forecast_days} data points after feature engineering.")
        
        results = []
        
        # For each point in the test set
        for i in range(len(engineered_df) - self.forecast_days):
            try:
                # Get the current row
                current_row = engineered_df.iloc[i]
                
                # Create features for prediction
                pred_df = self._create_future_df(current_row)
                
                # Make predictions for each day
                day_preds = []
                for day, model in enumerate(self.models, 1):
                    X = pred_df[self.feature_cols].values.reshape(1, -1)
                    pred = model.predict(X)[0]
                    day_preds.append(pred)
                    
                    # Update features for next day prediction if needed
                    if day < len(self.models):
                        # Shift lag features
                        for feature in self.features:
                            for lag in range(self.lookback_days, 1, -1):
                                lag_col = f"{feature}_lag_{lag}"
                                prev_lag_col = f"{feature}_lag_{lag-1}"
                                if lag_col in pred_df.columns and prev_lag_col in pred_df.columns:
                                    pred_df.loc[0, lag_col] = pred_df.loc[0, prev_lag_col]
                        
                        # Set prediction as new lag 1 value
                        pred_df.loc[0, f"{self.target_feature}_lag_1"] = pred
                
                # Get actual future values
                actual_values = df.iloc[i+1:i+self.forecast_days+1][self.target_feature].values
                
                # Compare predicted vs actual
                if len(actual_values) == len(day_preds):
                    results.append((actual_values, day_preds))
            
            except Exception as e:
                self.logger.warning(f"Error evaluating at index {i}: {str(e)}")
        
        if not results:
            raise ValueError("No valid evaluation results generated")
        
        # Calculate metrics across all predictions
        all_actual = np.concatenate([actual for actual, _ in results])
        all_pred = np.concatenate([pred for _, pred in results])
        
        mse = mean_squared_error(all_actual, all_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_actual, all_pred)
        
        # Calculate day-specific metrics
        day_metrics = []
        for day in range(self.forecast_days):
            day_actual = np.array([actual[day] for actual, _ in results if day < len(actual)])
            day_pred = np.array([pred[day] for _, pred in results if day < len(pred)])
            
            if len(day_actual) > 0:
                day_mse = mean_squared_error(day_actual, day_pred)
                day_rmse = np.sqrt(day_mse)
                day_mae = mean_absolute_error(day_actual, day_pred)
                day_mape = np.mean(np.abs((day_actual - day_pred) / day_actual)) * 100
                
                day_metrics.append({
                    'day': day + 1,
                    'mse': day_mse,
                    'rmse': day_rmse,
                    'mae': day_mae,
                    'mape': day_mape
                })
        
        # Overall MAPE
        mape = np.mean(np.abs((all_actual - all_pred) / all_actual)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'day_metrics': day_metrics
        }