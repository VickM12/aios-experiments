"""
Anomaly Prediction Module
Predicts when anomalies might occur based on historical patterns
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class AnomalyPredictor:
    """Predicts future anomalies based on historical telemetry patterns"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features for prediction"""
        df = df.copy()
        
        # Convert timestamp to datetime if it's a string
        if 'timestamp' in df.columns:
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create time features
        if 'timestamp' in df.columns:
            df['time_index'] = range(len(df))
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['second'] = df['timestamp'].dt.second
            df['time_of_day'] = df['hour'] * 3600 + df['minute'] * 60 + df['second']
        else:
            df['time_index'] = range(len(df))
            df['hour'] = 0
            df['minute'] = 0
            df['second'] = 0
            df['time_of_day'] = 0
        
        # Create lag features (previous values)
        for col in ['cpu_percent', 'memory_percent', 'disk_percent']:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1).fillna(df[col].iloc[0])
                df[f'{col}_lag2'] = df[col].shift(2).fillna(df[col].iloc[0])
                df[f'{col}_lag3'] = df[col].shift(3).fillna(df[col].iloc[0])
        
        # Moving averages
        for col in ['cpu_percent', 'memory_percent']:
            if col in df.columns:
                df[f'{col}_ma3'] = df[col].rolling(window=3, min_periods=1).mean()
                df[f'{col}_ma5'] = df[col].rolling(window=5, min_periods=1).mean()
        
        # Rate of change
        for col in ['cpu_percent', 'memory_percent']:
            if col in df.columns:
                df[f'{col}_roc'] = df[col].diff().fillna(0)
        
        return df
    
    def train_forecast_models(self, telemetry_data: List[Dict[str, Any]], 
                             anomaly_indices: List[int] = None) -> Dict[str, Any]:
        """Train models to forecast future metric values"""
        if len(telemetry_data) < 20:
            return {
                'success': False,
                'message': 'Insufficient data for training (need at least 20 samples)'
            }
        
        # Prepare features
        from ai_analyzer import TelemetryAnalyzer
        analyzer = TelemetryAnalyzer()
        df = analyzer.prepare_features(telemetry_data)
        
        # Add time features
        df = self.prepare_time_features(df)
        
        # Prepare training data (remove NaN rows)
        df_clean = df.dropna()
        if len(df_clean) < 10:
            return {
                'success': False,
                'message': 'Insufficient clean data after feature engineering'
            }
        
        # Train models for key metrics
        key_metrics = ['cpu_percent', 'memory_percent', 'disk_percent']
        available_metrics = [m for m in key_metrics if m in df_clean.columns]
        
        feature_cols = [col for col in df_clean.columns 
                       if col not in available_metrics + ['timestamp']]
        
        X = df_clean[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        self.models = {}
        model_performance = {}
        
        for metric in available_metrics:
            y = df_clean[metric].values
            
            # Use RandomForest for better non-linear patterns
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
            model.fit(X_scaled, y)
            
            # Calculate R² score
            y_pred = model.predict(X_scaled)
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            self.models[metric] = model
            model_performance[metric] = {
                'r2_score': float(r2),
                'mae': float(np.mean(np.abs(y - y_pred))),
                'rmse': float(np.sqrt(np.mean((y - y_pred) ** 2)))
            }
        
        self.is_trained = True
        
        return {
            'success': True,
            'models_trained': list(self.models.keys()),
            'performance': model_performance,
            'feature_count': len(feature_cols)
        }
    
    def predict_future_values(self, telemetry_data: List[Dict[str, Any]], 
                             steps_ahead: int = 10) -> Dict[str, Any]:
        """Predict future metric values"""
        if not self.is_trained:
            return {
                'success': False,
                'message': 'Models not trained. Call train_forecast_models() first.'
            }
        
        if len(telemetry_data) < 5:
            return {
                'success': False,
                'message': 'Insufficient data for prediction'
            }
        
        # Prepare features from recent data
        from ai_analyzer import TelemetryAnalyzer
        analyzer = TelemetryAnalyzer()
        df = analyzer.prepare_features(telemetry_data)
        df = self.prepare_time_features(df)
        
        # Get the last row as starting point
        df_clean = df.dropna()
        if len(df_clean) == 0:
            return {'success': False, 'message': 'No valid data for prediction'}
        
        last_row = df_clean.iloc[-1:].copy()
        feature_cols = [col for col in df_clean.columns 
                       if col not in ['cpu_percent', 'memory_percent', 'disk_percent', 'timestamp']]
        
        predictions = []
        current_row = last_row.copy()
        
        # Get last timestamp
        if 'timestamp' in df.columns:
            last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
        else:
            last_timestamp = datetime.now()
        
        # Estimate interval from data
        if len(telemetry_data) > 1:
            if 'timestamp' in telemetry_data[-1] and 'timestamp' in telemetry_data[-2]:
                try:
                    t1 = pd.to_datetime(telemetry_data[-1]['timestamp'])
                    t2 = pd.to_datetime(telemetry_data[-2]['timestamp'])
                    interval = (t1 - t2).total_seconds()
                except:
                    interval = 1.0
            else:
                interval = 1.0
        else:
            interval = 1.0
        
        for step in range(1, steps_ahead + 1):
            # Prepare features for this step
            X_step = current_row[feature_cols].values
            X_step_scaled = self.scaler.transform(X_step)
            
            # Predict each metric
            step_predictions = {}
            step_timestamp = last_timestamp + timedelta(seconds=interval * step)
            
            for metric, model in self.models.items():
                pred_value = model.predict(X_step_scaled)[0]
                step_predictions[metric] = float(pred_value)
            
            # Update current_row for next iteration (simple approach)
            for metric in step_predictions:
                if metric in current_row.columns:
                    current_row[metric] = step_predictions[metric]
                    # Update lag features
                    if f'{metric}_lag1' in current_row.columns:
                        current_row[f'{metric}_lag2'] = current_row[f'{metric}_lag1']
                        current_row[f'{metric}_lag1'] = step_predictions[metric]
            
            # Update time features
            if 'timestamp' in current_row.columns:
                current_row['timestamp'] = step_timestamp
            if 'hour' in current_row.columns:
                current_row['hour'] = step_timestamp.hour
                current_row['minute'] = step_timestamp.minute
                current_row['second'] = step_timestamp.second
                current_row['time_of_day'] = step_timestamp.hour * 3600 + step_timestamp.minute * 60 + step_timestamp.second
            
            predictions.append({
                'step': step,
                'timestamp': step_timestamp.isoformat(),
                'predictions': step_predictions
            })
        
        return {
            'success': True,
            'predictions': predictions,
            'steps_ahead': steps_ahead
        }
    
    def predict_anomaly_likelihood(self, telemetry_data: List[Dict[str, Any]], 
                                   anomaly_detector, 
                                   steps_ahead: int = 10) -> Dict[str, Any]:
        """Predict likelihood of future anomalies"""
        if not self.is_trained:
            return {
                'success': False,
                'message': 'Models not trained. Call train_forecast_models() first.'
            }
        
        # Get future value predictions
        future_predictions = self.predict_future_values(telemetry_data, steps_ahead)
        
        if not future_predictions.get('success'):
            return future_predictions
        
        # Prepare features for anomaly detection
        from ai_analyzer import TelemetryAnalyzer
        analyzer = TelemetryAnalyzer()
        
        # Get historical data for context
        df_historical = analyzer.prepare_features(telemetry_data)
        
        # Create feature vectors for future predictions
        anomaly_predictions = []
        
        for pred_step in future_predictions['predictions']:
            # Create a synthetic data point from predictions
            synthetic_data = {
                'cpu_percent': pred_step['predictions'].get('cpu_percent', 0),
                'memory_percent': pred_step['predictions'].get('memory_percent', 0),
                'disk_percent': pred_step['predictions'].get('disk_percent', 0),
            }
            
            # Use historical means for missing features
            for col in df_historical.columns:
                if col not in synthetic_data:
                    synthetic_data[col] = df_historical[col].mean()
            
            # Create feature vector matching historical format
            feature_vector = pd.DataFrame([synthetic_data])
            
            # Predict anomaly score
            try:
                anomaly_score = anomaly_detector.score_samples(feature_vector)[0]
                is_anomaly = anomaly_detector.predict(feature_vector)[0] == -1
                
                anomaly_predictions.append({
                    'step': pred_step['step'],
                    'timestamp': pred_step['timestamp'],
                    'anomaly_score': float(anomaly_score),
                    'is_anomaly': bool(is_anomaly),
                    'likelihood': 'high' if anomaly_score < -0.3 else 'medium' if anomaly_score < 0 else 'low',
                    'predicted_metrics': pred_step['predictions']
                })
            except Exception as e:
                anomaly_predictions.append({
                    'step': pred_step['step'],
                    'timestamp': pred_step['timestamp'],
                    'error': str(e)
                })
        
        # Calculate summary statistics
        predicted_anomalies = [p for p in anomaly_predictions if p.get('is_anomaly', False)]
        
        return {
            'success': True,
            'anomaly_predictions': anomaly_predictions,
            'summary': {
                'total_steps': steps_ahead,
                'predicted_anomalies': len(predicted_anomalies),
                'anomaly_percentage': (len(predicted_anomalies) / steps_ahead * 100) if steps_ahead > 0 else 0,
                'high_risk_steps': [p['step'] for p in anomaly_predictions if p.get('likelihood') == 'high']
            }
        }
    
    def analyze_anomaly_patterns(self, telemetry_data: List[Dict[str, Any]], 
                                anomaly_indices: List[int]) -> Dict[str, Any]:
        """Analyze patterns in when anomalies occur"""
        if not anomaly_indices or len(anomaly_indices) == 0:
            return {
                'patterns_found': False,
                'message': 'No anomalies to analyze'
            }
        
        # Prepare time-based features
        from ai_analyzer import TelemetryAnalyzer
        analyzer = TelemetryAnalyzer()
        df = analyzer.prepare_features(telemetry_data)
        df = self.prepare_time_features(df)
        
        # Analyze timing patterns
        anomaly_times = []
        for idx in anomaly_indices:
            if idx < len(df):
                row = df.iloc[idx]
                if 'timestamp' in row:
                    try:
                        ts = pd.to_datetime(row['timestamp'])
                        anomaly_times.append({
                            'hour': ts.hour,
                            'minute': ts.minute,
                            'day_of_week': ts.dayofweek,
                            'index': idx
                        })
                    except:
                        pass
        
        if not anomaly_times:
            return {
                'patterns_found': False,
                'message': 'Could not extract timing information'
            }
        
        # Find patterns
        hours = [t['hour'] for t in anomaly_times]
        minutes = [t['minute'] for t in anomaly_times]
        
        # Most common hour
        from collections import Counter
        hour_counts = Counter(hours)
        most_common_hour = hour_counts.most_common(1)[0] if hour_counts else None
        
        # Calculate intervals between anomalies
        intervals = []
        sorted_indices = sorted(anomaly_indices)
        for i in range(1, len(sorted_indices)):
            intervals.append(sorted_indices[i] - sorted_indices[i-1])
        
        avg_interval = np.mean(intervals) if intervals else None
        
        return {
            'patterns_found': True,
            'anomaly_count': len(anomaly_indices),
            'most_common_hour': most_common_hour[0] if most_common_hour else None,
            'hour_frequency': dict(hour_counts),
            'average_interval': float(avg_interval) if avg_interval else None,
            'min_interval': float(min(intervals)) if intervals else None,
            'max_interval': float(max(intervals)) if intervals else None,
        }

