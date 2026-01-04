"""
AI Analyzer Module
Analyzes telemetry data using machine learning and statistical methods
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class TelemetryAnalyzer:
    """AI-powered analyzer for telemetry data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination='auto', random_state=42)
        self.pca = PCA(n_components=2)
        self.is_fitted = False
        self.training_data = None
        
    def prepare_features(self, telemetry_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract and prepare features from telemetry data"""
        features = []
        
        for data in telemetry_data:
            feature_vector = {
                'cpu_percent': data.get('cpu', {}).get('cpu_percent', 0),
                'cpu_freq': data.get('cpu', {}).get('cpu_freq', {}).get('current', 0) or 0,
                'memory_percent': data.get('memory', {}).get('virtual_memory', {}).get('percent', 0),
                'memory_used_gb': (data.get('memory', {}).get('virtual_memory', {}).get('used', 0) or 0) / (1024**3),
                'swap_percent': data.get('memory', {}).get('swap_memory', {}).get('percent', 0),
                'disk_percent': data.get('disk', {}).get('disk_usage', {}).get('percent', 0),
                'disk_read_mb': (data.get('disk', {}).get('disk_io', {}).get('read_bytes', 0) or 0) / (1024**2),
                'disk_write_mb': (data.get('disk', {}).get('disk_io', {}).get('write_bytes', 0) or 0) / (1024**2),
                'network_sent_mb': (data.get('network', {}).get('network_io', {}).get('bytes_sent', 0) or 0) / (1024**2),
                'network_recv_mb': (data.get('network', {}).get('network_io', {}).get('bytes_recv', 0) or 0) / (1024**2),
                'process_count': data.get('processes', {}).get('total_processes', 0),
            }
            
            # Extract temperature data if available
            temp_data = data.get('temperature', {})
            if temp_data and 'error' not in temp_data:
                # Get average temperature from all sensors
                all_temps = []
                for sensor_name, entries in temp_data.items():
                    for entry in entries:
                        if 'current' in entry:
                            all_temps.append(entry['current'])
                feature_vector['avg_temperature'] = np.mean(all_temps) if all_temps else 0
            else:
                feature_vector['avg_temperature'] = 0
            
            features.append(feature_vector)
        
        return pd.DataFrame(features)
    
    def detect_anomalies(self, telemetry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect anomalies in telemetry data using Isolation Forest"""
        if len(telemetry_data) < 10:
            return {
                'anomalies_detected': 0,
                'anomaly_indices': [],
                'message': 'Insufficient data for anomaly detection (need at least 10 samples)'
            }
        
        df = self.prepare_features(telemetry_data)
        
        # Use score-based thresholding instead of fixed contamination
        # Fit on a subset of data if we have enough, otherwise use all
        if len(df) >= 20:
            # Use first 80% for training, last 20% for detection
            train_size = int(len(df) * 0.8)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            
            # Fit the anomaly detector on training data
            if not self.is_fitted or self.training_data is None:
                self.anomaly_detector.fit(train_df)
                self.training_data = train_df
                self.is_fitted = True
            
            # Get scores for test data
            test_scores = self.anomaly_detector.score_samples(test_df)
            # Use a threshold based on score distribution (lower = more anomalous)
            score_threshold = np.percentile(test_scores, 10)  # Bottom 10% are anomalies
            anomaly_mask = test_scores < score_threshold
            anomaly_indices = [train_size + i for i, is_anom in enumerate(anomaly_mask) if is_anom]
            
            # Get scores for all data for display
            train_scores = self.anomaly_detector.score_samples(train_df)
            all_scores = np.concatenate([train_scores, test_scores])
        else:
            # Not enough data - use adaptive thresholding
            if not self.is_fitted:
                self.anomaly_detector.fit(df)
                self.training_data = df
                self.is_fitted = True
            
            scores = self.anomaly_detector.score_samples(df)
            # Use percentile-based threshold (more conservative)
            score_threshold = np.percentile(scores, 5)  # Bottom 5% are anomalies
            anomaly_mask = scores < score_threshold
            anomaly_indices = [i for i, is_anom in enumerate(anomaly_mask) if is_anom]
            all_scores = scores
        
        # Calculate statistics for comparison
        df_mean = df.mean()
        df_std = df.std()
        
        # Analyze what makes each anomaly unusual
        anomaly_details = []
        for idx in anomaly_indices:
            anomaly_row = df.iloc[idx]
            deviations = {}
            
            # Calculate how many standard deviations each feature is from the mean
            for col in df.columns:
                mean_val = df_mean[col]
                std_val = df_std[col]
                actual_val = anomaly_row[col]
                
                if std_val > 0:  # Avoid division by zero
                    z_score = (actual_val - mean_val) / std_val
                    deviations[col] = {
                        'value': float(actual_val),
                        'mean': float(mean_val),
                        'std': float(std_val),
                        'z_score': float(z_score),
                        'deviation_percent': float((actual_val - mean_val) / mean_val * 100) if mean_val != 0 else 0
                    }
                else:
                    deviations[col] = {
                        'value': float(actual_val),
                        'mean': float(mean_val),
                        'std': 0,
                        'z_score': 0,
                        'deviation_percent': 0
                    }
            
            # Identify top contributing factors (highest absolute z-scores)
            contributing_factors = sorted(
                [(k, v['z_score']) for k, v in deviations.items()],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]  # Top 5 contributing factors
            
            anomaly_details.append({
                'index': idx,
                'timestamp': telemetry_data[idx].get('timestamp'),
                'score': float(all_scores[idx]),
                'features': anomaly_row.to_dict(),
                'deviations': deviations,
                'top_factors': [
                    {
                        'metric': factor[0],
                        'z_score': factor[1],
                        'deviation': deviations[factor[0]]['deviation_percent']
                    }
                    for factor in contributing_factors
                ]
            })
        
        return {
            'anomalies_detected': len(anomaly_indices),
            'anomaly_indices': anomaly_indices,
            'anomaly_scores': all_scores.tolist(),
            'anomaly_percentage': (len(anomaly_indices) / len(telemetry_data)) * 100,
            'details': anomaly_details
        }
    
    def cluster_analysis(self, telemetry_data: List[Dict[str, Any]], n_clusters: int = 3) -> Dict[str, Any]:
        """Perform clustering analysis to identify patterns"""
        if len(telemetry_data) < n_clusters:
            return {
                'clusters': [],
                'message': f'Insufficient data for clustering (need at least {n_clusters} samples)'
            }
        
        df = self.prepare_features(telemetry_data)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(df)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Analyze clusters
        clusters = []
        for i in range(n_clusters):
            cluster_indices = [j for j, label in enumerate(cluster_labels) if label == i]
            cluster_data = df.iloc[cluster_indices]
            
            clusters.append({
                'cluster_id': i,
                'size': len(cluster_indices),
                'centroid': kmeans.cluster_centers_[i].tolist(),
                'characteristics': {
                    'avg_cpu': float(cluster_data['cpu_percent'].mean()),
                    'avg_memory': float(cluster_data['memory_percent'].mean()),
                    'avg_disk': float(cluster_data['disk_percent'].mean()),
                },
                'indices': cluster_indices,
            })
        
        return {
            'n_clusters': n_clusters,
            'clusters': clusters,
            'cluster_labels': cluster_labels.tolist(),
        }
    
    def trend_analysis(self, telemetry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in telemetry data"""
        if len(telemetry_data) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        df = self.prepare_features(telemetry_data)
        
        trends = {}
        for column in df.columns:
            values = df[column].values
            if len(values) > 1:
                # Calculate linear trend
                x = np.arange(len(values))
                coeffs = np.polyfit(x, values, 1)
                slope = coeffs[0]
                
                # Calculate rate of change
                rate_of_change = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                
                trends[column] = {
                    'slope': float(slope),
                    'rate_of_change_percent': float(rate_of_change),
                    'current_value': float(values[-1]),
                    'initial_value': float(values[0]),
                    'trend': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                }
        
        return {
            'trends': trends,
            'summary': {
                'metrics_analyzed': len(trends),
                'increasing_metrics': sum(1 for t in trends.values() if t['trend'] == 'increasing'),
                'decreasing_metrics': sum(1 for t in trends.values() if t['trend'] == 'decreasing'),
                'stable_metrics': sum(1 for t in trends.values() if t['trend'] == 'stable'),
            }
        }
    
    def performance_insights(self, telemetry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance insights and recommendations"""
        if not telemetry_data:
            return {'message': 'No data available for analysis'}
        
        latest = telemetry_data[-1]
        df = self.prepare_features(telemetry_data)
        
        insights = {
            'current_status': {},
            'warnings': [],
            'recommendations': [],
        }
        
        # CPU Analysis
        cpu_percent = latest.get('cpu', {}).get('cpu_percent', 0)
        insights['current_status']['cpu'] = {
            'usage': cpu_percent,
            'status': 'high' if cpu_percent > 80 else 'moderate' if cpu_percent > 50 else 'normal'
        }
        if cpu_percent > 90:
            insights['warnings'].append('CPU usage is critically high (>90%)')
            insights['recommendations'].append('Consider closing resource-intensive applications or processes')
        
        # Memory Analysis
        mem_percent = latest.get('memory', {}).get('virtual_memory', {}).get('percent', 0)
        insights['current_status']['memory'] = {
            'usage': mem_percent,
            'status': 'high' if mem_percent > 80 else 'moderate' if mem_percent > 50 else 'normal'
        }
        if mem_percent > 90:
            insights['warnings'].append('Memory usage is critically high (>90%)')
            insights['recommendations'].append('Consider freeing up memory or closing unused applications')
        
        # Disk Analysis
        disk_percent = latest.get('disk', {}).get('disk_usage', {}).get('percent', 0)
        insights['current_status']['disk'] = {
            'usage': disk_percent,
            'status': 'high' if disk_percent > 80 else 'moderate' if disk_percent > 50 else 'normal'
        }
        if disk_percent > 90:
            insights['warnings'].append('Disk space is critically low (>90% used)')
            insights['recommendations'].append('Consider cleaning up disk space or removing unused files')
        
        # Temperature Analysis
        temp_data = latest.get('temperature', {})
        if temp_data and 'error' not in temp_data:
            all_temps = []
            for sensor_name, entries in temp_data.items():
                for entry in entries:
                    if 'current' in entry:
                        all_temps.append(entry['current'])
            
            if all_temps:
                avg_temp = np.mean(all_temps)
                insights['current_status']['temperature'] = {
                    'average': float(avg_temp),
                    'status': 'high' if avg_temp > 70 else 'moderate' if avg_temp > 60 else 'normal'
                }
                if avg_temp > 80:
                    insights['warnings'].append('System temperature is high (>80°C)')
                    insights['recommendations'].append('Check cooling system and ensure proper ventilation')
        
        # Network Analysis
        network_io = latest.get('network', {}).get('network_io', {})
        if network_io:
            sent_mb = network_io.get('bytes_sent', 0) / (1024**2)
            recv_mb = network_io.get('bytes_recv', 0) / (1024**2)
            connections = latest.get('network', {}).get('network_connections', 0)
            
            insights['current_status']['network'] = {
                'bytes_sent_mb': float(sent_mb),
                'bytes_recv_mb': float(recv_mb),
                'connections': int(connections),
                'status': 'active' if connections > 0 else 'idle'
            }
            
            # Check for network errors
            if network_io.get('errin', 0) > 100 or network_io.get('errout', 0) > 100:
                insights['warnings'].append(f'Network errors detected: {network_io.get("errin", 0)} in, {network_io.get("errout", 0)} out')
        
        # Power Analysis
        power_data = latest.get('power', {})
        if power_data:
            power_info = {}
            
            # GPU power
            if power_data.get('gpu'):
                gpu_powers = [gpu.get('power_draw_watts') for gpu in power_data['gpu'] if gpu.get('power_draw_watts')]
                if gpu_powers:
                    power_info['gpu_watts'] = float(np.mean(gpu_powers))
            
            # CPU RAPL power
            if power_data.get('rapl'):
                for domain, pwr in power_data['rapl'].items():
                    if 'package' in domain.lower():
                        power_info['cpu_energy_joules'] = float(pwr.get('energy_joules', 0))
            
            if power_info:
                insights['current_status']['power'] = power_info
        
        # Battery Analysis
        battery_data = latest.get('battery', {})
        if battery_data and 'error' not in battery_data:
            battery_percent = battery_data.get('percent', None)
            if battery_percent is not None:
                insights['current_status']['battery'] = {
                    'percent': float(battery_percent),
                    'power_plugged': battery_data.get('power_plugged', False),
                    'status': 'charging' if battery_data.get('power_plugged') else 'discharging'
                }
                
                if battery_percent < 20:
                    insights['warnings'].append('Battery level is critically low (<20%)')
                    insights['recommendations'].append('Consider plugging in the device or reducing power consumption')
                elif battery_percent < 10:
                    insights['warnings'].append('Battery level is extremely low (<10%)')
                    insights['recommendations'].append('Immediately plug in the device to prevent data loss')
        
        # Process Analysis
        process_data = latest.get('processes', {})
        if process_data:
            total_processes = process_data.get('total_processes', 0)
            top_processes = process_data.get('top_processes', [])
            
            # Find processes using significant resources
            high_cpu_processes = [p for p in top_processes if (p.get('cpu_percent', 0) or 0) > 10]
            high_mem_processes = [p for p in top_processes if (p.get('memory_percent', 0) or 0) > 5]
            
            insights['current_status']['processes'] = {
                'total': int(total_processes),
                'high_cpu_count': len(high_cpu_processes),
                'high_memory_count': len(high_mem_processes),
                'top_processes': top_processes[:5]  # Top 5 for reference
            }
            
            if high_cpu_processes:
                top_cpu_proc = high_cpu_processes[0]
                insights['warnings'].append(f"High CPU usage detected: {top_cpu_proc.get('name', 'unknown')} using {top_cpu_proc.get('cpu_percent', 0):.1f}%")
                insights['recommendations'].append(f"Consider investigating {top_cpu_proc.get('name', 'unknown')} (PID {top_cpu_proc.get('pid', 'N/A')}) if CPU usage is consistently high")
            
            if high_mem_processes:
                top_mem_proc = high_mem_processes[0]
                if (top_mem_proc.get('memory_percent', 0) or 0) > 10:
                    insights['warnings'].append(f"High memory usage detected: {top_mem_proc.get('name', 'unknown')} using {top_mem_proc.get('memory_percent', 0):.2f}%")
                    insights['recommendations'].append(f"Consider investigating {top_mem_proc.get('name', 'unknown')} (PID {top_mem_proc.get('pid', 'N/A')}) if memory usage is concerning")
        
        # Statistical Analysis
        if len(telemetry_data) > 1:
            insights['statistics'] = {
                'cpu_mean': float(df['cpu_percent'].mean()),
                'cpu_std': float(df['cpu_percent'].std()),
                'memory_mean': float(df['memory_percent'].mean()),
                'memory_std': float(df['memory_percent'].std()),
            }
        
        return insights
    
    def comprehensive_analysis(self, telemetry_data: List[Dict[str, Any]], 
                                include_predictions: bool = False,
                                prediction_steps: int = 10) -> Dict[str, Any]:
        """Perform comprehensive analysis combining all methods"""
        result = {
            'anomaly_detection': self.detect_anomalies(telemetry_data),
            'clustering': self.cluster_analysis(telemetry_data),
            'trend_analysis': self.trend_analysis(telemetry_data),
            'performance_insights': self.performance_insights(telemetry_data),
            'data_points_analyzed': len(telemetry_data),
        }
        
        # Add predictions if requested
        if include_predictions and len(telemetry_data) >= 20:
            try:
                from anomaly_predictor import AnomalyPredictor
                predictor = AnomalyPredictor()
                
                # Train models
                anomaly_indices = result['anomaly_detection'].get('anomaly_indices', [])
                train_result = predictor.train_forecast_models(telemetry_data, anomaly_indices)
                
                if train_result.get('success'):
                    # Predict future anomalies
                    anomaly_pred = predictor.predict_anomaly_likelihood(
                        telemetry_data, 
                        self.anomaly_detector,
                        steps_ahead=prediction_steps
                    )
                    
                    # Analyze patterns
                    patterns = predictor.analyze_anomaly_patterns(telemetry_data, anomaly_indices)
                    
                    result['predictions'] = {
                        'forecast_training': train_result,
                        'anomaly_predictions': anomaly_pred,
                        'anomaly_patterns': patterns
                    }
            except Exception as e:
                result['predictions'] = {
                    'error': f'Prediction failed: {str(e)}'
                }
        
        return result

