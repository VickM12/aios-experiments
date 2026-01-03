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
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.pca = PCA(n_components=2)
        self.is_fitted = False
        
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
        
        # Fit the anomaly detector
        if not self.is_fitted:
            self.anomaly_detector.fit(df)
            self.is_fitted = True
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = self.anomaly_detector.predict(df)
        anomaly_indices = [i for i, pred in enumerate(predictions) if pred == -1]
        
        # Calculate anomaly scores
        scores = self.anomaly_detector.score_samples(df)
        
        return {
            'anomalies_detected': len(anomaly_indices),
            'anomaly_indices': anomaly_indices,
            'anomaly_scores': scores.tolist(),
            'anomaly_percentage': (len(anomaly_indices) / len(telemetry_data)) * 100,
            'details': [
                {
                    'index': idx,
                    'timestamp': telemetry_data[idx].get('timestamp'),
                    'score': float(scores[idx]),
                    'features': df.iloc[idx].to_dict()
                }
                for idx in anomaly_indices
            ]
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
        
        # Statistical Analysis
        if len(telemetry_data) > 1:
            insights['statistics'] = {
                'cpu_mean': float(df['cpu_percent'].mean()),
                'cpu_std': float(df['cpu_percent'].std()),
                'memory_mean': float(df['memory_percent'].mean()),
                'memory_std': float(df['memory_percent'].std()),
            }
        
        return insights
    
    def comprehensive_analysis(self, telemetry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive analysis combining all methods"""
        return {
            'anomaly_detection': self.detect_anomalies(telemetry_data),
            'clustering': self.cluster_analysis(telemetry_data),
            'trend_analysis': self.trend_analysis(telemetry_data),
            'performance_insights': self.performance_insights(telemetry_data),
            'data_points_analyzed': len(telemetry_data),
        }

