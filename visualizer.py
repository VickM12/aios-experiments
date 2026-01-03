"""
Data Visualization Module
Creates visualizations of telemetry data and analysis results
"""
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any
import numpy as np


class TelemetryVisualizer:
    """Creates visualizations for telemetry data"""
    
    def __init__(self):
        self.fig_size = (12, 8)
    
    def prepare_dataframe(self, telemetry_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert telemetry data to pandas DataFrame"""
        records = []
        for data in telemetry_data:
            record = {
                'timestamp': data.get('timestamp', ''),
                'cpu_percent': data.get('cpu', {}).get('cpu_percent', 0),
                'memory_percent': data.get('memory', {}).get('virtual_memory', {}).get('percent', 0),
                'disk_percent': data.get('disk', {}).get('disk_usage', {}).get('percent', 0),
                'swap_percent': data.get('memory', {}).get('swap_memory', {}).get('percent', 0),
            }
            
            # Extract temperature if available
            temp_data = data.get('temperature', {})
            if temp_data and 'error' not in temp_data:
                all_temps = []
                for sensor_name, entries in temp_data.items():
                    for entry in entries:
                        if 'current' in entry:
                            all_temps.append(entry['current'])
                record['temperature'] = np.mean(all_temps) if all_temps else None
            else:
                record['temperature'] = None
            
            records.append(record)
        
        df = pd.DataFrame(records)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def plot_basic_metrics(self, telemetry_data: List[Dict[str, Any]], save_path: str = None):
        """Create basic metrics plot"""
        df = self.prepare_dataframe(telemetry_data)
        
        fig, axes = plt.subplots(2, 2, figsize=self.fig_size)
        fig.suptitle('System Telemetry Metrics', fontsize=16, fontweight='bold')
        
        # CPU Usage
        axes[0, 0].plot(df['timestamp'], df['cpu_percent'], color='blue', linewidth=2)
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_ylabel('Percentage')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].fill_between(df['timestamp'], df['cpu_percent'], alpha=0.3, color='blue')
        
        # Memory Usage
        axes[0, 1].plot(df['timestamp'], df['memory_percent'], color='green', linewidth=2)
        axes[0, 1].set_title('Memory Usage (%)')
        axes[0, 1].set_ylabel('Percentage')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].fill_between(df['timestamp'], df['memory_percent'], alpha=0.3, color='green')
        
        # Disk Usage
        axes[1, 0].plot(df['timestamp'], df['disk_percent'], color='orange', linewidth=2)
        axes[1, 0].set_title('Disk Usage (%)')
        axes[1, 0].set_ylabel('Percentage')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].fill_between(df['timestamp'], df['disk_percent'], alpha=0.3, color='orange')
        
        # Temperature (if available)
        if df['temperature'].notna().any():
            axes[1, 1].plot(df['timestamp'], df['temperature'], color='red', linewidth=2)
            axes[1, 1].set_title('Temperature (°C)')
            axes[1, 1].set_ylabel('Temperature')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].fill_between(df['timestamp'], df['temperature'], alpha=0.3, color='red')
        else:
            axes[1, 1].text(0.5, 0.5, 'Temperature data\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Temperature (°C)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_interactive_dashboard(self, telemetry_data: List[Dict[str, Any]], 
                                   analysis: Dict[str, Any] = None, 
                                   save_path: str = 'telemetry_dashboard.html'):
        """Create interactive Plotly dashboard"""
        df = self.prepare_dataframe(telemetry_data)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Disk Usage', 
                          'Temperature', 'Resource Overview', 'Trend Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.12
        )
        
        # CPU Usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cpu_percent'], 
                      mode='lines', name='CPU %', line=dict(color='blue', width=2),
                      fill='tozeroy'),
            row=1, col=1
        )
        
        # Memory Usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['memory_percent'], 
                      mode='lines', name='Memory %', line=dict(color='green', width=2),
                      fill='tozeroy'),
            row=1, col=2
        )
        
        # Disk Usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['disk_percent'], 
                      mode='lines', name='Disk %', line=dict(color='orange', width=2),
                      fill='tozeroy'),
            row=2, col=1
        )
        
        # Temperature
        if df['temperature'].notna().any():
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['temperature'], 
                          mode='lines', name='Temperature', line=dict(color='red', width=2),
                          fill='tozeroy'),
                row=2, col=2
            )
        
        # Resource Overview (current values)
        latest = telemetry_data[-1] if telemetry_data else {}
        resources = ['CPU', 'Memory', 'Disk']
        values = [
            latest.get('cpu', {}).get('cpu_percent', 0),
            latest.get('memory', {}).get('virtual_memory', {}).get('percent', 0),
            latest.get('disk', {}).get('disk_usage', {}).get('percent', 0),
        ]
        
        fig.add_trace(
            go.Bar(x=resources, y=values, name='Current Usage %',
                  marker=dict(color=['blue', 'green', 'orange'])),
            row=3, col=1
        )
        
        # Add anomaly markers if available
        if analysis and 'anomaly_detection' in analysis:
            anomaly_indices = analysis['anomaly_detection'].get('anomaly_indices', [])
            if anomaly_indices:
                anomaly_times = [df.iloc[idx]['timestamp'] for idx in anomaly_indices if idx < len(df)]
                anomaly_cpu = [df.iloc[idx]['cpu_percent'] for idx in anomaly_indices if idx < len(df)]
                
                fig.add_trace(
                    go.Scatter(x=anomaly_times, y=anomaly_cpu, 
                              mode='markers', name='Anomalies',
                              marker=dict(color='red', size=10, symbol='x')),
                    row=1, col=1
                )
        
        # Update layout
        fig.update_layout(
            title_text='AI Telemetry Dashboard',
            title_x=0.5,
            height=1000,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=1)
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
        fig.update_yaxes(title_text="Percentage (%)", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)
        fig.update_yaxes(title_text="Usage (%)", row=3, col=1)
        
        # Save or show
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")
        return fig
    
    def plot_anomaly_analysis(self, telemetry_data: List[Dict[str, Any]], 
                             analysis: Dict[str, Any], save_path: str = None):
        """Visualize anomaly detection results"""
        if 'anomaly_detection' not in analysis:
            print("No anomaly detection data available")
            return
        
        df = self.prepare_dataframe(telemetry_data)
        anomaly_data = analysis['anomaly_detection']
        anomaly_indices = anomaly_data.get('anomaly_indices', [])
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Anomaly Detection Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Time series with anomalies marked
        axes[0].plot(df['timestamp'], df['cpu_percent'], 'b-', label='CPU %', linewidth=2)
        axes[0].plot(df['timestamp'], df['memory_percent'], 'g-', label='Memory %', linewidth=2)
        
        if anomaly_indices:
            anomaly_times = [df.iloc[idx]['timestamp'] for idx in anomaly_indices if idx < len(df)]
            anomaly_cpu = [df.iloc[idx]['cpu_percent'] for idx in anomaly_indices if idx < len(df)]
            axes[0].scatter(anomaly_times, anomaly_cpu, color='red', s=100, 
                           marker='x', label='Anomalies', zorder=5)
        
        axes[0].set_title('System Metrics with Anomaly Markers')
        axes[0].set_ylabel('Percentage (%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Anomaly scores
        if 'anomaly_scores' in anomaly_data:
            scores = anomaly_data['anomaly_scores']
            axes[1].plot(df['timestamp'], scores, 'r-', linewidth=2)
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1].set_title('Anomaly Scores (lower = more anomalous)')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Anomaly Score')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Anomaly plot saved to {save_path}")
        else:
            plt.show()

