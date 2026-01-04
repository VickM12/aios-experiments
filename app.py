"""
Main Application
AI-powered telemetry monitoring and analysis application
"""
import json
import time
from datetime import datetime, timedelta
from telemetry_collector import TelemetryCollector
from ai_analyzer import TelemetryAnalyzer
from visualizer import TelemetryVisualizer
from os_integration import OSIntegration
from data_archive import DataArchive
from system_logs import SystemLogReader
import argparse


class TelemetryApp:
    """Main application for telemetry collection and AI analysis"""
    
    def __init__(self, enable_notifications: bool = False, enable_logging: bool = False,
                 enable_archiving: bool = False, archive_dir: str = "telemetry_archive",
                 retention_days: int = 30):
        self.collector = TelemetryCollector()
        self.analyzer = TelemetryAnalyzer()
        self.visualizer = TelemetryVisualizer()
        self.telemetry_history = []
        self.os_integration = OSIntegration(enable_notifications=enable_notifications, 
                                            enable_logging=enable_logging)
        self.archive = DataArchive(archive_dir=archive_dir, retention_days=retention_days) if enable_archiving else None
        self.log_reader = SystemLogReader() if enable_archiving else None
    
    def collect_sample(self, duration: int = 60, interval: float = 1.0, include_system_info: bool = True):
        """Collect telemetry samples"""
        print(f"Collecting telemetry data for {duration} seconds (interval: {interval}s)...")
        
        # Collect system info once at the start
        if include_system_info:
            print("Gathering system information...")
            system_info = self.collector.get_system_info()
            self._system_info = system_info
        
        # Collect continuous telemetry
        self.telemetry_history = self.collector.collect_continuous(duration, interval)
        
        # Add system info to first data point if collected
        if include_system_info and self.telemetry_history and hasattr(self, '_system_info'):
            self.telemetry_history[0]['system_info'] = self._system_info
        
        print(f"Collected {len(self.telemetry_history)} data points")
        return self.telemetry_history
    
    def display_system_info(self):
        """Display detailed system information"""
        if not hasattr(self, '_system_info'):
            print("Collecting system information...")
            self._system_info = self.collector.get_system_info()
        
        info = self._system_info
        print("\n" + "="*60)
        print("SYSTEM INFORMATION")
        print("="*60)
        
        # OS Info
        if 'os' in info:
            os_info = info['os']
            print("\n🖥️  OPERATING SYSTEM")
            print("-" * 60)
            print(f"   System: {os_info.get('system', 'Unknown')}")
            print(f"   Release: {os_info.get('release', 'Unknown')}")
            print(f"   Version: {os_info.get('version', 'Unknown')[:80]}")
            print(f"   Machine: {os_info.get('machine', 'Unknown')}")
            print(f"   Processor: {os_info.get('processor', 'Unknown')[:60]}")
        
        # CPU Info
        if 'cpu' in info:
            cpu_info = info['cpu']
            print("\n🔧 CPU")
            print("-" * 60)
            if 'model' in cpu_info:
                print(f"   Model: {cpu_info['model']}")
            if 'vendor' in cpu_info:
                print(f"   Vendor: {cpu_info['vendor']}")
            print(f"   Physical Cores: {cpu_info.get('physical_cores', 'Unknown')}")
            print(f"   Logical Cores: {cpu_info.get('logical_cores', 'Unknown')}")
            if 'frequency' in cpu_info:
                freq = cpu_info['frequency']
                print(f"   Frequency: {freq.get('current_mhz', 0):.0f} MHz "
                      f"(Min: {freq.get('min_mhz', 0):.0f}, Max: {freq.get('max_mhz', 0):.0f})")
        
        # Memory Info
        if 'memory' in info:
            mem_info = info['memory']
            print("\n💾 MEMORY")
            print("-" * 60)
            print(f"   Total: {mem_info.get('total_gb', 0):.2f} GB")
            if 'modules' in mem_info:
                print(f"   Modules: {len(mem_info['modules'])}")
                for i, module in enumerate(mem_info['modules'][:4], 1):  # Show first 4
                    mod_str = f"      {i}. {module.get('size', 'Unknown')}"
                    if 'type' in module:
                        mod_str += f" - {module['type']}"
                    if 'speed' in module:
                        mod_str += f" @ {module['speed']}"
                    print(mod_str)
        
        # Disk Info
        if 'disks' in info and isinstance(info['disks'], list):
            print("\n💿 DISKS")
            print("-" * 60)
            for disk in info['disks'][:5]:  # Show first 5
                print(f"   {disk.get('device', 'Unknown')} -> {disk.get('mountpoint', 'Unknown')}")
                print(f"      Type: {disk.get('fstype', 'Unknown')}, "
                      f"Size: {disk.get('total_gb', 0):.2f} GB, "
                      f"Used: {disk.get('percent', 0):.1f}%")
        
        # GPU Info
        if 'gpu' in info and isinstance(info['gpu'], list) and info['gpu']:
            print("\n🎮 GPU")
            print("-" * 60)
            for gpu in info['gpu']:
                print(f"   {gpu.get('vendor', 'Unknown')} {gpu.get('model', 'Unknown')}")
                if 'memory' in gpu:
                    print(f"      Memory: {gpu['memory']}")
                if 'driver' in gpu:
                    print(f"      Driver: {gpu['driver']}")
        
        # Network Interfaces
        if 'network_interfaces' in info and isinstance(info['network_interfaces'], list):
            print("\n🌐 NETWORK INTERFACES")
            print("-" * 60)
            active_interfaces = [iface for iface in info['network_interfaces'] if iface.get('is_up', False)]
            for iface in active_interfaces[:5]:  # Show first 5 active
                print(f"   {iface.get('name', 'Unknown')}: {'UP' if iface.get('is_up') else 'DOWN'}")
                for addr in iface.get('addresses', [])[:2]:  # Show first 2 addresses
                    if 'inet' in addr.get('family', '').lower() or 'AF_INET' in addr.get('family', ''):
                        print(f"      IP: {addr.get('address', 'Unknown')}")
        
        # System Info
        if 'system' in info:
            sys_info = info['system']
            print("\n🖥️  SYSTEM")
            print("-" * 60)
            print(f"   Hostname: {sys_info.get('hostname', 'Unknown')}")
            uptime = sys_info.get('uptime_seconds', 0)
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            print(f"   Uptime: {hours}h {minutes}m")
            if 'motherboard_model' in sys_info:
                print(f"   Motherboard: {sys_info.get('motherboard_manufacturer', '')} "
                      f"{sys_info.get('motherboard_model', '')}")
            if 'bios_version' in sys_info:
                print(f"   BIOS: {sys_info.get('bios_vendor', '')} "
                      f"v{sys_info.get('bios_version', '')}")
        
        print("\n" + "="*60)
    
    def analyze(self, include_predictions: bool = False, prediction_steps: int = 10):
        """Perform AI analysis on collected data"""
        if not self.telemetry_history:
            print("No telemetry data available. Please collect data first.")
            return None
        
        print("\n" + "="*60)
        print("AI ANALYSIS RESULTS")
        print("="*60)
        
        analysis = self.analyzer.comprehensive_analysis(
            self.telemetry_history,
            include_predictions=include_predictions,
            prediction_steps=prediction_steps
        )
        
        # Display results
        self._display_analysis(analysis)
        
        # Archive data if archiving is enabled
        if self.archive:
            # Get system logs for correlation
            system_logs = []
            if self.log_reader and self.telemetry_history:
                try:
                    start_time = datetime.fromisoformat(self.telemetry_history[0].get('timestamp', datetime.now().isoformat()))
                    # Only read logs from a short window (5 minutes before to 2 minutes after)
                    system_logs = self.log_reader.read_system_logs(
                        since=start_time - timedelta(minutes=5),
                        until=start_time + timedelta(minutes=2),
                        limit=200  # Limit to 200 most recent relevant logs
                    )
                except Exception as e:
                    # Silently fail if log reading has issues
                    pass
            
            # Archive session
            session_id = self.archive.archive_session(
                self.telemetry_history,
                analysis,
                getattr(self, '_system_info', None)
            )
            
            # Correlate with system logs
            if session_id and system_logs:
                self.archive.correlate_with_logs(session_id, system_logs)
                correlated_count = len(self.archive.get_correlated_events(session_id))
                print(f"\n💾 Archived session: {session_id}")
                print(f"   📁 Location: {self.archive.archive_dir}/{session_id}.json")
                if correlated_count > 0:
                    print(f"   🔗 Correlated with {correlated_count} system log events")
                print(f"\n💡 Next steps:")
                print(f"   • View session: python app.py --query-archive {session_id}")
                print(f"   • List all sessions: python app.py --list-sessions")
                print(f"   • Archive stats: python app.py --archive-stats")
        
        return analysis
    
    def _display_analysis(self, analysis: dict):
        """Display analysis results in a readable format"""
        
        # Performance Insights
        if 'performance_insights' in analysis:
            insights = analysis['performance_insights']
            print("\n📊 PERFORMANCE INSIGHTS")
            print("-" * 60)
            
            if 'current_status' in insights:
                status = insights['current_status']
                for metric, data in status.items():
                    status_icon = "🔴" if data.get('status') == 'high' else "🟡" if data.get('status') == 'moderate' else "🟢"
                    # Temperature uses 'average' and should display in Celsius
                    if metric == 'temperature':
                        value = data.get('average', 'N/A')
                        if value != 'N/A':
                            print(f"{status_icon} {metric.upper()}: {value:.2f}°C "
                                  f"({data.get('status', 'unknown')})")
                        else:
                            print(f"{status_icon} {metric.upper()}: {value} "
                                  f"({data.get('status', 'unknown')})")
                    else:
                        # Other metrics use 'usage' and display as percentage
                        value = data.get('usage', 'N/A')
                        if value != 'N/A':
                            print(f"{status_icon} {metric.upper()}: {value:.2f}% "
                                  f"({data.get('status', 'unknown')})")
                        else:
                            print(f"{status_icon} {metric.upper()}: {value} "
                                  f"({data.get('status', 'unknown')})")
            
            if insights.get('warnings'):
                print("\n⚠️  WARNINGS:")
                for warning in insights['warnings']:
                    print(f"   • {warning}")
                    # Send OS notification for warnings
                    self.os_integration.notify_warning("System Warning", warning)
                    
                    # Check for critical alerts
                    if 'CPU usage is critically high' in warning:
                        cpu_percent = status.get('cpu', {}).get('usage', 0)
                        self.os_integration.notify_critical_alert("CPU", cpu_percent, 90)
                    elif 'Memory usage is critically high' in warning:
                        mem_percent = status.get('memory', {}).get('usage', 0)
                        self.os_integration.notify_critical_alert("Memory", mem_percent, 90)
                    elif 'Disk space is critically low' in warning:
                        disk_percent = status.get('disk', {}).get('usage', 0)
                        self.os_integration.notify_critical_alert("Disk", disk_percent, 90)
                    elif 'System temperature is high' in warning:
                        temp = status.get('temperature', {}).get('average', 0)
                        self.os_integration.notify_critical_alert("Temperature", temp, 80, unit="°C")
            
            if insights.get('recommendations'):
                print("\n💡 RECOMMENDATIONS:")
                for rec in insights['recommendations']:
                    print(f"   • {rec}")
        
        # Anomaly Detection
        if 'anomaly_detection' in analysis:
            anomaly = analysis['anomaly_detection']
            print("\n🔍 ANOMALY DETECTION")
            print("-" * 60)
            if 'anomalies_detected' in anomaly:
                anomalies_count = anomaly['anomalies_detected']
                print(f"Anomalies detected: {anomalies_count} "
                      f"({anomaly.get('anomaly_percentage', 0):.2f}%)")
                # Send OS notification for anomalies
                if anomalies_count > 0:
                    self.os_integration.notify_anomaly(anomalies_count, {
                        'percentage': anomaly.get('anomaly_percentage', 0),
                        'details_count': len(anomaly.get('details', []))
                    })
                if anomaly.get('details'):
                    print("\nAnomaly Analysis:")
                    for detail in anomaly['details'][:10]:  # Show first 10
                        print(f"\n   📍 Anomaly at Index {detail['index']} "
                              f"(Score: {detail['score']:.4f})")
                        print(f"      Timestamp: {detail.get('timestamp', 'N/A')}")
                        
                        # Show top contributing factors
                        if 'top_factors' in detail and detail['top_factors']:
                            print(f"      🔎 Top Contributing Factors:")
                            for factor in detail['top_factors'][:3]:  # Top 3 factors
                                metric_name = factor['metric'].replace('_', ' ').title()
                                z_score = factor['z_score']
                                deviation = factor['deviation']
                                direction = "↑" if z_score > 0 else "↓"
                                
                                # Get the actual value and mean for context
                                if factor['metric'] in detail.get('deviations', {}):
                                    dev_info = detail['deviations'][factor['metric']]
                                    actual_val = dev_info['value']
                                    mean_val = dev_info['mean']
                                    
                                    # Format based on metric type
                                    if 'percent' in factor['metric'] or 'cpu' in factor['metric'] or 'memory' in factor['metric'] or 'disk' in factor['metric']:
                                        print(f"         • {metric_name}: {actual_val:.2f}% "
                                              f"(avg: {mean_val:.2f}%, {direction} {abs(deviation):.1f}%)")
                                    elif 'temperature' in factor['metric']:
                                        print(f"         • {metric_name}: {actual_val:.2f}°C "
                                              f"(avg: {mean_val:.2f}°C, {direction} {abs(deviation):.1f}%)")
                                    elif 'mb' in factor['metric'] or 'gb' in factor['metric']:
                                        print(f"         • {metric_name}: {actual_val:.2f} "
                                              f"(avg: {mean_val:.2f}, {direction} {abs(deviation):.1f}%)")
                                    else:
                                        print(f"         • {metric_name}: {actual_val:.2f} "
                                              f"(avg: {mean_val:.2f}, z-score: {z_score:.2f})")
            else:
                print(anomaly.get('message', 'No anomaly data'))
        
        # Trend Analysis
        if 'trend_analysis' in analysis:
            trends = analysis['trend_analysis']
            print("\n📈 TREND ANALYSIS")
            print("-" * 60)
            if 'summary' in trends:
                summary = trends['summary']
                print(f"Metrics analyzed: {summary.get('metrics_analyzed', 0)}")
                print(f"Increasing: {summary.get('increasing_metrics', 0)}")
                print(f"Decreasing: {summary.get('decreasing_metrics', 0)}")
                print(f"Stable: {summary.get('stable_metrics', 0)}")
            
            if 'trends' in trends:
                print("\nKey trends:")
                key_metrics = ['cpu_percent', 'memory_percent', 'disk_percent']
                for metric in key_metrics:
                    if metric in trends['trends']:
                        trend_data = trends['trends'][metric]
                        trend_arrow = "📈" if trend_data['trend'] == 'increasing' else "📉" if trend_data['trend'] == 'decreasing' else "➡️"
                        print(f"   {trend_arrow} {metric}: {trend_data['trend']} "
                              f"({trend_data['rate_of_change_percent']:+.2f}%)")
        
        # Clustering
        if 'clustering' in analysis:
            clustering = analysis['clustering']
            print("\n🎯 CLUSTERING ANALYSIS")
            print("-" * 60)
            if 'clusters' in clustering:
                print(f"Identified {clustering.get('n_clusters', 0)} clusters:")
                for cluster in clustering['clusters']:
                    print(f"   Cluster {cluster['cluster_id']}: {cluster['size']} samples")
                    chars = cluster.get('characteristics', {})
                    print(f"      Avg CPU: {chars.get('avg_cpu', 0):.2f}%, "
                          f"Avg Memory: {chars.get('avg_memory', 0):.2f}%")
            else:
                print(clustering.get('message', 'No clustering data'))
        
        # Predictions
        if 'predictions' in analysis:
            predictions = analysis['predictions']
            print("\n🔮 ANOMALY PREDICTIONS")
            print("-" * 60)
            
            if 'error' in predictions:
                print(f"   ⚠️  {predictions['error']}")
            else:
                # Training info
                if 'forecast_training' in predictions:
                    training = predictions['forecast_training']
                    if training.get('success'):
                        print(f"   ✅ Models trained: {', '.join(training.get('models_trained', []))}")
                        if 'performance' in training:
                            print("   Model Performance:")
                            for metric, perf in training['performance'].items():
                                print(f"      • {metric}: R²={perf.get('r2_score', 0):.3f}, "
                                      f"MAE={perf.get('mae', 0):.2f}")
                
                # Anomaly predictions
                if 'anomaly_predictions' in predictions:
                    pred_data = predictions['anomaly_predictions']
                    if pred_data.get('success'):
                        summary = pred_data.get('summary', {})
                        print(f"\n   📊 Future Anomaly Forecast ({summary.get('total_steps', 0)} steps ahead):")
                        print(f"      Predicted anomalies: {summary.get('predicted_anomalies', 0)} "
                              f"({summary.get('anomaly_percentage', 0):.1f}%)")
                        
                        high_risk = summary.get('high_risk_steps', [])
                        if high_risk:
                            print(f"      ⚠️  High-risk steps: {high_risk[:10]}")  # Show first 10
                        
                        # Show detailed predictions
                        anomaly_preds = pred_data.get('anomaly_predictions', [])
                        if anomaly_preds:
                            print(f"\n   🔍 Detailed Predictions (first 5 high-risk):")
                            high_risk_preds = [p for p in anomaly_preds if p.get('likelihood') == 'high'][:5]
                            for pred in high_risk_preds:
                                timestamp = pred.get('timestamp', 'N/A')
                                metrics = pred.get('predicted_metrics', {})
                                print(f"      Step {pred.get('step', '?')} ({timestamp[:19]}):")
                                print(f"         Risk: {pred.get('likelihood', 'unknown').upper()}, "
                                      f"Score: {pred.get('anomaly_score', 0):.3f}")
                                if metrics:
                                    cpu = metrics.get('cpu_percent', 0)
                                    mem = metrics.get('memory_percent', 0)
                                    print(f"         Predicted: CPU={cpu:.1f}%, Memory={mem:.1f}%")
                
                # Pattern analysis
                if 'anomaly_patterns' in predictions:
                    patterns = predictions['anomaly_patterns']
                    if patterns.get('patterns_found'):
                        print(f"\n   📈 Historical Anomaly Patterns:")
                        if patterns.get('most_common_hour') is not None:
                            print(f"      Most common hour: {patterns['most_common_hour']}:00")
                        if patterns.get('average_interval'):
                            print(f"      Average interval: {patterns['average_interval']:.1f} samples")
                        if patterns.get('min_interval') and patterns.get('max_interval'):
                            print(f"      Interval range: {patterns['min_interval']:.0f}-{patterns['max_interval']:.0f} samples")
        
        print("\n" + "="*60)
    
    def save_data(self, filename: str = None):
        """Save collected telemetry data to JSON file"""
        if not filename:
            filename = f"telemetry_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.telemetry_history, f, indent=2, default=str)
        print(f"Data saved to {filename}")
    
    def save_analysis(self, analysis: dict, filename: str = None):
        """Save analysis results to JSON file"""
        if not filename:
            filename = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"Analysis saved to {filename}")
    
    def visualize(self, analysis: dict = None, interactive: bool = True, show_plot: bool = False):
        """Generate visualizations of telemetry data"""
        if not self.telemetry_history:
            print("No telemetry data available. Please collect data first.")
            return
        
        print("\n📊 Generating visualizations...")
        
        if interactive:
            # Generate interactive Plotly dashboard
            dashboard_path = f"telemetry_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            self.visualizer.plot_interactive_dashboard(
                self.telemetry_history, 
                analysis, 
                save_path=dashboard_path
            )
            print(f"\n✅ Interactive dashboard saved to: {dashboard_path}")
            print(f"   Open it in your browser to view the graphs!")
        
        if show_plot:
            # Generate and show static matplotlib plots
            print("\n📈 Displaying static plots...")
            self.visualizer.plot_basic_metrics(self.telemetry_history)
        
        # Always generate static plot file
        plot_path = f"telemetry_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.visualizer.plot_basic_metrics(self.telemetry_history, save_path=plot_path)
        print(f"✅ Static plot saved to: {plot_path}")
        
        # Generate anomaly visualization if analysis available
        if analysis and 'anomaly_detection' in analysis:
            anomaly_path = f"anomaly_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.visualizer.plot_anomaly_analysis(self.telemetry_history, analysis, save_path=anomaly_path)
            print(f"✅ Anomaly plot saved to: {anomaly_path}")
    
    def real_time_monitor(self, duration: int = 60, interval: float = 1.0):
        """Real-time monitoring with live updates"""
        print(f"Starting real-time monitoring for {duration} seconds...")
        print("Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                data = self.collector.collect_all_telemetry()
                self.telemetry_history.append(data)
                
                # Display current metrics
                cpu = data.get('cpu', {}).get('cpu_percent', 0)
                mem = data.get('memory', {}).get('virtual_memory', {}).get('percent', 0)
                disk = data.get('disk', {}).get('disk_usage', {}).get('percent', 0)
                
                # Get fan speeds
                fan_info = []
                fans_data = data.get('fans', {})
                if isinstance(fans_data, dict) and 'message' not in fans_data:
                    for sensor_name, fan_list in fans_data.items():
                        if sensor_name.startswith('_') or sensor_name.endswith('_error'):
                            continue
                        if isinstance(fan_list, list):
                            for fan in fan_list[:2]:  # Show first 2 fans per sensor
                                if 'rpm' in fan:
                                    fan_info.append(f"{fan.get('label', 'Fan')}: {fan['rpm']} RPM")
                                elif 'current' in fan:
                                    fan_info.append(f"{fan.get('label', 'Fan')}: {fan['current']}")
                
                # Get power usage
                power_data = data.get('power', {})
                power_info = []
                if 'gpu' in power_data and isinstance(power_data['gpu'], list):
                    for gpu in power_data['gpu']:
                        if 'power_draw_watts' in gpu and gpu['power_draw_watts']:
                            power_info.append(f"GPU: {gpu['power_draw_watts']:.1f}W")
                if 'rapl' in power_data:
                    # Show package power if available
                    for domain, pwr in power_data['rapl'].items():
                        if 'package' in domain.lower():
                            power_info.append(f"CPU: {pwr.get('energy_joules', 0):.1f}J")
                
                # Build display string
                display_parts = [
                    f"[{datetime.now().strftime('%H:%M:%S')}]",
                    f"CPU: {cpu:5.1f}%",
                    f"Mem: {mem:5.1f}%",
                    f"Disk: {disk:5.1f}%"
                ]
                
                if fan_info:
                    display_parts.append(f"| {' | '.join(fan_info[:2])}")
                
                if power_info:
                    display_parts.append(f"| {' | '.join(power_info[:2])}")
                
                print(f"\r{' | '.join(display_parts)}", end='', flush=True)
                
                time.sleep(interval)
            
            print("\n\nMonitoring complete!")
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")


def main():
    parser = argparse.ArgumentParser(description='AI Telemetry Monitoring and Analysis')
    parser.add_argument('--mode', choices=['collect', 'analyze', 'monitor', 'full'], 
                       default='full', help='Operation mode')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Collection duration in seconds')
    parser.add_argument('--interval', type=float, default=1.0, 
                       help='Collection interval in seconds')
    parser.add_argument('--save-data', action='store_true', 
                       help='Save collected data to file')
    parser.add_argument('--save-analysis', action='store_true', 
                       help='Save analysis results to file')
    parser.add_argument('--visualize', action='store_true', 
                       help='Generate visualizations (interactive HTML dashboard)')
    parser.add_argument('--show-plot', action='store_true', 
                       help='Display static plots in a window (requires GUI)')
    parser.add_argument('--predict', action='store_true', 
                       help='Enable anomaly prediction (requires at least 20 data points)')
    parser.add_argument('--prediction-steps', type=int, default=10, 
                       help='Number of steps ahead to predict (default: 10)')
    parser.add_argument('--notifications', action='store_true',
                       help='Enable desktop notifications for alerts and anomalies')
    parser.add_argument('--system-logging', action='store_true',
                       help='Enable logging to system logs (syslog/journald)')
    parser.add_argument('--export-prometheus', type=str, metavar='FILE',
                       help='Export telemetry data to Prometheus format')
    parser.add_argument('--export-csv', type=str, metavar='FILE',
                       help='Export telemetry data to CSV format')
    parser.add_argument('--export-json', type=str, metavar='FILE',
                       help='Export data to JSON format')
    parser.add_argument('--generate-systemd', action='store_true',
                       help='Generate systemd service file for background monitoring')
    parser.add_argument('--archive', action='store_true',
                       help='Enable data archiving with retention')
    parser.add_argument('--archive-dir', type=str, default='telemetry_archive',
                       help='Directory for archived data (default: telemetry_archive)')
    parser.add_argument('--retention-days', type=int, default=30,
                       help='Number of days to retain archived data (default: 30)')
    parser.add_argument('--query-archive', type=str, metavar='SESSION_ID',
                       help='Query and display archived session')
    parser.add_argument('--list-sessions', action='store_true',
                       help='List all archived sessions')
    parser.add_argument('--archive-stats', action='store_true',
                       help='Show archive statistics')
    
    args = parser.parse_args()
    
    # Handle archive queries before creating app
    if args.query_archive or args.list_sessions or args.archive_stats:
        archive = DataArchive(archive_dir=args.archive_dir, retention_days=args.retention_days)
        
        if args.query_archive:
            session = archive.load_session(args.query_archive)
            if session:
                print(f"\n📦 Session: {args.query_archive}")
                print(f"   Start: {session.get('start_time')}")
                print(f"   End: {session.get('end_time')}")
                print(f"   Data Points: {len(session.get('telemetry_data', []))}")
                
                # Show correlated events
                events = archive.get_correlated_events(args.query_archive)
                if events:
                    print(f"\n🔗 Correlated System Log Events ({len(events)}):")
                    for event in events[:10]:  # Show top 10
                        print(f"   [{event['log_time']}] {event['source']} {event['level']}: {event['message'][:80]}")
                        print(f"      Correlation: {event['correlation_score']:.2f}")
            else:
                print(f"Session {args.query_archive} not found")
            return
        
        if args.list_sessions:
            sessions = archive.query_sessions()
            print(f"\n📚 Archived Sessions ({len(sessions)}):")
            for session in sessions[:20]:  # Show last 20
                print(f"   {session['session_id']}: {session['start_time']} - {session['data_points']} points, {session['anomaly_count']} anomalies")
            return
        
        if args.archive_stats:
            stats = archive.get_statistics()
            print("\n📊 Archive Statistics:")
            print(f"   Total Sessions: {stats.get('total_sessions', 0)}")
            print(f"   Total Data Points: {stats.get('total_data_points', 0):,}")
            print(f"   Total Anomalies: {stats.get('total_anomalies', 0)}")
            print(f"   Total Correlations: {stats.get('total_correlations', 0)}")
            if stats.get('oldest_session'):
                print(f"   Oldest: {stats['oldest_session']}")
                print(f"   Newest: {stats['newest_session']}")
            return
    
    # Generate systemd service file if requested
    if args.generate_systemd:
        os_int = OSIntegration(enable_notifications=False, enable_logging=False)
        service_content = os_int.create_systemd_service()
        service_file = 'aios-telemetry.service'
        with open(service_file, 'w') as f:
            f.write(service_content)
        print(f"Generated systemd service file: {service_file}")
        print(f"To install: sudo cp {service_file} /etc/systemd/system/")
        print(f"To enable: sudo systemctl enable {service_file.replace('.service', '')}")
        print(f"To start: sudo systemctl start {service_file.replace('.service', '')}")
        return
    
    app = TelemetryApp(enable_notifications=args.notifications, 
                       enable_logging=args.system_logging,
                       enable_archiving=args.archive,
                       archive_dir=args.archive_dir,
                       retention_days=args.retention_days)
    
    # Run maintenance tasks if archiving is enabled
    if args.archive:
        app.archive.compress_old_sessions()
        app.archive.cleanup_old_sessions()
    
    if args.mode == 'collect' or args.mode == 'full':
        # Display system info at start
        app.display_system_info()
        app.collect_sample(args.duration, args.interval)
        if args.save_data:
            app.save_data()
        
        # Export data if requested
        if args.export_prometheus:
            app.os_integration.export_to_prometheus(app.telemetry_history, args.export_prometheus)
        if args.export_csv:
            app.os_integration.export_to_csv(app.telemetry_history, args.export_csv)
        if args.export_json:
            app.os_integration.export_to_json(app.telemetry_history, args.export_json)
    
    if args.mode == 'analyze' or args.mode == 'full':
        if not app.telemetry_history:
            print("No data to analyze. Collecting sample data first...")
            app.collect_sample(args.duration, args.interval)
        
        analysis = app.analyze(
            include_predictions=args.predict,
            prediction_steps=args.prediction_steps
        )
        if args.save_analysis and analysis:
            app.save_analysis(analysis)
        
        # Generate visualizations if requested
        if args.visualize or args.show_plot:
            app.visualize(analysis, interactive=args.visualize, show_plot=args.show_plot)
    
    if args.mode == 'monitor':
        app.real_time_monitor(args.duration, args.interval)
        if args.save_data:
            app.save_data()
        if args.save_analysis:
            analysis = app.analyze(
                include_predictions=args.predict,
                prediction_steps=args.prediction_steps
            )
            if analysis:
                app.save_analysis(analysis)


if __name__ == '__main__':
    main()

