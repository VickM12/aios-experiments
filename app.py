"""
Main Application
AI-powered telemetry monitoring and analysis application
"""
import json
import time
from datetime import datetime
from telemetry_collector import TelemetryCollector
from ai_analyzer import TelemetryAnalyzer
import argparse


class TelemetryApp:
    """Main application for telemetry collection and AI analysis"""
    
    def __init__(self):
        self.collector = TelemetryCollector()
        self.analyzer = TelemetryAnalyzer()
        self.telemetry_history = []
    
    def collect_sample(self, duration: int = 60, interval: float = 1.0):
        """Collect telemetry samples"""
        print(f"Collecting telemetry data for {duration} seconds (interval: {interval}s)...")
        self.telemetry_history = self.collector.collect_continuous(duration, interval)
        print(f"Collected {len(self.telemetry_history)} data points")
        return self.telemetry_history
    
    def analyze(self):
        """Perform AI analysis on collected data"""
        if not self.telemetry_history:
            print("No telemetry data available. Please collect data first.")
            return None
        
        print("\n" + "="*60)
        print("AI ANALYSIS RESULTS")
        print("="*60)
        
        analysis = self.analyzer.comprehensive_analysis(self.telemetry_history)
        
        # Display results
        self._display_analysis(analysis)
        
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
                print(f"Anomalies detected: {anomaly['anomalies_detected']} "
                      f"({anomaly.get('anomaly_percentage', 0):.2f}%)")
                if anomaly.get('details'):
                    print("\nAnomaly details:")
                    for detail in anomaly['details'][:5]:  # Show first 5
                        print(f"   • Index {detail['index']}: Score {detail['score']:.4f}")
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
                
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"CPU: {cpu:5.1f}% | "
                      f"Memory: {mem:5.1f}% | "
                      f"Disk: {disk:5.1f}%", end='', flush=True)
                
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
    
    args = parser.parse_args()
    
    app = TelemetryApp()
    
    if args.mode == 'collect' or args.mode == 'full':
        app.collect_sample(args.duration, args.interval)
        if args.save_data:
            app.save_data()
    
    if args.mode == 'analyze' or args.mode == 'full':
        if not app.telemetry_history:
            print("No data to analyze. Collecting sample data first...")
            app.collect_sample(args.duration, args.interval)
        
        analysis = app.analyze()
        if args.save_analysis and analysis:
            app.save_analysis(analysis)
    
    if args.mode == 'monitor':
        app.real_time_monitor(args.duration, args.interval)
        if args.save_data:
            app.save_data()
        if args.save_analysis:
            analysis = app.analyze()
            if analysis:
                app.save_analysis(analysis)


if __name__ == '__main__':
    main()

