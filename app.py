"""
Main Application
AI-powered telemetry monitoring and analysis application
"""
import json
import time
from datetime import datetime
from telemetry_collector import TelemetryCollector
from ai_analyzer import TelemetryAnalyzer
from visualizer import TelemetryVisualizer
import argparse


class TelemetryApp:
    """Main application for telemetry collection and AI analysis"""
    
    def __init__(self):
        self.collector = TelemetryCollector()
        self.analyzer = TelemetryAnalyzer()
        self.visualizer = TelemetryVisualizer()
        self.telemetry_history = []
    
    def collect_sample(self, duration: int = 60, interval: float = 1.0):
        """Collect telemetry samples"""
        print(f"Collecting telemetry data for {duration} seconds (interval: {interval}s)...")
        self.telemetry_history = self.collector.collect_continuous(duration, interval)
        print(f"Collected {len(self.telemetry_history)} data points")
        return self.telemetry_history
    
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
    parser.add_argument('--visualize', action='store_true', 
                       help='Generate visualizations (interactive HTML dashboard)')
    parser.add_argument('--show-plot', action='store_true', 
                       help='Display static plots in a window (requires GUI)')
    parser.add_argument('--predict', action='store_true', 
                       help='Enable anomaly prediction (requires at least 20 data points)')
    parser.add_argument('--prediction-steps', type=int, default=10, 
                       help='Number of steps ahead to predict (default: 10)')
    
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

