# AI Telemetry Monitoring and Analysis App

An AI-powered application that collects machine telemetry data from system sensors and performs intelligent analysis using machine learning techniques.

## Features

- **Comprehensive Telemetry Collection**
  - CPU metrics (usage, frequency, per-core stats)
  - Memory metrics (virtual memory, swap)
  - Disk usage and I/O statistics
  - Network I/O and connections
  - Temperature sensors
  - Fan sensors
  - Battery information (if available)
  - Process metrics

- **AI-Powered Analysis**
  - **Anomaly Detection**: Uses Isolation Forest to identify unusual patterns
  - **Clustering Analysis**: K-means clustering to identify system states
  - **Trend Analysis**: Statistical analysis of metric trends over time
  - **Performance Insights**: Real-time recommendations and warnings

- **Visualization**
  - Static matplotlib plots
  - Interactive Plotly dashboards
  - Anomaly visualization

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

## Usage

### Basic Usage

```bash
# Collect data and analyze (default: 60 seconds)
python app.py

# Collect data for 120 seconds with 2-second intervals
python app.py --duration 120 --interval 2.0

# Real-time monitoring mode
python app.py --mode monitor --duration 300

# Save collected data and analysis
python app.py --save-data --save-analysis
```

### Command Line Options

- `--mode`: Operation mode
  - `collect`: Only collect data
  - `analyze`: Only analyze existing data
  - `monitor`: Real-time monitoring
  - `full`: Collect and analyze (default)

- `--duration`: Collection duration in seconds (default: 60)
- `--interval`: Collection interval in seconds (default: 1.0)
- `--save-data`: Save collected telemetry data to JSON file
- `--save-analysis`: Save analysis results to JSON file

### Programmatic Usage

```python
from telemetry_collector import TelemetryCollector
from ai_analyzer import TelemetryAnalyzer
from visualizer import TelemetryVisualizer

# Collect data
collector = TelemetryCollector()
data = collector.collect_continuous(duration=60, interval=1.0)

# Analyze
analyzer = TelemetryAnalyzer()
analysis = analyzer.comprehensive_analysis(data)

# Visualize
visualizer = TelemetryVisualizer()
visualizer.plot_basic_metrics(data, save_path='metrics.png')
visualizer.plot_interactive_dashboard(data, analysis, save_path='dashboard.html')
```

## Output Files

- `telemetry_data_YYYYMMDD_HHMMSS.json`: Raw telemetry data
- `analysis_results_YYYYMMDD_HHMMSS.json`: Analysis results
- `telemetry_dashboard.html`: Interactive dashboard
- `metrics.png`: Static plots

## Requirements

- Python 3.8+
- Linux system (for full sensor support)
- psutil for system metrics
- scikit-learn for ML analysis
- matplotlib/plotly for visualization

## Notes

- Some sensors (temperature, fans) may not be available on all systems
- Requires appropriate permissions to access system metrics
- The anomaly detector needs at least 10 data points to function effectively

## License

MIT License

