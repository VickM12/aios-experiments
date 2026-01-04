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
  - **Anomaly Prediction**: Forecasts future anomalies using machine learning models

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

# Enable anomaly prediction (forecasts future anomalies)
python app.py --predict --duration 120

# Predict 20 steps ahead with visualization
python app.py --predict --prediction-steps 20 --visualize

# Enable desktop notifications and system logging
python app.py --notifications --system-logging

# Export data to different formats
python app.py --export-csv telemetry.csv --export-json telemetry.json

# Generate systemd service file for background monitoring
python app.py --generate-systemd
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

### Anomaly Prediction

The app can predict when anomalies might occur in the future:

```bash
# Enable predictions (requires at least 20 data points)
python app.py --predict --duration 120

# Predict further ahead (20 steps)
python app.py --predict --prediction-steps 20

# Full analysis with predictions and visualization
python app.py --predict --visualize --duration 120
```

The prediction system:
- Trains Random Forest models on historical telemetry patterns
- Forecasts future metric values (CPU, Memory, Disk)
- Predicts anomaly likelihood for each future step
- Identifies historical patterns (e.g., most common hour for anomalies)
- Provides risk assessments (high/medium/low)

### Programmatic Usage

```python
from telemetry_collector import TelemetryCollector
from ai_analyzer import TelemetryAnalyzer
from visualizer import TelemetryVisualizer
from anomaly_predictor import AnomalyPredictor

# Collect data
collector = TelemetryCollector()
data = collector.collect_continuous(duration=60, interval=1.0)

# Analyze with predictions
analyzer = TelemetryAnalyzer()
analysis = analyzer.comprehensive_analysis(data, include_predictions=True, prediction_steps=10)

# Use predictor directly
predictor = AnomalyPredictor()
train_result = predictor.train_forecast_models(data)
future_predictions = predictor.predict_future_values(data, steps_ahead=10)

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
- **Cross-platform support**: Works on Linux, Windows, and macOS
  - **Linux**: Full sensor support (temperature, fans, RAPL power)
  - **Windows**: Core metrics + WMI for hardware info (requires `wmi` package)
  - **macOS**: Core metrics via psutil
- psutil for system metrics
- scikit-learn for ML analysis
- matplotlib/plotly for visualization

### Windows-Specific Requirements

On Windows, install additional packages for full hardware information:
```bash
pip install wmi pywin32
```

Note: Some features may be limited on Windows:
- Fan speeds (if not exposed via psutil)
- RAPL power monitoring (Linux only)
- Some temperature sensors (depends on hardware/drivers)

## GUI Application

The app includes a web-based GUI with chat interface:

```bash
python gui_app.py
```

This launches a web interface at `http://localhost:7860` with:
- **Real-time Monitoring**: Live telemetry collection and visualization
- **Chat Interface**: Ask AI questions about your system status
- **Interactive Dashboard**: Real-time plots and metrics
- **AI Analysis**: One-click analysis with recommendations

### Chat Commands

You can ask the AI chat:
- "What's my CPU usage?"
- "How's my memory?"
- "Are there any anomalies?"
- "What's the temperature?"
- "Give me recommendations"
- "Show me system status"

## OS Integration

The app supports safe OS-level integrations:

### Desktop Notifications
Enable desktop notifications for alerts and anomalies:
```bash
python app.py --notifications
```

### System Logging
Log events to system logs (syslog/journald on Linux):
```bash
python app.py --system-logging
```

### Data Export
Export telemetry data to various formats:
```bash
# CSV format
python app.py --export-csv data.csv

# JSON format
python app.py --export-json data.json

# Prometheus format
python app.py --export-prometheus metrics.prom
```

### Background Service (Linux)
Generate a systemd service file for background monitoring:
```bash
python app.py --generate-systemd
sudo cp aios-telemetry.service /etc/systemd/system/
sudo systemctl enable aios-telemetry
sudo systemctl start aios-telemetry
```

### Historical Data Archiving
Enable automatic archiving with system log correlation:
```bash
# Enable archiving (30 day retention by default)
python app.py --archive

# Custom retention period
python app.py --archive --retention-days 60 --archive-dir my_archive

# Query archived sessions
python app.py --list-sessions
python app.py --query-archive session_20240103_120000

# View archive statistics
python app.py --archive-stats
```

The archiving system:
- **Automatic compression**: Sessions older than 7 days are compressed
- **Retention policy**: Old sessions are automatically deleted after retention period
- **System log correlation**: Automatically correlates telemetry anomalies with system logs
- **Query interface**: Search and retrieve historical sessions
- **Database indexing**: Fast queries by time range, anomaly count, etc.

### System Log Integration
The app can read and correlate with system logs:
- **Linux**: journald (systemd) and syslog
- **Windows**: Event Log
- **macOS**: Unified logging system

Logs are automatically correlated with telemetry anomalies within a 5-minute window, helping identify what system events occurred during anomalies.

## Notes

- Some sensors (temperature, fans) may not be available on all systems
- Requires appropriate permissions to access system metrics
- The anomaly detector needs at least 10 data points to function effectively
- GUI requires Gradio (automatically installed with requirements)
- OS notifications require `notify-send` on Linux, `osascript` on macOS, or `win10toast` on Windows

## License

MIT License

