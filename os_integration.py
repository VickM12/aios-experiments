"""
OS Integration Module
Provides safe OS-level integrations for the telemetry monitoring app
"""
import platform
import logging
import logging.handlers
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import os


class OSIntegration:
    """Handles OS-level integrations safely"""
    
    def __init__(self, enable_notifications: bool = True, enable_logging: bool = True):
        self.is_linux = platform.system() == "Linux"
        self.is_windows = platform.system() == "Windows"
        self.is_mac = platform.system() == "Darwin"
        self.enable_notifications = enable_notifications
        self.enable_logging = enable_logging
        
        # Setup system logging
        if self.enable_logging:
            self._setup_system_logging()
    
    def _setup_system_logging(self):
        """Setup logging to system logs"""
        self.logger = logging.getLogger('aios_telemetry')
        self.logger.setLevel(logging.INFO)
        
        if self.is_linux:
            # Try to use journald if available
            try:
                from systemd import journal
                handler = journal.JournalHandler()
                handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
                self.logger.addHandler(handler)
            except ImportError:
                # Fall back to syslog
                try:
                    import syslog
                    handler = logging.handlers.SysLogHandler(address='/dev/log')
                    handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
                    self.logger.addHandler(handler)
                except:
                    # Use standard logging
                    handler = logging.StreamHandler()
                    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                    self.logger.addHandler(handler)
        else:
            # Windows/Mac use standard logging
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
    
    def send_notification(self, title: str, message: str, urgency: str = "normal"):
        """Send desktop notification"""
        if not self.enable_notifications:
            return
        
        try:
            if self.is_linux:
                # Use notify-send on Linux
                urgency_map = {"low": "low", "normal": "normal", "critical": "critical"}
                os.system(f'notify-send -u {urgency_map.get(urgency, "normal")} "{title}" "{message}"')
            elif self.is_mac:
                # Use osascript on macOS
                os.system(f'''osascript -e 'display notification "{message}" with title "{title}"' ''')
            elif self.is_windows:
                # Use Windows toast notifications (requires win10toast or similar)
                try:
                    from win10toast import ToastNotifier
                    toaster = ToastNotifier()
                    toaster.show_toast(title, message, duration=10)
                except ImportError:
                    # Fallback to simple message
                    print(f"NOTIFICATION: {title} - {message}")
        except Exception as e:
            # Silently fail if notifications aren't available
            pass
    
    def log_event(self, level: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log event to system logs"""
        if not self.enable_logging or not hasattr(self, 'logger'):
            return
        
        log_message = message
        if data:
            log_message += f" | Data: {json.dumps(data)}"
        
        if level == "error":
            self.logger.error(log_message)
        elif level == "warning":
            self.logger.warning(log_message)
        elif level == "info":
            self.logger.info(log_message)
        else:
            self.logger.debug(log_message)
    
    def notify_anomaly(self, anomaly_count: int, details: Optional[Dict[str, Any]] = None):
        """Send notification and log anomaly detection"""
        title = "⚠️ System Anomaly Detected"
        message = f"Detected {anomaly_count} anomaly/anomalies in system telemetry"
        
        self.send_notification(title, message, urgency="normal")
        self.log_event("warning", f"Anomaly detected: {anomaly_count} anomalies", details)
    
    def notify_critical_alert(self, metric: str, value: float, threshold: float, unit: str = "%"):
        """Send critical alert notification"""
        title = "🚨 Critical System Alert"
        message = f"{metric} is at {value:.1f}{unit} (threshold: {threshold:.1f}{unit})"
        
        self.send_notification(title, message, urgency="critical")
        self.log_event("error", f"Critical alert: {metric} = {value:.1f}{unit}", {
            "metric": metric,
            "value": value,
            "threshold": threshold
        })
    
    def notify_warning(self, metric: str, message: str):
        """Send warning notification"""
        title = f"⚠️ {metric} Warning"
        self.send_notification(title, message, urgency="normal")
        self.log_event("warning", f"{metric} warning: {message}")
    
    def export_to_prometheus(self, telemetry_data: List[Dict[str, Any]], output_file: str):
        """Export telemetry data to Prometheus format"""
        prometheus_lines = []
        
        for data in telemetry_data:
            timestamp = data.get('timestamp', datetime.now().isoformat())
            
            # CPU metrics
            if 'cpu' in data:
                cpu = data['cpu']
                prometheus_lines.append(f'cpu_usage_percent{{timestamp="{timestamp}"}} {cpu.get("cpu_percent", 0)}')
            
            # Memory metrics
            if 'memory' in data and 'virtual_memory' in data['memory']:
                mem = data['memory']['virtual_memory']
                prometheus_lines.append(f'memory_usage_percent{{timestamp="{timestamp}"}} {mem.get("percent", 0)}')
            
            # Disk metrics
            if 'disk' in data and 'disk_usage' in data['disk']:
                disk = data['disk']['disk_usage']
                prometheus_lines.append(f'disk_usage_percent{{timestamp="{timestamp}"}} {disk.get("percent", 0)}')
            
            # Temperature
            if 'temperature' in data:
                temp_data = data['temperature']
                if temp_data and 'error' not in temp_data:
                    all_temps = []
                    for sensor_name, entries in temp_data.items():
                        for entry in entries:
                            if 'current' in entry:
                                all_temps.append(entry['current'])
                    if all_temps:
                        avg_temp = sum(all_temps) / len(all_temps)
                        prometheus_lines.append(f'temperature_celsius{{timestamp="{timestamp}"}} {avg_temp}')
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(prometheus_lines))
        
        self.log_event("info", f"Exported {len(telemetry_data)} data points to Prometheus format: {output_file}")
    
    def export_to_json(self, data: Any, output_file: str):
        """Export data to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.log_event("info", f"Exported data to JSON: {output_file}")
    
    def export_to_csv(self, telemetry_data: List[Dict[str, Any]], output_file: str):
        """Export telemetry data to CSV format"""
        import csv
        
        if not telemetry_data:
            return
        
        # Get all possible keys
        all_keys = set()
        for data in telemetry_data:
            all_keys.update(self._flatten_dict(data).keys())
        
        # Write CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            
            for data in telemetry_data:
                flat_data = self._flatten_dict(data)
                writer.writerow(flat_data)
        
        self.log_event("info", f"Exported {len(telemetry_data)} data points to CSV: {output_file}")
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Handle lists by converting to string or taking first element
                if v and isinstance(v[0], dict):
                    items.extend(self._flatten_dict(v[0], new_key, sep=sep).items())
                else:
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def create_systemd_service(self, service_name: str = "aios-telemetry", 
                              user: Optional[str] = None, 
                              working_dir: Optional[str] = None) -> str:
        """Generate systemd service file content"""
        if not self.is_linux:
            return "# systemd service files are only available on Linux"
        
        if not working_dir:
            working_dir = os.getcwd()
        
        if not user:
            user = os.getenv('USER', 'root')
        
        service_content = f"""[Unit]
Description=AI Telemetry Monitoring Service
After=network.target

[Service]
Type=simple
User={user}
WorkingDirectory={working_dir}
ExecStart=/usr/bin/python3 {working_dir}/app.py --mode monitor --duration 3600
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
        return service_content

