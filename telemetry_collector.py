"""
Telemetry Collector Module
Collects system telemetry data from various sensors and system metrics
"""
import psutil
import time
from datetime import datetime
from typing import Dict, List, Any
import os


class TelemetryCollector:
    """Collects telemetry data from system sensors and metrics"""
    
    def __init__(self):
        self.collection_interval = 1.0  # seconds
        
    def get_cpu_metrics(self) -> Dict[str, Any]:
        """Collect CPU telemetry data"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'cpu_per_core': psutil.cpu_percent(interval=0.1, percpu=True),
            'cpu_freq': {
                'current': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                'min': psutil.cpu_freq().min if psutil.cpu_freq() else None,
                'max': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            },
            'cpu_count': {
                'physical': psutil.cpu_count(logical=False),
                'logical': psutil.cpu_count(logical=True),
            },
            'cpu_stats': dict(psutil.cpu_stats()._asdict()) if hasattr(psutil, 'cpu_stats') else {},
        }
    
    def get_memory_metrics(self) -> Dict[str, Any]:
        """Collect memory telemetry data"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            'virtual_memory': {
                'total': mem.total,
                'available': mem.available,
                'used': mem.used,
                'free': mem.free,
                'percent': mem.percent,
            },
            'swap_memory': {
                'total': swap.total,
                'used': swap.used,
                'free': swap.free,
                'percent': swap.percent,
            },
        }
    
    def get_disk_metrics(self) -> Dict[str, Any]:
        """Collect disk telemetry data"""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        return {
            'disk_usage': {
                'total': disk_usage.total,
                'used': disk_usage.used,
                'free': disk_usage.free,
                'percent': disk_usage.percent,
            },
            'disk_io': dict(disk_io._asdict()) if disk_io else {},
        }
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """Collect network telemetry data"""
        net_io = psutil.net_io_counters()
        net_connections = len(psutil.net_connections())
        return {
            'network_io': dict(net_io._asdict()) if net_io else {},
            'network_connections': net_connections,
        }
    
    def get_temperature_sensors(self) -> Dict[str, Any]:
        """Collect temperature sensor data"""
        temps = {}
        try:
            if hasattr(psutil, "sensors_temperatures"):
                sensor_data = psutil.sensors_temperatures()
                for name, entries in sensor_data.items():
                    temps[name] = [
                        {
                            'label': entry.label,
                            'current': entry.current,
                            'high': entry.high,
                            'critical': entry.critical,
                        }
                        for entry in entries
                    ]
        except Exception as e:
            temps['error'] = str(e)
        return temps
    
    def get_fan_sensors(self) -> Dict[str, Any]:
        """Collect fan sensor data"""
        fans = {}
        try:
            if hasattr(psutil, "sensors_fans"):
                fan_data = psutil.sensors_fans()
                for name, entries in fan_data.items():
                    fans[name] = [
                        {
                            'label': entry.label,
                            'current': entry.current,
                        }
                        for entry in entries
                    ]
        except Exception as e:
            fans['error'] = str(e)
        return fans
    
    def get_battery_info(self) -> Dict[str, Any]:
        """Collect battery telemetry data"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return {
                    'percent': battery.percent,
                    'secsleft': battery.secsleft,
                    'power_plugged': battery.power_plugged,
                }
        except Exception:
            pass
        return {}
    
    def get_process_metrics(self) -> Dict[str, Any]:
        """Collect process-related metrics"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by CPU usage
        processes.sort(key=lambda x: x.get('cpu_percent', 0) or 0, reverse=True)
        
        return {
            'total_processes': len(processes),
            'top_processes': processes[:10],  # Top 10 by CPU
        }
    
    def collect_all_telemetry(self) -> Dict[str, Any]:
        """Collect all available telemetry data"""
        timestamp = datetime.now().isoformat()
        
        return {
            'timestamp': timestamp,
            'cpu': self.get_cpu_metrics(),
            'memory': self.get_memory_metrics(),
            'disk': self.get_disk_metrics(),
            'network': self.get_network_metrics(),
            'temperature': self.get_temperature_sensors(),
            'fans': self.get_fan_sensors(),
            'battery': self.get_battery_info(),
            'processes': self.get_process_metrics(),
        }
    
    def collect_continuous(self, duration: int = 60, interval: float = 1.0) -> List[Dict[str, Any]]:
        """Collect telemetry data continuously for a specified duration"""
        self.collection_interval = interval
        data_points = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            data_points.append(self.collect_all_telemetry())
            time.sleep(interval)
        
        return data_points

