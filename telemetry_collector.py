"""
Telemetry Collector Module
Collects system telemetry data from various sensors and system metrics
"""
import psutil
import time
import platform
import subprocess
import socket
from datetime import datetime
from typing import Dict, List, Any, Optional
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
    
    def get_system_info(self) -> Dict[str, Any]:
        """Collect detailed system hardware and OS information"""
        info = {
            'os': {},
            'cpu': {},
            'memory': {},
            'disks': [],
            'network_interfaces': [],
            'gpu': [],
            'system': {},
        }
        
        # OS Information
        try:
            info['os'] = {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'platform': platform.platform(),
                'python_version': platform.python_version(),
            }
        except Exception as e:
            info['os']['error'] = str(e)
        
        # CPU Details
        try:
            # Try to get CPU model from /proc/cpuinfo
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    for line in cpuinfo.split('\n'):
                        if 'model name' in line.lower():
                            info['cpu']['model'] = line.split(':')[1].strip()
                            break
                        if 'vendor_id' in line.lower():
                            info['cpu']['vendor'] = line.split(':')[1].strip()
                            break
                        if 'cpu family' in line.lower():
                            info['cpu']['family'] = line.split(':')[1].strip()
                            break
            except:
                pass
            
            # CPU architecture
            info['cpu']['architecture'] = platform.machine()
            info['cpu']['physical_cores'] = psutil.cpu_count(logical=False)
            info['cpu']['logical_cores'] = psutil.cpu_count(logical=True)
            
            # CPU frequency info
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                info['cpu']['frequency'] = {
                    'current_mhz': cpu_freq.current,
                    'min_mhz': cpu_freq.min,
                    'max_mhz': cpu_freq.max,
                }
        except Exception as e:
            info['cpu']['error'] = str(e)
        
        # Memory Details
        try:
            mem = psutil.virtual_memory()
            info['memory'] = {
                'total_gb': round(mem.total / (1024**3), 2),
                'available_gb': round(mem.available / (1024**3), 2),
                'total_bytes': mem.total,
            }
            
            # Try to get memory info from dmidecode (requires root)
            try:
                result = subprocess.run(
                    ['dmidecode', '-t', 'memory'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    # Parse memory modules
                    modules = []
                    current_module = {}
                    for line in result.stdout.split('\n'):
                        if 'Memory Device' in line:
                            if current_module:
                                modules.append(current_module)
                            current_module = {}
                        elif 'Size:' in line and 'No Module' not in line:
                            size_str = line.split(':')[1].strip()
                            current_module['size'] = size_str
                        elif 'Type:' in line:
                            current_module['type'] = line.split(':')[1].strip()
                        elif 'Speed:' in line:
                            current_module['speed'] = line.split(':')[1].strip()
                    if current_module:
                        modules.append(current_module)
                    if modules:
                        info['memory']['modules'] = modules
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                pass  # dmidecode not available or requires root
        except Exception as e:
            info['memory']['error'] = str(e)
        
        # Disk Details (all partitions)
        try:
            partitions = psutil.disk_partitions()
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info = {
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'opts': partition.opts,
                        'total_gb': round(usage.total / (1024**3), 2),
                        'used_gb': round(usage.used / (1024**3), 2),
                        'free_gb': round(usage.free / (1024**3), 2),
                        'percent': usage.percent,
                    }
                    info['disks'].append(disk_info)
                except PermissionError:
                    pass
        except Exception as e:
            info['disks'] = {'error': str(e)}
        
        # Network Interfaces
        try:
            net_if_addrs = psutil.net_if_addrs()
            net_if_stats = psutil.net_if_stats()
            
            for interface_name, addresses in net_if_addrs.items():
                interface_info = {
                    'name': interface_name,
                    'addresses': [],
                    'is_up': False,
                    'speed_mbps': None,
                }
                
                # Get addresses
                for addr in addresses:
                    interface_info['addresses'].append({
                        'family': str(addr.family),
                        'address': addr.address,
                        'netmask': addr.netmask if addr.netmask else None,
                    })
                
                # Get interface stats
                if interface_name in net_if_stats:
                    stats = net_if_stats[interface_name]
                    interface_info['is_up'] = stats.isup
                    interface_info['speed_mbps'] = stats.speed
                
                info['network_interfaces'].append(interface_info)
        except Exception as e:
            info['network_interfaces'] = {'error': str(e)}
        
        # GPU Information
        try:
            # Try nvidia-smi
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3:
                            info['gpu'].append({
                                'vendor': 'NVIDIA',
                                'model': parts[0],
                                'memory': parts[1],
                                'driver': parts[2],
                            })
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                pass
            
            # Try lspci for other GPUs
            try:
                result = subprocess.run(
                    ['lspci'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'vga' in line.lower() or '3d' in line.lower() or 'display' in line.lower():
                            gpu_info = line.split(':')[2].strip() if ':' in line else line.strip()
                            if not any(gpu.get('model', '').lower() in gpu_info.lower() 
                                      for gpu in info['gpu']):
                                info['gpu'].append({
                                    'vendor': 'Unknown',
                                    'model': gpu_info,
                                })
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                pass
        except Exception as e:
            info['gpu'] = {'error': str(e)}
        
        # System Information
        try:
            info['system'] = {
                'hostname': socket.gethostname(),
                'uptime_seconds': time.time() - psutil.boot_time(),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            }
            
            # Try to get motherboard/BIOS info
            try:
                result = subprocess.run(
                    ['dmidecode', '-t', 'baseboard', '-t', 'bios'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    output = result.stdout
                    if 'Base Board' in output or 'Motherboard' in output:
                        for line in output.split('\n'):
                            if 'Manufacturer:' in line:
                                info['system']['motherboard_manufacturer'] = line.split(':')[1].strip()
                            elif 'Product Name:' in line:
                                info['system']['motherboard_model'] = line.split(':')[1].strip()
                            elif 'Version:' in line and 'motherboard' not in info['system']:
                                info['system']['motherboard_version'] = line.split(':')[1].strip()
                    
                    if 'BIOS' in output:
                        for line in output.split('\n'):
                            if 'Vendor:' in line:
                                info['system']['bios_vendor'] = line.split(':')[1].strip()
                            elif 'Version:' in line and 'bios' not in info['system']:
                                info['system']['bios_version'] = line.split(':')[1].strip()
                            elif 'Release Date:' in line:
                                info['system']['bios_date'] = line.split(':')[1].strip()
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                pass  # dmidecode not available or requires root
        except Exception as e:
            info['system']['error'] = str(e)
        
        return info
    
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

