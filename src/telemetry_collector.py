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
        self.is_windows = platform.system() == 'Windows'
        self.is_linux = platform.system() == 'Linux'
        self.is_mac = platform.system() == 'Darwin'
        
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
        """Collect fan sensor data from multiple sources"""
        fans = {}
        detection_methods = []
        
        # Try psutil first
        try:
            if hasattr(psutil, "sensors_fans"):
                fan_data = psutil.sensors_fans()
                if fan_data:
                    for name, entries in fan_data.items():
                        fans[name] = [
                            {
                                'label': entry.label,
                                'current': entry.current,
                            }
                            for entry in entries
                        ]
                    detection_methods.append('psutil')
        except Exception as e:
            fans['psutil_error'] = str(e)
        
        # Also check /sys/class/hwmon/ directly for more fan data (Linux only)
        if self.is_linux:
            try:
                import glob
                hwmon_paths = glob.glob('/sys/class/hwmon/hwmon*/fan*_input')
                
                for fan_path in hwmon_paths:
                    try:
                        # Get fan number and hwmon name
                        hwmon_dir = '/'.join(fan_path.split('/')[:-1])
                        hwmon_name_path = os.path.join(hwmon_dir, 'name')
                        fan_num = fan_path.split('_')[0].split('fan')[-1]
                        
                        # Read fan speed
                        with open(fan_path, 'r') as f:
                            fan_rpm = int(f.read().strip())
                        
                        # Get hwmon name
                        hwmon_name = 'unknown'
                        if os.path.exists(hwmon_name_path):
                            with open(hwmon_name_path, 'r') as f:
                                hwmon_name = f.read().strip()
                        
                        # Get fan label if available
                        fan_label_path = os.path.join(hwmon_dir, f'fan{fan_num}_label')
                        fan_label = f'fan{fan_num}'
                        if os.path.exists(fan_label_path):
                            with open(fan_label_path, 'r') as f:
                                fan_label = f.read().strip()
                        
                        # Get max speed if available
                        fan_max_path = os.path.join(hwmon_dir, f'fan{fan_num}_max')
                        fan_max = None
                        if os.path.exists(fan_max_path):
                            with open(fan_max_path, 'r') as f:
                                fan_max = int(f.read().strip())
                        
                        # Store fan data
                        if hwmon_name not in fans:
                            fans[hwmon_name] = []
                        
                        fan_info = {
                            'label': fan_label,
                            'rpm': fan_rpm,
                        }
                        if fan_max:
                            fan_info['max_rpm'] = fan_max
                            fan_info['percent'] = round((fan_rpm / fan_max) * 100, 1) if fan_max > 0 else 0
                        
                        fans[hwmon_name].append(fan_info)
                        detection_methods.append('hwmon')
                    except (ValueError, IOError, PermissionError):
                        pass
            except Exception as e:
                fans['hwmon_error'] = str(e)
        
        # Try sensors command as fallback (Linux/Unix)
        if not self.is_windows:
            try:
                result = subprocess.run(
                    ['sensors'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    # Parse fan speeds from sensors output
                    for line in result.stdout.split('\n'):
                        if 'fan' in line.lower() and 'rpm' in line.lower():
                            # Extract fan info
                            parts = line.split(':')
                            if len(parts) == 2:
                                fan_name = parts[0].strip()
                                fan_value = parts[1].strip()
                                # Extract RPM number
                                import re
                                rpm_match = re.search(r'(\d+)\s*RPM', fan_value, re.IGNORECASE)
                                if rpm_match:
                                    if 'sensors' not in fans:
                                        fans['sensors'] = []
                                    fans['sensors'].append({
                                        'label': fan_name,
                                        'rpm': int(rpm_match.group(1)),
                                        'raw': fan_value
                                    })
                                    detection_methods.append('sensors_command')
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                pass
        
        # Return results
        if fans and not any(key.endswith('_error') for key in fans.keys()):
            if detection_methods:
                fans['_detection_methods'] = detection_methods
            return fans
        else:
            # No fans found - return informative message
            return {
                'message': 'No fan sensors found',
                'checked_methods': ['psutil', 'hwmon', 'sensors_command'],
                'note': 'Fans may be controlled by BIOS/EC and not exposed via standard Linux interfaces. This is common on laptops and some desktops.'
            }
    
    def get_battery_info(self) -> Dict[str, Any]:
        """Collect battery telemetry data"""
        battery_info = {}
        try:
            battery = psutil.sensors_battery()
            if battery:
                battery_info = {
                    'percent': battery.percent,
                    'secsleft': battery.secsleft,
                    'power_plugged': battery.power_plugged,
                }
        except Exception:
            pass
        
        # Also check /sys/class/power_supply/ for more detailed battery info (Linux only)
        if self.is_linux:
            try:
                import glob
                battery_paths = glob.glob('/sys/class/power_supply/BAT*/')
                
                for bat_path in battery_paths:
                    bat_name = os.path.basename(bat_path.rstrip('/'))
                    bat_data = {}
                    
                    # Read various battery attributes
                    attrs = ['capacity', 'energy_now', 'energy_full', 'power_now', 
                            'voltage_now', 'current_now', 'status']
                    
                    for attr in attrs:
                        attr_path = os.path.join(bat_path, attr)
                        if os.path.exists(attr_path):
                            try:
                                with open(attr_path, 'r') as f:
                                    value = f.read().strip()
                                    # Convert to numeric if possible
                                    try:
                                        if 'now' in attr or 'full' in attr:
                                            # These are in micro-wh or micro-ah
                                            bat_data[attr] = int(value) / 1_000_000  # Convert to Wh/Ah
                                        else:
                                            bat_data[attr] = int(value) if value.isdigit() else value
                                    except ValueError:
                                        bat_data[attr] = value
                            except (IOError, PermissionError):
                                pass
                    
                    if bat_data:
                        battery_info[bat_name] = bat_data
            except Exception:
                pass
        
        return battery_info
    
    def get_power_usage(self) -> Dict[str, Any]:
        """Collect power usage information from various sources"""
        power_info = {}
        
        # 1. RAPL (Intel Running Average Power Limit) - CPU power
        try:
            import glob
            rapl_paths = glob.glob('/sys/devices/virtual/powercap/intel-rapl/intel-rapl:*/energy_uj')
            
            rapl_power = {}
            for rapl_path in rapl_paths:
                try:
                    # Get domain name
                    domain_path = os.path.join(os.path.dirname(rapl_path), 'name')
                    domain_name = 'unknown'
                    if os.path.exists(domain_path):
                        with open(domain_path, 'r') as f:
                            domain_name = f.read().strip()
                    
                    # Read energy (in microjoules)
                    with open(rapl_path, 'r') as f:
                        energy_uj = int(f.read().strip())
                    
                    # Convert to Joules
                    rapl_power[domain_name] = {
                        'energy_joules': energy_uj / 1_000_000,
                        'energy_uj': energy_uj,
                    }
                except (ValueError, IOError, PermissionError):
                    pass
            
            if rapl_power:
                power_info['rapl'] = rapl_power
        except Exception:
            pass
        
        # 2. Power supply information (Linux only)
        if self.is_linux:
            try:
                import glob
                psu_paths = glob.glob('/sys/class/power_supply/*/')
                
                for psu_path in psu_paths:
                    psu_name = os.path.basename(psu_path.rstrip('/'))
                    
                    # Skip batteries (already handled)
                    if psu_name.startswith('BAT'):
                        continue
                    
                    psu_data = {}
                    
                    # Read power-related attributes
                    attrs = ['power_now', 'current_now', 'voltage_now', 'energy_now', 
                            'energy_full', 'type', 'online', 'status']
                    
                    for attr in attrs:
                        attr_path = os.path.join(psu_path, attr)
                        if os.path.exists(attr_path):
                            try:
                                with open(attr_path, 'r') as f:
                                    value = f.read().strip()
                                    # Convert to numeric if possible
                                    try:
                                        if 'now' in attr or 'full' in attr:
                                            # These are in micro-units
                                            psu_data[attr] = int(value) / 1_000_000
                                        elif attr in ['online']:
                                            psu_data[attr] = value == '1'
                                        else:
                                            psu_data[attr] = int(value) if value.isdigit() else value
                                    except ValueError:
                                        psu_data[attr] = value
                            except (IOError, PermissionError):
                                pass
                    
                    if psu_data:
                        power_info['power_supplies'] = power_info.get('power_supplies', {})
                        power_info['power_supplies'][psu_name] = psu_data
            except Exception:
                pass
        
        # 3. Try nvidia-smi for GPU power (if NVIDIA GPU)
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw,power.limit', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                gpu_power = []
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2:
                        # Helper function to safely convert to float
                        def safe_float(value):
                            if not value or value.upper() in ['N/A', 'NAN', 'NONE', '']:
                                return None
                            # Remove brackets and other non-numeric characters
                            value = value.strip('[]/')
                            try:
                                return float(value)
                            except (ValueError, TypeError):
                                return None
                        
                        power_draw = safe_float(parts[0])
                        power_limit = safe_float(parts[1])
                        
                        # Only add if we have at least one valid value
                        if power_draw is not None or power_limit is not None:
                            gpu_power.append({
                                'power_draw_watts': power_draw,
                                'power_limit_watts': power_limit,
                            })
                if gpu_power:
                    power_info['gpu'] = gpu_power
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        return power_info if power_info else {'message': 'No power usage data available'}
    
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
            # Try to get CPU model from /proc/cpuinfo (Linux) or WMI (Windows)
            if self.is_linux:
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
            elif self.is_windows:
                # Use WMI on Windows to get CPU info
                try:
                    import wmi
                    c = wmi.WMI()
                    for processor in c.Win32_Processor():
                        info['cpu']['model'] = processor.Name
                        info['cpu']['vendor'] = processor.Manufacturer
                        break
                except ImportError:
                    # wmi not installed, try alternative
                    try:
                        result = subprocess.run(
                            ['wmic', 'cpu', 'get', 'name'],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if result.returncode == 0:
                            lines = result.stdout.strip().split('\n')
                            if len(lines) > 1:
                                info['cpu']['model'] = lines[1].strip()
                    except:
                        pass
                except Exception:
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
            
            # Try to get memory info from dmidecode (Linux, requires root) or WMI (Windows)
            if self.is_linux:
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
            
            # Try lspci for other GPUs (Linux) or WMI (Windows)
            if self.is_linux:
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
            
            # Try to get motherboard/BIOS info (Linux: dmidecode, Windows: WMI)
            if self.is_linux:
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
            'power': self.get_power_usage(),
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

