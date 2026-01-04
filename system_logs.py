"""
System Log Integration Module
Reads and parses system logs from various sources
"""
import platform
import subprocess
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging


class SystemLogReader:
    """Reads system logs from various sources"""
    
    def __init__(self):
        self.is_linux = platform.system() == "Linux"
        self.is_windows = platform.system() == "Windows"
        self.is_mac = platform.system() == "Darwin"
        self.logger = logging.getLogger('aios_logs')
    
    def read_journald_logs(self, since: Optional[datetime] = None, 
                          until: Optional[datetime] = None,
                          priority: Optional[str] = None,
                          unit: Optional[str] = None,
                          limit: int = 1000) -> List[Dict[str, Any]]:
        """Read logs from systemd journal (Linux)"""
        if not self.is_linux:
            return []
        
        try:
            cmd = ['journalctl', '--no-pager', '--output=json', '--lines', str(limit)]
            
            if since:
                # Use relative time format for better performance
                time_diff = datetime.now() - since
                if time_diff.total_seconds() < 3600:
                    cmd.extend(['--since', f'-{int(time_diff.total_seconds()/60)}m'])
                else:
                    cmd.extend(['--since', since.isoformat()])
            else:
                # Default to last 10 minutes (shorter window for better performance)
                cmd.extend(['--since', '-10m'])
            
            if until:
                cmd.extend(['--until', until.isoformat()])
            
            if priority:
                cmd.extend(['--priority', priority])
            
            if unit:
                cmd.extend(['--unit', unit])
            
            # Shorter timeout and limit results
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                self.logger.warning(f"journalctl failed: {result.stderr}")
                return []
            
            logs = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                try:
                    import json
                    entry = json.loads(line)
                    
                    # Parse timestamp
                    timestamp_str = entry.get('__REALTIME_TIMESTAMP', '')
                    if timestamp_str:
                        # journald timestamps are in microseconds
                        timestamp = datetime.fromtimestamp(int(timestamp_str) / 1000000)
                    else:
                        timestamp = datetime.now()
                    
                    logs.append({
                        'timestamp': timestamp.isoformat(),
                        'source': 'journald',
                        'level': self._parse_journal_priority(entry.get('PRIORITY', 6)),
                        'message': entry.get('MESSAGE', ''),
                        'unit': entry.get('_SYSTEMD_UNIT', ''),
                        'hostname': entry.get('_HOSTNAME', ''),
                        'pid': entry.get('_PID', ''),
                        'raw': entry
                    })
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.debug(f"Failed to parse journal entry: {e}")
                    continue
            
            return logs
        except FileNotFoundError:
            self.logger.warning("journalctl not found")
            return []
        except subprocess.TimeoutExpired:
            self.logger.warning("journalctl timed out")
            return []
        except Exception as e:
            self.logger.error(f"Error reading journald logs: {e}")
            return []
    
    def _parse_journal_priority(self, priority: int) -> str:
        """Convert journald priority number to level string"""
        priority_map = {
            0: 'EMERG',
            1: 'ALERT',
            2: 'CRIT',
            3: 'ERR',
            4: 'WARNING',
            5: 'NOTICE',
            6: 'INFO',
            7: 'DEBUG'
        }
        return priority_map.get(priority, 'INFO')
    
    def read_syslog(self, log_file: Optional[str] = None,
                   since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Read logs from syslog file (Linux)"""
        if not self.is_linux:
            return []
        
        # Try common syslog locations
        possible_locations = [
            log_file,  # User-specified
            '/var/log/syslog',  # Debian/Ubuntu
            '/var/log/messages',  # RHEL/Fedora/CentOS
            '/var/log/system.log',  # Some systems
        ]
        
        log_file_to_use = None
        for location in possible_locations:
            if location and os.path.exists(location):
                log_file_to_use = location
                break
        
        if not log_file_to_use:
            return []
        
        try:
            logs = []
            with open(log_file_to_use, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # Parse syslog format: Jan  3 12:00:00 hostname service: message
                    match = re.match(r'(\w+)\s+(\d+)\s+(\d+):(\d+):(\d+)\s+(\S+)\s+(.+?):\s+(.*)', line)
                    if match:
                        month_str, day, hour, minute, second, hostname, service, message = match.groups()
                        
                        # Parse timestamp (approximate year)
                        year = datetime.now().year
                        try:
                            timestamp = datetime.strptime(
                                f"{year} {month_str} {day} {hour}:{minute}:{second}",
                                "%Y %b %d %H:%M:%S"
                            )
                        except ValueError:
                            continue
                        
                        if since and timestamp < since:
                            continue
                        
                        # Determine log level from message
                        level = 'INFO'
                        if any(word in message.upper() for word in ['ERROR', 'FAILED', 'CRITICAL']):
                            level = 'ERROR'
                        elif any(word in message.upper() for word in ['WARNING', 'WARN']):
                            level = 'WARNING'
                        
                        logs.append({
                            'timestamp': timestamp.isoformat(),
                            'source': 'syslog',
                            'level': level,
                            'message': message,
                            'service': service,
                            'hostname': hostname
                        })
            
            return logs
        except PermissionError:
            self.logger.warning(f"Permission denied reading {log_file}")
            return []
        except FileNotFoundError:
            self.logger.warning(f"Syslog file not found: {log_file}")
            return []
        except Exception as e:
            self.logger.error(f"Error reading syslog: {e}")
            return []
    
    def read_windows_event_log(self, log_name: str = 'System',
                              since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Read Windows Event Log"""
        if not self.is_windows:
            return []
        
        try:
            import win32evtlog
            import win32evtlogutil
            
            logs = []
            hand = win32evtlog.OpenEventLog(None, log_name)
            
            try:
                flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
                events = win32evtlog.ReadEventLog(hand, flags, 0)
                
                for event in events:
                    if event.TimeGenerated < since:
                        break
                    
                    # Parse event
                    level_map = {
                        1: 'ERROR',
                        2: 'WARNING',
                        4: 'INFO',
                        8: 'INFO',
                        16: 'INFO'
                    }
                    
                    logs.append({
                        'timestamp': event.TimeGenerated.isoformat(),
                        'source': 'eventlog',
                        'level': level_map.get(event.EventType, 'INFO'),
                        'message': win32evtlogutil.SafeFormatMessage(event, log_name),
                        'event_id': event.EventID,
                        'category': event.EventCategory
                    })
            finally:
                win32evtlog.CloseEventLog(hand)
            
            return logs
        except ImportError:
            self.logger.warning("pywin32 not available for Event Log access")
            return []
        except Exception as e:
            self.logger.error(f"Error reading Windows Event Log: {e}")
            return []
    
    def read_macos_logs(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Read macOS system logs"""
        if not self.is_mac:
            return []
        
        try:
            since_str = since.strftime('%Y-%m-%d %H:%M:%S') if since else '1 hour ago'
            cmd = ['log', 'show', '--style', 'json', '--predicate', 
                   f'eventDate >= "{since_str}"']
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return []
            
            import json
            logs = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    timestamp_str = entry.get('timestamp', '')
                    
                    # Parse macOS log timestamp
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    logs.append({
                        'timestamp': timestamp.isoformat(),
                        'source': 'macos_log',
                        'level': entry.get('level', 'INFO'),
                        'message': entry.get('message', ''),
                        'subsystem': entry.get('subsystem', ''),
                        'category': entry.get('category', '')
                    })
                except (json.JSONDecodeError, ValueError):
                    continue
            
            return logs
        except FileNotFoundError:
            return []
        except Exception as e:
            self.logger.error(f"Error reading macOS logs: {e}")
            return []
    
    def read_system_logs(self, since: Optional[datetime] = None,
                        until: Optional[datetime] = None,
                        priority: Optional[str] = None,
                        limit: int = 500) -> List[Dict[str, Any]]:
        """Read system logs from appropriate source for current OS"""
        logs = []
        
        if self.is_linux:
            # Try journald first, fall back to syslog
            try:
                logs = self.read_journald_logs(since=since, until=until, priority=priority, limit=limit)
            except Exception as e:
                self.logger.debug(f"journald read failed: {e}")
                logs = []
            
            # Only try syslog if journald returned nothing or very few logs
            if len(logs) < 10:
                try:
                    syslog_logs = self.read_syslog(since=since)
                    logs.extend(syslog_logs)
                except Exception as e:
                    self.logger.debug(f"syslog read failed: {e}")
        elif self.is_windows:
            logs = self.read_windows_event_log(since=since)
        elif self.is_mac:
            logs = self.read_macos_logs(since=since)
        
        return logs
    
    def filter_logs_by_keywords(self, logs: List[Dict[str, Any]], 
                               keywords: List[str]) -> List[Dict[str, Any]]:
        """Filter logs by keywords (useful for finding relevant system events)"""
        filtered = []
        for log in logs:
            message = log.get('message', '').upper()
            if any(keyword.upper() in message for keyword in keywords):
                filtered.append(log)
        return filtered
    
    def get_error_logs(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get only error-level logs"""
        return [log for log in logs if log.get('level') in ['ERROR', 'CRIT', 'ALERT', 'EMERG']]
    
    def get_warning_logs(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get warning-level logs"""
        return [log for log in logs if log.get('level') in ['WARNING', 'WARN']]

