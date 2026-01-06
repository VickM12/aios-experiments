"""
Data Archiving and Historical Retention Module
Manages long-term storage and retrieval of telemetry data with system log correlation
"""
import json
import sqlite3
import os
import gzip
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging


class DataArchive:
    """Manages historical telemetry data with retention policies"""
    
    def __init__(self, archive_dir: str = "telemetry_archive", 
                 retention_days: int = 30,
                 compress_after_days: int = 7):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(exist_ok=True)
        
        self.retention_days = retention_days
        self.compress_after_days = compress_after_days
        
        # Database for metadata and queries
        self.db_path = self.archive_dir / "archive.db"
        self._init_database()
        
        self.logger = logging.getLogger('aios_archive')
    
    def _init_database(self):
        """Initialize SQLite database for metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Telemetry data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS telemetry_sessions (
                session_id TEXT PRIMARY KEY,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                data_points INTEGER,
                file_path TEXT,
                compressed BOOLEAN DEFAULT 0,
                system_info TEXT
            )
        ''')
        
        # Anomalies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP,
                anomaly_index INTEGER,
                score REAL,
                metrics TEXT,
                FOREIGN KEY (session_id) REFERENCES telemetry_sessions(session_id)
            )
        ''')
        
        # System log correlations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS log_correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                telemetry_timestamp TIMESTAMP,
                log_timestamp TIMESTAMP,
                log_source TEXT,
                log_level TEXT,
                log_message TEXT,
                correlation_score REAL,
                FOREIGN KEY (session_id) REFERENCES telemetry_sessions(session_id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_time ON telemetry_sessions(start_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_session ON anomalies(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_session ON log_correlations(session_id)')
        
        conn.commit()
        conn.close()
    
    def archive_session(self, telemetry_data: List[Dict[str, Any]], 
                       analysis: Optional[Dict[str, Any]] = None,
                       system_info: Optional[Dict[str, Any]] = None) -> str:
        """Archive a telemetry session"""
        if not telemetry_data:
            return None
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = telemetry_data[0].get('timestamp', datetime.now().isoformat())
        end_time = telemetry_data[-1].get('timestamp', datetime.now().isoformat())
        
        # Save data to JSON file
        data_file = self.archive_dir / f"{session_id}.json"
        with open(data_file, 'w') as f:
            json.dump({
                'session_id': session_id,
                'start_time': start_time,
                'end_time': end_time,
                'telemetry_data': telemetry_data,
                'analysis': analysis,
                'system_info': system_info
            }, f, indent=2, default=str)
        
        # Store metadata in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO telemetry_sessions 
            (session_id, start_time, end_time, data_points, file_path, system_info)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, start_time, end_time, len(telemetry_data), 
              str(data_file), json.dumps(system_info) if system_info else None))
        
        # Store anomalies if available
        if analysis and 'anomaly_detection' in analysis:
            anomaly_data = analysis['anomaly_detection']
            if anomaly_data.get('anomalies_detected', 0) > 0:
                for detail in anomaly_data.get('details', []):
                    cursor.execute('''
                        INSERT INTO anomalies 
                        (session_id, timestamp, anomaly_index, score, metrics)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (session_id, 
                          detail.get('timestamp', start_time),
                          detail.get('index', 0),
                          detail.get('score', 0.0),
                          json.dumps(detail.get('features', {}))))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Archived session {session_id} with {len(telemetry_data)} data points")
        return session_id
    
    def compress_old_sessions(self):
        """Compress sessions older than compress_after_days"""
        cutoff_date = datetime.now() - timedelta(days=self.compress_after_days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT session_id, file_path FROM telemetry_sessions
            WHERE start_time < ? AND compressed = 0
        ''', (cutoff_date.isoformat(),))
        
        compressed_count = 0
        for row in cursor.fetchall():
            session_id, file_path = row
            json_path = Path(file_path)
            
            if json_path.exists():
                # Compress the file
                gz_path = json_path.with_suffix('.json.gz')
                with open(json_path, 'rb') as f_in:
                    with gzip.open(gz_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove original and update database
                json_path.unlink()
                cursor.execute('''
                    UPDATE telemetry_sessions 
                    SET file_path = ?, compressed = 1
                    WHERE session_id = ?
                ''', (str(gz_path), session_id))
                compressed_count += 1
        
        conn.commit()
        conn.close()
        
        if compressed_count > 0:
            self.logger.info(f"Compressed {compressed_count} old sessions")
    
    def cleanup_old_sessions(self):
        """Remove sessions older than retention_days"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT session_id, file_path FROM telemetry_sessions
            WHERE start_time < ?
        ''', (cutoff_date.isoformat(),))
        
        deleted_count = 0
        for row in cursor.fetchall():
            session_id, file_path = row
            file_path_obj = Path(file_path)
            
            # Delete file
            if file_path_obj.exists():
                file_path_obj.unlink()
            
            # Delete from database (cascade will handle related records)
            cursor.execute('DELETE FROM telemetry_sessions WHERE session_id = ?', (session_id,))
            deleted_count += 1
        
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            self.logger.info(f"Deleted {deleted_count} old sessions")
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT file_path, compressed FROM telemetry_sessions WHERE session_id = ?', 
                      (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        file_path, compressed = row
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            return None
        
        # Load data (handle compressed files)
        if compressed:
            with gzip.open(file_path_obj, 'rt') as f:
                return json.load(f)
        else:
            with open(file_path_obj, 'r') as f:
                return json.load(f)
    
    def query_sessions(self, start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      min_anomalies: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query sessions by time range and criteria"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT session_id, start_time, end_time, data_points FROM telemetry_sessions WHERE 1=1'
        params = []
        
        if start_time:
            query += ' AND start_time >= ?'
            params.append(start_time.isoformat())
        
        if end_time:
            # Include sessions that start on or before the end date
            # This ensures we capture all sessions that occurred during the date range
            query += ' AND start_time <= ?'
            params.append(end_time.isoformat())
        
        query += ' ORDER BY start_time DESC'
        
        cursor.execute(query, params)
        sessions = []
        
        for row in cursor.fetchall():
            session_id, start, end, points = row
            
            # Check anomaly count if requested
            if min_anomalies is not None:
                cursor.execute('SELECT COUNT(*) FROM anomalies WHERE session_id = ?', (session_id,))
                anomaly_count = cursor.fetchone()[0]
                if anomaly_count < min_anomalies:
                    continue
            
            sessions.append({
                'session_id': session_id,
                'start_time': start,
                'end_time': end,
                'data_points': points,
                'anomaly_count': self._get_anomaly_count(cursor, session_id)
            })
        
        conn.close()
        return sessions
    
    def _get_anomaly_count(self, cursor, session_id: str) -> int:
        """Get anomaly count for a session"""
        cursor.execute('SELECT COUNT(*) FROM anomalies WHERE session_id = ?', (session_id,))
        return cursor.fetchone()[0]
    
    def correlate_with_logs(self, session_id: str, log_entries: List[Dict[str, Any]]):
        """Correlate telemetry session with system log entries"""
        session = self.load_session(session_id)
        if not session:
            return
        
        telemetry_data = session.get('telemetry_data', [])
        anomalies = session.get('analysis', {}).get('anomaly_detection', {}).get('details', [])
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Correlate anomalies with logs
        for anomaly in anomalies:
            anomaly_time = datetime.fromisoformat(anomaly.get('timestamp', ''))
            
            # Find logs within ±5 minutes of anomaly
            time_window = timedelta(minutes=5)
            for log_entry in log_entries:
                log_time = datetime.fromisoformat(log_entry.get('timestamp', ''))
                
                if abs(anomaly_time - log_time) <= time_window:
                    # Calculate correlation score (closer = higher score)
                    time_diff = abs((anomaly_time - log_time).total_seconds())
                    score = max(0, 1.0 - (time_diff / 300.0))  # Normalize to 0-1
                    
                    cursor.execute('''
                        INSERT INTO log_correlations
                        (session_id, telemetry_timestamp, log_timestamp, log_source, 
                         log_level, log_message, correlation_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (session_id,
                          anomaly_time.isoformat(),
                          log_time.isoformat(),
                          log_entry.get('source', 'unknown'),
                          log_entry.get('level', 'INFO'),
                          log_entry.get('message', ''),
                          score))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Correlated {len(anomalies)} anomalies with system logs for session {session_id}")
    
    def get_correlated_events(self, session_id: str) -> List[Dict[str, Any]]:
        """Get correlated log events for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT telemetry_timestamp, log_timestamp, log_source, log_level, 
                   log_message, correlation_score
            FROM log_correlations
            WHERE session_id = ?
            ORDER BY correlation_score DESC
        ''', (session_id,))
        
        events = []
        for row in cursor.fetchall():
            events.append({
                'telemetry_time': row[0],
                'log_time': row[1],
                'source': row[2],
                'level': row[3],
                'message': row[4],
                'correlation_score': row[5]
            })
        
        conn.close()
        return events
    
    def get_correlated_events_count(self, session_id: str) -> int:
        """Get count of correlated log events for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM log_correlations
            WHERE session_id = ?
        ''', (session_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_correlated_events_counts(self, session_ids: List[str]) -> Dict[str, int]:
        """Get counts of correlated log events for multiple sessions"""
        if not session_ids:
            return {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Use IN clause for efficient query
        placeholders = ','.join('?' * len(session_ids))
        cursor.execute(f'''
            SELECT session_id, COUNT(*) as count
            FROM log_correlations
            WHERE session_id IN ({placeholders})
            GROUP BY session_id
        ''', session_ids)
        
        counts = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        
        # Ensure all session_ids have a count (even if 0)
        return {sid: counts.get(sid, 0) for sid in session_ids}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get archive statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total sessions
        cursor.execute('SELECT COUNT(*) FROM telemetry_sessions')
        stats['total_sessions'] = cursor.fetchone()[0]
        
        # Total data points
        cursor.execute('SELECT SUM(data_points) FROM telemetry_sessions')
        stats['total_data_points'] = cursor.fetchone()[0] or 0
        
        # Total anomalies
        cursor.execute('SELECT COUNT(*) FROM anomalies')
        stats['total_anomalies'] = cursor.fetchone()[0]
        
        # Date range
        cursor.execute('SELECT MIN(start_time), MAX(end_time) FROM telemetry_sessions')
        row = cursor.fetchone()
        if row[0]:
            stats['oldest_session'] = row[0]
            stats['newest_session'] = row[1]
        
        # Correlations
        cursor.execute('SELECT COUNT(*) FROM log_correlations')
        stats['total_correlations'] = cursor.fetchone()[0]
        
        conn.close()
        return stats

