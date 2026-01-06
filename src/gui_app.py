"""
GUI Application with Chat Interface
Web-based GUI for telemetry monitoring and AI chat
"""
import os
import gradio as gr
import json
import threading
import time
import csv
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from .telemetry_collector import TelemetryCollector
from .ai_analyzer import TelemetryAnalyzer
from .visualizer import TelemetryVisualizer
from .llm_analyzer import LLMAnalyzer
from .data_archive import DataArchive
from .system_logs import SystemLogReader
from .config_manager import ConfigManager
import plotly.graph_objects as go
import pandas as pd
import platform


class TelemetryGUI:
    """GUI application with chat interface"""
    
    def __init__(self, enable_archiving: bool = True, llm_provider: str = None, llm_model: str = None):
        self.config = ConfigManager()
        self.collector = TelemetryCollector()
        self.analyzer = TelemetryAnalyzer()
        self.visualizer = TelemetryVisualizer()
        # Load LLM config from preferences or use provided/env vars
        provider = llm_provider or self.config.get_llm_provider() or os.getenv("LLM_PROVIDER", "ollama")
        model = llm_model or self.config.get_llm_model() or os.getenv("LLM_MODEL", None)
        self.llm_analyzer = LLMAnalyzer(provider=provider, model=model)
        self.telemetry_history = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.system_info = None
        self.monitoring_status = "Ready"
        self.monitoring_start_time = None
        self.monitoring_duration = None
        self.is_live_monitoring = False
        self.monitoring_completed = False
        # Load archive settings from config
        archive_enabled = enable_archiving if enable_archiving is not None else self.config.get("archive.enabled", True)
        self.archive = DataArchive(
            retention_days=self.config.get("archive.retention_days", 30)
        ) if archive_enabled else None
        self.log_reader = SystemLogReader() if archive_enabled else None
        self.last_archived_session = None
    
    def update_llm_provider(self, provider: str, model: str = None) -> Tuple[str, str]:
        """Update the LLM provider and model"""
        try:
            # For llamacpp, convert colon to dash in model names (gemma3:1b -> gemma3-1b.gguf)
            if provider == "llamacpp" and model:
                # If model has colon, convert to dash and add .gguf if needed
                if ":" in model:
                    model = model.replace(":", "-")
                # Add .gguf extension if not present
                if not model.endswith(".gguf"):
                    model = model + ".gguf"
            
            self.llm_analyzer = LLMAnalyzer(provider=provider, model=model)
            status = "Connected" if self.llm_analyzer.is_available() else "Not available"
            model_info = self.llm_analyzer.get_model_info()
            return status, model_info
        except Exception as e:
            return f"Error: {str(e)}", "N/A"
    
    def get_available_models(self, provider: str) -> List[str]:
        """Get list of available models for a provider"""
        if provider == "ollama":
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    return [m.get('name', '') for m in models if m.get('name')]
            except:
                pass
            return ["llama3.2", "llama2", "alpaca", "gemma3:1b"]
        elif provider == "llamacpp":
            # List models in models directory
            from pathlib import Path
            project_root = Path(__file__).parent.parent
            models_dir = project_root / "models"
            if models_dir.exists():
                models = [f.name for f in models_dir.glob("*.gguf")]
                if models:
                    return models
            # Return default even if no models found (user can type custom name)
            return ["gemma3-1b.gguf"]
        elif provider == "openai":
            return ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        elif provider == "anthropic":
            return ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]
        return []
        
    def get_system_info_display(self):
        """Get formatted system information (compact version for main page)"""
        if not self.system_info:
            self.system_info = self.collector.get_system_info()
        
        info = self.system_info
        html = "<div style='font-family: monospace; font-size: 0.9em;'>"
        
        if 'os' in info:
            os_info = info['os']
            html += f"<p><strong>OS:</strong> {os_info.get('system', 'Unknown')} {os_info.get('release', '')}</p>"
        
        if 'cpu' in info:
            cpu_info = info['cpu']
            html += f"<p><strong>CPU:</strong> {cpu_info.get('model', 'Unknown')[:40]}</p>"
            html += f"<p><strong>Cores:</strong> {cpu_info.get('physical_cores', '?')}P/{cpu_info.get('logical_cores', '?')}L</p>"
        
        if 'memory' in info:
            mem_info = info['memory']
            html += f"<p><strong>Memory:</strong> {mem_info.get('total_gb', 0):.1f} GB</p>"
        
        html += "</div>"
        return html
    
    def get_detected_system_type(self) -> str:
        """Get detected system type"""
        system = platform.system()
        if system == "Windows":
            return "Windows"
        elif system == "Linux":
            return "Linux"
        elif system == "Darwin":
            return "macOS"
        return "Unknown"
    
    def get_welcome_page_content(self) -> str:
        """Generate welcome page content with system detection"""
        if not self.system_info:
            self.system_info = self.collector.get_system_info()
        
        info = self.system_info
        system_type = self.get_detected_system_type()
        
        # Build system details in a compact format with visible labels
        details = []
        if 'os' in info:
            os_info = info['os']
            os_str = f"{os_info.get('system', 'Unknown')} {os_info.get('release', '')}".strip()
            if os_str and os_str != 'Unknown':
                details.append(f"<span style='color: #1e40af; font-weight: 600;'>OS:</span> <span style='color: #374151;'>{os_str}</span>")
            arch = os_info.get('machine', '')
            if arch and arch != 'Unknown':
                details.append(f"<span style='color: #1e40af; font-weight: 600;'>Arch:</span> <span style='color: #374151;'>{arch}</span>")
        
        if 'cpu' in info:
            cpu_info = info['cpu']
            cpu_model = cpu_info.get('model', 'Unknown')
            if cpu_model and cpu_model != 'Unknown':
                # Truncate long CPU names
                if len(cpu_model) > 50:
                    cpu_model = cpu_model[:47] + "..."
                details.append(f"<span style='color: #1e40af; font-weight: 600;'>CPU:</span> <span style='color: #374151;'>{cpu_model}</span>")
            cores = f"{cpu_info.get('physical_cores', '?')}P/{cpu_info.get('logical_cores', '?')}L"
            if cores != '?P/?L':
                details.append(f"<span style='color: #1e40af; font-weight: 600;'>Cores:</span> <span style='color: #374151;'>{cores}</span>")
        
        if 'memory' in info:
            mem_info = info['memory']
            mem_gb = mem_info.get('total_gb', 0)
            if mem_gb > 0:
                details.append(f"<span style='color: #1e40af; font-weight: 600;'>RAM:</span> <span style='color: #374151;'>{mem_gb:.1f} GB</span>")
        
        if 'gpu' in info and isinstance(info['gpu'], list) and info['gpu']:
            gpu_list = []
            for gpu in info['gpu'][:1]:  # Show only first GPU
                vendor = gpu.get('vendor', '').strip()
                model = gpu.get('model', '').strip()
                if vendor or model:
                    gpu_str = f"{vendor} {model}".strip()
                    if len(gpu_str) > 40:
                        gpu_str = gpu_str[:37] + "..."
                    gpu_list.append(gpu_str)
            if gpu_list:
                details.append(f"<span style='color: #1e40af; font-weight: 600;'>GPU:</span> <span style='color: #374151;'>{gpu_list[0]}</span>")
        
        details_html = " <span style='color: #9ca3af; margin: 0 4px;'>•</span> ".join(details) if details else "<span style='color: #6b7280;'>System information unavailable</span>"
        
        html = f"""
        <div style='max-width: 800px; margin: 0 auto; padding: 20px;'>
            <h1 style='text-align: center; color: #2563eb; margin-bottom: 30px;'>🚀 Welcome to AIOS Telemetry Monitor</h1>
            
            <div style='background: #f0f9ff; border-left: 4px solid #2563eb; padding: 12px 15px; margin: 20px 0; border-radius: 4px;'>
                <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 10px;'>
                    <h2 style='margin: 0; font-size: 1.1em; color: #1e3a8a; font-weight: 600;'><i class='fas fa-desktop' style='margin-right: 6px;'></i> Detected System</h2>
                    <span style='font-size: 1.1em; font-weight: bold; color: #1e40af; background: #dbeafe; padding: 2px 8px; border-radius: 3px;'>{system_type}</span>
                </div>
                <div style='font-size: 0.95em; color: #1f2937; line-height: 1.8; font-family: system-ui, -apple-system, sans-serif;'>
                    {details_html}
                </div>
            </div>
            
            <div style='margin: 30px 0;'>
                <h2 style='font-size: 1.1em;'><i class='fas fa-clipboard-list' style='margin-right: 6px;'></i> Quick Setup</h2>
                <p style='color: #6b7280;'>Configure your preferences below. You can change these settings anytime from the Settings tab.</p>
            </div>
        </div>
        """
        return html
    
    def chat_with_ai(self, message: str, history: List) -> Tuple[str, List]:
        """Chat interface for asking questions about telemetry"""
        if not message.strip():
            return "", history
        
        # Ensure history is in the correct format (list of dicts with role/content)
        # Gradio Chatbot expects: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        if history:
            new_history = []
            for item in history:
                if item is None:
                    continue
                # Handle string format (legacy or corrupted)
                if isinstance(item, str):
                    try:
                        import json
                        # Try to parse if it's a JSON string
                        parsed = json.loads(item)
                        if isinstance(parsed, list):
                            item = parsed
                        elif isinstance(parsed, dict):
                            item = parsed
                    except:
                        # If it's just a plain string, skip it (invalid format)
                        continue
                
                # Handle tuple/list format (old Gradio format)
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    user_msg = item[0] if item[0] is not None else ""
                    assistant_msg = item[1] if item[1] is not None else ""
                    # Extract text from dict format if needed
                    if isinstance(user_msg, dict):
                        user_msg = user_msg.get('text', user_msg.get('content', ''))
                    if isinstance(assistant_msg, dict):
                        assistant_msg = assistant_msg.get('text', assistant_msg.get('content', ''))
                    user_msg = str(user_msg).strip()
                    assistant_msg = str(assistant_msg).strip()
                    if user_msg:
                        new_history.append({"role": "user", "content": user_msg})
                    if assistant_msg:
                        new_history.append({"role": "assistant", "content": assistant_msg})
                # Handle dict format (new Gradio format)
                elif isinstance(item, dict):
                    # Handle nested dicts (corrupted format)
                    if 'text' in item and isinstance(item['text'], (dict, list, str)):
                        text = item['text']
                        # If text is a string that looks like JSON, try to parse it
                        if isinstance(text, str) and (text.startswith('[') or text.startswith('{')):
                            try:
                                import json
                                text = json.loads(text)
                            except:
                                pass
                        # If text is still a dict/list, extract the actual content
                        if isinstance(text, dict):
                            text = text.get('text', text.get('content', str(text)))
                        elif isinstance(text, list) and len(text) > 0:
                            # Get first item's text
                            first = text[0]
                            if isinstance(first, dict):
                                text = first.get('text', first.get('content', str(first)))
                            else:
                                text = str(first)
                        content = str(text).strip()
                    else:
                        content = item.get("content") or item.get("text", "")
                        if isinstance(content, (dict, list)):
                            # Extract from nested structure
                            if isinstance(content, dict):
                                content = content.get('text', content.get('content', str(content)))
                            elif isinstance(content, list) and len(content) > 0:
                                first = content[0]
                                if isinstance(first, dict):
                                    content = first.get('text', first.get('content', str(first)))
                                else:
                                    content = str(first)
                        content = str(content).strip()
                    
                    if content:
                        new_history.append({
                            "role": item.get("role", "user"),
                            "content": content
                        })
            history = new_history
        else:
            history = []
        
        # Get current telemetry state
        current_data = self.telemetry_history[-1] if self.telemetry_history else None
        
        # Use LLM for all questions if available - it's smart enough to understand any question
        response = ""
        
        # Try LLM first if available and we have data
        if self.llm_analyzer.is_available() and len(self.telemetry_history) > 0:
            try:
                # Get analysis for context
                analysis = None
                if len(self.telemetry_history) >= 10:
                    analysis = self.analyzer.comprehensive_analysis(self.telemetry_history)
                
                # Use LLM to answer the question with full context
                response = self.llm_analyzer.answer_question(
                    message,
                    self.telemetry_history,
                    analysis,
                    self.system_info
                )
                # If we got a valid response, use it
                if response and str(response).strip():
                    # Update history and return
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": response})
                    return "", history
            except Exception as e:
                # LLM failed, fall through to basic responses
                response = f"Error getting LLM response: {str(e)}\n\n"
        
        # Fallback to basic responses if LLM not available, failed, or no data
        if not current_data:
            if not response:
                response = "No telemetry data available. Please start monitoring first."
            if not self.llm_analyzer.is_available():
                response += "\n\n*Note: For detailed AI responses, configure an LLM provider in Settings.*"
        else:
            # Provide a simple fallback message
            if not response:
                response = "I can help you with questions about your system telemetry. "
                response += "For detailed responses, please configure an LLM provider in Settings."
        
        # If we still don't have a response, provide a helpful message
        if not response or not response.strip():
            response = "I can help you with questions about your system telemetry. "
            if not self.llm_analyzer.is_available():
                response += "For detailed AI responses, please configure an LLM provider in Settings."
            elif len(self.telemetry_history) == 0:
                response += "Please start monitoring to collect telemetry data."
        
        # All topic-specific handlers removed - LLM handles everything intelligently
        # The old keyword-based handlers were removed because the LLM can understand any question
        
        # Ensure response is always a string (never None)
        if response is None:
            response = "I'm sorry, I couldn't generate a response. Please try again."
        response = str(response).strip()
        if not response:
            response = "I'm sorry, I couldn't generate a response. Please try again."
        
        # Clean message - extract from nested format if needed
        clean_message = str(message).strip()
        if clean_message.startswith('[') or clean_message.startswith('{'):
            try:
                import json
                parsed = json.loads(clean_message)
                if isinstance(parsed, list) and len(parsed) > 0:
                    first = parsed[0]
                    if isinstance(first, dict):
                        clean_message = first.get('text', first.get('content', str(first)))
                    else:
                        clean_message = str(first)
                elif isinstance(parsed, dict):
                    clean_message = parsed.get('text', parsed.get('content', str(parsed)))
            except:
                pass  # Keep original if parsing fails
        
        # Add messages in correct format (Gradio Chatbot expects list of dicts)
        history.append({"role": "user", "content": clean_message})
        history.append({"role": "assistant", "content": response})
        return "", history
    
    def start_monitoring(self, duration: int, interval: float, live: bool = False):
        """Start telemetry monitoring"""
        if self.is_monitoring:
            return "Monitoring already in progress!", gr.update()
        
        self.is_monitoring = True
        self.is_live_monitoring = live
        self.telemetry_history = []
        self.monitoring_start_time = time.time()
        self.monitoring_duration = None if live else duration
        self.monitoring_completed = False
        
        def monitor():
            if live:
                # Live monitoring - runs until stopped
                self.monitoring_status = "<i class='fas fa-circle' style='color: #10b981; margin-right: 4px;'></i> Live monitoring active..."
                while self.is_monitoring:
                    data = self.collector.collect_all_telemetry()
                    self.telemetry_history.append(data)
                    elapsed = time.time() - self.monitoring_start_time
                    self.monitoring_status = f"<i class='fas fa-circle' style='color: #10b981; margin-right: 4px;'></i> Live monitoring... ({len(self.telemetry_history)} data points, {elapsed:.0f}s elapsed)"
                    time.sleep(interval)
            else:
                # Timed monitoring
                end_time = time.time() + duration
                while time.time() < end_time and self.is_monitoring:
                    data = self.collector.collect_all_telemetry()
                    self.telemetry_history.append(data)
                    elapsed = time.time() - self.monitoring_start_time
                    remaining = max(0, end_time - time.time())
                    self.monitoring_status = f"<i class='fas fa-circle' style='color: #10b981; margin-right: 4px;'></i> Monitoring... ({len(self.telemetry_history)} points, {remaining:.0f}s remaining)"
                    time.sleep(interval)
            
            # Monitoring ended
            if self.is_monitoring:  # Only update if it ended naturally (not stopped manually)
                points = len(self.telemetry_history)
                elapsed = time.time() - self.monitoring_start_time
                self.monitoring_status = f"<i class='fas fa-check-circle' style='color: #10b981; margin-right: 4px;'></i> Monitoring completed! Collected {points} data points over {elapsed:.0f} seconds."
                self.monitoring_completed = True
            self.is_monitoring = False
            self.is_live_monitoring = False
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        
        if live:
            status_msg = "<i class='fas fa-circle' style='color: #10b981; margin-right: 4px;'></i> Live monitoring started! Click Stop to end."
        else:
            status_msg = f"<i class='fas fa-circle' style='color: #10b981; margin-right: 4px;'></i> Monitoring started for {duration} seconds!"
        
        return status_msg, gr.update()
    
    def stop_monitoring(self):
        """Stop telemetry monitoring"""
        was_monitoring = self.is_monitoring
        self.is_monitoring = False
        self.is_live_monitoring = False
        self.monitoring_completed = False
        
        if was_monitoring:
            points = len(self.telemetry_history)
            elapsed = time.time() - self.monitoring_start_time if self.monitoring_start_time else 0
            status_msg = f"<i class='fas fa-stop-circle' style='color: #ef4444; margin-right: 4px;'></i> Monitoring stopped. Collected {points} data points over {elapsed:.0f} seconds."
            self.monitoring_status = status_msg
            return status_msg, gr.update()
        else:
            return "No monitoring in progress.", gr.update()
    
    def get_monitoring_status(self):
        """Get current monitoring status"""
        if self.is_monitoring:
            return self.monitoring_status
        elif self.monitoring_completed or "<i class='fas fa-check-circle'" in self.monitoring_status:
            return self.monitoring_status  # Keep completion message
        elif "<i class='fas fa-stop-circle'" in self.monitoring_status:
            return self.monitoring_status  # Keep stop message
        else:
            return "Ready - Click 'Start Monitoring' to begin"
    
    def get_latest_metrics(self):
        """Get latest telemetry metrics for display"""
        if not self.telemetry_history:
            return "No data collected yet. Start monitoring first."
        
        latest = self.telemetry_history[-1]
        cpu = latest.get('cpu', {}).get('cpu_percent', 0)
        mem = latest.get('memory', {}).get('virtual_memory', {}).get('percent', 0)
        disk = latest.get('disk', {}).get('disk_usage', {}).get('percent', 0)
        
        html = f"""
        <div style='font-family: monospace; font-size: 14px;'>
            <h3><i class='fas fa-chart-line' style='margin-right: 6px;'></i> Current Metrics</h3>
            <p><strong>CPU:</strong> {cpu:.1f}%</p>
            <p><strong>Memory:</strong> {mem:.1f}%</p>
            <p><strong>Disk:</strong> {disk:.1f}%</p>
            <p><strong>Data Points:</strong> {len(self.telemetry_history)}</p>
        </div>
        """
        return html
    
    def update_plot(self):
        """Update the telemetry plot"""
        if len(self.telemetry_history) < 2:
            return None
        
        df = self.visualizer.prepare_dataframe(self.telemetry_history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cpu_percent'],
            mode='lines',
            name='CPU %',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['memory_percent'],
            mode='lines',
            name='Memory %',
            line=dict(color='green', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['disk_percent'],
            mode='lines',
            name='Disk %',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title='Real-time Telemetry',
            xaxis_title='Time',
            yaxis_title='Percentage (%)',
            height=400
        )
        
        return fig
    
    def analyze_data(self, use_llm: bool = False, include_predictions: bool = False, prediction_steps: int = 10):
        """Run AI analysis on collected data
        
        Args:
            use_llm: If True, include LLM-powered insights and explanations
            include_predictions: If True, include anomaly predictions
            prediction_steps: Number of steps ahead to predict (requires at least 20 data points)
        """
        if len(self.telemetry_history) < 10:
            return "Need at least 10 data points for analysis. Please collect more data."
        
        # Check if we have enough data for predictions
        if include_predictions and len(self.telemetry_history) < 20:
            return f"Need at least 20 data points for predictions. Currently have {len(self.telemetry_history)}. Please collect more data."
        
        # Show initial status
        status_msg = "🔄 Running AI analysis..."
        if include_predictions:
            status_msg += " (including predictions - this may take a moment)"
        if use_llm:
            status_msg += " (with LLM insights - this may take longer)"
        
        # Run analysis
        analysis = self.analyzer.comprehensive_analysis(
            self.telemetry_history,
            include_predictions=include_predictions,
            prediction_steps=prediction_steps
        )
        
        # Archive if enabled
        system_logs = []
        if self.archive:
            try:
                # Get system logs for correlation
                if self.log_reader and self.telemetry_history:
                    from datetime import timedelta
                    start_time = datetime.fromisoformat(self.telemetry_history[0].get('timestamp', datetime.now().isoformat()))
                    system_logs = self.log_reader.read_system_logs(
                        since=start_time - timedelta(minutes=5),
                        until=start_time + timedelta(minutes=2),
                        limit=200
                    )
                
                # Archive session
                session_id = self.archive.archive_session(
                    self.telemetry_history,
                    analysis,
                    self.system_info
                )
                
                # Correlate with system logs
                if session_id and system_logs:
                    self.archive.correlate_with_logs(session_id, system_logs)
                    self.last_archived_session = session_id
            except Exception as e:
                # Silently fail archiving if there's an error
                pass
        
        # Format analysis results
        result = "## AI Analysis Results\n\n"
        
        # Add LLM insights if requested and available
        if use_llm:
            result += "### <i class='fas fa-brain' style='margin-right: 6px;'></i> LLM Analysis\n\n"
            if self.llm_analyzer.is_available():
                try:
                    result += "*Generating LLM insights... This may take a moment.*\n\n"
                    llm_insights = self.llm_analyzer.analyze_performance(
                        self.telemetry_history,
                        analysis,
                        self.system_info
                    )
                    if llm_insights and str(llm_insights).strip():
                        result += str(llm_insights).strip() + "\n\n"
                    else:
                        result += "<i class='fas fa-exclamation-triangle' style='color: #f59e0b; margin-right: 4px;'></i> *LLM returned empty response. Showing standard analysis below.*\n\n"
                except Exception as e:
                    result += f"<i class='fas fa-exclamation-triangle' style='color: #f59e0b; margin-right: 4px;'></i> *LLM analysis error: {str(e)}*\n"
                    result += "*Showing standard analysis below.*\n\n"
            else:
                result += "<i class='fas fa-exclamation-triangle' style='color: #f59e0b; margin-right: 4px;'></i> *LLM not available.*\n"
                result += "*To enable LLM insights, configure an LLM provider in the Settings tab.*\n"
                result += "*Showing standard analysis below.*\n\n"
            result += "---\n\n"
        
        # Performance Insights
        insights = analysis.get('performance_insights', {})
        if 'current_status' in insights:
            result += "### <i class='fas fa-chart-line' style='margin-right: 6px;'></i> Performance Status\n"
            for metric, data in insights['current_status'].items():
                status_icon = "<i class='fas fa-circle' style='color: #ef4444; margin-right: 4px;'></i>" if data.get('status') == 'high' else "<i class='fas fa-circle' style='color: #f59e0b; margin-right: 4px;'></i>" if data.get('status') == 'moderate' else "<i class='fas fa-circle' style='color: #10b981; margin-right: 4px;'></i>"
                
                # Handle different metric types
                if metric == 'temperature':
                    value = data.get('average', 'N/A')
                    if isinstance(value, (int, float)):
                        result += f"{status_icon} **{metric.upper()}**: {value:.2f}°C ({data.get('status', 'unknown')})\n"
                    else:
                        result += f"{status_icon} **{metric.upper()}**: {value} ({data.get('status', 'unknown')})\n"
                elif metric == 'network':
                    sent_mb = data.get('bytes_sent_mb', 0)
                    recv_mb = data.get('bytes_recv_mb', 0)
                    connections = data.get('connections', 0)
                    result += f"{status_icon} **{metric.upper()}**: {sent_mb:.1f} MB sent, {recv_mb:.1f} MB recv ({connections} connections)\n"
                elif metric == 'power':
                    power_str = []
                    if 'gpu_watts' in data:
                        power_str.append(f"GPU: {data['gpu_watts']:.1f}W")
                    if 'cpu_energy_joules' in data:
                        power_str.append(f"CPU: {data['cpu_energy_joules']:.1f}J")
                    if power_str:
                        result += f"{status_icon} **{metric.upper()}**: {', '.join(power_str)}\n"
                    else:
                        result += f"{status_icon} **{metric.upper()}**: Not available\n"
                elif metric == 'battery':
                    percent = data.get('percent', 'N/A')
                    status = data.get('status', 'unknown')
                    plugged = data.get('power_plugged', False)
                    result += f"{status_icon} **{metric.upper()}**: {percent}% ({status}, {'Plugged in' if plugged else 'Unplugged'})\n"
                elif metric == 'processes':
                    total = data.get('total', 0)
                    high_cpu = data.get('high_cpu_count', 0)
                    high_mem = data.get('high_memory_count', 0)
                    result += f"{status_icon} **{metric.upper()}**: {total} total"
                    if high_cpu > 0 or high_mem > 0:
                        result += f" ({high_cpu} high CPU, {high_mem} high memory)"
                    result += "\n"
                    # Show top processes
                    top_processes = data.get('top_processes', [])[:5]
                    if top_processes:
                        result += "  Top processes:\n"
                        for proc in top_processes:
                            proc_name = proc.get('name', 'unknown')
                            proc_cpu = proc.get('cpu_percent', 0) or 0
                            proc_mem = proc.get('memory_percent', 0) or 0
                            if proc_cpu > 0 or proc_mem > 0.1:
                                result += f"    • {proc_name}: CPU {proc_cpu:.1f}%, Mem {proc_mem:.2f}%\n"
                else:
                    # Default: CPU, Memory, Disk (percentages)
                    value = data.get('usage', data.get('average', 'N/A'))
                    if isinstance(value, (int, float)):
                        result += f"{status_icon} **{metric.upper()}**: {value:.2f}% ({data.get('status', 'unknown')})\n"
                    else:
                        result += f"{status_icon} **{metric.upper()}**: {value} ({data.get('status', 'unknown')})\n"
        
        # Anomaly Detection
        anomaly = analysis.get('anomaly_detection', {})
        if anomaly.get('anomalies_detected', 0) > 0:
            result += f"\n### <i class='fas fa-search' style='margin-right: 6px;'></i> Anomalies Detected\n"
            result += f"- Found {anomaly['anomalies_detected']} anomalies ({anomaly.get('anomaly_percentage', 0):.1f}%)\n"
        
        # Predictions
        if 'predictions' in analysis:
            predictions = analysis['predictions']
            if 'error' in predictions:
                result += f"\n### <i class='fas fa-crystal-ball' style='margin-right: 6px;'></i> Anomaly Predictions\n"
                result += f"<i class='fas fa-exclamation-triangle' style='color: #f59e0b; margin-right: 4px;'></i> {predictions['error']}\n"
            elif 'anomaly_predictions' in predictions:
                pred_data = predictions['anomaly_predictions']
                if pred_data.get('success'):
                    summary = pred_data.get('summary', {})
                    result += f"\n### <i class='fas fa-crystal-ball' style='margin-right: 6px;'></i> Anomaly Predictions ({summary.get('total_steps', 0)} steps ahead)\n"
                    result += f"- **Predicted anomalies:** {summary.get('predicted_anomalies', 0)} ({summary.get('anomaly_percentage', 0):.1f}%)\n"
                    
                    high_risk = summary.get('high_risk_steps', [])
                    if high_risk:
                        result += f"- **<i class='fas fa-exclamation-triangle' style='color: #f59e0b; margin-right: 4px;'></i> High-risk steps:** {', '.join(map(str, high_risk[:10]))}\n"
                    
                    # Show detailed predictions for high-risk steps
                    anomaly_preds = pred_data.get('anomaly_predictions', [])
                    if anomaly_preds:
                        high_risk_preds = [p for p in anomaly_preds if p.get('likelihood') == 'high'][:5]
                        if high_risk_preds:
                            result += f"\n**High-Risk Predictions:**\n"
                            for pred in high_risk_preds:
                                timestamp = pred.get('timestamp', 'N/A')
                                if isinstance(timestamp, str) and len(timestamp) > 19:
                                    timestamp = timestamp[:19]
                                metrics = pred.get('predicted_metrics', {})
                                result += f"- **Step {pred.get('step', '?')}** ({timestamp}):\n"
                                result += f"  - Risk: {pred.get('likelihood', 'unknown').upper()}, Score: {pred.get('anomaly_score', 0):.3f}\n"
                                if metrics:
                                    cpu = metrics.get('cpu_percent', 0)
                                    mem = metrics.get('memory_percent', 0)
                                    disk = metrics.get('disk_percent', 0)
                                    result += f"  - Predicted: CPU={cpu:.1f}%, Memory={mem:.1f}%, Disk={disk:.1f}%\n"
                    
                    # Show anomaly patterns if available
                    if 'anomaly_patterns' in predictions:
                        patterns = predictions['anomaly_patterns']
                        if patterns.get('success') and patterns.get('patterns'):
                            result += f"\n**Anomaly Patterns Detected:**\n"
                            for pattern in patterns['patterns'][:3]:  # Show top 3 patterns
                                result += f"- {pattern.get('description', 'Pattern')}\n"
        
        # Recommendations
        if insights.get('recommendations'):
            result += f"\n### <i class='fas fa-lightbulb' style='margin-right: 6px;'></i> Recommendations\n"
            for rec in insights['recommendations']:
                result += f"- {rec}\n"
        
        # Archive info
        if self.archive and self.last_archived_session:
            result += f"\n### <i class='fas fa-archive' style='margin-right: 6px;'></i> Archive\n"
            result += f"- Session archived: `{self.last_archived_session}`\n"
            if system_logs:
                correlated = len(self.archive.get_correlated_events(self.last_archived_session))
                result += f"- Correlated with {correlated} system log events\n"
        
        return result
    
    def list_archived_sessions(self, limit: int = 50, offset: int = 0):
        """List archived sessions with pagination - returns DataFrame, summary, and total count"""
        if not self.archive:
            return pd.DataFrame(), "Archiving is not enabled.", 0
        
        # Store full sessions list for row click handling
        self._current_sessions_df = None
        
        # Query all sessions (ordered by start_time DESC to get latest first)
        all_sessions = self.archive.query_sessions(
            start_time=None,
            end_time=None,
            min_anomalies=None
        )
        
        if not all_sessions:
            return pd.DataFrame(), "No archived sessions found.", 0
        
        # Get correlated event counts for all sessions
        all_session_ids = [s['session_id'] for s in all_sessions]
        correlation_counts = self.archive.get_correlated_events_counts(all_session_ids)
        
        # Create DataFrame with session data including checkbox column
        table_data = []
        for session in all_sessions:
            session_id = session['session_id']
            correlation_count = correlation_counts.get(session_id, 0)
            start_time_str = session['start_time']
            # Extract just the date part for display
            try:
                date_display = datetime.fromisoformat(start_time_str.replace('Z', '+00:00')).strftime('%Y-%m-%d')
            except:
                date_display = start_time_str[:10] if len(start_time_str) >= 10 else start_time_str
            
            table_data.append({
                'Select': False,  # Checkbox column for export selection
                'Session ID': session_id,
                'Date': date_display,
                'Data Points': session['data_points'],
                'Anomalies': session['anomaly_count'],
                'Log Correlations': correlation_count
            })
        
        # Create full DataFrame
        full_df = pd.DataFrame(table_data)
        
        # Apply pagination
        total_count = len(full_df)
        paginated_df = full_df.iloc[offset:offset + limit].copy()
        
        # Store the full DataFrame for row click handling
        self._current_sessions_df = paginated_df.copy()
        
        # Create summary text
        start_idx = offset + 1 if total_count > 0 else 0
        end_idx = min(offset + limit, total_count)
        summary = f"**Showing {start_idx}-{end_idx} of {total_count} sessions**"
        
        return paginated_df, summary, total_count
    
    def query_archived_session(self, session_id: str, show_telemetry: bool = True):
        """Query a specific archived session - always shows all telemetry data"""
        if not self.archive:
            return "Archiving is not enabled."
        
        if not session_id or not session_id.strip():
            return "Please enter a session ID."
        
        session = self.archive.load_session(session_id.strip())
        if not session:
            return f"Session '{session_id}' not found."
        
        result = f"## <i class='fas fa-box' style='margin-right: 6px;'></i> Session: {session_id}\n\n"
        result += f"**Start Time:** {session.get('start_time')}\n"
        result += f"**End Time:** {session.get('end_time')}\n"
        result += f"**Data Points:** {len(session.get('telemetry_data', []))}\n\n"
        
        # Always show all telemetry data
        telemetry_data = session.get('telemetry_data', [])
        if telemetry_data:
            result += f"### <i class='fas fa-database' style='margin-right: 6px;'></i> Telemetry Data ({len(telemetry_data)} points)\n\n"
            # Show summary statistics for key metrics
            if len(telemetry_data) > 0:
                # Calculate averages for key metrics
                cpu_values = [d.get('cpu', {}).get('cpu_percent', 0) for d in telemetry_data if d.get('cpu', {}).get('cpu_percent')]
                mem_values = [d.get('memory', {}).get('virtual_memory', {}).get('percent', 0) for d in telemetry_data if d.get('memory', {}).get('virtual_memory', {}).get('percent')]
                
                if cpu_values:
                    result += f"**CPU Usage:**\n"
                    result += f"- Average: {sum(cpu_values) / len(cpu_values):.1f}%\n"
                    result += f"- Min: {min(cpu_values):.1f}%, Max: {max(cpu_values):.1f}%\n\n"
                
                if mem_values:
                    result += f"**Memory Usage:**\n"
                    result += f"- Average: {sum(mem_values) / len(mem_values):.1f}%\n"
                    result += f"- Min: {min(mem_values):.1f}%, Max: {max(mem_values):.1f}%\n\n"
                
                # Show all data points in a table format
                result += f"**All Data Points:**\n\n"
                result += "| # | Timestamp | CPU % | Memory % | Disk % |\n"
                result += "|---|-----------|-------|----------|--------|\n"
                for i, data_point in enumerate(telemetry_data, 1):
                    timestamp = data_point.get('timestamp', 'N/A')
                    cpu = data_point.get('cpu', {}).get('cpu_percent', 'N/A')
                    mem = data_point.get('memory', {}).get('virtual_memory', {}).get('percent', 'N/A')
                    disk = data_point.get('disk', {}).get('disk_usage', {}).get('percent', 'N/A')
                    # Format values
                    cpu_str = f"{cpu:.1f}" if isinstance(cpu, (int, float)) else str(cpu)
                    mem_str = f"{mem:.1f}" if isinstance(mem, (int, float)) else str(mem)
                    disk_str = f"{disk:.1f}" if isinstance(disk, (int, float)) else str(disk)
                    result += f"| {i} | {timestamp} | {cpu_str} | {mem_str} | {disk_str} |\n"
                result += "\n"
        
        # Show analysis summary if available
        analysis = session.get('analysis', {})
        if analysis:
            anomaly = analysis.get('anomaly_detection', {})
            if anomaly.get('anomalies_detected', 0) > 0:
                result += f"### <i class='fas fa-search' style='margin-right: 6px;'></i> Anomalies\n"
                result += f"- Detected: {anomaly['anomalies_detected']} ({anomaly.get('anomaly_percentage', 0):.1f}%)\n\n"
        
        # Show correlated log events
        events = self.archive.get_correlated_events(session_id.strip())
        if events:
            result += f"### <i class='fas fa-link' style='margin-right: 6px;'></i> Correlated System Log Events ({len(events)})\n\n"
            for event in events[:10]:  # Show top 10
                result += f"**[{event['log_time']}]** {event['source']} - {event['level']}\n"
                result += f"- {event['message'][:100]}{'...' if len(event['message']) > 100 else ''}\n"
                result += f"- Correlation: {event['correlation_score']:.2f}\n\n"
        else:
            result += "### <i class='fas fa-link' style='margin-right: 6px;'></i> Correlated Log Events\n"
            result += "No correlated log events found.\n"
        
        return result
    
    def export_session(self, session_id: str, format_type: str = "json") -> str:
        """Export a single session to CSV or JSON"""
        if not self.archive:
            return "Archiving is not enabled."
        
        if not session_id or not session_id.strip():
            return "Please enter a session ID."
        
        session = self.archive.load_session(session_id.strip())
        if not session:
            return f"Session '{session_id}' not found."
        
        telemetry_data = session.get('telemetry_data', [])
        if not telemetry_data:
            return f"Session '{session_id}' has no telemetry data to export."
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{session_id}_{timestamp}.{format_type}"
            filepath = Path(filename)
            
            if format_type == "csv":
                # Flatten and export to CSV
                all_keys = set()
                for data in telemetry_data:
                    all_keys.update(self._flatten_dict(data).keys())
                
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                    writer.writeheader()
                    for data in telemetry_data:
                        flat_data = self._flatten_dict(data)
                        writer.writerow(flat_data)
            else:  # JSON
                with open(filepath, 'w') as f:
                    json.dump(session, f, indent=2, default=str)
            
            return f"**Export successful!**\n\nExported {len(telemetry_data)} data points to: `{filename}`"
        except Exception as e:
            return f"**Export failed:** {str(e)}"
    
    def export_selected_sessions(self, table_data, format_type: str = "json") -> Tuple[str, Optional[str]]:
        """Export selected sessions from table (where Select=True) to CSV or JSON - returns (status_message, file_path)"""
        if not self.archive:
            return "Archiving is not enabled.", None
        
        # Handle DataFrame input - get rows where Select column is True
        if isinstance(table_data, pd.DataFrame):
            if table_data.empty:
                return "Please select at least one session to export (check the 'Select' column).", None
            
            # Filter rows where Select is True
            if 'Select' in table_data.columns:
                selected_rows = table_data[table_data['Select'] == True]
                if selected_rows.empty:
                    return "Please check the 'Select' column for sessions you want to export.", None
                session_ids = selected_rows['Session ID'].tolist()
            else:
                # Fallback: if no Select column, export all visible rows
                if 'Session ID' in table_data.columns:
                    session_ids = table_data['Session ID'].tolist()
                else:
                    return "Invalid table format.", None
        else:
            return "Invalid selection format.", None
        
        exported = []
        failed = []
        all_telemetry = []
        
        for session_id in session_ids:
            session = self.archive.load_session(session_id)
            if session:
                telemetry_data = session.get('telemetry_data', [])
                if telemetry_data:
                    all_telemetry.extend(telemetry_data)
                    exported.append(session_id)
                else:
                    failed.append(f"{session_id} (no data)")
            else:
                failed.append(f"{session_id} (not found)")
        
        if not all_telemetry:
            return f"**No data to export.**\n\nFailed sessions: {', '.join(failed) if failed else 'None'}", None
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"exported_sessions_{timestamp}.{format_type}"
            filepath = Path(filename)
            
            if format_type == "csv":
                # Flatten and export to CSV
                all_keys = set()
                for data in all_telemetry:
                    all_keys.update(self._flatten_dict(data).keys())
                
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                    writer.writeheader()
                    for data in all_telemetry:
                        flat_data = self._flatten_dict(data)
                        writer.writerow(flat_data)
            else:  # JSON
                export_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'sessions_exported': exported,
                    'total_data_points': len(all_telemetry),
                    'telemetry_data': all_telemetry
                }
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            result = f"**Export successful!**\n\n"
            result += f"Exported {len(exported)} sessions with {len(all_telemetry)} total data points\n"
            result += f"File: `{filename}`\n\n"
            if failed:
                result += f"**Failed sessions:** {', '.join(failed)}"
            
            return result, str(filepath.absolute())
        except Exception as e:
            return f"**Export failed:** {str(e)}", None
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for CSV export"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                if v and isinstance(v[0], dict):
                    items.extend(self._flatten_dict(v[0], new_key, sep=sep).items())
                else:
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_archive_stats(self):
        """Get archive statistics"""
        if not self.archive:
            return "Archiving is not enabled."
        
        stats = self.archive.get_statistics()
        result = "## <i class='fas fa-chart-pie' style='margin-right: 6px;'></i> Archive Statistics\n\n"
        result += f"**Total Sessions:** {stats.get('total_sessions', 0)}\n"
        result += f"**Total Data Points:** {stats.get('total_data_points', 0):,}\n"
        result += f"**Total Anomalies:** {stats.get('total_anomalies', 0)}\n"
        result += f"**Total Correlations:** {stats.get('total_correlations', 0)}\n\n"
        
        if stats.get('oldest_session'):
            result += f"**Oldest Session:** {stats['oldest_session']}\n"
            result += f"**Newest Session:** {stats['newest_session']}\n"
        
        return result
    
    def _icon(self, icon_name: str, style: str = "fas") -> str:
        """Generate Font Awesome icon HTML for Markdown/HTML components"""
        return f"<i class='{style} fa-{icon_name}' style='margin-right: 6px;'></i>"
    
    def _icon_unicode(self, icon_name: str) -> str:
        """Get Unicode symbol for buttons/tabs (components that don't support HTML)"""
        icon_map = {
            'play': '▶️',
            'stop': '⏹️',
            'sync': '🔄',
            'search': '🔍',
            'brain': '🤖',
            'cog': '⚙️',
            'hand-wave': '👋',
            'chart-line': '📊',
            'archive': '📦',
            'list': '📚',
            'chart-pie': '📊',
            'save': '💾',
            'check-circle': '✅',
            'check': '✓',
            'comments': '💬',
            'paper-plane': '📤',
            'chart-bar': '📈',
            'crystal-ball': '🔮',
            'lightbulb': '💡',
            'desktop': '🖥️',
            'clipboard-list': '📋',
            'link': '🔗',
            'box': '📦',
            'hourglass-half': '⏳',
            'robot': '🤖'
        }
        return icon_map.get(icon_name, '•')
    
    def create_interface(self):
        """Create the Gradio interface"""
        # Add Font Awesome for icons with CSS-based icon injection for buttons
        custom_css = """
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
        
        .fa-icon { margin-right: 6px; display: inline-block; }
        
        /* Add Font Awesome icons to buttons using CSS classes */
        .icon-play::before { content: "\\f04b"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 6px; }
        .icon-stop::before { content: "\\f04d"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 6px; }
        .icon-sync::before { content: "\\f021"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 6px; }
        .icon-search::before { content: "\\f002"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 6px; }
        .icon-brain::before { content: "\\f5dc"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 6px; }
        .icon-save::before { 
            content: "\\f0c7"; 
            font-family: "Font Awesome 6 Free"; 
            font-weight: 900; 
            margin-right: 6px; 
            display: inline !important;
            vertical-align: middle;
        }
        button.icon-save {
            display: inline-flex !important;
            align-items: center !important;
        }
        .icon-check-circle::before { content: "\\f058"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 6px; }
        .icon-paper-plane::before { content: "\\f1d8"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 6px; }
        .icon-list::before { content: "\\f03a"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 6px; }
        .icon-chart-pie::before { content: "\\f200"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 6px; }
        .icon-chevron-left::before { content: "\\f053"; font-family: "Font Awesome 6 Free"; font-weight: 900; }
        .icon-chevron-right::before { content: "\\f054"; font-family: "Font Awesome 6 Free"; font-weight: 900; }
        .icon-times::before { content: "\\f00d"; font-family: "Font Awesome 6 Free"; font-weight: 900; }
        
        /* Make pagination and close buttons narrower */
        button.icon-chevron-left, button.icon-chevron-right, button.icon-times {
            min-width: 24px !important;
            width: 24px !important;
            max-width: 24px !important;
            padding: 4px 2px !important;
        }
        
        /* Save buttons should fit their text content without wrapping */
        button.icon-save {
            width: auto !important;
            min-width: auto !important;
            max-width: 200px !important;
            white-space: nowrap !important;
            padding: 2px 2px !important;
        }
        
        /* Compact radio buttons - bring them closer to export button */
        .compact-radio {
            margin-left: 8px !important;
        }
        .compact-radio .wrap {
            gap: 4px !important;
        }
        
        /* Add icons to tabs using JavaScript-injected classes */
        .tab-icon-welcome::before { content: "\\f4ad"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 6px; }
        .tab-icon-settings::before { content: "\\f013"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 6px; }
        .tab-icon-monitoring::before { content: "\\f201"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 6px; }
        .tab-icon-archive::before { content: "\\f187"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 6px; }
        
        /* Add icon to checkbox label */
        .checkbox-icon-predictions::before { content: "\\f6e8"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 6px; }
        """
        
        # JavaScript to add icon classes to tabs and checkbox
        custom_js = """
        function addTabIcons() {
            // Add icons to tab buttons
            setTimeout(() => {
                const tabButtons = document.querySelectorAll('.tab-nav button, button[role="tab"]');
                tabButtons.forEach(btn => {
                    const text = btn.textContent.trim();
                    if (text === 'Welcome') {
                        btn.classList.add('tab-icon-welcome');
                    } else if (text === 'Settings') {
                        btn.classList.add('tab-icon-settings');
                    } else if (text === 'Monitoring') {
                        btn.classList.add('tab-icon-monitoring');
                    } else if (text === 'Archive') {
                        btn.classList.add('tab-icon-archive');
                    }
                });
                
                // Add icon to checkbox label
                const checkboxes = document.querySelectorAll('label');
                checkboxes.forEach(label => {
                    if (label.textContent.includes('Include Predictions')) {
                        label.classList.add('checkbox-icon-predictions');
                    }
                });
            }, 100);
        }
        
        document.addEventListener('DOMContentLoaded', addTabIcons);
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', addTabIcons);
        } else {
            addTabIcons();
        }
        
        // Re-run after Gradio updates
        if (window.gradio) {
            const originalUpdate = window.gradio.update;
            window.gradio.update = function(...args) {
                const result = originalUpdate.apply(this, args);
                setTimeout(addTabIcons, 100);
                return result;
            };
        }
        """
        
        with gr.Blocks(title="AI Telemetry Monitor", theme=gr.themes.Soft(), css=custom_css, head=f"<script>{custom_js}</script>") as app:
            gr.HTML("<h1><i class='fas fa-robot' style='margin-right: 8px;'></i> AI Telemetry Monitoring & Analysis</h1>")
            gr.Markdown("Monitor your system in real-time and chat with AI about your telemetry data")
            
            # Define shared components that need to be updated from multiple tabs
            llm_status_display = gr.Markdown(
                value=f"**LLM:** {self.llm_analyzer.provider.upper()} - {self.llm_analyzer.get_model_info()}\n"
                      f"**Status:** {'Connected' if self.llm_analyzer.is_available() else 'Not available'}\n\n"
                      f"*Note: Configure LLM settings in the Settings tab*"
            )
            
            # Create tabs for different sections
            with gr.Tabs():
                # Welcome/Settings tab (show first if first run, otherwise show as Settings)
                if self.config.is_first_run():
                    tab_name = "Welcome"
                else:
                    tab_name = "Settings"
                with gr.TabItem(tab_name):
                    welcome_content = gr.HTML(value=self.get_welcome_page_content())
                    
                    gr.Markdown(f"## {self._icon('brain')} LLM Configuration")
                    gr.Markdown("Configure your AI assistant provider and model")
                    
                    with gr.Row():
                        llm_provider_dropdown = gr.Dropdown(
                            choices=["ollama", "llamacpp", "openai", "anthropic"],
                            value=self.llm_analyzer.provider,
                            label="Provider",
                            interactive=True
                        )
                        llm_model_dropdown = gr.Dropdown(
                            choices=self.get_available_models(self.llm_analyzer.provider),
                            value=self.llm_analyzer.model,
                            label="Model",
                            interactive=True,
                            allow_custom_value=True
                        )
                    
                    def update_model_choices(provider):
                        """Update model dropdown when provider changes"""
                        models = self.get_available_models(provider)
                        default_model = models[0] if models else ""
                        return {
                            "choices": models,
                            "value": default_model
                        }
                    
                    llm_provider_dropdown.change(
                        fn=update_model_choices,
                        inputs=llm_provider_dropdown,
                        outputs=llm_model_dropdown
                    )
                    
                    llm_status = gr.Textbox(
                        value="Connected" if self.llm_analyzer.is_available() else "Not available",
                        label="Status",
                        interactive=False
                    )
                    llm_model_info = gr.Textbox(
                        value=self.llm_analyzer.get_model_info(),
                        label="Current Model",
                        interactive=False
                    )
                    
                    llm_hint = gr.Markdown(value="")
                    
                    def apply_llm_settings(provider, model):
                        """Apply LLM provider and model changes and save to config"""
                        status, model_info = self.update_llm_provider(provider, model)
                        # Save to config
                        self.config.set_llm_config(provider, model)
                        # Update the monitoring tab display
                        status_icon = "<i class='fas fa-check-circle' style='color: #10b981; margin-right: 4px;'></i>" if self.llm_analyzer.is_available() else "<i class='fas fa-times-circle' style='color: #ef4444; margin-right: 4px;'></i>"
                        status_text = "Connected" if self.llm_analyzer.is_available() else "Not available"
                        updated_display = (
                            f"**<i class='fas fa-brain' style='margin-right: 4px;'></i> LLM:** {self.llm_analyzer.provider.upper()} - {self.llm_analyzer.get_model_info()}\n"
                            f"**Status:** {status_icon} {status_text}\n\n"
                            f"<i class='fas fa-lightbulb' style='margin-right: 4px;'></i> *Configure LLM settings in the Settings tab*"
                        )
                        return status, model_info, updated_display
                    
                    def update_hint(provider, status):
                        """Update hint as string for Markdown component"""
                        if "check-circle" in status or "Connected" in status:
                            return ""
                        if provider == "llamacpp":
                            return "<i class='fas fa-lightbulb' style='margin-right: 4px;'></i> *Local model not found. Download a model first: `python scripts/download_model.py`*"
                        elif provider == "ollama":
                            return "<i class='fas fa-lightbulb' style='margin-right: 4px;'></i> *Ollama not detected. Start Ollama (`ollama serve`) or use local model.*"
                        elif provider in ["openai", "anthropic"]:
                            return f"<i class='fas fa-lightbulb' style='margin-right: 4px;'></i> *Set {provider.upper()}_API_KEY environment variable.*"
                        else:
                            return "<i class='fas fa-lightbulb' style='margin-right: 4px;'></i> *Check your configuration.*"
                    
                    apply_llm_btn = gr.Button("Save LLM Settings", variant="primary", elem_classes=["icon-save"])
                    apply_llm_btn.click(
                        fn=apply_llm_settings,
                        inputs=[llm_provider_dropdown, llm_model_dropdown],
                        outputs=[llm_status, llm_model_info, llm_status_display]
                    )
                    apply_llm_btn.click(
                        fn=update_hint,
                        inputs=[llm_provider_dropdown, llm_status],
                        outputs=llm_hint
                    )
                    
                    # Show initial hint if not available
                    if not self.llm_analyzer.is_available():
                        initial_hint = ""
                        if self.llm_analyzer.provider == "llamacpp":
                            initial_hint = "<i class='fas fa-lightbulb' style='margin-right: 4px;'></i> *Local model not found. Download a model first: `python scripts/download_model.py`*"
                        elif self.llm_analyzer.provider == "ollama":
                            initial_hint = "<i class='fas fa-lightbulb' style='margin-right: 4px;'></i> *Ollama not detected. Start Ollama (`ollama serve`) or use local model.*"
                        elif self.llm_analyzer.provider in ["openai", "anthropic"]:
                            initial_hint = f"<i class='fas fa-lightbulb' style='margin-right: 4px;'></i> *Set {self.llm_analyzer.provider.upper()}_API_KEY environment variable.*"
                        if initial_hint:
                            gr.Markdown(initial_hint)
                    
                    # Monitoring preferences
                    gr.Markdown(f"## {self._icon('chart-line')} Monitoring Preferences")
                    with gr.Row():
                        default_duration = gr.Slider(
                            minimum=10,
                            maximum=3600,
                            value=self.config.get("monitoring.default_duration", 60),
                            step=10,
                            label="Default Duration (seconds)"
                        )
                        default_interval = gr.Slider(
                            minimum=0.5,
                            maximum=10,
                            value=self.config.get("monitoring.default_interval", 1.0),
                            step=0.5,
                            label="Default Interval (seconds)"
                        )
                    
                    auto_archive = gr.Checkbox(
                        value=self.config.get("archive.enabled", True),
                        label="Enable Automatic Archiving"
                    )
                    
                    def save_monitoring_prefs(duration, interval, archive):
                        """Save monitoring preferences"""
                        try:
                            self.config.set("monitoring.default_duration", int(duration))
                            self.config.set("monitoring.default_interval", float(interval))
                            self.config.set("archive.enabled", archive)
                            self.config.save_config()
                            return f"{self._icon('check-circle')} Preferences saved successfully!"
                        except Exception as e:
                            return f"{self._icon('exclamation-triangle')} Error saving preferences: {str(e)}"
                    
                    with gr.Row():
                        save_prefs_btn = gr.Button("Save Preferences", variant="primary", elem_classes=["icon-save"], scale=1)
                        prefs_status = gr.Markdown()
                    save_prefs_btn.click(
                        fn=save_monitoring_prefs,
                        inputs=[default_duration, default_interval, auto_archive],
                        outputs=prefs_status
                    )
                    
                    # First-time setup completion
                    if self.config.is_first_run():
                        gr.Markdown("---")
                        gr.Markdown("### 🎉 Setup Complete!")
                        gr.Markdown("Configure your preferences above, then click 'Save Preferences' to continue.")
                        
                        def complete_setup():
                            """Mark setup as complete"""
                            self.config.mark_setup_complete()
                            return f"{self._icon('check-circle')} Setup complete! You can now use the Monitoring tab."
                        
                        complete_btn = gr.Button("Complete Setup", variant="primary", elem_classes=["icon-check-circle"])
                        setup_status = gr.Markdown()
                        complete_btn.click(
                            fn=complete_setup,
                            inputs=None,
                            outputs=setup_status
                        )
                
                with gr.TabItem("Monitoring"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            system_info = gr.HTML(value=self.get_system_info_display())
                            latest_metrics = gr.HTML(value="No data yet")
                            
                            with gr.Row():
                                duration = gr.Slider(
                                    minimum=10,
                                    maximum=3600,
                                    value=self.config.get("monitoring.default_duration", 60),
                                    step=10,
                                    label="Duration (seconds)"
                                )
                                interval = gr.Slider(
                                    minimum=0.5,
                                    maximum=10,
                                    value=self.config.get("monitoring.default_interval", 1.0),
                                    step=0.5,
                                    label="Interval (seconds)"
                                )
                            
                            live_monitoring = gr.Checkbox(
                                value=False,
                                label="Live Monitoring (continuous until stopped)",
                                info="Enable for continuous monitoring without time limit"
                            )
                            
                            with gr.Row():
                                start_btn = gr.Button("Start Monitoring", variant="primary", elem_classes=["icon-play"])
                                stop_btn = gr.Button("Stop Monitoring", elem_classes=["icon-stop"])
                                refresh_btn = gr.Button("Refresh", variant="secondary", elem_classes=["icon-sync"])
                            
                            status = gr.Markdown(
                                value="Ready - Click 'Start Monitoring' to begin"
                            )
                            
                            # Helpful tip about checking completion
                            gr.Markdown(
                                f"<i class='fas fa-lightbulb' style='margin-right: 4px;'></i> **Tip:** Click 'Refresh' to check if monitoring has completed.",
                                visible=True
                            )
                        
                        with gr.Column(scale=2):
                            plot = gr.Plot(label="Telemetry Plot")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown(f"## {self._icon('comments')} Chat with AI")
                            chatbot = gr.Chatbot(label="Ask questions about your system")
                            msg = gr.Textbox(
                                label="Message",
                                placeholder="Ask about CPU, memory, anomalies, recommendations...",
                                lines=2
                            )
                            msg.submit(self.chat_with_ai, [msg, chatbot], [msg, chatbot])
                            chat_btn = gr.Button("Send", variant="primary", elem_classes=["icon-paper-plane"])
                            chat_btn.click(self.chat_with_ai, [msg, chatbot], [msg, chatbot])
                        
                        with gr.Column(scale=1):
                            gr.Markdown(f"## {self._icon('chart-bar')} Analysis")
                            with gr.Row():
                                analyze_btn = gr.Button("Run AI Analysis", variant="secondary", elem_classes=["icon-search"])
                                analyze_llm_btn = gr.Button("AI + LLM", variant="primary", elem_classes=["icon-brain"])
                            
                            # Prediction options
                            with gr.Row():
                                enable_predictions = gr.Checkbox(
                                    value=False,
                                    label="Include Predictions",
                                    info="Predict future anomalies (requires 20+ data points)",
                                    elem_classes=["checkbox-icon-predictions"]
                                )
                                prediction_steps = gr.Slider(
                                    minimum=5,
                                    maximum=50,
                                    value=10,
                                    step=5,
                                    label="Prediction Steps",
                                    visible=False
                                )
                            
                            # Show prediction steps slider when predictions are enabled
                            def toggle_prediction_steps(enable):
                                return gr.update(visible=enable)
                            
                            enable_predictions.change(
                                fn=toggle_prediction_steps,
                                inputs=enable_predictions,
                                outputs=prediction_steps
                            )
                            
                            analysis_output = gr.Markdown()
                            
                            # Show current LLM status (read-only, full config in Settings)
                            # Note: llm_status_display is defined at the top level and updated from Settings tab
                            # Display it here in the Monitoring tab
                            llm_status_display
                
                # Archive tab (if archiving is enabled)
                if self.archive:
                    with gr.TabItem("Archive"):
                        gr.Markdown("## Historical Data Archive")
                        gr.Markdown("View and query archived telemetry sessions with system log correlations")
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                # Toolbar for actions
                                with gr.Row():
                                    archive_stats_btn = gr.Button("Archive Statistics", variant="secondary", elem_classes=["icon-chart-pie"])
                                
                                # Consolidated export section above table
                                with gr.Row():
                                    export_selected_btn = gr.Button("Export", variant="primary", elem_classes=["icon-save"], scale=1, min_width=80)
                                    export_format = gr.Radio(
                                        choices=["json", "csv"],
                                        value="json",
                                        label="Format",
                                        scale=1,
                                        container=False,
                                        elem_classes=["compact-radio"]
                                    )
                                    export_file = gr.File(
                                        label="",
                                        visible=False,
                                        scale=1
                                    )
                                export_output = gr.Markdown(visible=False)
                                
                                # Session table
                                sessions_summary = gr.Markdown()
                                sessions_table = gr.Dataframe(
                                    label="Archived Sessions (Click a row to view details, use column headers to sort)",
                                    headers=["Select", "Session ID", "Date", "Data Points", "Anomalies", "Log Correlations"],
                                    datatype=["bool", "str", "str", "number", "number", "number"],
                                    interactive=True,  # Required for checkbox column to work
                                    wrap=True,
                                    row_count=(10, "dynamic"),
                                    col_count=(6, "fixed"),
                                    type="pandas"
                                )
                                
                                # Note: Gradio Dataframe doesn't support per-column interactivity
                                # We keep interactive=True to allow checkbox selection in the "Select" column
                                # Other columns are read-only by default (users can't edit non-bool columns easily)
                                
                                # Spacer between table and pagination
                                gr.Markdown("")  # Spacer (no scale parameter)
                                # Pagination controls - bottom right
                                with gr.Row():
                                    gr.Markdown("")  # Spacer (no scale parameter)
                                    with gr.Row(scale=1):
                                        prev_page_btn = gr.Button("", variant="secondary", scale=1, min_width=32, interactive=True, elem_classes=["icon-chevron-left"])
                                        page_info = gr.Markdown("Page 1", elem_classes=["text-center"])
                                        next_page_btn = gr.Button("", variant="secondary", scale=1, min_width=32, interactive=True, elem_classes=["icon-chevron-right"])
                            
                            with gr.Column(scale=1):
                                with gr.Row():
                                    gr.Markdown("### Session Details")
                                    close_details_btn = gr.Button("", variant="secondary", scale=1, min_width=12, visible=False, elem_classes=["icon-times"])
                                gr.Markdown("*Click a row in the table to view session details*")
                                session_output = gr.Markdown()
                        
                        # Archive button actions
                        # Store reference to full table for row click handling
                        full_table_data = [None]  # Use list to allow modification in nested functions
                        current_page = [1]  # Track current page (1-indexed)
                        page_size = 50  # Records per page
                        total_sessions_count = [0]  # Track total count
                        
                        def load_page(page_num: int):
                            """Load a specific page of sessions"""
                            offset = (page_num - 1) * page_size
                            df, summary, total = self.list_archived_sessions(limit=page_size, offset=offset)
                            full_table_data[0] = df.copy() if isinstance(df, pd.DataFrame) and not df.empty else None
                            total_sessions_count[0] = total
                            current_page[0] = page_num
                            
                            # Calculate page info
                            total_pages = (total + page_size - 1) // page_size if total > 0 else 1
                            page_info_text = f"**Page {page_num} of {total_pages}**"
                            
                            # Update button states
                            prev_disabled = page_num <= 1
                            next_disabled = page_num >= total_pages
                            
                            return (
                                df,
                                summary,
                                page_info_text,
                                gr.update(interactive=not prev_disabled),
                                gr.update(interactive=not next_disabled)
                            )
                        
                        def go_to_prev_page():
                            """Go to previous page"""
                            try:
                                # Auto-load if table is empty (first interaction)
                                if full_table_data[0] is None:
                                    return load_page(1)
                                if current_page[0] > 1:
                                    return load_page(current_page[0] - 1)
                                return load_page(current_page[0])
                            except Exception as e:
                                # If error, try to load first page
                                return load_page(1)
                        
                        def go_to_next_page():
                            """Go to next page"""
                            try:
                                # Auto-load if table is empty (first interaction)
                                if full_table_data[0] is None:
                                    return load_page(1)
                                total_pages = (total_sessions_count[0] + page_size - 1) // page_size if total_sessions_count[0] > 0 else 1
                                if current_page[0] < total_pages:
                                    return load_page(current_page[0] + 1)
                                return load_page(current_page[0])
                            except Exception as e:
                                # If error, try to load first page
                                return load_page(1)
                        
                        # Pagination button handlers
                        prev_page_btn.click(
                            fn=go_to_prev_page,
                            inputs=None,
                            outputs=[sessions_table, sessions_summary, page_info, prev_page_btn, next_page_btn]
                        )
                        next_page_btn.click(
                            fn=go_to_next_page,
                            inputs=None,
                            outputs=[sessions_table, sessions_summary, page_info, prev_page_btn, next_page_btn]
                        )
                        
                        # Auto-load first page on app load using app.load event
                        def init_archive_data():
                            """Initialize archive with first page"""
                            return load_page(1)
                        
                        # Add app.load event for archive initialization
                        # This will be called when the app loads
                        app.load(
                            fn=init_archive_data,
                            inputs=None,
                            outputs=[sessions_table, sessions_summary, page_info, prev_page_btn, next_page_btn]
                        )
                        
                        # Also trigger load when pagination buttons are first clicked (fallback for Windows/Chrome)
                        # The buttons will auto-load if data is None, so this handles the case where app.load doesn't fire
                        
                        # Handle row click to show details
                        def show_session_details(evt: gr.SelectData):
                            """Show session details when row is clicked"""
                            if evt is None:
                                return "", gr.update(visible=False)
                            
                            try:
                                # Get row index from the select event
                                row_idx = evt.index[0] if hasattr(evt, 'index') and evt.index and len(evt.index) > 0 else None
                                
                                if row_idx is not None and full_table_data[0] is not None:
                                    # Get session ID from the stored full table
                                    if len(full_table_data[0]) > row_idx:
                                        session_id = full_table_data[0].iloc[row_idx]['Session ID']
                                        details = self.query_archived_session(session_id, show_telemetry=True)
                                        return details, gr.update(visible=True)
                                
                                # Fallback: try to get from evt.value
                                if hasattr(evt, 'value') and evt.value is not None:
                                    if isinstance(evt.value, pd.DataFrame) and len(evt.value) > 0:
                                        session_id = evt.value.iloc[0]['Session ID']
                                        details = self.query_archived_session(session_id, show_telemetry=True)
                                        return details, gr.update(visible=True)
                                
                                return "", gr.update(visible=False)
                            except Exception as e:
                                return f"Error loading session: {str(e)}", gr.update(visible=False)
                        
                        # Handle row selection via select event (click on row, not header)
                        def on_table_select(evt: gr.SelectData):
                            """Handle row click - only if not clicking header"""
                            # Check if clicking on a data row (index[0] >= 0 means data row, -1 or None means header)
                            if evt and hasattr(evt, 'index') and evt.index:
                                row_idx = evt.index[0]
                                if row_idx is not None and row_idx >= 0:
                                    return show_session_details(evt)
                            return "", gr.update(visible=False)
                        
                        try:
                            sessions_table.select(
                                fn=on_table_select,
                                outputs=[session_output, close_details_btn]
                            )
                        except:
                            # Fallback: if select doesn't work, we'll handle it differently
                            pass
                        
                        # Close details button
                        def close_details():
                            """Close the session details view"""
                            return "", gr.update(visible=False)
                        
                        close_details_btn.click(
                            fn=close_details,
                            inputs=None,
                            outputs=[session_output, close_details_btn]
                        )
                        # Archive stats button
                        def get_stats():
                            """Get archive stats"""
                            stats = self.get_archive_stats()
                            # Keep current table state
                            current_df = full_table_data[0] if full_table_data[0] is not None else pd.DataFrame()
                            total_pages = (total_sessions_count[0] + page_size - 1) // page_size if total_sessions_count[0] > 0 else 1
                            page_info_text = f"**Page {current_page[0]} of {total_pages}**" if total_sessions_count[0] > 0 else "Page 1"
                            prev_disabled = current_page[0] <= 1
                            next_disabled = current_page[0] >= total_pages
                            return (
                                current_df,
                                stats,
                                page_info_text,
                                gr.update(interactive=not prev_disabled),
                                gr.update(interactive=not next_disabled)
                            )
                        
                        archive_stats_btn.click(
                            fn=get_stats,
                            inputs=None,
                            outputs=[sessions_table, sessions_summary, page_info, prev_page_btn, next_page_btn]
                        )
                        
                        # Auto-load first page - use app.load event
                        def init_archive_on_load():
                            """Initialize archive table with first page on app load"""
                            return load_page(1)
                        
                        # Add to app.load - this will be set up after the app is created
                        # Store function reference for app.load
                        archive_load_func = init_archive_on_load
                        archive_load_outputs = [sessions_table, sessions_summary, page_info, prev_page_btn, next_page_btn]
                        
                        # Export selected sessions - consolidated
                        def export_with_download(selected_df, fmt):
                            status, filepath = self.export_selected_sessions(selected_df, fmt)
                            if filepath:
                                return (
                                    gr.update(value=status, visible=True),
                                    gr.update(value=filepath, visible=True, label="Download")
                                )
                            else:
                                return (
                                    gr.update(value=status, visible=True),
                                    gr.update(visible=False)
                                )
                        
                        export_selected_btn.click(
                            fn=export_with_download,
                            inputs=[sessions_table, export_format],
                            outputs=[export_output, export_file]
                        )
            
            # Auto-update on page load
            def load_initial_data():
                status_text = self.get_monitoring_status()
                plot_fig = self.update_plot()
                metrics_html = self.get_latest_metrics()
                return status_text, plot_fig, metrics_html
            
            # Initial load for monitoring tab
            app.load(
                fn=load_initial_data,
                inputs=None,
                outputs=[status, plot, latest_metrics]
            )
            
            # Auto-load archive data on app load (if archiving is enabled)
            # Note: The archive components are in a nested scope, so we need to
            # initialize them differently. We'll use a workaround by making the
            # archive initialization happen when the Archive tab is first accessed.
            # For now, users can click pagination buttons or archive stats to load data.
            # A better solution would be to use Gradio's tab select events, but that's
            # not easily available. The data will load when pagination buttons are clicked.
            
            def start_monitoring_wrapper(duration, interval, live):
                """Start monitoring wrapper"""
                status_msg, _ = self.start_monitoring(duration, interval, live)
                plot_fig = self.update_plot()
                metrics_html = self.get_latest_metrics()
                # Update status immediately
                self.monitoring_status = status_msg
                self.monitoring_completed = False  # Reset completion flag
                return status_msg, plot_fig, metrics_html
            
            def stop_monitoring_wrapper():
                """Stop monitoring wrapper"""
                status_msg, _ = self.stop_monitoring()
                plot_fig = self.update_plot()
                metrics_html = self.get_latest_metrics()
                return status_msg, plot_fig, metrics_html
            
            def update_during_monitoring():
                """Update status and display during active monitoring"""
                if self.is_monitoring:
                    status_text = self.get_monitoring_status()
                    plot_fig = self.update_plot()
                    metrics_html = self.get_latest_metrics()
                    return status_text, plot_fig, metrics_html
                else:
                    # Return current state without updating
                    return gr.update(), gr.update(), gr.update()
            
            # Button actions
            start_btn.click(
                fn=start_monitoring_wrapper,
                inputs=[duration, interval, live_monitoring],
                outputs=[status, plot, latest_metrics]
            )
            def refresh_with_status():
                """Refresh that also updates status"""
                status_text = self.get_monitoring_status()
                plot_fig = self.update_plot()
                metrics_html = self.get_latest_metrics()
                return status_text, plot_fig, metrics_html
            
            refresh_btn.click(
                fn=refresh_with_status,
                inputs=None,
                outputs=[status, plot, latest_metrics]
            )
            stop_btn.click(
                fn=stop_monitoring_wrapper,
                inputs=None,
                outputs=[status, plot, latest_metrics]
            )
            
            # Auto-refresh during monitoring (updates every 2 seconds)
            def auto_refresh():
                """Auto-refresh - always checks and updates status"""
                # Always get current status to catch completion
                status_text = self.get_monitoring_status()
                plot_fig = self.update_plot()
                metrics_html = self.get_latest_metrics()
                return status_text, plot_fig, metrics_html
            
            # Use app.load with every parameter for periodic updates
            # Note: This feature may not be available in all Gradio versions
            # If it doesn't work, the Refresh button will update the status
            try:
                app.load(
                    fn=auto_refresh,
                    inputs=None,
                    outputs=[status, plot, latest_metrics],
                    every=2.0  # Update every 2 seconds
                )
            except (TypeError, AttributeError, ValueError) as e:
                # Auto-refresh not supported - user will need to click Refresh
                # This is expected in some Gradio versions
                pass
            def run_standard_analysis(enable_pred, steps):
                """Run standard AI analysis without LLM"""
                # Show immediate feedback
                status = "## <i class='fas fa-sync fa-spin' style='margin-right: 6px;'></i> Running AI Analysis...\n\n"
                if enable_pred:
                    status += "<i class='fas fa-hourglass-half' style='margin-right: 6px;'></i> *Training prediction models and analyzing data... This may take a moment.*\n\n"
                else:
                    status += "<i class='fas fa-hourglass-half' style='margin-right: 6px;'></i> *Analyzing telemetry data...*\n\n"
                yield status
                
                # Run the actual analysis
                result = self.analyze_data(use_llm=False, include_predictions=enable_pred, prediction_steps=int(steps))
                yield result
            
            def run_llm_analysis(enable_pred, steps):
                """Run AI analysis with LLM insights"""
                # Show immediate feedback
                status = "## <i class='fas fa-sync fa-spin' style='margin-right: 6px;'></i> Running AI + LLM Analysis...\n\n"
                if enable_pred:
                    status += "<i class='fas fa-hourglass-half' style='margin-right: 6px;'></i> *Training prediction models, analyzing data, and generating LLM insights... This may take longer.*\n\n"
                else:
                    status += "<i class='fas fa-hourglass-half' style='margin-right: 6px;'></i> *Analyzing data and generating LLM insights... This may take a moment.*\n\n"
                yield status
                
                # Run the actual analysis
                result = self.analyze_data(use_llm=True, include_predictions=enable_pred, prediction_steps=int(steps))
                yield result
            
            analyze_btn.click(
                fn=run_standard_analysis,
                inputs=[enable_predictions, prediction_steps],
                outputs=analysis_output
            )
            analyze_llm_btn.click(
                fn=run_llm_analysis,
                inputs=[enable_predictions, prediction_steps],
                outputs=analysis_output
            )
        
        return app


def main():
    """Launch the GUI application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIOS Telemetry GUI")
    parser.add_argument("--llm-provider", default=None, 
                       choices=["ollama", "llamacpp", "openai", "anthropic"],
                       help="LLM provider to use (default: ollama, or LLM_PROVIDER env var)")
    parser.add_argument("--llm-model", default=None,
                       help="LLM model name (e.g., 'gemma3-1b.gguf' for llamacpp)")
    parser.add_argument("--no-archive", action="store_true",
                       help="Disable data archiving")
    
    args = parser.parse_args()
    
    gui = TelemetryGUI(
        enable_archiving=not args.no_archive,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model
    )
    app = gui.create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()

