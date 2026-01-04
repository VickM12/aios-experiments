"""
GUI Application with Chat Interface
Web-based GUI for telemetry monitoring and AI chat
"""
import os
import gradio as gr
import json
import threading
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from .telemetry_collector import TelemetryCollector
from .ai_analyzer import TelemetryAnalyzer
from .visualizer import TelemetryVisualizer
from .llm_analyzer import LLMAnalyzer
from .data_archive import DataArchive
from .system_logs import SystemLogReader
import plotly.graph_objects as go
import pandas as pd


class TelemetryGUI:
    """GUI application with chat interface"""
    
    def __init__(self, enable_archiving: bool = True, llm_provider: str = None, llm_model: str = None):
        self.collector = TelemetryCollector()
        self.analyzer = TelemetryAnalyzer()
        self.visualizer = TelemetryVisualizer()
        # Allow provider to be set via environment variable or parameter
        provider = llm_provider or os.getenv("LLM_PROVIDER", "ollama")
        model = llm_model or os.getenv("LLM_MODEL", None)
        self.llm_analyzer = LLMAnalyzer(provider=provider, model=model)
        self.telemetry_history = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.system_info = None
        self.archive = DataArchive() if enable_archiving else None
        self.log_reader = SystemLogReader() if enable_archiving else None
        self.last_archived_session = None
        
    def get_system_info_display(self):
        """Get formatted system information"""
        if not self.system_info:
            self.system_info = self.collector.get_system_info()
        
        info = self.system_info
        html = "<div style='font-family: monospace;'>"
        html += "<h2>🖥️ System Information</h2>"
        
        if 'os' in info:
            os_info = info['os']
            html += f"<p><strong>OS:</strong> {os_info.get('system', 'Unknown')} {os_info.get('release', '')}</p>"
            html += f"<p><strong>Machine:</strong> {os_info.get('machine', 'Unknown')}</p>"
        
        if 'cpu' in info:
            cpu_info = info['cpu']
            html += f"<p><strong>CPU:</strong> {cpu_info.get('model', 'Unknown')}</p>"
            html += f"<p><strong>Cores:</strong> {cpu_info.get('physical_cores', '?')} physical, {cpu_info.get('logical_cores', '?')} logical</p>"
        
        if 'memory' in info:
            mem_info = info['memory']
            html += f"<p><strong>Memory:</strong> {mem_info.get('total_gb', 0):.2f} GB</p>"
        
        if 'gpu' in info and isinstance(info['gpu'], list) and info['gpu']:
            for gpu in info['gpu']:
                html += f"<p><strong>GPU:</strong> {gpu.get('vendor', '')} {gpu.get('model', '')}</p>"
        
        html += "</div>"
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
        
        # Simple AI responses based on keywords and context
        message_lower = message.lower()
        response = ""
        
        if "cpu" in message_lower or "processor" in message_lower:
            # Check if user wants detailed explanation
            wants_details = any(word in message_lower for word in ['tell me', 'explain', 'more', 'detail', 'in detail', 'about', 'what', 'how', 'why', 'please'])
            
            if wants_details and self.llm_analyzer.is_available() and len(self.telemetry_history) > 0:
                # Use LLM for detailed CPU explanation
                try:
                    analysis = None
                    if len(self.telemetry_history) >= 10:
                        analysis = self.analyzer.comprehensive_analysis(self.telemetry_history)
                    response = self.llm_analyzer.answer_question(
                        message,
                        self.telemetry_history,
                        analysis,
                        self.system_info
                    )
                except Exception as e:
                    # Fall back to basic info
                    if current_data:
                        cpu = current_data.get('cpu', {})
                        cpu_percent = cpu.get('cpu_percent', 0)
                        response = f"**CPU Status:**\n"
                        response += f"- Current usage: {cpu_percent:.1f}%\n"
                        response += f"- Status: {'High' if cpu_percent > 80 else 'Moderate' if cpu_percent > 50 else 'Normal'}\n"
                        if cpu.get('cpu_freq', {}).get('current'):
                            response += f"- Frequency: {cpu['cpu_freq']['current']:.0f} MHz\n"
                        response += f"\n⚠️ Error getting LLM response: {str(e)}"
                    else:
                        response = "No telemetry data available. Please start monitoring first."
            elif current_data:
                cpu = current_data.get('cpu', {})
                cpu_percent = cpu.get('cpu_percent', 0)
                response = f"**CPU Status:**\n"
                response += f"- Current usage: {cpu_percent:.1f}%\n"
                response += f"- Status: {'High' if cpu_percent > 80 else 'Moderate' if cpu_percent > 50 else 'Normal'}\n"
                if cpu.get('cpu_freq', {}).get('current'):
                    response += f"- Frequency: {cpu['cpu_freq']['current']:.0f} MHz\n"
                if wants_details and not self.llm_analyzer.is_available():
                    response += f"\n💡 *For detailed explanations, make sure the LLM is available.*"
                elif wants_details:
                    response += f"\n💡 *Ask 'Tell me more about the CPU' or 'Explain the CPU in detail' for LLM analysis.*"
            else:
                response = "No telemetry data available. Please start monitoring first."
        
        elif "memory" in message_lower or "ram" in message_lower:
            if current_data:
                mem = current_data.get('memory', {}).get('virtual_memory', {})
                mem_percent = mem.get('percent', 0)
                mem_gb = mem.get('used', 0) / (1024**3)
                response = f"**Memory Status:**\n"
                response += f"- Usage: {mem_percent:.1f}% ({mem_gb:.2f} GB used)\n"
                response += f"- Status: {'High' if mem_percent > 80 else 'Moderate' if mem_percent > 50 else 'Normal'}\n"
            else:
                response = "No telemetry data available. Please start monitoring first."
        
        elif "network" in message_lower or "internet" in message_lower or "bandwidth" in message_lower:
            if current_data:
                network_io = current_data.get('network', {}).get('network_io', {})
                if network_io:
                    sent_mb = network_io.get('bytes_sent', 0) / (1024**2)
                    recv_mb = network_io.get('bytes_recv', 0) / (1024**2)
                    sent_gb = sent_mb / 1024
                    recv_gb = recv_mb / 1024
                    packets_sent = network_io.get('packets_sent', 0)
                    packets_recv = network_io.get('packets_recv', 0)
                    connections = current_data.get('network', {}).get('network_connections', 0)
                    
                    response = f"**Network Status:**\n"
                    response += f"- Data sent: {sent_gb:.2f} GB ({sent_mb:.1f} MB)\n"
                    response += f"- Data received: {recv_gb:.2f} GB ({recv_mb:.1f} MB)\n"
                    response += f"- Packets sent: {packets_sent:,}\n"
                    response += f"- Packets received: {packets_recv:,}\n"
                    response += f"- Active connections: {connections}\n"
                    if network_io.get('errin', 0) > 0 or network_io.get('errout', 0) > 0:
                        response += f"- ⚠️ Errors: {network_io.get('errin', 0)} in, {network_io.get('errout', 0)} out\n"
                else:
                    response = "Network data not available."
            else:
                response = "No telemetry data available. Please start monitoring first."
        
        elif "power" in message_lower or "watt" in message_lower or "energy" in message_lower:
            if current_data:
                power_data = current_data.get('power', {})
                if power_data:
                    response = f"**Power Usage:**\n"
                    
                    # GPU power
                    if power_data.get('gpu'):
                        for i, gpu in enumerate(power_data['gpu']):
                            if 'power_draw_watts' in gpu and gpu['power_draw_watts']:
                                response += f"- GPU {i+1}: {gpu['power_draw_watts']:.1f}W\n"
                    
                    # CPU RAPL power
                    if power_data.get('rapl'):
                        for domain, pwr in power_data['rapl'].items():
                            if 'package' in domain.lower():
                                energy = pwr.get('energy_joules', 0)
                                response += f"- CPU Package: {energy:.1f}J\n"
                    
                    if not power_data.get('gpu') and not power_data.get('rapl'):
                        response += "Power monitoring not available on this system.\n"
                else:
                    response = "Power data not available."
            else:
                response = "No telemetry data available. Please start monitoring first."
        
        elif "battery" in message_lower:
            if current_data:
                battery_data = current_data.get('battery', {})
                if battery_data and 'error' not in battery_data:
                    percent = battery_data.get('percent', 'N/A')
                    plugged = battery_data.get('power_plugged', False)
                    status = "Charging" if plugged else "Discharging"
                    secsleft = battery_data.get('secsleft', None)
                    
                    response = f"**Battery Status:**\n"
                    response += f"- Charge: {percent}%\n"
                    response += f"- Status: {status}\n"
                    if secsleft and secsleft != -1:
                        hours = secsleft // 3600
                        minutes = (secsleft % 3600) // 60
                        response += f"- Time remaining: {hours}h {minutes}m\n"
                else:
                    response = "Battery data not available (system may not have a battery)."
            else:
                response = "No telemetry data available. Please start monitoring first."
        
        elif "process" in message_lower or "processes" in message_lower or "top process" in message_lower or "what's using" in message_lower:
            # Check if user wants detailed explanation
            wants_details = any(word in message_lower for word in ['tell me', 'explain', 'more', 'detail', 'in detail', 'about', 'what', 'how', 'why'])
            
            if wants_details and self.llm_analyzer.is_available() and len(self.telemetry_history) > 0:
                # Use LLM for detailed process explanation
                try:
                    analysis = None
                    if len(self.telemetry_history) >= 10:
                        analysis = self.analyzer.comprehensive_analysis(self.telemetry_history)
                    response = self.llm_analyzer.answer_question(
                        message,
                        self.telemetry_history,
                        analysis,
                        self.system_info
                    )
                except Exception as e:
                    # Fall back to basic info
                    if current_data:
                        process_data = current_data.get('processes', {})
                        if process_data:
                            total = process_data.get('total_processes', 0)
                            top_processes = process_data.get('top_processes', [])[:10]
                            response = f"**Process Information:**\n"
                            response += f"- Total processes: {total}\n\n"
                            response += f"**Top CPU-consuming processes:**\n"
                            for i, proc in enumerate(top_processes, 1):
                                proc_name = proc.get('name', 'unknown')
                                proc_cpu = proc.get('cpu_percent', 0) or 0
                                proc_mem = proc.get('memory_percent', 0) or 0
                                proc_pid = proc.get('pid', 'N/A')
                                if proc_cpu > 0 or proc_mem > 0.1:
                                    response += f"{i}. {proc_name} (PID {proc_pid}): CPU {proc_cpu:.1f}%, Memory {proc_mem:.2f}%\n"
                            response += f"\n⚠️ Error getting LLM response: {str(e)}"
                        else:
                            response = "Process data not available."
                    else:
                        response = "No telemetry data available. Please start monitoring first."
            elif current_data:
                process_data = current_data.get('processes', {})
                if process_data:
                    total = process_data.get('total_processes', 0)
                    top_processes = process_data.get('top_processes', [])[:10]
                    
                    response = f"**Process Information:**\n"
                    response += f"- Total processes: {total}\n\n"
                    response += f"**Top CPU-consuming processes:**\n"
                    for i, proc in enumerate(top_processes, 1):
                        proc_name = proc.get('name', 'unknown')
                        proc_cpu = proc.get('cpu_percent', 0) or 0
                        proc_mem = proc.get('memory_percent', 0) or 0
                        proc_pid = proc.get('pid', 'N/A')
                        if proc_cpu > 0 or proc_mem > 0.1:
                            response += f"{i}. {proc_name} (PID {proc_pid}): CPU {proc_cpu:.1f}%, Memory {proc_mem:.2f}%\n"
                    if wants_details and not self.llm_analyzer.is_available():
                        response += f"\n\n💡 *For detailed explanations, make sure the LLM is available.*"
                    elif wants_details:
                        response += f"\n\n💡 *Ask 'Tell me more about the processes' or 'Explain the processes in detail' for LLM analysis.*"
                else:
                    response = "Process data not available."
            else:
                response = "No telemetry data available. Please start monitoring first."
        
        elif "anomaly" in message_lower or "problem" in message_lower or "issue" in message_lower or "anomalies" in message_lower:
            if len(self.telemetry_history) >= 10:
                try:
                    analysis = self.analyzer.comprehensive_analysis(self.telemetry_history)
                    anomaly = analysis.get('anomaly_detection', {})
                    if anomaly.get('anomalies_detected', 0) > 0:
                        # Check if user wants detailed explanation (keywords like "tell", "explain", "more", "detail", "llm")
                        wants_details = any(word in message_lower for word in ['tell', 'explain', 'more', 'detail', 'what', 'why', 'how', 'llm', 'about'])
                        
                        if wants_details and self.llm_analyzer.is_available():
                            # Use LLM for detailed explanation
                            try:
                                llm_response = self.llm_analyzer.explain_anomalies(self.telemetry_history, analysis, self.system_info)
                                if llm_response and str(llm_response).strip():
                                    response = str(llm_response).strip()
                                else:
                                    # LLM returned empty, fall back to basic info
                                    response = f"**Anomaly Detection:**\n"
                                    response += f"- Found {anomaly['anomalies_detected']} anomalies ({anomaly.get('anomaly_percentage', 0):.1f}%)\n"
                                    if anomaly.get('details'):
                                        response += f"- Latest anomaly at index {anomaly['details'][-1]['index']}\n"
                                    response += f"\n⚠️ LLM returned empty response. Showing basic info."
                            except Exception as e:
                                response = f"**Anomaly Detection:**\n"
                                response += f"- Found {anomaly['anomalies_detected']} anomalies ({anomaly.get('anomaly_percentage', 0):.1f}%)\n"
                                if anomaly.get('details'):
                                    response += f"- Latest anomaly at index {anomaly['details'][-1]['index']}\n"
                                response += f"\n⚠️ Error getting LLM explanation: {str(e)}"
                        else:
                            # Basic anomaly info
                            response = f"**Anomaly Detection:**\n"
                            response += f"- Found {anomaly['anomalies_detected']} anomalies ({anomaly.get('anomaly_percentage', 0):.1f}%)\n"
                            if anomaly.get('details'):
                                response += f"- Latest anomaly at index {anomaly['details'][-1]['index']}\n"
                            if self.llm_analyzer.is_available():
                                response += f"\n💡 *Ask 'Tell me more about the anomalies' or 'Explain the anomalies' for detailed LLM analysis.*"
                            elif wants_details:
                                response += f"\n💡 *LLM not available. For detailed explanations, make sure Ollama is running or set API keys.*"
                    else:
                        response = "**No anomalies detected.** System appears to be operating normally."
                except Exception as e:
                    response = f"Error analyzing anomalies: {str(e)}"
            else:
                response = "Need at least 10 data points for anomaly detection. Please collect more data."
        
        elif "temperature" in message_lower or "temp" in message_lower:
            if current_data:
                temp_data = current_data.get('temperature', {})
                if temp_data and 'error' not in temp_data:
                    all_temps = []
                    for sensor_name, entries in temp_data.items():
                        for entry in entries:
                            if 'current' in entry:
                                all_temps.append(entry['current'])
                    if all_temps:
                        avg_temp = sum(all_temps) / len(all_temps)
                        response = f"**Temperature Status:**\n"
                        response += f"- Average: {avg_temp:.1f}°C\n"
                        response += f"- Status: {'High' if avg_temp > 70 else 'Moderate' if avg_temp > 60 else 'Normal'}\n"
                    else:
                        response = "Temperature sensors not available."
                else:
                    response = "Temperature data not available."
            else:
                response = "No telemetry data available."
        
        elif "recommendation" in message_lower or "suggest" in message_lower or "advice" in message_lower:
            if len(self.telemetry_history) >= 5:
                analysis = self.analyzer.comprehensive_analysis(self.telemetry_history)
                insights = analysis.get('performance_insights', {})
                if insights.get('recommendations'):
                    response = "**Recommendations:**\n"
                    for rec in insights['recommendations']:
                        response += f"- {rec}\n"
                else:
                    response = "**System is operating normally.** No recommendations at this time."
            else:
                response = "Please collect more data for recommendations."
        
        elif "summary" in message_lower or "overview" in message_lower or "status" in message_lower:
            if current_data:
                cpu = current_data.get('cpu', {}).get('cpu_percent', 0)
                mem = current_data.get('memory', {}).get('virtual_memory', {}).get('percent', 0)
                disk = current_data.get('disk', {}).get('disk_usage', {}).get('percent', 0)
                response = f"**System Status Summary:**\n"
                response += f"- CPU: {cpu:.1f}%\n"
                response += f"- Memory: {mem:.1f}%\n"
                response += f"- Disk: {disk:.1f}%\n"
                response += f"- Data points collected: {len(self.telemetry_history)}\n"
            else:
                response = "No data available. Start monitoring to see status."
        
        else:
            # For general questions, try using LLM if available
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
                except Exception as e:
                    response = f"I encountered an error: {str(e)}\n\n"
                    response += "I can help you with:\n"
                    response += "- CPU status and usage\n"
                    response += "- Memory/RAM information\n"
                    response += "- Temperature monitoring\n"
                    response += "- Anomaly detection\n"
                    response += "- System recommendations\n"
                    response += "- Overall system status"
            else:
                # Check for common question patterns that should use LLM
                llm_keywords = ['tell me', 'explain', 'why', 'how', 'what is', 'what are', 'describe', 'analyze', 'llm']
                if any(keyword in message_lower for keyword in llm_keywords) and len(self.telemetry_history) > 0:
                    if self.llm_analyzer.is_available():
                        try:
                            analysis = None
                            if len(self.telemetry_history) >= 10:
                                analysis = self.analyzer.comprehensive_analysis(self.telemetry_history)
                            response = self.llm_analyzer.answer_question(
                                message,
                                self.telemetry_history,
                                analysis
                            )
                        except Exception as e:
                            response = f"Error using LLM: {str(e)}\n\n"
                            response += "I can help you with:\n"
                            response += "- CPU status and usage\n"
                            response += "- Memory/RAM information\n"
                            response += "- Temperature monitoring\n"
                            response += "- Anomaly detection\n"
                            response += "- System recommendations\n"
                            response += "- Overall system status"
                    else:
                        response = "I can help you with:\n"
                        response += "- CPU status and usage\n"
                        response += "- Memory/RAM information\n"
                        response += "- Temperature monitoring\n"
                        response += "- Anomaly detection\n"
                        response += "- System recommendations\n"
                        response += "- Overall system status\n\n"
                        response += "💡 *For more detailed AI responses, make sure Ollama is running or set API keys.*\n\n"
                        response += "Try asking: 'What's my CPU usage?' or 'Are there any anomalies?'"
                else:
                    response = "I can help you with:\n"
                    response += "- CPU status and usage\n"
                    response += "- Memory/RAM information\n"
                    response += "- Temperature monitoring\n"
                    response += "- Anomaly detection\n"
                    response += "- System recommendations\n"
                    response += "- Overall system status\n\n"
                    if not self.llm_analyzer.is_available():
                        response += "💡 *For more detailed AI responses, make sure Ollama is running or set API keys.*\n\n"
                    response += "Try asking: 'What's my CPU usage?' or 'Are there any anomalies?'"
        
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
    
    def start_monitoring(self, duration: int, interval: float):
        """Start telemetry monitoring"""
        if self.is_monitoring:
            return "Monitoring already in progress!", gr.update()
        
        self.is_monitoring = True
        self.telemetry_history = []
        
        def monitor():
            end_time = time.time() + duration
            while time.time() < end_time and self.is_monitoring:
                data = self.collector.collect_all_telemetry()
                self.telemetry_history.append(data)
                time.sleep(interval)
            self.is_monitoring = False
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        
        return f"Monitoring started for {duration} seconds!", gr.update()
    
    def stop_monitoring(self):
        """Stop telemetry monitoring"""
        self.is_monitoring = False
        return "Monitoring stopped.", gr.update()
    
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
            <h3>📊 Current Metrics</h3>
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
    
    def analyze_data(self):
        """Run AI analysis on collected data"""
        if len(self.telemetry_history) < 10:
            return "Need at least 10 data points for analysis. Please collect more data."
        
        analysis = self.analyzer.comprehensive_analysis(self.telemetry_history)
        
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
        
        # Performance Insights
        insights = analysis.get('performance_insights', {})
        if 'current_status' in insights:
            result += "### 📊 Performance Status\n"
            for metric, data in insights['current_status'].items():
                status_icon = "🔴" if data.get('status') == 'high' else "🟡" if data.get('status') == 'moderate' else "🟢"
                
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
            result += f"\n### 🔍 Anomalies Detected\n"
            result += f"- Found {anomaly['anomalies_detected']} anomalies ({anomaly.get('anomaly_percentage', 0):.1f}%)\n"
        
        # Recommendations
        if insights.get('recommendations'):
            result += f"\n### 💡 Recommendations\n"
            for rec in insights['recommendations']:
                result += f"- {rec}\n"
        
        # Archive info
        if self.archive and self.last_archived_session:
            result += f"\n### 💾 Archive\n"
            result += f"- Session archived: `{self.last_archived_session}`\n"
            if system_logs:
                correlated = len(self.archive.get_correlated_events(self.last_archived_session))
                result += f"- Correlated with {correlated} system log events\n"
        
        return result
    
    def list_archived_sessions(self):
        """List all archived sessions"""
        if not self.archive:
            return "Archiving is not enabled."
        
        sessions = self.archive.query_sessions()
        if not sessions:
            return "No archived sessions found."
        
        result = "## 📚 Archived Sessions\n\n"
        result += f"Total: {len(sessions)} sessions\n\n"
        
        for session in sessions[:20]:  # Show last 20
            result += f"### {session['session_id']}\n"
            result += f"- **Start:** {session['start_time']}\n"
            result += f"- **End:** {session['end_time']}\n"
            result += f"- **Data Points:** {session['data_points']}\n"
            result += f"- **Anomalies:** {session['anomaly_count']}\n\n"
        
        if len(sessions) > 20:
            result += f"\n*Showing 20 of {len(sessions)} sessions*\n"
        
        return result
    
    def query_archived_session(self, session_id: str):
        """Query a specific archived session"""
        if not self.archive:
            return "Archiving is not enabled."
        
        if not session_id or not session_id.strip():
            return "Please enter a session ID."
        
        session = self.archive.load_session(session_id.strip())
        if not session:
            return f"Session '{session_id}' not found."
        
        result = f"## 📦 Session: {session_id}\n\n"
        result += f"**Start Time:** {session.get('start_time')}\n"
        result += f"**End Time:** {session.get('end_time')}\n"
        result += f"**Data Points:** {len(session.get('telemetry_data', []))}\n\n"
        
        # Show analysis summary if available
        analysis = session.get('analysis', {})
        if analysis:
            anomaly = analysis.get('anomaly_detection', {})
            if anomaly.get('anomalies_detected', 0) > 0:
                result += f"### 🔍 Anomalies\n"
                result += f"- Detected: {anomaly['anomalies_detected']} ({anomaly.get('anomaly_percentage', 0):.1f}%)\n\n"
        
        # Show correlated log events
        events = self.archive.get_correlated_events(session_id.strip())
        if events:
            result += f"### 🔗 Correlated System Log Events ({len(events)})\n\n"
            for event in events[:10]:  # Show top 10
                result += f"**[{event['log_time']}]** {event['source']} - {event['level']}\n"
                result += f"- {event['message'][:100]}{'...' if len(event['message']) > 100 else ''}\n"
                result += f"- Correlation: {event['correlation_score']:.2f}\n\n"
        else:
            result += "### 🔗 Correlated Log Events\n"
            result += "No correlated log events found.\n"
        
        return result
    
    def get_archive_stats(self):
        """Get archive statistics"""
        if not self.archive:
            return "Archiving is not enabled."
        
        stats = self.archive.get_statistics()
        result = "## 📊 Archive Statistics\n\n"
        result += f"**Total Sessions:** {stats.get('total_sessions', 0)}\n"
        result += f"**Total Data Points:** {stats.get('total_data_points', 0):,}\n"
        result += f"**Total Anomalies:** {stats.get('total_anomalies', 0)}\n"
        result += f"**Total Correlations:** {stats.get('total_correlations', 0)}\n\n"
        
        if stats.get('oldest_session'):
            result += f"**Oldest Session:** {stats['oldest_session']}\n"
            result += f"**Newest Session:** {stats['newest_session']}\n"
        
        return result
    
    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="AI Telemetry Monitor", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🤖 AI Telemetry Monitoring & Analysis")
            gr.Markdown("Monitor your system in real-time and chat with AI about your telemetry data")
            
            # Create tabs for different sections
            with gr.Tabs():
                with gr.TabItem("📊 Monitoring"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            system_info = gr.HTML(value=self.get_system_info_display())
                            latest_metrics = gr.HTML(value="No data yet")
                            
                            with gr.Row():
                                duration = gr.Slider(minimum=10, maximum=3600, value=60, step=10, label="Duration (seconds)")
                                interval = gr.Slider(minimum=0.5, maximum=10, value=1.0, step=0.5, label="Interval (seconds)")
                            
                            with gr.Row():
                                start_btn = gr.Button("▶️ Start Monitoring", variant="primary")
                                stop_btn = gr.Button("⏹️ Stop Monitoring")
                                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                            
                            status = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Column(scale=2):
                            plot = gr.Plot(label="Telemetry Plot")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## 💬 Chat with AI")
                            chatbot = gr.Chatbot(label="Ask questions about your system")
                            msg = gr.Textbox(
                                label="Message",
                                placeholder="Ask about CPU, memory, anomalies, recommendations...",
                                lines=2
                            )
                            msg.submit(self.chat_with_ai, [msg, chatbot], [msg, chatbot])
                            chat_btn = gr.Button("Send", variant="primary")
                            chat_btn.click(self.chat_with_ai, [msg, chatbot], [msg, chatbot])
                        
                        with gr.Column(scale=1):
                            gr.Markdown("## 📈 Analysis")
                            with gr.Row():
                                analyze_btn = gr.Button("🔍 Run AI Analysis", variant="secondary")
                                analyze_llm_btn = gr.Button("🤖 AI + LLM", variant="primary")
                            analysis_output = gr.Markdown()
                            model_info = self.llm_analyzer.get_model_info()
                            provider_name = self.llm_analyzer.provider.upper()
                            gr.Markdown(f"**🤖 LLM Provider:** {provider_name}")
                            gr.Markdown(f"**Model:** {model_info}")
                            if not self.llm_analyzer.is_available():
                                if self.llm_analyzer.provider == "llamacpp":
                                    gr.Markdown("💡 *Local model not found. Download a model first: `python scripts/download_model.py`*")
                                elif self.llm_analyzer.provider == "ollama":
                                    gr.Markdown("💡 *Ollama not detected. Start Ollama (`ollama serve`) or use local model with `--llm-provider llamacpp`*")
                                else:
                                    gr.Markdown("💡 *LLM not available. Check API keys or model configuration.*")
                
                # Archive tab (if archiving is enabled)
                if self.archive:
                    with gr.TabItem("📦 Archive"):
                        gr.Markdown("## Historical Data Archive")
                        gr.Markdown("View and query archived telemetry sessions with system log correlations")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Archive Management")
                                list_sessions_btn = gr.Button("📚 List All Sessions", variant="primary")
                                archive_stats_btn = gr.Button("📊 Archive Statistics", variant="secondary")
                                sessions_output = gr.Markdown()
                            
                            with gr.Column(scale=1):
                                gr.Markdown("### Query Session")
                                session_id_input = gr.Textbox(
                                    label="Session ID",
                                    placeholder="e.g., session_20260103_141413",
                                    lines=1
                                )
                                query_session_btn = gr.Button("🔍 Query Session", variant="primary")
                                session_output = gr.Markdown()
                        
                        # Archive button actions
                        list_sessions_btn.click(
                            fn=self.list_archived_sessions,
                            inputs=None,
                            outputs=sessions_output
                        )
                        archive_stats_btn.click(
                            fn=self.get_archive_stats,
                            inputs=None,
                            outputs=sessions_output
                        )
                        query_session_btn.click(
                            fn=self.query_archived_session,
                            inputs=session_id_input,
                            outputs=session_output
                        )
                        session_id_input.submit(
                            fn=self.query_archived_session,
                            inputs=session_id_input,
                            outputs=session_output
                        )
            
            # Auto-update on page load
            def load_initial_data():
                return self.update_plot(), self.get_latest_metrics()
            
            app.load(
                fn=load_initial_data,
                inputs=None,
                outputs=[plot, latest_metrics]
            )
            
            # Button actions
            start_btn.click(
                fn=self.start_monitoring,
                inputs=[duration, interval],
                outputs=[status, plot]
            )
            refresh_btn.click(
                fn=load_initial_data,
                inputs=None,
                outputs=[plot, latest_metrics]
            )
            stop_btn.click(
                fn=self.stop_monitoring,
                inputs=None,
                outputs=[status]
            )
            analyze_btn.click(
                fn=self.analyze_data,
                inputs=None,
                outputs=analysis_output
            )
            analyze_llm_btn.click(
                fn=self.analyze_data,
                inputs=None,
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

