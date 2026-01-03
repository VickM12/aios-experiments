"""
GUI Application with Chat Interface
Web-based GUI for telemetry monitoring and AI chat
"""
import gradio as gr
import json
import threading
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from telemetry_collector import TelemetryCollector
from ai_analyzer import TelemetryAnalyzer
from visualizer import TelemetryVisualizer
from llm_analyzer import LLMAnalyzer
import plotly.graph_objects as go
import pandas as pd


class TelemetryGUI:
    """GUI application with chat interface"""
    
    def __init__(self):
        self.collector = TelemetryCollector()
        self.analyzer = TelemetryAnalyzer()
        self.visualizer = TelemetryVisualizer()
        self.llm_analyzer = LLMAnalyzer(provider="ollama")
        self.telemetry_history = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.system_info = None
        
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
        if history and isinstance(history[0], (tuple, list)):
            # Convert old tuple format to new dict format
            new_history = []
            for item in history:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    new_history.append({"role": "user", "content": item[0]})
                    new_history.append({"role": "assistant", "content": item[1]})
                elif isinstance(item, dict):
                    new_history.append(item)
            history = new_history
        
        # Get current telemetry state
        current_data = self.telemetry_history[-1] if self.telemetry_history else None
        
        # Simple AI responses based on keywords and context
        message_lower = message.lower()
        response = ""
        
        if "cpu" in message_lower or "processor" in message_lower:
            if current_data:
                cpu = current_data.get('cpu', {})
                cpu_percent = cpu.get('cpu_percent', 0)
                response = f"**CPU Status:**\n"
                response += f"- Current usage: {cpu_percent:.1f}%\n"
                response += f"- Status: {'High' if cpu_percent > 80 else 'Moderate' if cpu_percent > 50 else 'Normal'}\n"
                if cpu.get('cpu_freq', {}).get('current'):
                    response += f"- Frequency: {cpu['cpu_freq']['current']:.0f} MHz\n"
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
        
        elif "anomaly" in message_lower or "problem" in message_lower or "issue" in message_lower:
            if len(self.telemetry_history) >= 10:
                analysis = self.analyzer.comprehensive_analysis(self.telemetry_history)
                anomaly = analysis.get('anomaly_detection', {})
                if anomaly.get('anomalies_detected', 0) > 0:
                    response = f"**Anomaly Detection:**\n"
                    response += f"- Found {anomaly['anomalies_detected']} anomalies ({anomaly.get('anomaly_percentage', 0):.1f}%)\n"
                    if anomaly.get('details'):
                        response += f"- Latest anomaly at index {anomaly['details'][-1]['index']}\n"
                else:
                    response = "**No anomalies detected.** System appears to be operating normally."
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
            response = "I can help you with:\n"
            response += "- CPU status and usage\n"
            response += "- Memory/RAM information\n"
            response += "- Temperature monitoring\n"
            response += "- Anomaly detection\n"
            response += "- System recommendations\n"
            response += "- Overall system status\n\n"
            response += "Try asking: 'What's my CPU usage?' or 'Are there any anomalies?'"
        
        # Add messages in new format
        history.append({"role": "user", "content": message})
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
        
        # Format analysis results
        result = "## AI Analysis Results\n\n"
        
        # Performance Insights
        insights = analysis.get('performance_insights', {})
        if 'current_status' in insights:
            result += "### 📊 Performance Status\n"
            for metric, data in insights['current_status'].items():
                status_icon = "🔴" if data.get('status') == 'high' else "🟡" if data.get('status') == 'moderate' else "🟢"
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
        
        return result
    
    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="AI Telemetry Monitor", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🤖 AI Telemetry Monitoring & Analysis")
            gr.Markdown("Monitor your system in real-time and chat with AI about your telemetry data")
            
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
                    gr.Markdown(f"**🤖 LLM:** {model_info}")
                    if not self.llm_analyzer.is_available():
                        gr.Markdown("💡 *Ollama not detected. Start Ollama or set OPENAI_API_KEY/ANTHROPIC_API_KEY*")
            
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
        
        return app


def main():
    """Launch the GUI application"""
    gui = TelemetryGUI()
    app = gui.create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()

