"""
LLM Integration for Telemetry Analysis
Provides natural language explanations and insights using LLMs
"""
import os
import subprocess
import requests
from typing import Dict, List, Any, Optional
import json


class LLMAnalyzer:
    """Uses LLMs to provide natural language analysis of telemetry data"""
    
    def __init__(self, provider: str = "ollama", model: str = None):
        self.provider = provider
        self.model = model or self._get_default_model()
        self.api_key = None
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._setup_client()
    
    def _get_default_model(self) -> str:
        """Get default model based on provider"""
        defaults = {
            "ollama": "llama3.2",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307"
        }
        return defaults.get(self.provider, "llama3.2")
    
    def _setup_client(self):
        """Setup the LLM client"""
        if self.provider == "ollama":
            # Try using the ollama Python library first
            try:
                import ollama
                # Try to list models using the library
                try:
                    models_list = ollama.list()
                    model_names = [m.get('name', '') for m in models_list.get('models', [])]
                except:
                    # Fall back to API call
                    response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        models = response.json().get('models', [])
                        model_names = [m.get('name', '') for m in models]
                    else:
                        model_names = []
                
                if not model_names:
                    # No models available via library, try direct API
                    try:
                        response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
                        if response.status_code == 200:
                            models = response.json().get('models', [])
                            model_names = [m.get('name', '') for m in models]
                    except:
                        model_names = []
                    
                    if not model_names:
                        # Still no models - but set client to "ollama" anyway
                        # User might need to pull models, but we'll allow them to try
                        self.client = "ollama"
                        # Try common model names that might work
                        common_models = ['llama3.2', 'llama2', 'alpaca', 'llama3.2:latest', 'llama2:latest', 'alpaca:latest']
                        self.model = self.model or 'llama3.2'  # Use default
                    else:
                        self.client = "ollama"
                        self._select_model(model_names)
                else:
                    self.client = "ollama"
                    self._select_model(model_names)
            except ImportError:
                # ollama library not installed, use API only
                try:
                    response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        models = response.json().get('models', [])
                        model_names = [m.get('name', '') for m in models]
                        if model_names:
                            self.client = "ollama"
                            self._select_model(model_names)
                        else:
                            self.client = None
                    else:
                        self.client = None
                except Exception:
                    self.client = None
            except Exception as e:
                # Any other error, try API as fallback
                try:
                    response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        models = response.json().get('models', [])
                        model_names = [m.get('name', '') for m in models]
                        if model_names:
                            self.client = "ollama"
                            self._select_model(model_names)
                        else:
                            self.client = None
                    else:
                        self.client = None
                except:
                    self.client = None
    
    def _select_model(self, model_names: list):
        """Select the best model from available models"""
        model_found = False
        # Check if default model is available
        for name in model_names:
            if self.model in name or name.startswith(self.model):
                # Keep full name with tag (e.g., "llama3.2:latest")
                self.model = name
                model_found = True
                break
        
        if not model_found:
            # Prefer llama3.2, llama2, or alpaca (in that order)
            preferred = ['llama3.2', 'llama2', 'alpaca']
            for pref in preferred:
                for name in model_names:
                    if pref in name.lower():
                        # Keep full name with tag if available
                        self.model = name
                        model_found = True
                        break
                if model_found:
                    break
            
            # If still not found, use first available (keep full name)
            if not model_found and model_names:
                self.model = model_names[0]
        elif self.provider == "openai":
            try:
                import openai
                self.api_key = os.getenv("OPENAI_API_KEY")
                if self.api_key:
                    self.client = openai.OpenAI(api_key=self.api_key)
                else:
                    self.client = None
            except ImportError:
                self.client = None
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
                if self.api_key:
                    self.client = Anthropic(api_key=self.api_key)
                else:
                    self.client = None
            except ImportError:
                self.client = None
    
    def is_available(self) -> bool:
        """Check if LLM is available"""
        if self.provider == "ollama":
            # Check if Ollama server is running
            if self.client == "ollama":
                return True
            # Even if client not set, check if server is accessible
            try:
                response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=1)
                if response.status_code == 200:
                    # Server is running, allow usage (models might be available even if not listed)
                    self.client = "ollama"
                    if not self.model:
                        self.model = self._get_default_model()
                    return True
            except:
                pass
            return False
        return self.client is not None and self.api_key is not None
    
    def get_model_info(self) -> str:
        """Get information about the current model"""
        if not self.is_available():
            return "LLM not available"
        if self.provider == "ollama":
            return f"Ollama: {self.model}" if self.model else "Ollama: No model selected"
        elif self.provider == "openai":
            return f"OpenAI: {self.model}"
        elif self.provider == "anthropic":
            return f"Anthropic: {self.model}"
        return f"{self.provider}: {self.model}"
    
    def explain_anomalies(self, telemetry_data: List[Dict[str, Any]], 
                         analysis: Dict[str, Any],
                         system_info: Dict[str, Any] = None) -> str:
        """Get LLM explanation of detected anomalies"""
        if not self.is_available():
            return "LLM not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
        
        anomaly_data = analysis.get('anomaly_detection', {})
        if not anomaly_data.get('anomalies_detected', 0):
            return "No anomalies detected. System appears to be operating normally."
        
        # Build comprehensive context
        context = self._build_comprehensive_context(telemetry_data, analysis, system_info)
        
        context_str = json.dumps(context, indent=1, default=str)
        
        prompt = f"""You are analyzing system telemetry data. The system has detected {anomaly_data.get('anomalies_detected', 0)} anomalies out of {len(telemetry_data)} data points.

Telemetry Data:
{context_str}

Please provide:
1. A clear explanation of what these anomalies likely represent
2. Whether this is concerning or normal system behavior
3. Specific recommendations if action is needed
4. What metrics are most unusual

Keep the response concise and actionable."""

        try:
            if self.provider == "ollama":
                # Try using ollama library first
                try:
                    import ollama
                    response = ollama.generate(
                        model=self.model,
                        prompt=f"You are a system monitoring expert analyzing telemetry data.\n\n{prompt}",
                        options={
                            "temperature": 0.7,
                            "num_predict": 1000
                        }
                    )
                    result = response.get('response', 'Error: No response from model')
                    return result.strip() if result else 'Error: Empty response from model'
                except ImportError:
                    # Fall back to API
                    pass
                except Exception as e:
                    # If library call fails, try API
                    pass
                
                # Fall back to API call
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": f"You are a system monitoring expert analyzing telemetry data.\n\n{prompt}",
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 1000
                        }
                    },
                    timeout=120
                )
                if response.status_code == 200:
                    result = response.json().get('response', 'Error: No response from model')
                    return result.strip() if result else 'Error: Empty response from model'
                else:
                    error_text = response.text[:200] if hasattr(response, 'text') else str(response.status_code)
                    return f"Error: Ollama returned status {response.status_code}: {error_text}"
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a system monitoring expert analyzing telemetry data."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
        except Exception as e:
            return f"Error getting LLM explanation: {str(e)}"
    
    def analyze_performance(self, telemetry_data: List[Dict[str, Any]], 
                           analysis: Dict[str, Any],
                           system_info: Dict[str, Any] = None) -> str:
        """Get LLM analysis of overall system performance"""
        if not self.is_available():
            return "LLM not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
        
        # Build comprehensive context
        context = self._build_comprehensive_context(telemetry_data, analysis, system_info)
        
        prompt = f"""Analyze this system's performance based on comprehensive telemetry data:

Complete System Context:
{json.dumps(context, indent=2, default=str)}

Provide:
1. Overall system health assessment
2. Any concerning patterns or trends
3. Specific recommendations for optimization
4. Expected vs actual performance
5. Resource utilization analysis (CPU, memory, disk, network, processes)
6. Temperature and power considerations
7. System-specific insights based on hardware configuration

Keep response concise and technical but understandable, referencing specific data points and trends."""

        try:
            if self.provider == "ollama":
                # Try using ollama library first
                try:
                    import ollama
                    response = ollama.generate(
                        model=self.model,
                        prompt=f"You are a system performance analyst.\n\n{prompt}",
                        options={
                            "temperature": 0.7,
                            "num_predict": 1000
                        }
                    )
                    result = response.get('response', 'Error: No response from model')
                    return result.strip() if result else 'Error: Empty response from model'
                except (ImportError, Exception):
                    pass
                
                # Fall back to API call
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": f"You are a system performance analyst.\n\n{prompt}",
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 1000
                        }
                    },
                    timeout=120
                )
                if response.status_code == 200:
                    result = response.json().get('response', 'Error: No response from model')
                    return result.strip() if result else 'Error: Empty response from model'
                else:
                    error_text = response.text[:200] if hasattr(response, 'text') else str(response.status_code)
                    return f"Error: Ollama returned status {response.status_code}: {error_text}"
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a system performance analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
        except Exception as e:
            return f"Error getting LLM analysis: {str(e)}"
    
    def _build_comprehensive_context(self, telemetry_data: List[Dict[str, Any]], 
                                    analysis: Dict[str, Any] = None,
                                    system_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build comprehensive but concise context from all available telemetry data"""
        context = {
            "session_info": {
                "data_points": len(telemetry_data),
                "time_range": {
                    "start": telemetry_data[0].get('timestamp') if telemetry_data else None,
                    "end": telemetry_data[-1].get('timestamp') if telemetry_data else None
                }
            }
        }
        
        # Latest telemetry snapshot (key metrics only)
        if telemetry_data:
            latest = telemetry_data[-1]
            cpu = latest.get('cpu', {})
            mem = latest.get('memory', {}).get('virtual_memory', {})
            disk = latest.get('disk', {}).get('disk_usage', {})
            network_io = latest.get('network', {}).get('network_io', {})
            processes = latest.get('processes', {})
            
            context["latest_telemetry"] = {
                "timestamp": latest.get('timestamp'),
                "cpu_percent": cpu.get('cpu_percent', 0),
                "cpu_freq_mhz": cpu.get('cpu_freq', {}).get('current', 0),
                "memory_percent": mem.get('percent', 0),
                "memory_used_gb": round(mem.get('used', 0) / (1024**3), 2),
                "disk_percent": disk.get('percent', 0),
                "network_sent_mb": round(network_io.get('bytes_sent', 0) / (1024**2), 1) if network_io else 0,
                "network_recv_mb": round(network_io.get('bytes_recv', 0) / (1024**2), 1) if network_io else 0,
                "network_connections": latest.get('network', {}).get('network_connections', 0),
                "total_processes": processes.get('total_processes', 0),
                "top_processes": processes.get('top_processes', [])[:5]  # Only top 5
            }
            
            # Temperature summary
            temp_data = latest.get('temperature', {})
            if temp_data and 'error' not in temp_data:
                all_temps = []
                for sensor_name, entries in temp_data.items():
                    for entry in entries:
                        if 'current' in entry and entry['current']:
                            all_temps.append(entry['current'])
                if all_temps:
                    context["latest_telemetry"]["avg_temperature_c"] = round(sum(all_temps) / len(all_temps), 1)
            
            # Battery summary
            battery = latest.get('battery', {})
            if battery and 'error' not in battery:
                context["latest_telemetry"]["battery_percent"] = battery.get('percent')
                context["latest_telemetry"]["battery_plugged"] = battery.get('power_plugged', False)
            
            # Power summary
            power = latest.get('power', {})
            if power:
                power_info = {}
                
                # GPU power
                if power.get('gpu'):
                    gpu_powers = [g.get('power_draw_watts') for g in power['gpu'] if g.get('power_draw_watts')]
                    if gpu_powers:
                        power_info["gpu_watts"] = round(sum(gpu_powers) / len(gpu_powers), 1)
                
                # CPU RAPL power
                if power.get('rapl'):
                    for domain, pwr in power['rapl'].items():
                        if 'package' in domain.lower():
                            energy = pwr.get('energy_joules', 0)
                            if energy:
                                power_info["cpu_energy_joules"] = round(energy, 1)
                
                # Power supplies
                if power.get('power_supplies'):
                    supplies = []
                    for name, supply in power['power_supplies'].items():
                        if supply.get('online'):
                            supplies.append({
                                "name": name,
                                "type": supply.get('type', 'Unknown'),
                                "status": supply.get('status', 'Unknown')
                            })
                    if supplies:
                        power_info["power_supplies"] = supplies
                
                if power_info:
                    context["latest_telemetry"]["power"] = power_info
        
        # Telemetry summary (statistics across all data points)
        if len(telemetry_data) > 1:
            cpu_values = [d.get('cpu', {}).get('cpu_percent', 0) for d in telemetry_data]
            mem_values = [d.get('memory', {}).get('virtual_memory', {}).get('percent', 0) for d in telemetry_data]
            disk_values = [d.get('disk', {}).get('disk_usage', {}).get('percent', 0) for d in telemetry_data]
            
            # Power statistics
            gpu_power_values = []
            cpu_energy_values = []
            for d in telemetry_data:
                power_data = d.get('power', {})
                if power_data.get('gpu'):
                    for gpu in power_data['gpu']:
                        if gpu.get('power_draw_watts'):
                            gpu_power_values.append(gpu['power_draw_watts'])
                if power_data.get('rapl'):
                    for domain, pwr in power_data['rapl'].items():
                        if 'package' in domain.lower():
                            energy = pwr.get('energy_joules', 0)
                            if energy:
                                cpu_energy_values.append(energy)
            
            context["telemetry_summary"] = {
                "cpu": {
                    "min": round(min(cpu_values), 1) if cpu_values else 0,
                    "max": round(max(cpu_values), 1) if cpu_values else 0,
                    "avg": round(sum(cpu_values) / len(cpu_values), 1) if cpu_values else 0
                },
                "memory": {
                    "min": round(min(mem_values), 1) if mem_values else 0,
                    "max": round(max(mem_values), 1) if mem_values else 0,
                    "avg": round(sum(mem_values) / len(mem_values), 1) if mem_values else 0
                },
                "disk": {
                    "min": round(min(disk_values), 1) if disk_values else 0,
                    "max": round(max(disk_values), 1) if disk_values else 0,
                    "avg": round(sum(disk_values) / len(disk_values), 1) if disk_values else 0
                }
            }
            
            # Add power summary if available
            if gpu_power_values:
                context["telemetry_summary"]["gpu_power_watts"] = {
                    "min": round(min(gpu_power_values), 1),
                    "max": round(max(gpu_power_values), 1),
                    "avg": round(sum(gpu_power_values) / len(gpu_power_values), 1)
                }
            
            if cpu_energy_values:
                context["telemetry_summary"]["cpu_energy_joules"] = {
                    "min": round(min(cpu_energy_values), 1),
                    "max": round(max(cpu_energy_values), 1),
                    "avg": round(sum(cpu_energy_values) / len(cpu_energy_values), 1)
                }
        
        # Analysis results (summarized, not full details)
        if analysis:
            anomaly = analysis.get('anomaly_detection', {})
            trends = analysis.get('trend_analysis', {}).get('trends', {})
            insights = analysis.get('performance_insights', {})
            
            context["analysis"] = {
                "anomalies_detected": anomaly.get('anomalies_detected', 0),
                "anomaly_percentage": round(anomaly.get('anomaly_percentage', 0), 1),
                "top_anomaly_factors": [f.get('metric') for f in anomaly.get('details', [{}])[0].get('top_factors', [])[:3]] if anomaly.get('details') else [],
                "key_trends": {
                    k: {"trend": v.get('trend'), "rate": round(v.get('rate_of_change_percent', 0), 1)} 
                    for k, v in list(trends.items())[:5]  # Only top 5 trends
                },
                "performance_status": insights.get('current_status', {}),
                "warnings": insights.get('warnings', [])[:3],  # Only top 3 warnings
                "recommendations": insights.get('recommendations', [])[:3]  # Only top 3 recommendations
            }
        
        # System information (key details only)
        if system_info:
            context["system_info"] = {
                "os": system_info.get('os', {}).get('system', 'Unknown'),
                "os_release": system_info.get('os', {}).get('release', 'Unknown'),
                "cpu_model": system_info.get('cpu', {}).get('model', 'Unknown'),
                "cpu_cores": f"{system_info.get('cpu', {}).get('physical_cores', '?')} physical, {system_info.get('cpu', {}).get('logical_cores', '?')} logical",
                "memory_gb": round(system_info.get('memory', {}).get('total_gb', 0), 1),
                "gpu": [g.get('model', 'Unknown') for g in system_info.get('gpu', [])[:1]],  # Only first GPU
                "hostname": system_info.get('system', {}).get('hostname', 'Unknown')
            }
        
        return context
    
    def answer_question(self, question: str, telemetry_data: List[Dict[str, Any]], 
                       analysis: Dict[str, Any] = None,
                       system_info: Dict[str, Any] = None) -> str:
        """Answer a question about the telemetry data"""
        if not self.is_available():
            return "LLM not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
        
        # Build comprehensive context with ALL available data
        context = self._build_comprehensive_context(telemetry_data, analysis, system_info)
        
        # Create a more concise prompt to avoid token limits
        context_str = json.dumps(context, indent=1, default=str)
        
        prompt = f"""You are a system performance analyst. Answer the user's question using the telemetry data below.

Question: {question}

Telemetry Data:
{context_str}

Provide a clear, accurate answer based on the data. Reference specific values when relevant."""

        try:
            if self.provider == "ollama":
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": f"You are a helpful system monitoring assistant.\n\n{prompt}",
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 1000
                        }
                    },
                    timeout=120
                )
                if response.status_code == 200:
                    result = response.json().get('response', 'Error: No response from model')
                    return result.strip() if result else 'Error: Empty response from model'
                else:
                    error_text = response.text[:200] if hasattr(response, 'text') else str(response.status_code)
                    return f"Error: Ollama returned status {response.status_code}: {error_text}"
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful system monitoring assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
        except Exception as e:
            return f"Error getting answer: {str(e)}"

