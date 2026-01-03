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
                         analysis: Dict[str, Any]) -> str:
        """Get LLM explanation of detected anomalies"""
        if not self.is_available():
            return "LLM not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
        
        anomaly_data = analysis.get('anomaly_detection', {})
        if not anomaly_data.get('anomalies_detected', 0):
            return "No anomalies detected. System appears to be operating normally."
        
        # Prepare context
        latest = telemetry_data[-1] if telemetry_data else {}
        cpu = latest.get('cpu', {}).get('cpu_percent', 0)
        mem = latest.get('memory', {}).get('virtual_memory', {}).get('percent', 0)
        
        # Get anomaly details
        anomaly_details = anomaly_data.get('details', [])[:5]  # Top 5
        
        prompt = f"""You are analyzing system telemetry data. The system has detected {anomaly_data.get('anomalies_detected', 0)} anomalies out of {len(telemetry_data)} data points.

Current system state:
- CPU usage: {cpu:.1f}%
- Memory usage: {mem:.1f}%

Anomaly details:
{json.dumps(anomaly_details, indent=2)}

Please provide:
1. A clear explanation of what these anomalies likely represent
2. Whether this is concerning or normal system behavior
3. Specific recommendations if action is needed
4. What metrics are most unusual

Keep the response concise and actionable."""

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a system monitoring expert analyzing telemetry data."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
        except Exception as e:
            return f"Error getting LLM explanation: {str(e)}"
    
    def analyze_performance(self, telemetry_data: List[Dict[str, Any]], 
                           analysis: Dict[str, Any]) -> str:
        """Get LLM analysis of overall system performance"""
        if not self.is_available():
            return "LLM not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
        
        insights = analysis.get('performance_insights', {})
        trends = analysis.get('trend_analysis', {})
        
        latest = telemetry_data[-1] if telemetry_data else {}
        cpu = latest.get('cpu', {}).get('cpu_percent', 0)
        mem = latest.get('memory', {}).get('virtual_memory', {}).get('percent', 0)
        disk = latest.get('disk', {}).get('disk_usage', {}).get('percent', 0)
        
        prompt = f"""Analyze this system's performance based on telemetry data:

Current metrics:
- CPU: {cpu:.1f}%
- Memory: {mem:.1f}%
- Disk: {disk:.1f}%

Performance status:
{json.dumps(insights.get('current_status', {}), indent=2)}

Trends:
{json.dumps(trends.get('summary', {}), indent=2)}

Provide:
1. Overall system health assessment
2. Any concerning patterns
3. Specific recommendations for optimization
4. Expected vs actual performance

Keep response concise and technical but understandable."""

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
                            "num_predict": 500
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
                            "num_predict": 500
                        }
                    },
                    timeout=60
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
                    max_tokens=500
                )
                return response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
        except Exception as e:
            return f"Error getting LLM analysis: {str(e)}"
    
    def answer_question(self, question: str, telemetry_data: List[Dict[str, Any]], 
                       analysis: Dict[str, Any] = None) -> str:
        """Answer a question about the telemetry data"""
        if not self.is_available():
            return "LLM not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
        
        # Prepare context
        latest = telemetry_data[-1] if telemetry_data else {}
        context = {
            "current_cpu": latest.get('cpu', {}).get('cpu_percent', 0),
            "current_memory": latest.get('memory', {}).get('virtual_memory', {}).get('percent', 0),
            "data_points": len(telemetry_data),
        }
        
        if analysis:
            context["anomalies"] = analysis.get('anomaly_detection', {}).get('anomalies_detected', 0)
            context["insights"] = analysis.get('performance_insights', {}).get('current_status', {})
        
        prompt = f"""Answer this question about system telemetry data:

Question: {question}

Context:
{json.dumps(context, indent=2)}

Provide a clear, helpful answer based on the telemetry data."""

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
                            "num_predict": 300
                        }
                    },
                    timeout=60
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
                    max_tokens=300
                )
                return response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
        except Exception as e:
            return f"Error getting answer: {str(e)}"

