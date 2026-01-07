"""
LLM Integration for Telemetry Analysis
Provides natural language explanations and insights using LLMs
Supports Ollama API, OpenAI, Anthropic, and local llama-cpp-python models
"""
import os
import subprocess
import requests
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
import json


class LLMAnalyzer:
    """Uses LLMs to provide natural language analysis of telemetry data"""
    
    def __init__(self, provider: str = "ollama", model: str = None, model_path: Optional[str] = None):
        self.provider = provider
        self.model = model or self._get_default_model()
        self.model_path = model_path
        self.api_key = None
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llama_cpp_model = None
        self.client = None  # Initialize client to None
        # Thread lock for llama-cpp models (not thread-safe)
        self._llm_lock = threading.Lock()
        self._setup_client()
    
    def _get_default_model(self) -> str:
        """Get default model based on provider"""
        defaults = {
            "ollama": "llama3.2",
            "llamacpp": "gemma3-1b.gguf",
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
        elif self.provider == "llamacpp":
            # Setup llama-cpp-python
            try:
                from llama_cpp import Llama
                
                # Determine model path
                # __file__ is src/llm_analyzer.py, so parent.parent is project root
                project_root = Path(__file__).parent.parent
                
                if self.model_path:
                    model_path = Path(self.model_path)
                    if not model_path.is_absolute():
                        # If relative path, make it relative to project root
                        model_path = project_root / model_path
                else:
                    # Look in models directory
                    models_dir = project_root / "models"
                    model_path = models_dir / self.model
                
                # Resolve to absolute path for better error messages
                model_path = model_path.resolve()
                
                if not model_path.exists():
                    # Try without .gguf extension if model doesn't have one
                    if not model_path.suffix or model_path.suffix != '.gguf':
                        model_path_with_ext = model_path.with_suffix('.gguf')
                        if model_path_with_ext.exists():
                            model_path = model_path_with_ext
                        else:
                            # Try adding .gguf if no extension
                            if not model_path.suffix:
                                model_path = model_path.with_suffix('.gguf')
                    
                    if not model_path.exists():
                        print(f"Warning: Model file not found at {model_path}")
                        print(f"  Searched in: {project_root / 'models'}")
                        print(f"  Model name: {self.model}")
                        print(f"  Please download the model first using: python scripts/download_model.py")
                        self.client = None
                        self.llama_cpp_model = None
                        return
                
                # Model file exists, try to load it
                print(f"Loading llama-cpp model from {model_path}")
                try:
                    self.llama_cpp_model = Llama(
                        model_path=str(model_path),
                        n_ctx=4096,  # Context window
                        n_threads=4,  # Number of threads
                        verbose=False
                    )
                    self.client = "llamacpp"
                    print(f"[OK] Successfully loaded model: {model_path.name}")
                except Exception as load_error:
                    print(f"Error loading model file: {load_error}")
                    print(f"  File exists: {model_path.exists()}")
                    print(f"  File size: {model_path.stat().st_size / (1024**2):.1f} MB" if model_path.exists() else "N/A")
                    import traceback
                    traceback.print_exc()
                    self.client = None
                    self.llama_cpp_model = None
            except ImportError as import_err:
                print("Warning: llama-cpp-python not installed. Install with: pip install llama-cpp-python")
                print(f"  Import error: {import_err}")
                self.client = None
                self.llama_cpp_model = None
            except Exception as e:
                print(f"Error setting up llama-cpp model: {e}")
                import traceback
                traceback.print_exc()
                self.client = None
                self.llama_cpp_model = None
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
        elif self.provider == "llamacpp":
            # Setup llama-cpp-python
            try:
                from llama_cpp import Llama
                
                # Determine model path
                # __file__ is src/llm_analyzer.py, so parent.parent is project root
                project_root = Path(__file__).parent.parent
                
                if self.model_path:
                    model_path = Path(self.model_path)
                    if not model_path.is_absolute():
                        # If relative path, make it relative to project root
                        model_path = project_root / model_path
                else:
                    # Look in models directory
                    models_dir = project_root / "models"
                    model_path = models_dir / self.model
                
                # Resolve to absolute path for better error messages
                model_path = model_path.resolve()
                
                if not model_path.exists():
                    # Try without .gguf extension if model doesn't have one
                    if not model_path.suffix or model_path.suffix != '.gguf':
                        model_path_with_ext = model_path.with_suffix('.gguf')
                        if model_path_with_ext.exists():
                            model_path = model_path_with_ext
                        else:
                            # Try adding .gguf if no extension
                            if not model_path.suffix:
                                model_path = model_path.with_suffix('.gguf')
                    
                    if not model_path.exists():
                        print(f"Warning: Model file not found at {model_path}")
                        print(f"  Searched in: {project_root / 'models'}")
                        print(f"  Model name: {self.model}")
                        print(f"  Please download the model first using: python scripts/download_model.py")
                        self.client = None
                        self.llama_cpp_model = None
                        return
                
                # Model file exists, try to load it
                print(f"Loading llama-cpp model from {model_path}")
                try:
                    self.llama_cpp_model = Llama(
                        model_path=str(model_path),
                        n_ctx=4096,  # Context window
                        n_threads=4,  # Number of threads
                        verbose=False
                    )
                    self.client = "llamacpp"
                    print(f"[OK] Successfully loaded model: {model_path.name}")
                except Exception as load_error:
                    print(f"Error loading model file: {load_error}")
                    print(f"  File exists: {model_path.exists()}")
                    print(f"  File size: {model_path.stat().st_size / (1024**2):.1f} MB" if model_path.exists() else "N/A")
                    import traceback
                    traceback.print_exc()
                    self.client = None
                    self.llama_cpp_model = None
            except ImportError as import_err:
                print("Warning: llama-cpp-python not installed. Install with: pip install llama-cpp-python")
                print(f"  Import error: {import_err}")
                self.client = None
                self.llama_cpp_model = None
            except Exception as e:
                print(f"Error setting up llama-cpp model: {e}")
                import traceback
                traceback.print_exc()
                self.client = None
                self.llama_cpp_model = None
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
        if self.provider == "llamacpp":
            # For llamacpp, just check if model is loaded (client might not be set yet)
            return self.llama_cpp_model is not None
        elif self.provider == "ollama":
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
            if self.provider == "llamacpp" and self.llama_cpp_model:
                # Use llama-cpp-python for local inference (thread-safe with lock)
                full_prompt = f"You are a system monitoring expert analyzing telemetry data.\n\n{prompt}"
                with self._llm_lock:  # Prevent concurrent access to model
                    response = self.llama_cpp_model(
                        full_prompt,
                        max_tokens=1000,
                        temperature=0.7,
                        stop=["\n\n\n", "Human:", "User:"],
                        echo=False
                    )
                # Handle llama-cpp-python response format (same as Linux - keep it simple)
                result = response.get('choices', [{}])[0].get('text', '')
                result = result.strip() if result else ''
                
                if not result:
                    return 'Error: Empty response from model. The model may need more context or the prompt may be too long.'
                return result
            elif self.provider == "ollama":
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
        except requests.exceptions.Timeout:
            return self._generate_fallback_response(context, "LLM request timed out while analyzing anomalies")
        except requests.exceptions.RequestException as e:
            return self._generate_fallback_response(context, f"LLM connection error: {str(e)}")
        except Exception as e:
            return self._generate_fallback_response(context, f"LLM error: {str(e)}")
    
    def analyze_performance(self, telemetry_data: List[Dict[str, Any]], 
                           analysis: Dict[str, Any],
                           system_info: Dict[str, Any] = None) -> str:
        """Get LLM analysis of overall system performance"""
        if not self.is_available():
            return "LLM not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
        
        # Build comprehensive context
        context = self._build_comprehensive_context(telemetry_data, analysis, system_info)
        
        context_str = json.dumps(context, indent=1, default=str)
        # Limit context size for small models - they can't handle too much data
        if len(context_str) > 1200:
            context_str = context_str[:1200] + "... (truncated for model size)"
        
        prompt = f"""You are analyzing system telemetry data. Write a clear analysis in plain text paragraphs (not JSON, not code blocks).

Telemetry Data:
{context_str}

Your response should include:
1. Overall system health status (good/moderate/needs attention)
2. Any concerning patterns or anomalies you notice
3. Specific optimization recommendations if needed
4. Brief summary of resource utilization (CPU, memory, disk)

Format your response as clear paragraphs. Do not use JSON format. Do not use code blocks. Write naturally."""

        try:
            if self.provider == "llamacpp" and self.llama_cpp_model:
                # Use llama-cpp-python for local inference (thread-safe with lock)
                # Don't add extra prefix - the prompt already has clear instructions
                with self._llm_lock:  # Prevent concurrent access to model
                    response = self.llama_cpp_model(
                        prompt,
                        max_tokens=400,  # Reasonable limit for small models
                        temperature=0.7,
                        stop=["\n\n\n", "Human:", "User:"],  # Basic stop sequences only
                        echo=False
                    )
                # Handle llama-cpp-python response format (same as Linux - keep it simple)
                result = response.get('choices', [{}])[0].get('text', '')
                result = result.strip() if result else ''
                
                if not result:
                    return 'Error: Empty response from model. The model may need more context or the prompt may be too long.'
                return result
            elif self.provider == "ollama":
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
        except requests.exceptions.Timeout:
            return self._generate_fallback_response(context, "LLM request timed out")
        except requests.exceptions.RequestException as e:
            return self._generate_fallback_response(context, f"LLM connection error: {str(e)}")
        except Exception as e:
            return self._generate_fallback_response(context, f"LLM error: {str(e)}")
    
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
    
    def _generate_fallback_response(self, context: Dict[str, Any], error_msg: str) -> str:
        """Generate a fallback response with telemetry summary when LLM fails"""
        response = f"[WARNING] **{error_msg}**\n\n"
        response += "Here's a summary of your system telemetry:\n\n"
        
        # Latest telemetry
        latest = context.get('latest_telemetry', {})
        if latest:
            response += "**Current System Status:**\n"
            if 'cpu_percent' in latest:
                response += f"- CPU: {latest['cpu_percent']:.1f}%\n"
            if 'memory_percent' in latest:
                response += f"- Memory: {latest['memory_percent']:.1f}%\n"
            if 'disk_percent' in latest:
                response += f"- Disk: {latest['disk_percent']:.1f}%\n"
            if 'avg_temperature_c' in latest:
                response += f"- Temperature: {latest['avg_temperature_c']:.1f}°C\n"
            if 'network_sent_mb' in latest and 'network_recv_mb' in latest:
                response += f"- Network: {latest['network_sent_mb']:.1f} MB sent, {latest['network_recv_mb']:.1f} MB received\n"
            if 'power' in latest:
                power = latest['power']
                if 'gpu_watts' in power:
                    response += f"- GPU Power: {power['gpu_watts']:.1f}W\n"
                if 'cpu_energy_joules' in power:
                    response += f"- CPU Energy: {power['cpu_energy_joules']:.1f}J\n"
            if 'battery_percent' in latest:
                response += f"- Battery: {latest['battery_percent']}% ({'Charging' if latest.get('battery_plugged') else 'Discharging'})\n"
            if 'total_processes' in latest:
                response += f"- Total Processes: {latest['total_processes']}\n"
                if 'top_processes' in latest and latest['top_processes']:
                    response += "- Top Processes:\n"
                    for proc in latest['top_processes'][:3]:
                        proc_cpu = proc.get('cpu_percent', 0) or 0
                        if proc_cpu > 0:
                            response += f"  • {proc.get('name', 'unknown')} (PID {proc.get('pid', 'N/A')}): {proc_cpu:.1f}% CPU\n"
            response += "\n"
        
        # Summary statistics
        summary = context.get('telemetry_summary', {})
        if summary:
            response += "**Session Summary:**\n"
            if 'cpu' in summary:
                cpu = summary['cpu']
                response += f"- CPU: {cpu['min']:.1f}% - {cpu['max']:.1f}% (avg: {cpu['avg']:.1f}%)\n"
            if 'memory' in summary:
                mem = summary['memory']
                response += f"- Memory: {mem['min']:.1f}% - {mem['max']:.1f}% (avg: {mem['avg']:.1f}%)\n"
            if 'gpu_power_watts' in summary:
                gpu = summary['gpu_power_watts']
                response += f"- GPU Power: {gpu['min']:.1f}W - {gpu['max']:.1f}W (avg: {gpu['avg']:.1f}W)\n"
            if 'cpu_energy_joules' in summary:
                cpu_e = summary['cpu_energy_joules']
                response += f"- CPU Energy: {cpu_e['min']:.1f}J - {cpu_e['max']:.1f}J (avg: {cpu_e['avg']:.1f}J)\n"
            response += "\n"
        
        # Analysis results
        analysis = context.get('analysis', {})
        if analysis:
            if analysis.get('anomalies_detected', 0) > 0:
                response += f"**Anomalies:** {analysis['anomalies_detected']} detected ({analysis.get('anomaly_percentage', 0):.1f}%)\n"
                if analysis.get('top_anomaly_factors'):
                    response += f"- Top factors: {', '.join(analysis['top_anomaly_factors'])}\n"
            if analysis.get('warnings'):
                response += "**Warnings:**\n"
                for warning in analysis['warnings']:
                    response += f"- {warning}\n"
            if analysis.get('recommendations'):
                response += "**Recommendations:**\n"
                for rec in analysis['recommendations']:
                    response += f"- {rec}\n"
            response += "\n"
        
        # System info
        sys_info = context.get('system_info', {})
        if sys_info:
            response += "**System Information:**\n"
            response += f"- OS: {sys_info.get('os', 'Unknown')} {sys_info.get('os_release', '')}\n"
            response += f"- CPU: {sys_info.get('cpu_model', 'Unknown')}\n"
            response += f"- Cores: {sys_info.get('cpu_cores', 'Unknown')}\n"
            response += f"- Memory: {sys_info.get('memory_gb', 0):.1f} GB\n"
            if sys_info.get('gpu'):
                response += f"- GPU: {sys_info['gpu'][0] if sys_info['gpu'] else 'Unknown'}\n"
            response += "\n"
        
        response += "💡 *Tip: The LLM timed out, but you can still see your system status above. Try asking a more specific question or wait a moment and try again.*"
        
        return response
    
    def answer_question(self, question: str, telemetry_data: List[Dict[str, Any]], 
                       analysis: Dict[str, Any] = None,
                       system_info: Dict[str, Any] = None,
                       conversation_history: List[Dict[str, str]] = None) -> str:
        """Answer a question about the telemetry data with optional conversation history"""
        if not self.is_available():
            return "LLM not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
        
        # Build comprehensive context with ALL available data
        context = self._build_comprehensive_context(telemetry_data, analysis, system_info)
        
        # Create a more concise prompt to avoid token limits
        # Limit context size for small models - only include essential info
        if len(telemetry_data) == 0:
            # No telemetry data - just use system info for general questions
            if system_info:
                context_str = f"System: {system_info.get('os', 'Unknown')} {system_info.get('os_release', '')}, CPU: {system_info.get('cpu_model', 'Unknown')}, RAM: {system_info.get('memory_gb', 0):.1f} GB"
                if system_info.get('gpu'):
                    context_str += f", GPU: {system_info['gpu'][0] if system_info['gpu'] else 'None'}"
            else:
                context_str = "No system information available."
        else:
            # Limit context to latest snapshot and summary stats only
            context_str = json.dumps(context, indent=1, default=str)
            # Truncate if too long (keep first 2000 chars for small models)
            if len(context_str) > 2000:
                context_str = context_str[:2000] + "... (truncated)"
        
        # Include conversation history if provided
        history_context = ""
        if conversation_history and len(conversation_history) > 0:
            # Format conversation history (the current question is not in history yet, so include all)
            history_messages = []
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content and content.strip():
                    history_messages.append(f"{role.capitalize()}: {content}")
            
            if history_messages:
                # Include last 6 messages (3 exchanges) for context to avoid token limits
                history_context = "\n\nPrevious conversation:\n" + "\n".join(history_messages[-6:])
        
        # Simplify prompt for general questions without telemetry data
        if len(telemetry_data) == 0:
            prompt = f"""You are a helpful system monitoring assistant. Answer the user's question about their device.{history_context}

Question: {question}

System Information:
{context_str}

Provide a clear, helpful answer. If you don't have specific data, provide general information about the topic."""
        else:
            prompt = f"""You are a system performance analyst. Answer the user's question using the telemetry data below.{history_context}

Question: {question}

Telemetry Data:
{context_str}

Provide a clear, accurate answer based on the data. Reference specific values when relevant. If the question refers to previous conversation, use that context to provide a more helpful answer."""

        try:
            if self.provider == "llamacpp" and self.llama_cpp_model:
                # Use llama-cpp-python for local inference (thread-safe with lock)
                full_prompt = f"You are a helpful system monitoring assistant.\n\n{prompt}"
                with self._llm_lock:  # Prevent concurrent access to model
                    response = self.llama_cpp_model(
                        full_prompt,
                        max_tokens=1000,
                        temperature=0.7,
                        stop=["\n\n\n", "Human:", "User:"],
                        echo=False
                    )
                # Handle llama-cpp-python response format (same as Linux - keep it simple)
                result = response.get('choices', [{}])[0].get('text', '')
                result = result.strip() if result else ''
                
                if not result:
                    return 'Error: Empty response from model. The model may need more context or the prompt may be too long.'
                return result
            elif self.provider == "ollama":
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
        except requests.exceptions.Timeout:
            return self._generate_fallback_response(context, "LLM request timed out")
        except requests.exceptions.RequestException as e:
            return self._generate_fallback_response(context, f"LLM connection error: {str(e)}")
        except Exception as e:
            return self._generate_fallback_response(context, f"LLM error: {str(e)}")

