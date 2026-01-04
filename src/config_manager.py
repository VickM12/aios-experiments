"""
Configuration Manager
Handles saving and loading user preferences
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages application configuration and preferences"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return self._default_config()
        return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "first_run": True,
            "llm": {
                "provider": "ollama",
                "model": None
            },
            "monitoring": {
                "default_duration": 60,
                "default_interval": 1.0,
                "auto_archive": True
            },
            "archive": {
                "enabled": True,
                "retention_days": 30
            },
            "notifications": {
                "enabled": False
            }
        }
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'llm.provider')"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default
    
    def set(self, key: str, value: Any):
        """Set a configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def is_first_run(self) -> bool:
        """Check if this is the first run"""
        return self.config.get("first_run", True)
    
    def mark_setup_complete(self):
        """Mark first-time setup as complete"""
        self.config["first_run"] = False
        self.save_config()
    
    def get_llm_provider(self) -> str:
        """Get configured LLM provider"""
        return self.get("llm.provider", "ollama")
    
    def get_llm_model(self) -> Optional[str]:
        """Get configured LLM model"""
        return self.get("llm.model")
    
    def set_llm_config(self, provider: str, model: Optional[str] = None):
        """Set LLM configuration"""
        self.set("llm.provider", provider)
        if model:
            self.set("llm.model", model)
        self.save_config()

