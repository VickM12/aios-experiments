#!/usr/bin/env python3
"""
Download Model Weights from Ollama
Downloads model weights from Ollama and saves them locally for use with llama-cpp-python
"""
import os
import sys
import json
import requests
import shutil
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODELS_DIR = Path(__file__).parent.parent / "models"


def get_ollama_models() -> list:
    """Get list of available models from Ollama"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m.get('name', '') for m in data.get('models', [])]
        return []
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []


def pull_model(model_name: str) -> bool:
    """Pull a model from Ollama"""
    print(f"Pulling model '{model_name}' from Ollama...")
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=300
        )
        
        if response.status_code != 200:
            print(f"Error: Failed to pull model. Status: {response.status_code}")
            return False
        
        # Stream the response
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if 'status' in data:
                        print(f"  {data['status']}")
                    if 'completed' in data and data.get('completed', 0) > 0:
                        total = data.get('total', 1)
                        completed = data.get('completed', 0)
                        percent = (completed / total * 100) if total > 0 else 0
                        print(f"  Progress: {percent:.1f}% ({completed}/{total})")
                except json.JSONDecodeError:
                    pass
        
        print(f"✓ Model '{model_name}' pulled successfully")
        return True
    except Exception as e:
        print(f"Error pulling model: {e}")
        return False


def get_model_path(model_name: str) -> Optional[Path]:
    """Get the path to model files in Ollama's storage"""
    # Ollama stores models in ~/.ollama/models/blobs/
    home = Path.home()
    ollama_models = home / ".ollama" / "models" / "blobs"
    
    if not ollama_models.exists():
        # Try alternative location
        ollama_models = home / ".local" / "share" / "ollama" / "models" / "blobs"
    
    if not ollama_models.exists():
        print(f"Warning: Could not find Ollama models directory")
        return None
    
    # Get model manifest to find the actual model file
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/show", 
                               json={"name": model_name},
                               timeout=10)
        if response.status_code == 200:
            model_info = response.json()
            # The model file is typically the largest blob
            # We'll need to find it by checking the manifest
            pass
    except Exception as e:
        print(f"Error getting model info: {e}")
    
    # For now, we'll use a simpler approach: copy from Ollama's cache
    # The actual model file will be in the blobs directory
    return ollama_models


def find_model_file(model_name: str, ollama_models_dir: Path) -> Optional[Path]:
    """Find the actual model file (GGUF) in Ollama's storage"""
    # Ollama stores models as blobs with SHA256 hashes
    # We need to find the model file by checking manifests or by size
    # For gemma3:1b, we'll look for the largest .gguf file or check the manifest
    
    # Strategy: Find the largest file in the blobs directory
    # Model files are typically the largest files (> 50MB for small models, > 100MB for larger ones)
    # For gemma3:1b, it should be around 1-2GB
    largest_file = None
    largest_size = 0
    min_size = 50 * 1024 * 1024  # 50MB minimum
    
    print(f"Searching for model file in {ollama_models_dir}...")
    blob_files = list(ollama_models_dir.glob("*"))
    print(f"Found {len(blob_files)} files in Ollama storage")
    
    for blob_file in blob_files:
        if blob_file.is_file():
            try:
                size = blob_file.stat().st_size
                if size > largest_size and size >= min_size:
                    largest_size = size
                    largest_file = blob_file
            except (OSError, PermissionError):
                continue
    
    if largest_file:
        print(f"Found model file: {largest_file.name} ({largest_size / (1024**3):.2f} GB)")
    else:
        print(f"Warning: No model file found (searched for files > {min_size / (1024**2):.0f} MB)")
    
    return largest_file


def copy_model_to_local(model_name: str, source_path: Path, dest_name: Optional[str] = None) -> Optional[Path]:
    """Copy model file from Ollama storage to local models directory"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use model name as filename, or provided name
    if dest_name is None:
        # Clean model name for filename (gemma3:1b -> gemma3-1b.gguf)
        dest_name = model_name.replace(":", "-") + ".gguf"
    
    dest_path = MODELS_DIR / dest_name
    
    print(f"Copying model from {source_path} to {dest_path}...")
    try:
        shutil.copy2(source_path, dest_path)
        print(f"✓ Model saved to {dest_path}")
        return dest_path
    except Exception as e:
        print(f"Error copying model: {e}")
        return None


def download_model(model_name: str = "gemma3:1b", save_name: Optional[str] = None) -> Optional[Path]:
    """
    Download model weights from Ollama and save locally
    
    Args:
        model_name: Name of the model in Ollama (e.g., "gemma3:1b")
        save_name: Optional custom name for the saved file
    
    Returns:
        Path to the saved model file, or None if failed
    """
    print(f"Downloading model: {model_name}")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print(f"Models directory: {MODELS_DIR}\n")
    
    # Check if Ollama is running
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code != 200:
            print("Error: Ollama is not responding. Make sure Ollama is running.")
            return None
    except Exception as e:
        print(f"Error: Cannot connect to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        return None
    
    # Check if model is available
    available_models = get_ollama_models()
    if model_name not in available_models:
        print(f"Model '{model_name}' not found in Ollama.")
        print(f"Available models: {', '.join(available_models) if available_models else 'None'}")
        print(f"\nPulling model from Ollama...")
        if not pull_model(model_name):
            return None
        # Refresh list
        available_models = get_ollama_models()
        if model_name not in available_models:
            print(f"Error: Model '{model_name}' still not available after pulling")
            return None
    
    # Find model file in Ollama storage
    ollama_models_dir = get_model_path(model_name)
    if ollama_models_dir is None:
        print("Error: Could not locate Ollama models directory")
        return None
    
    model_file = find_model_file(model_name, ollama_models_dir)
    if model_file is None:
        print("Error: Could not find model file in Ollama storage")
        print("You may need to manually locate the model file")
        return None
    
    print(f"Found model file: {model_file} ({model_file.stat().st_size / (1024**2):.1f} MB)")
    
    # Copy to local models directory
    saved_path = copy_model_to_local(model_name, model_file, save_name)
    
    if saved_path:
        print(f"\n✓ Success! Model saved to: {saved_path}")
        print(f"  You can now use this model with llama-cpp-python")
        return saved_path
    else:
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download model weights from Ollama")
    parser.add_argument("--model", default="gemma3:1b", help="Model name to download (default: gemma3:1b)")
    parser.add_argument("--save-as", help="Custom name for saved model file")
    parser.add_argument("--ollama-url", default=OLLAMA_BASE_URL, help="Ollama base URL")
    
    args = parser.parse_args()
    
    if args.ollama_url != OLLAMA_BASE_URL:
        os.environ["OLLAMA_BASE_URL"] = args.ollama_url
    
    result = download_model(args.model, args.save_as)
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)

