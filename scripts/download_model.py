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
        
        print(f"[OK] Model '{model_name}' pulled successfully")
        return True
    except Exception as e:
        print(f"Error pulling model: {e}")
        return False


def get_model_path(model_name: str) -> Optional[Path]:
    """Get the path to model files in Ollama's storage"""
    # Ollama stores models in ~/.ollama/models/blobs/ (Linux/Mac)
    # or C:\Users\<user>\.ollama\models\blobs\ (Windows)
    home = Path.home()
    ollama_models = home / ".ollama" / "models" / "blobs"
    
    if not ollama_models.exists():
        # Try alternative location (Linux)
        ollama_models = home / ".local" / "share" / "ollama" / "models" / "blobs"
    
    if not ollama_models.exists():
        print(f"Warning: Could not find Ollama models directory")
        return None
    
    return ollama_models


def get_model_manifest(model_name: str) -> Optional[dict]:
    """Get model manifest from Ollama API to find the correct blob"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/show", 
                               json={"name": model_name},
                               timeout=10)
        if response.status_code == 200:
            manifest = response.json()
            # Debug: print manifest structure
            if manifest:
                print(f"Manifest keys: {list(manifest.keys())}")
                # Print the full manifest for debugging (can be verbose, but helpful)
                import json
                print(f"Full manifest (first 2000 chars): {json.dumps(manifest, indent=2)[:2000]}")
            return manifest
    except Exception as e:
        print(f"Error getting model manifest: {e}")
    return None


def verify_gguf_file(file_path: Path) -> bool:
    """Verify that a file is a valid GGUF file by checking its header"""
    try:
        with open(file_path, 'rb') as f:
            # GGUF files start with specific magic bytes
            # Check for GGUF magic: 0x46554747 (little-endian "GGUF")
            header = f.read(4)
            if len(header) < 4:
                return False
            # Check for GGUF magic bytes
            if header == b'GGUF' or header == b'\x47\x47\x55\x46':
                return True
            # Also check reversed (big-endian)
            if header == b'FUGC' or header == b'\x46\x55\x47\x43':
                return True
        return False
    except Exception:
        return False


def find_model_file(model_name: str, ollama_models_dir: Path) -> Optional[Path]:
    """Find the actual model file (GGUF) in Ollama's storage for the specific model"""
    # Ollama stores models as blobs with SHA256 hashes
    # Strategy: Use manifest to find the correct blob, then fall back to finding largest file
    
    print(f"Searching for model file for '{model_name}' in {ollama_models_dir}...")
    
    # Try to get model manifest to find the correct blob
    manifest = get_model_manifest(model_name)
    model_blob_hash = None
    
    if manifest:
        # The manifest contains digest information that we can use to find the correct blob
        print(f"Retrieved manifest for {model_name}")
        
        # Try multiple strategies to find the model blob hash
        # Strategy 1: Direct digest field
        if 'digest' in manifest:
            model_blob_hash = manifest['digest']
            print(f"Found digest in manifest: {model_blob_hash[:16]}...")
        
        # Strategy 2: Check 'model' nested dict
        if not model_blob_hash and 'model' in manifest:
            model_info = manifest['model']
            if isinstance(model_info, dict):
                if 'digest' in model_info:
                    model_blob_hash = model_info['digest']
                    print(f"Found digest in manifest.model: {model_blob_hash[:16]}...")
                # Check for 'from' field which might contain the base model hash
                if not model_blob_hash and 'from' in model_info:
                    from_field = model_info['from']
                    import re
                    # Handle both sha256: and sha- formats (Windows uses sha-)
                    sha256_match = re.search(r'sha256[-:]?([a-f0-9]{64})', str(from_field), re.IGNORECASE)
                    if sha256_match:
                        model_blob_hash = sha256_match.group(1)
                        print(f"Found hash in manifest.model.from: {model_blob_hash[:16]}...")
        
        # Strategy 3: Check 'layers' array - find the largest layer (model weights)
        if not model_blob_hash:
            for key in ['layers', 'blobs', 'weights', 'config', 'model_layers']:
                if key in manifest:
                    data = manifest[key]
                    if isinstance(data, list) and len(data) > 0:
                        print(f"Found {key} array with {len(data)} items")
                        # Find the largest layer/blob by size
                        largest_size = 0
                        for item in data:
                            if isinstance(item, dict):
                                size = item.get('size', item.get('Size', 0))
                                digest = item.get('digest') or item.get('Digest') or item.get('hash') or item.get('Hash')
                                if size > largest_size and digest:
                                    largest_size = size
                                    model_blob_hash = digest
                        if model_blob_hash:
                            print(f"Found largest layer digest from {key}: {model_blob_hash[:16]}... (size: {largest_size / (1024**2):.1f} MB)")
                            break
        
        # Strategy 4: Check modelfile for hash references
        if not model_blob_hash and 'modelfile' in manifest:
            modelfile_str = str(manifest.get('modelfile', ''))
            import re
            # Look for sha256:hash or sha-hash patterns (Windows uses sha-)
            sha256_matches = re.findall(r'sha256[-:]?([a-f0-9]{64})', modelfile_str, re.IGNORECASE)
            if sha256_matches:
                # Use the first match (usually the model hash)
                model_blob_hash = sha256_matches[0]
                print(f"Found hash in modelfile: {model_blob_hash[:16]}...")
        
        # Strategy 5: Check 'config' or other nested structures
        if not model_blob_hash:
            # Recursively search for digest/hash fields
            def find_hash_recursive(obj, depth=0):
                if depth > 3:  # Limit recursion depth
                    return None
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key.lower() in ['digest', 'hash', 'sha256'] and isinstance(value, str):
                            # Check if it looks like a hash
                            # Handle sha256:, sha256-, sha-, and plain hash formats
                            cleaned = value.replace('sha256:', '').replace('sha256-', '').replace('sha-', '').replace(':', '').replace('-', '')
                            if len(cleaned) >= 32 and all(c in '0123456789abcdefABCDEF' for c in cleaned):
                                return cleaned
                        result = find_hash_recursive(value, depth + 1)
                        if result:
                            return result
                elif isinstance(obj, list):
                    for item in obj:
                        result = find_hash_recursive(item, depth + 1)
                        if result:
                            return result
                return None
            
            found_hash = find_hash_recursive(manifest)
            if found_hash:
                model_blob_hash = found_hash
                print(f"Found hash via recursive search: {model_blob_hash[:16]}...")
    
    # Get all blob files
    blob_files = list(ollama_models_dir.glob("*"))
    print(f"Found {len(blob_files)} files in Ollama storage")
    
    # If we have a hash from manifest, try to find the matching blob
    if model_blob_hash:
        # Clean the hash - remove prefixes and normalize (handle sha256:, sha256-, sha-, etc.)
        hash_clean = model_blob_hash.replace('sha256:', '').replace('sha256-', '').replace('sha-', '').replace(':', '').replace('-', '').lower().strip()
        print(f"Looking for blob with hash: {hash_clean[:16]}... (full: {hash_clean})")
        
        # Try exact match first
        exact_match = None
        partial_matches = []
        
        for blob_file in blob_files:
            if blob_file.is_file():
                blob_name = blob_file.name.lower()
                # Remove any prefixes from blob filename
                # Remove all hash prefixes (sha256-, sha256:, sha-, etc.)
                blob_hash = blob_name.replace('sha256-', '').replace('sha256:', '').replace('sha-', '').replace(':', '').replace('-', '').strip()
                
                # Exact match (full hash)
                if blob_hash == hash_clean or blob_name == hash_clean:
                    exact_match = blob_file
                    print(f"Found exact matching blob file: {blob_file.name}")
                    break
                
                # Partial match (hash starts with or contains the search hash, or vice versa)
                if len(hash_clean) >= 16:  # Only try partial if we have a reasonable hash length
                    if blob_hash.startswith(hash_clean[:16]) or hash_clean.startswith(blob_hash[:16]):
                        partial_matches.append((blob_file, len(blob_hash)))
        
        if exact_match:
            # Verify it's a valid GGUF before returning
            if verify_gguf_file(exact_match):
                print(f"Verified: blob file is a valid GGUF")
                return exact_match
            else:
                print(f"Warning: Exact match found but file is not a valid GGUF, continuing search...")
        
        # If we have partial matches, prefer the one with the longest hash (most specific)
        if partial_matches:
            partial_matches.sort(key=lambda x: x[1], reverse=True)
            # Try each partial match, verify it's a GGUF
            for blob_file, _ in partial_matches:
                if verify_gguf_file(blob_file):
                    print(f"Found and verified partial matching blob file: {blob_file.name}")
                    return blob_file
            # If none are valid GGUF, use the best match anyway (might still work)
            best_match = partial_matches[0][0]
            print(f"Found partial matching blob file: {best_match.name} (best of {len(partial_matches)} matches, not verified as GGUF)")
            return best_match
        
        print(f"Warning: Could not find blob file matching hash {hash_clean[:16]}...")
    
    # Fallback: Find the largest file in the blobs directory
    # Model files are typically the largest files (> 50MB for small models, > 100MB for larger ones)
    largest_file = None
    largest_size = 0
    min_size = 50 * 1024 * 1024  # 50MB minimum
    
    if not model_blob_hash:
        print(f"[WARNING] Could not extract blob hash from manifest.")
        print(f"   This means we can't identify the exact blob file for '{model_name}'.")
        print(f"   Falling back to size-based search (may pick wrong file if multiple models exist).")
        print(f"   Consider using HuggingFace download instead for more reliable results.\n")
    
    # Sort by size and check if files are valid GGUF
    valid_gguf_files = []
    for blob_file in blob_files:
        if blob_file.is_file():
            try:
                size = blob_file.stat().st_size
                if size >= min_size:
                    # Check if it's a valid GGUF file
                    if verify_gguf_file(blob_file):
                        valid_gguf_files.append((blob_file, size))
                    elif size > largest_size:
                        # Keep track of largest file even if not verified yet
                        largest_size = size
                        largest_file = blob_file
            except (OSError, PermissionError):
                continue
    
    # Prefer verified GGUF files
    if valid_gguf_files:
        # Get the largest verified GGUF file
        valid_gguf_files.sort(key=lambda x: x[1], reverse=True)
        largest_file = valid_gguf_files[0][0]
        largest_size = valid_gguf_files[0][1]
        print(f"Found verified GGUF file: {largest_file.name} ({largest_size / (1024**3):.2f} GB)")
                        print(f"[WARNING] Using size-based selection - this may not be the correct file for '{model_name}'")
        print(f"   If this is wrong, try downloading from HuggingFace instead.")
    elif largest_file:
        print(f"Found potential model file: {largest_file.name} ({largest_size / (1024**3):.2f} GB)")
                        print(f"[WARNING] File not verified as GGUF and may not be correct for '{model_name}'")
        print(f"   File format will be verified after copying.")
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
        
        # Verify the copied file is a valid GGUF
        if not verify_gguf_file(dest_path):
            print(f"Warning: Copied file does not appear to be a valid GGUF file.")
            print(f"This may happen if Ollama stores models in a different format.")
            print(f"Attempting to use Ollama's export functionality...")
            
            # Try using Ollama's API to export the model properly
            try:
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "",
                        "stream": False
                    },
                    timeout=5
                )
                # This won't work for export, but let's try a different approach
                # Actually, we should use ollama show to get model info and find the right blob
                pass
            except:
                pass
            
            # Ollama doesn't export models as GGUF - they're in a proprietary format
            print(f"\n[ERROR] The file copied from Ollama is not a valid GGUF file.")
            print(f"\n[WARNING] Important: Ollama stores models in a proprietary format.")
            print(f"   Ollama does NOT provide direct GGUF file downloads.")
            print(f"   The blob files in Ollama's storage cannot be used directly with llama-cpp-python.")
            print(f"\n✅ Solution: Download GGUF models directly from their original sources")
            print(f"\nRecommended approach:")
            print(f"1. Visit HuggingFace and search for GGUF versions of your model:")
            print(f"   https://huggingface.co/models?search=gguf")
            print(f"2. For gemma3:1b, try: https://huggingface.co/google/gemma-2-1b-it")
            print(f"3. Download the .gguf file and place it in: {MODELS_DIR}")
            print(f"\nOr use TheBloke's quantized models (many GGUF options):")
            print(f"   https://huggingface.co/TheBloke")
            
            # Clean up the invalid file
            try:
                dest_path.unlink()
            except:
                pass
            return None
        
        print(f"[OK] Model saved to {dest_path}")
        return dest_path
    except Exception as e:
        print(f"Error copying model: {e}")
        return None


def download_from_huggingface(model_name: str, save_name: Optional[str] = None) -> Optional[Path]:
    """
    Download GGUF model directly from HuggingFace
    
    Args:
        model_name: Model name (e.g., "gemma-2-1b-it" or HuggingFace repo path)
        save_name: Optional custom name for the saved file
    
    Returns:
        Path to the saved model file, or None if failed
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed. Install it with: pip install huggingface_hub")
        return None
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if save_name is None:
        save_name = model_name.replace(":", "-").replace("/", "-") + ".gguf"
    
    dest_path = MODELS_DIR / save_name
    
    print(f"Downloading {model_name} from HuggingFace...")
    print(f"This may take a while depending on model size...")
    
    try:
        # List files in the repo to find GGUF files
        from huggingface_hub import list_repo_files
        
        print(f"Searching for GGUF files in {model_name}...")
        repo_files = list_repo_files(repo_id=model_name, repo_type="model")
        
        # Find all .gguf files
        gguf_files = [f for f in repo_files if f.endswith('.gguf')]
        
        if not gguf_files:
            raise Exception(f"No .gguf files found in repository {model_name}")
        
        # Prefer quantized models (Q4, Q5, Q6, Q8) or the largest file
        # Sort by preference: Q4_K_M > Q5_K_M > Q6_K > Q8_0 > others
        def sort_key(filename):
            filename_lower = filename.lower()
            if 'q4_k_m' in filename_lower:
                return (0, filename)
            elif 'q5_k_m' in filename_lower:
                return (1, filename)
            elif 'q6_k' in filename_lower:
                return (2, filename)
            elif 'q8_0' in filename_lower:
                return (3, filename)
            else:
                return (4, filename)
        
        gguf_files.sort(key=sort_key)
        selected_file = gguf_files[0]
        
        print(f"Found {len(gguf_files)} GGUF file(s), downloading: {selected_file}")
        
        # Download the selected GGUF file
        downloaded_path = hf_hub_download(
            repo_id=model_name,
            filename=selected_file,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )
        
        # Rename to desired name if different
        downloaded_path = Path(downloaded_path)
        if downloaded_path.name != save_name:
            shutil.move(str(downloaded_path), str(dest_path))
        else:
            dest_path = downloaded_path
        
        print(f"[OK] Model downloaded to: {dest_path}")
        return dest_path
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        print(f"\nYou may need to:")
        print(f"1. Specify the full HuggingFace repo path (e.g., 'google/gemma-2-1b-it')")
        print(f"2. Or download manually from: https://huggingface.co/models?search=gguf")
        return None


def download_model(model_name: str = "gemma3:1b", save_name: Optional[str] = None, 
                   source: str = "auto", ollama_url: Optional[str] = None) -> Optional[Path]:
    """
    Download model weights and save locally
    
    Args:
        model_name: Name of the model (e.g., "gemma3:1b" for Ollama, or HuggingFace repo path)
        save_name: Optional custom name for the saved file
        source: Source to download from - "auto" (try Ollama first, fallback to HuggingFace),
                "ollama" (Ollama only), or "huggingface" (HuggingFace only)
        ollama_url: Optional custom Ollama server URL (default: http://localhost:11434)
                    Can be a remote server like http://remote-server:11434
    
    Returns:
        Path to the saved model file, or None if failed
    """
    # Use custom Ollama URL if provided
    global OLLAMA_BASE_URL
    original_url = OLLAMA_BASE_URL
    if ollama_url:
        OLLAMA_BASE_URL = ollama_url
    
    try:
        if source == "huggingface":
            return download_from_huggingface(model_name, save_name)
        
        # Try Ollama first (if source is "auto" or "ollama")
        if source in ["auto", "ollama"]:
            print(f"Attempting to download via Ollama API...")
            print(f"Ollama URL: {OLLAMA_BASE_URL}")
            print(f"Models directory: {MODELS_DIR}\n")
            
            # Check if Ollama server is accessible (local or remote)
            ollama_connected = False
            original_ollama_url = OLLAMA_BASE_URL
            
            try:
                response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
                if response.status_code == 200:
                    print(f"[OK] Connected to Ollama server at {OLLAMA_BASE_URL}")
                    ollama_connected = True
                else:
                    raise Exception(f"Ollama returned status {response.status_code}")
            except Exception as e:
                # If local Ollama failed and we're in auto mode, try to find/use remote Ollama
                if source == "auto" and (OLLAMA_BASE_URL.startswith("http://localhost") or 
                                         OLLAMA_BASE_URL.startswith("http://127.0.0.1")):
                    print(f"[WARNING] Local Ollama server not available: {e}")
                    
                    # Try remote Ollama server if configured via environment variable
                    remote_ollama = os.getenv("OLLAMA_REMOTE_URL")
                    if remote_ollama and not remote_ollama.startswith("http://localhost") and not remote_ollama.startswith("http://127.0.0.1"):
                        if remote_ollama != OLLAMA_BASE_URL:  # Don't try the same URL again
                            print(f"Trying configured remote Ollama server: {remote_ollama}...")
                            try:
                                response = requests.get(f"{remote_ollama}/api/tags", timeout=5)
                                if response.status_code == 200:
                                    print(f"[OK] Connected to remote Ollama server at {remote_ollama}")
                                    OLLAMA_BASE_URL = remote_ollama
                                    ollama_connected = True
                                else:
                                    raise Exception(f"Remote Ollama returned status {response.status_code}")
                            except Exception as remote_e:
                                print(f"  Remote Ollama also unavailable: {remote_e}")
                    
                    # Try Ollama cloud API as fallback
                    if not ollama_connected:
                        print(f"Trying Ollama cloud API: https://ollama.com/api...")
                        try:
                            response = requests.get("https://ollama.com/api/tags", timeout=5)
                            if response.status_code == 200:
                                print(f"[OK] Connected to Ollama cloud API")
                                OLLAMA_BASE_URL = "https://ollama.com"
                                ollama_connected = True
                            else:
                                raise Exception(f"Ollama cloud API returned status {response.status_code}")
                        except Exception as cloud_e:
                            print(f"  Ollama cloud API also unavailable: {cloud_e}")
                    
                    if not ollama_connected:
                        print(f"[WARNING] No Ollama servers available (local, remote, or cloud)")
                        print(f"Falling back to HuggingFace download...\n")
                        return download_from_huggingface(model_name, save_name)
                elif source == "ollama":
                    print(f"[ERROR] Cannot connect to Ollama server: {e}")
                    print(f"\nTo use Ollama:")
                    print(f"1. Install Ollama locally: https://ollama.com/download")
                    print(f"2. Start Ollama server: ollama serve")
                    print(f"3. Or use a remote Ollama server:")
                    print(f"   - Set OLLAMA_BASE_URL environment variable to remote URL")
                    print(f"   - Example: export OLLAMA_BASE_URL=http://your-ollama-server:11434")
                    print(f"4. Or use Ollama cloud API: https://ollama.com/api")
                    return None
                else:
                    # Auto mode but not localhost - already tried, fallback to HuggingFace
                    print(f"[WARNING] Ollama server not available: {e}")
                    print(f"Falling back to HuggingFace download...\n")
                    return download_from_huggingface(model_name, save_name)
            
            print(f"Note: Ollama stores models in a proprietary format.")
            print(f"If extraction fails, we'll suggest HuggingFace as an alternative.\n")
            
            # Check if model is available
            available_models = get_ollama_models()
            if model_name not in available_models:
                print(f"Model '{model_name}' not found in Ollama.")
                print(f"Available models: {', '.join(available_models) if available_models else 'None'}")
                print(f"\nPulling model from Ollama API...")
                if not pull_model(model_name):
                    if source == "auto":
                        print(f"\n[WARNING] Failed to pull from Ollama, falling back to HuggingFace...\n")
                        return download_from_huggingface(model_name, save_name)
                    return None
                # Refresh list
                available_models = get_ollama_models()
                if model_name not in available_models:
                    if source == "auto":
                        print(f"\n[WARNING] Model still not available, falling back to HuggingFace...\n")
                        return download_from_huggingface(model_name, save_name)
                    print(f"Error: Model '{model_name}' still not available after pulling")
                    return None
        
        # Find model file in Ollama storage (only works if Ollama is local)
        # For remote Ollama servers, we can't access the blob storage
        if OLLAMA_BASE_URL.startswith("http://localhost") or OLLAMA_BASE_URL.startswith("http://127.0.0.1"):
            ollama_models_dir = get_model_path(model_name)
            if ollama_models_dir is None:
                print("Error: Could not locate Ollama models directory")
                if source == "auto":
                    print(f"Falling back to HuggingFace...\n")
                    return download_from_huggingface(model_name, save_name)
                return None
            
            model_file = find_model_file(model_name, ollama_models_dir)
            if model_file is None:
                print("Error: Could not find model file in Ollama storage")
                if source == "auto":
                    print(f"Falling back to HuggingFace...\n")
                    return download_from_huggingface(model_name, save_name)
                return None
            
            print(f"Found model file: {model_file} ({model_file.stat().st_size / (1024**2):.1f} MB)")
            
            # Copy to local models directory
            saved_path = copy_model_to_local(model_name, model_file, save_name)
            
            if saved_path:
                print(f"\n[OK] Success! Model saved to: {saved_path}")
                print(f"  You can now use this model with llama-cpp-python")
                return saved_path
            else:
                if source == "auto":
                    print(f"\n[WARNING] Failed to extract GGUF from Ollama, falling back to HuggingFace...\n")
                    return download_from_huggingface(model_name, save_name)
                return None
        else:
            # Remote Ollama server - can't access blob storage directly
            print(f"[WARNING] Remote Ollama server detected: {OLLAMA_BASE_URL}")
            print(f"Cannot access blob storage on remote server.")
            print(f"Falling back to HuggingFace download...\n")
            return download_from_huggingface(model_name, save_name)
    finally:
        # Restore original URL
        OLLAMA_BASE_URL = original_url


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

