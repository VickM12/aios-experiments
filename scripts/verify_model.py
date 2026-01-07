#!/usr/bin/env python3
"""
Verify and diagnose GGUF model files
Helps identify issues with downloaded models
"""
import sys
from pathlib import Path
import struct

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS_DIR = Path(__file__).parent.parent / "models"


def read_gguf_metadata(file_path: Path) -> dict:
    """Read GGUF file metadata to get model information"""
    info = {
        "valid": False,
        "size_mb": 0,
        "quantization": "unknown",
        "architecture": "unknown",
        "error": None
    }
    
    try:
        info["size_mb"] = file_path.stat().st_size / (1024**2)
        
        with open(file_path, 'rb') as f:
            # Check magic bytes
            magic = f.read(4)
            if magic != b'GGUF':
                info["error"] = f"Invalid magic bytes: {magic.hex()}"
                return info
            
            # Read version (uint32)
            version = struct.unpack('<I', f.read(4))[0]
            info["version"] = version
            
            # Read tensor count (uint64)
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            info["tensor_count"] = tensor_count
            
            # Read metadata count (uint64)
            metadata_count = struct.unpack('<Q', f.read(8))[0]
            info["metadata_count"] = metadata_count
            
            # Try to read some metadata to get quantization info
            # This is a simplified parser - full GGUF parsing is more complex
            try:
                # Read metadata KV pairs (simplified)
                for _ in range(min(metadata_count, 20)):  # Limit to first 20 to avoid reading too much
                    # Read key length (uint64)
                    key_len = struct.unpack('<Q', f.read(8))[0]
                    if key_len > 1000:  # Sanity check
                        break
                    key = f.read(key_len).decode('utf-8', errors='ignore')
                    
                    # Read value type (uint32)
                    value_type = struct.unpack('<I', f.read(4))[0]
                    
                    # Skip value based on type (simplified - just skip)
                    if value_type == 8:  # String
                        str_len = struct.unpack('<Q', f.read(8))[0]
                        if str_len < 1000:
                            value = f.read(str_len).decode('utf-8', errors='ignore')
                            if 'quant' in key.lower() or 'q4' in value.lower() or 'q5' in value.lower() or 'q8' in value.lower():
                                info["quantization"] = value
                            if 'arch' in key.lower() or 'model' in key.lower():
                                if 'arch' in key.lower():
                                    info["architecture"] = value
            except:
                pass  # If metadata parsing fails, that's okay
            
            info["valid"] = True
    except Exception as e:
        info["error"] = str(e)
    
    return info


def verify_model_file(model_path: Path) -> dict:
    """Verify a model file and return diagnostic information"""
    result = {
        "path": str(model_path),
        "exists": model_path.exists(),
        "size_mb": 0,
        "size_gb": 0,
        "is_valid_gguf": False,
        "metadata": None,
        "recommendations": []
    }
    
    if not model_path.exists():
        result["recommendations"].append("Model file does not exist")
        return result
    
    result["size_mb"] = model_path.stat().st_size / (1024**2)
    result["size_gb"] = model_path.stat().st_size / (1024**3)
    
    # Check if it's a valid GGUF
    try:
        with open(model_path, 'rb') as f:
            header = f.read(4)
            result["is_valid_gguf"] = (header == b'GGUF')
    except:
        pass
    
    if result["is_valid_gguf"]:
        result["metadata"] = read_gguf_metadata(model_path)
    
    # Provide recommendations based on size
    if result["size_gb"] < 0.5:
        result["recommendations"].append("Model file is very small (< 0.5 GB) - might be corrupted or wrong quantization")
    elif result["size_gb"] > 5:
        result["recommendations"].append("Model file is very large (> 5 GB) - might be wrong model or unquantized")
    
    # For gemma3:1b, expected sizes:
    # Q2: ~200-300 MB
    # Q4_K_M: ~500-700 MB
    # Q5_K_M: ~700-900 MB
    # Q8_0: ~1.2-1.5 GB
    # F16 (unquantized): ~2-3 GB
    
    if "gemma" in model_path.name.lower() and "1b" in model_path.name.lower():
        if 0.5 <= result["size_gb"] <= 0.9:
            result["recommendations"].append("Size looks reasonable for Q4/Q5 quantized gemma3:1b")
        elif result["size_gb"] < 0.5:
            result["recommendations"].append("Size is small - might be Q2 quantization (lower quality)")
        elif result["size_gb"] > 1.5:
            result["recommendations"].append("Size is large - might be Q8 or unquantized (better quality)")
    
    return result


def find_backups(model_name: str = "gemma3-1b.gguf") -> list:
    """Find backup files for a model"""
    backups = []
    base_name = Path(model_name).stem
    
    for file in MODELS_DIR.glob(f"{base_name}_backup_*.gguf"):
        backups.append({
            "path": file,
            "size_mb": file.stat().st_size / (1024**2),
            "size_gb": file.stat().st_size / (1024**3),
            "modified": file.stat().st_mtime
        })
    
    return sorted(backups, key=lambda x: x["modified"], reverse=True)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify GGUF model files")
    parser.add_argument("--model", default="gemma3-1b.gguf", help="Model filename to verify")
    parser.add_argument("--list-backups", action="store_true", help="List available backup files")
    
    args = parser.parse_args()
    
    model_path = MODELS_DIR / args.model
    
    print(f"Verifying model: {model_path}")
    print("=" * 60)
    
    result = verify_model_file(model_path)
    
    print(f"Path: {result['path']}")
    print(f"Exists: {result['exists']}")
    if result['exists']:
        print(f"Size: {result['size_mb']:.1f} MB ({result['size_gb']:.2f} GB)")
        print(f"Valid GGUF: {result['is_valid_gguf']}")
        
        if result['metadata']:
            meta = result['metadata']
            print(f"\nMetadata:")
            print(f"  Valid: {meta['valid']}")
            if meta['valid']:
                print(f"  Version: {meta.get('version', 'unknown')}")
                print(f"  Tensors: {meta.get('tensor_count', 'unknown')}")
                print(f"  Quantization: {meta.get('quantization', 'unknown')}")
                print(f"  Architecture: {meta.get('architecture', 'unknown')}")
            if meta.get('error'):
                print(f"  Error: {meta['error']}")
        
        if result['recommendations']:
            print(f"\nRecommendations:")
            for rec in result['recommendations']:
                print(f"  - {rec}")
    
    if args.list_backups or not result['exists']:
        print(f"\n" + "=" * 60)
        print("Checking for backup files...")
        backups = find_backups(args.model)
        if backups:
            print(f"Found {len(backups)} backup(s):")
            for i, backup in enumerate(backups, 1):
                from datetime import datetime
                mod_time = datetime.fromtimestamp(backup['modified']).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  {i}. {backup['path'].name}")
                print(f"     Size: {backup['size_mb']:.1f} MB ({backup['size_gb']:.2f} GB)")
                print(f"     Modified: {mod_time}")
                print(f"     To restore: cp '{backup['path']}' '{model_path}'")
        else:
            print("No backup files found")


if __name__ == "__main__":
    main()

