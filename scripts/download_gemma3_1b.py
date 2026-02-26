#!/usr/bin/env python3
"""
Download Gemma 3 1B-IT model in GGUF format for use with llama-cpp-python

This script downloads a quantized GGUF version of google/gemma-3-1b-it
Uses HF_TOKEN from environment variables for authentication.

Model info: https://huggingface.co/google/gemma-3-1b-it
GGUF version: https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF
"""

import os
import sys
from pathlib import Path

def main():
    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set!")
        print("Please set your Hugging Face token:")
        print("  Windows: $env:HF_TOKEN='your_token_here'")
        print("  Linux/Mac: export HF_TOKEN='your_token_here'")
        print("\nGet your token at: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    print(f"[OK] Found HF_TOKEN: {hf_token[:8]}...")
    
    # Try to import huggingface_hub
    try:
        from huggingface_hub import hf_hub_download, HfApi
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download, HfApi
    
    # Model configuration
    # Using bartowski's quantized GGUF version (popular, well-tested)
    repo_id = "bartowski/google_gemma-3-1b-it-GGUF"
    
    # Available quantizations for 1B model:
    # - Q8_0: Best quality, ~1.1GB
    # - Q6_K_L: Great quality, ~900MB  
    # - Q4_K_M: Good balance, ~700MB
    # - F16: Full precision, ~2GB
    
    # For 1B model, Q8_0 is small enough to use best quality
    filename = "google_gemma-3-1b-it-Q8_0.gguf"
    
    # Destination
    script_dir = Path(__file__).parent.parent
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    output_path = models_dir / "gemma3-1b-it-Q8_0.gguf"
    
    print(f"\n{'='*60}")
    print("Downloading Gemma 3 1B-IT (Q8_0 quantization - best quality)")
    print(f"{'='*60}")
    print(f"Source: {repo_id}")
    print(f"File: {filename}")
    print(f"Destination: {output_path}")
    print(f"Size: ~1.1 GB")
    print(f"{'='*60}\n")
    
    # Check if already exists
    if output_path.exists():
        size_gb = output_path.stat().st_size / (1024**3)
        print(f"[!] Model already exists: {output_path}")
        print(f"    Size: {size_gb:.2f} GB")
        response = input("Download again? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return
    
    try:
        # Verify token and repo access
        print("Verifying Hugging Face access...")
        api = HfApi(token=hf_token)
        
        # List available files
        print(f"Checking available files in {repo_id}...")
        try:
            files = api.list_repo_files(repo_id, token=hf_token)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            print(f"Found {len(gguf_files)} GGUF files available:")
            for f in gguf_files[:5]:  # Show first 5
                print(f"  - {f}")
            if len(gguf_files) > 5:
                print(f"  ... and {len(gguf_files) - 5} more")
        except Exception as e:
            print(f"Warning: Could not list files: {e}")
        
        # Download the model
        print(f"\nDownloading {filename}...")
        print("This may take a while depending on your connection...")
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=hf_token,
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        
        # Rename to simpler name
        downloaded = Path(downloaded_path)
        if downloaded.exists() and downloaded != output_path:
            if output_path.exists():
                output_path.unlink()
            downloaded.rename(output_path)
        
        print(f"\n{'='*60}")
        print("[OK] Download complete!")
        print(f"{'='*60}")
        print(f"Model saved to: {output_path}")
        size_gb = output_path.stat().st_size / (1024**3)
        print(f"Size: {size_gb:.2f} GB")
        
        print(f"\n{'='*60}")
        print("To use this model:")
        print(f"{'='*60}")
        print("1. In the GUI Settings tab:")
        print("   - Set Provider to: llamacpp")
        print("   - Set Model to: gemma3-1b-it-Q8_0.gguf")
        print("   - Click 'Save LLM Settings'")
        print("\nOr run with command line:")
        print(f"   python gui.py --llm-provider llamacpp --llm-model gemma3-1b-it-Q8_0.gguf")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've accepted the Gemma license at:")
        print("   https://huggingface.co/google/gemma-3-1b-it")
        print("2. Check your HF_TOKEN is valid")
        print("3. Check your internet connection")
        
        # Try alternative repo
        print("\nTrying alternative download source...")
        alt_repos = [
            ("unsloth/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q8_0.gguf"),
            ("lmstudio-community/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q8_0.gguf"),
        ]
        
        for alt_repo, alt_file in alt_repos:
            try:
                print(f"Trying {alt_repo}...")
                downloaded_path = hf_hub_download(
                    repo_id=alt_repo,
                    filename=alt_file,
                    token=hf_token,
                    local_dir=models_dir,
                    local_dir_use_symlinks=False
                )
                print(f"[OK] Downloaded from {alt_repo}")
                break
            except Exception as alt_e:
                print(f"  Failed: {alt_e}")
                continue
        
        sys.exit(1)


if __name__ == "__main__":
    main()

