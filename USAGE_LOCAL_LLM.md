# Using Local LLM (llama-cpp-python)

This guide explains how to use the local llama-cpp-python model instead of Ollama.

## Step 1: Download the Model

First, download a model from Ollama and save it locally:

```bash
# Download gemma3:1b (default)
python scripts/download_model.py --model gemma3:1b

# Or download a different model
python scripts/download_model.py --model llama3.2
```

This will:
1. Pull the model from Ollama (if not already available)
2. Find the model file in Ollama's storage
3. Copy it to `models/gemma3-1b.gguf` (or similar)

## Step 2: Install llama-cpp-python

Make sure llama-cpp-python is installed:

```bash
pip install llama-cpp-python
```

## Step 3: Run with Local Model

### Option A: Command Line Arguments

```bash
# Use local llama-cpp model
python gui.py --llm-provider llamacpp --llm-model gemma3-1b.gguf

# Or just use the default model name
python gui.py --llm-provider llamacpp
```

### Option B: Environment Variables

```bash
# Set environment variables
export LLM_PROVIDER=llamacpp
export LLM_MODEL=gemma3-1b.gguf

# Run GUI
python gui.py
```

### Option C: Programmatic Usage

```python
from src.gui_app import TelemetryGUI

# Create GUI with local model
gui = TelemetryGUI(
    llm_provider="llamacpp",
    llm_model="gemma3-1b.gguf"  # Optional, defaults to gemma3-1b.gguf
)

app = gui.create_interface()
app.launch()
```

## CLI Usage (if LLM is used in CLI)

For the CLI application, you can also use the local model:

```python
from src.llm_analyzer import LLMAnalyzer

# Create analyzer with local model
llm = LLMAnalyzer(
    provider="llamacpp",
    model="gemma3-1b.gguf"  # Looks in models/ directory
)

# Use it for analysis
response = llm.answer_question("What's my CPU usage?", telemetry_data, analysis)
```

## Model Location

The local model should be in the `models/` directory:

```
AIOS-experiment/
└── models/
    └── gemma3-1b.gguf  # Your downloaded model
```

If you specify a model name without a path, it will look in `models/` directory. You can also specify a full path:

```python
llm = LLMAnalyzer(
    provider="llamacpp",
    model_path="/path/to/your/model.gguf"
)
```

## Advantages of Local Model

- **Offline**: No need for Ollama server or internet connection
- **Privacy**: All inference happens locally
- **Performance**: Can be faster for small models
- **Control**: Full control over model loading and inference parameters

## Troubleshooting

### Model Not Found

If you get "Model file not found":
1. Check that the model file exists in `models/` directory
2. Verify the filename matches what you specified
3. Try specifying the full path with `model_path` parameter

### Import Error

If you get "llama-cpp-python not installed":
```bash
pip install llama-cpp-python
```

### Slow Performance

For better performance with llama-cpp-python:
- Use smaller models (1B-3B parameters)
- Adjust `n_threads` in the LLMAnalyzer initialization
- Consider using GPU acceleration (requires CUDA/ROCm build)

## Switching Between Providers

You can easily switch between providers:

```bash
# Ollama (requires ollama serve)
python gui.py --llm-provider ollama

# Local llama-cpp
python gui.py --llm-provider llamacpp

# OpenAI (requires API key)
export OPENAI_API_KEY=your_key
python gui.py --llm-provider openai

# Anthropic (requires API key)
export ANTHROPIC_API_KEY=your_key
python gui.py --llm-provider anthropic
```

