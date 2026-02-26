FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies (skip llama-cpp-python and platform-specific packages in Docker)
RUN pip install --no-cache-dir \
    psutil>=5.9.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    matplotlib>=3.7.0 \
    plotly>=5.14.0 \
    flask>=2.3.0 \
    python-dotenv>=1.0.0 \
    gradio>=4.0.0 \
    requests>=2.31.0

# Copy application code
COPY . .

# Expose Gradio port
EXPOSE 7860

# Default environment: use Docker Model Runner
ENV LLM_PROVIDER=docker
ENV LLM_MODEL=ai/gemma3
# Docker Model Runner is on the host — accessible via host.docker.internal
ENV DOCKER_MODEL_RUNNER_URL=http://host.docker.internal:12434/engines/llama.cpp/v1

# Run the GUI app
CMD ["python", "gui.py"]

