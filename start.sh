#!/bin/bash

export OLLAMA_MODELS="/runpod-volume"

# Start up Ollama
ollama serve &
sleep 10

# Check if model is available
ollama list | grep "llama3:8b"
if [ $? -ne 0 ]; then
    echo "⬇️ Model llama3:8b not found, installing..."

    # Install model
    ollama pull llama3:8b

    # Run ollama models to ensure it is available
    ollama run llama3:8b
fi

# Run the Python handler
python3 -u rp_lb_handler.py