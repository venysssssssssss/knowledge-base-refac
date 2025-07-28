#!/bin/bash

# Download Mistral 7B AWQ Model
# This script downloads the optimized AWQ quantized version

echo "🚀 Downloading Mistral 7B AWQ Model..."

# Create models directory
mkdir -p models

# Option 1: Using huggingface-cli (recommended)
echo "Installing Hugging Face CLI..."
pip install huggingface_hub[cli]

echo "Downloading Mistral 7B Instruct AWQ..."
huggingface-cli download \
    TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
    --local-dir ./models/mistral-7b-instruct-v0.2 \
    --local-dir-use-symlinks False

# Alternative Option 2: Using git lfs (if preferred)
# git lfs install
# git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-AWQ ./models/mistral-7b-instruct-v0.2

echo "✅ Model download completed!"
echo "📍 Model location: ./models/mistral-7b-instruct-v0.2"

# Verify download
if [ -d "./models/mistral-7b-instruct-v0.2" ]; then
    echo "🔍 Verifying download..."
    ls -la ./models/mistral-7b-instruct-v0.2/
    echo "✅ Download verified!"
else
    echo "❌ Download failed - directory not found"
    exit 1
fi

echo "🎯 Next steps:"
echo "1. Run: docker compose up -d qdrant redis minio"
echo "2. Test model: python ai-services/inference/test_load.py"
echo "3. Start AI service: docker compose up ai-services"
