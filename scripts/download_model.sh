#!/bin/bash

# Download Mistral 7B Model
# This script downloads the official Mistral 7B Instruct v0.2 model

echo "🚀 Downloading Mistral 7B Model..."

# Create models directory
mkdir -p models

# Install git lfs if not available
echo "📦 Installing git lfs..."
git lfs install

echo "Downloading Mistral 7B Instruct v0.2 via git..."
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 ./models/mistral-7b-instruct-v0.2

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
