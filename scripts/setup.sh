#!/bin/bash

# Quick Setup Script for Mistral 7B Knowledge Base
# Run this to get started quickly

echo "🚀 Knowledge Base Setup - Mistral 7B"
echo "=================================="

# Step 1: Check system requirements
echo "📋 Checking system requirements..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Check if NVIDIA Docker is available (for GPU support)
if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
    echo "⚠️  NVIDIA Docker not available. GPU acceleration will not work."
    echo "   Install NVIDIA Container Toolkit for GPU support."
else
    echo "✅ NVIDIA Docker detected"
fi

echo "✅ System requirements OK"

# Step 2: Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{qdrant,redis,minio}
mkdir -p models
mkdir -p logs

echo "✅ Directories created"

# Step 3: Download model (optional - can be done separately)
read -p "📥 Download Mistral 7B model now? (y/N): " download_model

if [[ $download_model =~ ^[Yy]$ ]]; then
    echo "🔄 Downloading Mistral 7B AWQ model..."
    cd "$(dirname "$0")" && ./download_model.sh
    cd - > /dev/null
else
    echo "⏭️  Skipping model download. Run ./scripts/download_model.sh later."
fi

# Step 4: Start infrastructure services
echo "🐳 Starting infrastructure services..."
docker compose up -d qdrant redis minio

echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if services are running
if docker compose ps | grep -q "qdrant.*Up"; then
    echo "✅ Qdrant running on http://localhost:6333"
else
    echo "❌ Qdrant failed to start"
fi

if docker compose ps | grep -q "redis.*Up"; then
    echo "✅ Redis running on localhost:6379"
else
    echo "❌ Redis failed to start"
fi

if docker compose ps | grep -q "minio.*Up"; then
    echo "✅ MinIO running on http://localhost:9001"
    echo "   Default credentials: minioadmin / minioadmin123"
else
    echo "❌ MinIO failed to start"
fi

# Step 5: Instructions for next steps
echo ""
echo "🎯 Next Steps:"
echo "=============="
echo "1. 📥 Download model (if not done): ./scripts/download_model.sh"
echo "2. 🧪 Test model loading: python ai-services/inference/test_load.py"
echo "3. 🚀 Start AI service: docker compose up ai-services"
echo "4. 🌐 Access services:"
echo "   - Qdrant Dashboard: http://localhost:6333/dashboard"
echo "   - MinIO Console: http://localhost:9001"
echo "   - AI API (when running): http://localhost:8000/docs"
echo ""
echo "🔧 Troubleshooting:"
echo "- Logs: docker compose logs [service-name]"
echo "- Stop all: docker compose down"
echo "- Clean restart: docker compose down && docker compose up -d"

echo ""
echo "✨ Setup completed! Your knowledge base infrastructure is ready."
