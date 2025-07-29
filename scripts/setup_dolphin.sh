#!/bin/bash

# Setup Dolphin model in Ollama for embeddings
echo "🐬 Setting up Dolphin model for embeddings..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama is not running. Please start Ollama first."
    echo "   Run: ollama serve"
    exit 1
fi

echo "✅ Ollama is running"

# List current models
echo "📋 Current models in Ollama:"
ollama list

# Check if dolphin-mistral is already installed
if ollama list | grep -q "dolphin-mistral"; then
    echo "✅ Dolphin-Mistral already installed"
else
    echo "📥 Installing Dolphin-Mistral model..."
    ollama pull dolphin-mistral:latest
    
    if [ $? -eq 0 ]; then
        echo "✅ Dolphin-Mistral installed successfully"
    else
        echo "❌ Failed to install Dolphin-Mistral"
        echo "🔄 Trying alternative Dolphin model..."
        ollama pull dolphin-llama3:latest
        
        if [ $? -eq 0 ]; then
            echo "✅ Dolphin-Llama3 installed successfully"
        else
            echo "❌ Failed to install Dolphin models"
            echo "ℹ️  Will use fallback embedding method"
        fi
    fi
fi

# Test embedding generation
echo "🧪 Testing Dolphin embedding generation..."

# Test if embeddings API works
EMBED_TEST=$(curl -s -X POST http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dolphin-mistral:latest",
    "prompt": "This is a test for embedding generation"
  }')

if echo "$EMBED_TEST" | grep -q "embedding"; then
    echo "✅ Dolphin embeddings working correctly"
    echo "📊 Embedding dimension: $(echo "$EMBED_TEST" | grep -o '"embedding":\[[^]]*\]' | grep -o '[0-9.,-]*' | tr ',' '\n' | wc -l)"
else
    echo "⚠️  Embeddings API may not be available"
    echo "   Response: $EMBED_TEST"
    echo "ℹ️  Will use fallback hash-based embeddings"
fi

echo ""
echo "🎯 Setup completed!"
echo ""
echo "Available models for embeddings:"
ollama list | grep -E "(dolphin|mistral)"
echo ""
echo "Next steps:"
echo "1. Start the document processor: cd ai-services/document-processor && uvicorn pdf_processor:app --port 8001"
echo "2. Upload PDFs and generate embeddings"
echo "3. The system will automatically use Dolphin for embeddings when available"
