#!/bin/bash

# Install dependencies for Dolphin + SentenceTransformers RAG system
echo "📦 Installing dependencies for Dolphin RAG system..."

# Update Poetry dependencies
echo "🔄 Updating Poetry dependencies..."
poetry install

# Install additional packages for embeddings
echo "🧠 Installing SentenceTransformers and related packages..."
poetry add sentence-transformers
poetry add pillow
poetry add transformers
poetry add torch

# Install ByteDance Dolphin for document parsing (optional)
echo "🐬 Installing ByteDance Dolphin for document parsing..."
echo "Note: You can also clone and setup Dolphin manually from https://github.com/ByteDance/Dolphin"

# Create directory for Dolphin if not exists
if [ ! -d "./external/dolphin" ]; then
    echo "📁 Creating directory for external Dolphin..."
    mkdir -p ./external/dolphin
    
    echo "📥 Cloning ByteDance Dolphin repository..."
    cd ./external/dolphin
    git clone https://github.com/ByteDance/Dolphin.git .
    
    # Install Dolphin requirements if available
    if [ -f "requirements.txt" ]; then
        echo "📦 Installing Dolphin requirements..."
        pip install -r requirements.txt
    fi
    
    cd ../../..
fi

echo "✅ Dependencies installation completed!"
echo ""
echo "🚀 Next steps:"
echo "1. Start Qdrant: docker compose up -d qdrant"
echo "2. Start document processor: cd ai-services/document-processor && python pdf_processor.py"
echo "3. Test embedding generation with SentenceTransformers"
echo ""
echo "📚 About the setup:"
echo "- Using SentenceTransformers for embeddings (all-MiniLM-L6-v2)"
echo "- ByteDance Dolphin available for advanced document parsing"
echo "- Fallback to hash-based embeddings if models fail"
