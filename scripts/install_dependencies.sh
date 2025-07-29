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

# Install ByteDance Dolphin for document parsing
echo "🐬 Setting up ByteDance Dolphin for document parsing..."

# Initialize and update submodules if not done yet
if [ ! -f "./external/dolphin/README.md" ]; then
    echo "� Initializing Dolphin submodule..."
    git submodule init
    git submodule update
else
    echo "✅ Dolphin submodule already initialized"
fi

# Install Dolphin requirements if available
if [ -f "./external/dolphin/requirements.txt" ]; then
    echo "📦 Installing Dolphin requirements..."
    poetry run pip install -r ./external/dolphin/requirements.txt
else
    echo "⚠️ Dolphin requirements.txt not found, skipping..."
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
