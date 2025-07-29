#!/usr/bin/env python3
"""
Test script for the RAG Knowledge Base pipeline
Tests PDF upload, search, and Q&A functionality
"""

import asyncio
import httpx
import time
from pathlib import Path

# Service URLs
MISTRAL_SERVICE_URL = "http://localhost:8000"
DOCUMENT_PROCESSOR_URL = "http://localhost:8001"
RAG_SERVICE_URL = "http://localhost:8002"

async def test_services_health():
    """Test if all services are healthy"""
    print("🔍 Testing service health...")
    
    services = {
        "Mistral Service": MISTRAL_SERVICE_URL,
        "Document Processor": DOCUMENT_PROCESSOR_URL,
        "RAG Service": RAG_SERVICE_URL
    }
    
    async with httpx.AsyncClient() as client:
        for name, url in services.items():
            try:
                response = await client.get(f"{url}/health", timeout=5.0)
                if response.status_code == 200:
                    print(f"✅ {name}: Healthy")
                else:
                    print(f"❌ {name}: Unhealthy (Status: {response.status_code})")
            except Exception as e:
                print(f"❌ {name}: Error - {e}")

async def test_mistral_direct():
    """Test Mistral service directly"""
    print("\n🤖 Testing Mistral service...")
    
    async with httpx.AsyncClient() as client:
        try:
            payload = {
                "question": "Olá! Como você está?",
                "context": "",
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            response = await client.post(f"{MISTRAL_SERVICE_URL}/query", json=payload, timeout=30.0)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Mistral Response: {result['answer'][:100]}...")
                print(f"   Tokens used: {result['tokens_used']}")
                print(f"   Processing time: {result['processing_time']:.2f}s")
            else:
                print(f"❌ Mistral Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Mistral Error: {e}")

async def test_document_upload():
    """Test document upload (if PDF available)"""
    print("\n📄 Testing document upload...")
    
    # Create a simple test PDF content
    test_content = """
    Este é um documento de teste para o sistema de conhecimento.
    
    O sistema utiliza Mistral 7B para responder perguntas baseadas em documentos.
    
    Principais características:
    - Processamento de PDFs
    - Geração de embeddings
    - Busca por similaridade
    - Geração de respostas contextualizadas
    
    Para utilizar o sistema:
    1. Faça upload de documentos PDF
    2. Faça perguntas sobre o conteúdo
    3. Receba respostas baseadas nos documentos
    """
    
    # Create a simple text file for testing
    test_file_path = Path("/tmp/test_document.txt")
    test_file_path.write_text(test_content)
    
    print("📝 Created test document with sample content")
    print("   Note: For PDF upload, you would need an actual PDF file")
    
    return test_content

async def test_rag_query():
    """Test RAG query"""
    print("\n🔄 Testing RAG query...")
    
    async with httpx.AsyncClient() as client:
        try:
            payload = {
                "question": "Como funciona o sistema de conhecimento?",
                "max_tokens": 200,
                "temperature": 0.7,
                "search_limit": 3,
                "score_threshold": 0.5
            }
            
            response = await client.post(f"{RAG_SERVICE_URL}/ask", json=payload, timeout=60.0)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ RAG Response:")
                print(f"   Question: {result['question']}")
                print(f"   Answer: {result['answer'][:200]}...")
                print(f"   Sources found: {len(result['sources'])}")
                print(f"   Total time: {result['processing_time']:.2f}s")
                print(f"   Search time: {result['search_time']:.2f}s")
                print(f"   Generation time: {result['generation_time']:.2f}s")
                
                if result['sources']:
                    print("   📚 Sources:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"      {i}. Score: {source['score']:.2f}")
                        print(f"         Preview: {source['content_preview'][:100]}...")
                else:
                    print("   📚 No sources found - answered without context")
                    
            else:
                print(f"❌ RAG Error: {response.text}")
                
        except Exception as e:
            print(f"❌ RAG Error: {e}")

async def test_qdrant_status():
    """Test Qdrant collection status"""
    print("\n🗄️ Testing Qdrant collection...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{DOCUMENT_PROCESSOR_URL}/collections/info", timeout=10.0)
            
            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    print(f"⚠️ Qdrant Status: {result['error']}")
                else:
                    print(f"✅ Qdrant Collection: {result['collection_name']}")
                    print(f"   Vectors count: {result['vectors_count']}")
                    print(f"   Status: {result['status']}")
            else:
                print(f"❌ Qdrant Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Qdrant Error: {e}")

async def main():
    """Run all tests"""
    print("🚀 Testing RAG Knowledge Base Pipeline")
    print("=" * 50)
    
    await test_services_health()
    await test_mistral_direct()
    await test_document_upload()
    await test_qdrant_status()
    await test_rag_query()
    
    print("\n" + "=" * 50)
    print("✨ Test completed!")
    print("\n📝 Next steps:")
    print("1. Upload actual PDF documents via: POST /upload-pdf")
    print("2. Ask questions via: POST /ask")
    print("3. Monitor services via: GET /health endpoints")

if __name__ == "__main__":
    asyncio.run(main())
