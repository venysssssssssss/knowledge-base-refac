#!/usr/bin/env python3
"""
Test script to verify document search functionality is working correctly
"""

import asyncio
import httpx
import argparse
import json
from typing import Dict, Any, List

async def test_document_search(client: httpx.AsyncClient, doc_processor_url: str, query: str, limit: int = 5) -> Dict[str, Any]:
    """Test document search directly"""
    try:
        payload = {
            "query": query,
            "limit": limit,
            "score_threshold": 0.6
        }
        print(f"🔍 Testing document search: '{query}'")
        response = await client.post(f"{doc_processor_url}/search", json=payload, timeout=10.0)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Document search successful")
            if isinstance(result, dict) and "chunks" in result:
                chunks = result["chunks"]
                print(f"📄 Found {len(chunks)} chunks")
                for i, chunk in enumerate(chunks[:3], 1):
                    print(f"\n--- Chunk {i} ---")
                    print(f"Score: {chunk.get('score', 'N/A')}")
                    print(f"Content: {chunk.get('content', '')[:100]}...")
                    print(f"Metadata: {json.dumps(chunk.get('metadata', {}), indent=2)}")
            else:
                print(f"📄 Found {len(result)} chunks")
                for i, chunk in enumerate(result[:3], 1):
                    print(f"\n--- Chunk {i} ---")
                    print(f"Score: {chunk.get('score', 'N/A')}")
                    print(f"Content: {chunk.get('content', '')[:100]}...")
                    print(f"Metadata: {json.dumps(chunk.get('metadata', {}), indent=2)}")
            return result
        else:
            print(f"❌ Document search failed: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        print(f"❌ Document search exception: {e}")
        return {}

async def run_test(doc_processor_url: str, query: str, limit: int):
    """Run the test"""
    async with httpx.AsyncClient() as client:
        # Test health endpoint first
        try:
            health_response = await client.get(f"{doc_processor_url}/health", timeout=5.0)
            if health_response.status_code == 200:
                print(f"✅ Document processor is healthy")
            else:
                print(f"⚠️ Document processor returned {health_response.status_code}: {health_response.text}")
        except Exception as e:
            print(f"❌ Document processor health check failed: {e}")
        
        # Run the document search test
        await test_document_search(client, doc_processor_url, query, limit)

def main():
    parser = argparse.ArgumentParser(description="Test document search functionality")
    parser.add_argument("--url", default="http://localhost:8001", help="Document processor URL")
    parser.add_argument("--query", default="quem pode solicitar alteração cadastral", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    print("🚀 Testing document search")
    asyncio.run(run_test(args.url, args.query, args.limit))
    print("✅ Test completed")

if __name__ == "__main__":
    main()
