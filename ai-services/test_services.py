#!/usr/bin/env python3
"""
Test script to verify all services are functioning correctly
"""

import argparse
import asyncio
import json
from typing import Any, Dict

import httpx


async def test_service_health(
    client: httpx.AsyncClient, url: str, name: str
) -> bool:
    """Test if a service is healthy by calling its /health endpoint"""
    try:
        response = await client.get(f'{url}/health', timeout=5.0)
        if response.status_code == 200:
            print(f'✅ {name} service is healthy')
            return True
        else:
            print(
                f'❌ {name} service returned {response.status_code}: {response.text}'
            )
            return False
    except Exception as e:
        print(f'❌ {name} service is not accessible: {e}')
        return False


async def test_rag_query(
    client: httpx.AsyncClient, rag_url: str, question: str
) -> Dict[str, Any]:
    """Test the RAG service with a question"""
    try:
        payload = {
            'question': question,
            'max_tokens': 512,
            'temperature': 0.7,
            'search_limit': 5,
            'score_threshold': 0.6,
        }
        print(f"🔍 Testing RAG query: '{question}'")
        response = await client.post(
            f'{rag_url}/ask', json=payload, timeout=30.0
        )
        if response.status_code == 200:
            result = response.json()
            print(f'✅ RAG query successful')
            print(f"📝 Answer: {result.get('answer', 'No answer')[:100]}...")
            print(f"🔗 Sources: {len(result.get('sources', []))} documents")
            return result
        else:
            print(
                f'❌ RAG query failed: {response.status_code} - {response.text}'
            )
            return {}
    except Exception as e:
        print(f'❌ RAG query exception: {e}')
        return {}


async def test_document_search(
    client: httpx.AsyncClient, doc_processor_url: str, query: str
) -> Dict[str, Any]:
    """Test document search directly"""
    try:
        payload = {'query': query, 'limit': 5, 'score_threshold': 0.6}
        print(f"🔍 Testing document search: '{query}'")
        response = await client.post(
            f'{doc_processor_url}/search', json=payload, timeout=10.0
        )
        if response.status_code == 200:
            result = response.json()
            print(f'✅ Document search successful')
            return result
        else:
            print(
                f'❌ Document search failed: {response.status_code} - {response.text}'
            )
            return {}
    except Exception as e:
        print(f'❌ Document search exception: {e}')
        return {}


async def run_tests(rag_url: str, doc_processor_url: str, mistral_url: str):
    """Run all tests"""
    async with httpx.AsyncClient() as client:
        # 1. Test all services health
        rag_healthy = await test_service_health(client, rag_url, 'RAG')
        doc_healthy = await test_service_health(
            client, doc_processor_url, 'Document Processor'
        )
        mistral_healthy = await test_service_health(
            client, mistral_url, 'Mistral'
        )

        if not all([rag_healthy, doc_healthy, mistral_healthy]):
            print('⚠️ Some services are not healthy, tests may fail')

        # 2. Test document search directly
        question = 'quem pode solicitar alteração cadastral'
        doc_result = await test_document_search(
            client, doc_processor_url, question
        )

        # Check if search returned any results
        if (
            not doc_result
            or 'chunks' not in doc_result
            or not doc_result['chunks']
        ):
            print(
                '⚠️ Document search returned no results, RAG may not work properly'
            )

        # 3. Test RAG query
        rag_result = await test_rag_query(client, rag_url, question)

        # 4. Print detailed diagnostic info
        print('\n📊 Diagnostic Summary:')
        print(
            f"- Document search returned {len(doc_result.get('chunks', []))} chunks"
        )
        if doc_result and 'chunks' in doc_result and doc_result['chunks']:
            print(
                f"- Top search score: {doc_result['chunks'][0].get('score', 'N/A')}"
            )
        print(
            f"- RAG response time: {rag_result.get('processing_time', 'N/A')}s"
        )
        print(f"- RAG tokens used: {rag_result.get('tokens_used', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description='Test RAG system components')
    parser.add_argument(
        '--rag', default='http://localhost:8002', help='RAG service URL'
    )
    parser.add_argument(
        '--doc', default='http://localhost:8001', help='Document processor URL'
    )
    parser.add_argument(
        '--mistral',
        default='http://localhost:8003',
        help='Mistral service URL',
    )
    parser.add_argument(
        '--question',
        default='quem pode solicitar alteração cadastral',
        help='Test question',
    )

    args = parser.parse_args()

    print('🚀 Starting service tests')
    asyncio.run(run_tests(args.rag, args.doc, args.mistral))
    print('✅ Tests completed')


if __name__ == '__main__':
    main()
