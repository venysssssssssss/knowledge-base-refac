"""
RAG (Retrieval-Augmented Generation) Service
Combines document search with Mistral 7B for contextual answers
"""

import asyncio
import logging
import time
import httpx
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs
MISTRAL_SERVICE_URL = "http://127.0.0.1:8000"  # Local quando rodando direto
DOCUMENT_PROCESSOR_URL = "http://127.0.0.1:8001"

class RAGRequest(BaseModel):
    question: str
    max_tokens: int = 512
    temperature: float = 0.7
    search_limit: int = 3
    score_threshold: float = 0.7

class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    tokens_used: int
    processing_time: float
    search_time: float
    generation_time: float

class RAGService:
    def __init__(self):
        self.http_client = None
    
    async def initialize(self):
        """Initialize HTTP client"""
        self.http_client = httpx.AsyncClient(timeout=60.0)
        logger.info("✅ RAG service initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.http_client:
            await self.http_client.aclose()
    
    async def search_documents(self, query: str, limit: int = 3, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            search_payload = {
                "query": query,
                "limit": limit,
                "score_threshold": score_threshold
            }
            
            response = await self.http_client.post(
                f"{DOCUMENT_PROCESSOR_URL}/search",
                json=search_payload
            )
            
            if response.status_code != 200:
                logger.error(f"Document search failed: {response.text}")
                return []
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def generate_answer(self, question: str, context: str, max_tokens: int = 512, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate answer using Mistral 7B"""
        try:
            mistral_payload = {
                "question": question,
                "context": context,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = await self.http_client.post(
                f"{MISTRAL_SERVICE_URL}/query",
                json=mistral_payload
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Mistral service error: {response.text}")
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")
    
    def build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Build context from search results"""
        if not search_results:
            return ""
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            content = result.get("content", "")
            filename = result.get("metadata", {}).get("filename", "Unknown")
            score = result.get("score", 0)
            
            context_parts.append(f"Documento {i} (Fonte: {filename}, Relevância: {score:.2f}):\n{content}")
        
        return "\n\n".join(context_parts)
    
    async def process_rag_query(self, request: RAGRequest) -> RAGResponse:
        """Process a complete RAG query"""
        start_time = time.time()
        
        # Step 1: Search for relevant documents
        search_start = time.time()
        search_results = await self.search_documents(
            query=request.question,
            limit=request.search_limit,
            score_threshold=request.score_threshold
        )
        search_time = time.time() - search_start
        
        logger.info(f"Found {len(search_results)} relevant documents in {search_time:.2f}s")
        
        # Step 2: Build context
        context = self.build_context(search_results)
        
        # Step 3: Generate answer with Mistral
        generation_start = time.time()
        if context:
            mistral_response = await self.generate_answer(
                question=request.question,
                context=context,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
        else:
            # Fallback: answer without context
            logger.warning("No relevant documents found, answering without context")
            mistral_response = await self.generate_answer(
                question=request.question,
                context="",
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        # Step 4: Prepare sources information
        sources = []
        for result in search_results:
            source = {
                "content_preview": result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
                "score": result.get("score", 0),
                "metadata": result.get("metadata", {})
            }
            sources.append(source)
        
        return RAGResponse(
            question=request.question,
            answer=mistral_response.get("answer", ""),
            sources=sources,
            tokens_used=mistral_response.get("tokens_used", 0),
            processing_time=total_time,
            search_time=search_time,
            generation_time=generation_time
        )

# Global service instance
rag_service = RAGService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""
    # Startup
    await rag_service.initialize()
    
    # Test connections
    try:
        async with httpx.AsyncClient() as client:
            # Test Mistral service
            mistral_response = await client.get(f"{MISTRAL_SERVICE_URL}/health")
            if mistral_response.status_code == 200:
                logger.info("✅ Mistral service is accessible")
            else:
                logger.warning("⚠️ Mistral service not accessible")
            
            # Test Document processor
            doc_response = await client.get(f"{DOCUMENT_PROCESSOR_URL}/health")
            if doc_response.status_code == 200:
                logger.info("✅ Document processor is accessible")
            else:
                logger.warning("⚠️ Document processor not accessible")
                
    except Exception as e:
        logger.error(f"Service connection test failed: {e}")
    
    yield
    
    # Shutdown
    await rag_service.cleanup()

# FastAPI app
app = FastAPI(
    title="RAG Knowledge Base API",
    description="Retrieval-Augmented Generation for document-based Q&A",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "rag-service",
        "dependencies": {
            "mistral_service": MISTRAL_SERVICE_URL,
            "document_processor": DOCUMENT_PROCESSOR_URL
        }
    }

@app.post("/ask", response_model=RAGResponse)
async def ask_question(request: RAGRequest):
    """Ask a question with RAG (Retrieval-Augmented Generation)"""
    try:
        return await rag_service.process_rag_query(request)
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/services/status")
async def get_services_status():
    """Check status of all dependent services"""
    status = {
        "mistral_service": {"url": MISTRAL_SERVICE_URL, "status": "unknown"},
        "document_processor": {"url": DOCUMENT_PROCESSOR_URL, "status": "unknown"}
    }
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check Mistral service
            try:
                response = await client.get(f"{MISTRAL_SERVICE_URL}/health")
                status["mistral_service"]["status"] = "healthy" if response.status_code == 200 else "unhealthy"
                status["mistral_service"]["details"] = response.json() if response.status_code == 200 else response.text
            except Exception as e:
                status["mistral_service"]["status"] = "error"
                status["mistral_service"]["error"] = str(e)
            
            # Check Document processor
            try:
                response = await client.get(f"{DOCUMENT_PROCESSOR_URL}/health")
                status["document_processor"]["status"] = "healthy" if response.status_code == 200 else "unhealthy"
                status["document_processor"]["details"] = response.json() if response.status_code == 200 else response.text
            except Exception as e:
                status["document_processor"]["status"] = "error"
                status["document_processor"]["error"] = str(e)
    
    except Exception as e:
        logger.error(f"Error checking services: {e}")
    
    return status

if __name__ == "__main__":
    uvicorn.run(
        "rag_service:app",
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
