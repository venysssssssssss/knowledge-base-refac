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

# Configuração básica de logging estruturado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

# Princípios SOLID aplicados
# SRP: Cada classe/função tem responsabilidade única
# OCP: Classes abertas para extensão, fechadas para modificação
# LSP: Subclasses podem substituir superclasses
# ISP: Interfaces específicas para cada operação
# DIP: Dependa de abstrações

class RagLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def info(self, msg):
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)

    def debug(self, msg):
        self.logger.debug(msg)
        
    def warning(self, msg):
        self.logger.warning(msg)
        
    def warn(self, msg):
        self.logger.warning(msg)

# Exemplo de uso do logger
rag_logger = RagLogger(__name__)
rag_logger.info('RAG Service iniciado.')

# Service URLs
MISTRAL_SERVICE_URL = "http://10.117.0.19:8003"  # Updated to avoid port conflict
DOCUMENT_PROCESSOR_URL = "http://10.117.0.19:8001"

class RAGRequest(BaseModel):
    question: str
    max_tokens: int = 512
    temperature: float = 0.7
    search_limit: int = 8
    score_threshold: float = 0.5
    document_id: str = None  # Novo campo opcional para filtrar por documento

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
        rag_logger.info("✅ RAG service initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.http_client:
            await self.http_client.aclose()
    
    async def search_documents(self, query: str, limit: int = 3, score_threshold: float = 0.5, document_id: str = None) -> List[Dict[str, Any]]:
        """Search for relevant documents, optionally filtering by document_id"""
        try:
            search_payload = {
                "query": query,
                "limit": limit,
                "score_threshold": score_threshold
            }
            if document_id:
                search_payload["document_id"] = document_id
            response = await self.http_client.post(
                f"{DOCUMENT_PROCESSOR_URL}/search",
                json=search_payload
            )
            if response.status_code != 200:
                rag_logger.error(f"Document search failed: {response.text}")
                return []
            
            # Handle different response formats - the document processor might return
            # either a list of chunks directly or a dict with 'chunks' key
            response_data = response.json()
            if isinstance(response_data, dict) and "chunks" in response_data:
                return response_data["chunks"]
            return response_data
        except Exception as e:
            rag_logger.error(f"Error searching documents: {e}")
            return []
    
    def build_context_intelligent(self, search_results: List[Dict[str, Any]], question: str) -> str:
        """
        Construção inteligente de contexto com:
        1. Ranking por relevância
        2. Deduplicação semântica  
        3. Estruturação hierárquica
        4. Otimização de token usage
        """
        if not search_results:
            return ""
        
        # 1. Filtrar e ranquear resultados
        filtered_results = []
        for result in search_results:
            content = result.get("content", "").strip()
            score = result.get("score", 0)
            
            # Filtrar conteúdo muito curto ou irrelevante
            if len(content) < 20 or score < 0.3:
                continue
                
            filtered_results.append(result)
        
        # 2. Agrupar por tipo de informação
        categorized_content = {
            "solicitantes": [],
            "prazos": [],
            "documentos": [], 
            "procedimentos": [],
            "outros": []
        }
        
        question_lower = question.lower()
        
        for result in filtered_results:
            content = result.get("content", "")
            content_lower = content.lower()
            
            # Categorização baseada em palavras-chave
            if any(word in content_lower for word in ["titular", "procurador", "curador", "tutor", "solicitar"]):
                categorized_content["solicitantes"].append(result)
            elif any(word in content_lower for word in ["prazo", "tempo", "hora", "dias", "zendesk"]):
                categorized_content["prazos"].append(result)
            elif any(word in content_lower for word in ["documento", "certidão", "identificação", "foto"]):
                categorized_content["documentos"].append(result)
            elif any(word in content_lower for word in ["proceder", "procedimento", "processo", "como"]):
                categorized_content["procedimentos"].append(result)
            else:
                categorized_content["outros"].append(result)
        
        # 3. Construir contexto estruturado
        context_parts = []
        
        # Priorizar categoria mais relevante para a pergunta
        category_priority = ["outros"]  # Default
        
        if any(word in question_lower for word in ["quem", "pode", "solicitar"]):
            category_priority = ["solicitantes", "procedimentos", "outros", "documentos", "prazos"]
        elif any(word in question_lower for word in ["prazo", "tempo", "quanto tempo", "quando"]):
            category_priority = ["prazos", "procedimentos", "outros", "solicitantes", "documentos"]
        elif any(word in question_lower for word in ["documento", "apresentar", "necessário"]):
            category_priority = ["documentos", "procedimentos", "outros", "solicitantes", "prazos"]
        elif any(word in question_lower for word in ["como", "proceder", "procedimento"]):
            category_priority = ["procedimentos", "outros", "documentos", "prazos", "solicitantes"]
        
        # 4. Construir contexto seguindo prioridade
        used_content = set()
        total_chars = 0
        max_chars = 2000  # Limitar tamanho do contexto
        
        for category in category_priority:
            if total_chars >= max_chars:
                break
                
            category_results = categorized_content[category]
            # Ordenar por score dentro da categoria
            category_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            for i, result in enumerate(category_results):
                content = result.get("content", "")
                content_key = content[:50]  # Chave para deduplicação
                
                if content_key in used_content or total_chars + len(content) > max_chars:
                    continue
                
                used_content.add(content_key)
                filename = result.get("metadata", {}).get("filename", "ICATU")
                score = result.get("score", 0)
                
                # Formatação inteligente baseada na categoria
                if category == "prazos":
                    prefix = "⏰ PRAZO"
                elif category == "documentos":
                    prefix = "📄 DOCUMENTAÇÃO"
                elif category == "solicitantes":
                    prefix = "👥 SOLICITANTES"
                elif category == "procedimentos":
                    prefix = "📋 PROCEDIMENTO"
                else:
                    prefix = "📌 INFORMAÇÃO"
                
                context_part = f"{prefix} (Relevância: {score:.1f}):\n{content}"
                context_parts.append(context_part)
                total_chars += len(content)
                
                # Limitar número de itens por categoria
                if len([p for p in context_parts if prefix in p]) >= 3:
                    break
        
        final_context = "\n\n".join(context_parts)
        
        # 5. Adicionar metadados de contexto
        context_metadata = f"""
RESUMO DO CONTEXTO:
- Total de fontes: {len(context_parts)}
- Categorias cobertas: {[cat for cat in category_priority if categorized_content[cat]]}
- Foco principal: {category_priority[0]}

CONTEÚDO TÉCNICO:
{final_context}
"""
        
        return context_metadata.strip()

    async def generate_answer_enhanced(self, question: str, context: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        """Generate enhanced answer using Mistral with intelligent prompting"""
        try:
            # Advanced prompt engineering with CoT reasoning
            enhanced_prompt = f"""Você é um especialista em procedimentos ICATU Seguros com acesso a documentos técnicos.

METODOLOGIA DE ANÁLISE:
1. LEIA o contexto fornecido cuidadosamente
2. IDENTIFIQUE palavras-chave relevantes na pergunta  
3. BUSQUE informações específicas no contexto
4. CONFIRME se a informação está disponível
5. RESPONDA de forma precisa e completa

REGRAS CRÍTICAS:
- Use APENAS informações do contexto fornecido
- Se não encontrar a informação: "A informação solicitada não está disponível nos documentos fornecidos."
- Mantenha terminologia técnica ICATU
- Seja direto e objetivo
- Cite prazos e procedimentos exatos quando disponíveis

CONTEXTO TÉCNICO ICATU:
{context}

PERGUNTA ESPECÍFICA: {question}

RESPOSTA FINAL (baseada exclusivamente no contexto):"""
            
            mistral_payload = {
                "question": question,
                "context": enhanced_prompt,
                "max_tokens": max_tokens,
                "temperature": min(temperature, 0.2),  # Temperatura baixa para precisão máxima
                "instructions": "Responda de forma precisa baseado exclusivamente no contexto fornecido."
            }
            
            response = await self.http_client.post(
                f"{MISTRAL_SERVICE_URL}/query",
                json=mistral_payload
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Mistral service error: {response.text}")
            
            result = response.json()
            return result.get("answer", "")
            
        except Exception as e:
            rag_logger.error(f"Error generating answer: {e}")
            raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")

    async def generate_answer(self, question: str, context: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        """Generate answer using the enhanced method - compatibility wrapper"""
        return await self.generate_answer_enhanced(question, context, max_tokens, temperature)
    
    async def process_rag_query(self, request: RAGRequest) -> RAGResponse:
        """Process a complete RAG query, optionally filtering by document_id"""
        start_time = time.time()
        # Step 1: Search for relevant documents
        search_start = time.time()
        search_results = await self.search_documents(
            query=request.question,
            limit=min(request.search_limit, 8),
            score_threshold=max(request.score_threshold, 0.2),  # Threshold mais baixo
            document_id=request.document_id
        )
        search_time = time.time() - search_start
        rag_logger.info(f"Found {len(search_results)} relevant documents in {search_time:.2f}s")
        
        # Step 2: Build context
        context = self.build_context_intelligent(search_results, request.question)
        
        # Log the context length for debugging
        rag_logger.debug(f"Built context with {len(context)} characters")
        
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
            rag_logger.warning("No relevant documents found, answering without context")
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
                rag_logger.info("✅ Mistral service is accessible")
            else:
                rag_logger.warning("⚠️ Mistral service not accessible")
            
            # Test Document processor
            doc_response = await client.get(f"{DOCUMENT_PROCESSOR_URL}/health")
            if doc_response.status_code == 200:
                rag_logger.info("✅ Document processor is accessible")
            else:
                rag_logger.warning("⚠️ Document processor not accessible")
                
    except Exception as e:
        rag_logger.error(f"Service connection test failed: {e}")
    
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
        rag_logger.info(f"Recebida pergunta: {request.question}")
        response = await rag_service.process_rag_query(request)
        rag_logger.info(f"Resposta gerada para pergunta: {request.question}")
        return response
    except Exception as e:
        rag_logger.error(f"RAG query failed: {e}")
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
        rag_logger.error(f"Error checking services: {e}")
    
    return status

if __name__ == "__main__":
    uvicorn.run(
        "rag_service:app",
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
