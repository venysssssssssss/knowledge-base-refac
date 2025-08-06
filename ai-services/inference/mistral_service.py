"""
Mistral 7B Service with Ollama
Optimized for knowledge base Q&A
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configuração básica de logging estruturado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)

# Princípios SOLID aplicados
# SRP: Cada classe/função tem responsabilidade única
# OCP: Classes abertas para extensão, fechadas para modificação
# LSP: Subclasses podem substituir superclasses
# ISP: Interfaces específicas para cada operação
# DIP: Dependa de abstrações


class InferenceLogger:
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
inference_logger = InferenceLogger(__name__)
inference_logger.info('Serviço de inferência iniciado.')

# Ollama configuration
OLLAMA_BASE_URL = 'http://10.117.0.19:11434'  # Endereço correto do Ollama
MODEL_NAME = 'mistral:latest'


class QueryRequest(BaseModel):
    question: str
    context: str = ''
    max_tokens: int = 512
    temperature: float = 0.7
    instructions: str = ''  # Novo campo para instruções específicas


class QueryResponse(BaseModel):
    answer: str
    tokens_used: int
    processing_time: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup"""
    # Startup
    inference_logger.info('Starting Mistral 7B service with Ollama...')

    # Test Ollama connection
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f'{OLLAMA_BASE_URL}/api/tags')
            if response.status_code == 200:
                models = response.json()
                model_names = [
                    model['name'] for model in models.get('models', [])
                ]
                if MODEL_NAME in model_names:
                    inference_logger.info(
                        f'✅ {MODEL_NAME} is available in Ollama'
                    )
                else:
                    inference_logger.info(
                        f'⚠️ {MODEL_NAME} not found. Available models: {model_names}'
                    )
            else:
                inference_logger.error('❌ Failed to connect to Ollama')
    except Exception as e:
        inference_logger.error(f'❌ Ollama connection error: {e}')

    yield

    # Shutdown
    inference_logger.info('Shutting down Mistral service...')


app = FastAPI(
    title='Mistral 7B Knowledge Base API',
    description='AI service for document Q&A using Mistral 7B',
    version='1.0.0',
    lifespan=lifespan,
)


@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return {'status': 'healthy', 'model': 'mistral-7b-instruct'}


@app.post('/query', response_model=QueryResponse)
async def query_model(request: QueryRequest):
    """
    Gera resposta baseada na pergunta e nos chunks de contexto vindos do Qdrant.
    O modelo Mistral 7B responderá EXCLUSIVAMENTE com base no contexto fornecido.
    """
    start_time = time.time()
    # Formata o prompt usando os chunks e instruções
    prompt = format_mistral_prompt(
        request.question, [request.context], request.instructions
    )
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                'model': MODEL_NAME,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': request.temperature,
                    'num_predict': request.max_tokens,
                    'top_p': 0.9,
                },
            }
            response = await client.post(
                f'{OLLAMA_BASE_URL}/api/generate', json=payload
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500, detail=f'Ollama error: {response.text}'
                )
            result = response.json()
            answer = result.get('response', '').strip()
            if not answer:
                raise HTTPException(
                    status_code=500, detail='No response generated'
                )
            processing_time = time.time() - start_time
            tokens_used = len(answer.split()) * 1.3  # Estimativa simples
            return QueryResponse(
                answer=answer,
                tokens_used=int(tokens_used),
                processing_time=processing_time,
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail='Request timeout')
    except Exception as e:
        inference_logger.error(f'Generation error: {e}')
        raise HTTPException(
            status_code=500, detail=f'Generation failed: {str(e)}'
        )


def format_mistral_prompt(
    question: str, context_chunks: List[str] = None, instructions: str = ''
) -> str:
    """
    Formata o prompt para o Mistral Instruct, usando os chunks do Qdrant como contexto.
    Inclui instrução explícita para responder SOMENTE com base no contexto fornecido.
    """
    context_chunks = context_chunks or []
    default_instructions = """
    Você é um assistente de documentos preciso e confiável. 
    Responda à pergunta EXCLUSIVAMENTE com base no contexto fornecido abaixo.
    Não invente informações, não extrapole, não utilize conhecimento externo.
    Se a resposta não estiver claramente no contexto, diga: "A informação solicitada não está disponível nos documentos fornecidos."
    """

    # Use instruções personalizadas se fornecidas
    final_instructions = instructions if instructions else default_instructions

    if context_chunks:
        context = '\n'.join(context_chunks)
        prompt = f"""<s>[INST] {final_instructions}\n\nContexto:\n{context}\n\nPergunta: {question}\n\nResponda apenas com base nas informações do contexto fornecido. [/INST]"""
    else:
        prompt = f'<s>[INST] {question} [/INST]'
    return prompt


@app.get('/stats')
async def get_stats():
    """Get model statistics"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f'{OLLAMA_BASE_URL}/api/tags')
            if response.status_code == 200:
                models = response.json()
                return {
                    'model': MODEL_NAME,
                    'status': 'running',
                    'available_models': [
                        model['name'] for model in models.get('models', [])
                    ],
                    'ollama_url': OLLAMA_BASE_URL,
                }
            else:
                return {
                    'model': MODEL_NAME,
                    'status': 'error',
                    'error': 'Cannot connect to Ollama',
                }
    except Exception as e:
        return {'model': MODEL_NAME, 'status': 'error', 'error': str(e)}


if __name__ == '__main__':
    uvicorn.run(
        'mistral_service:app',
        host='0.0.0.0',
        port=8003,  # Changed from 8003 to avoid conflict
        log_level='info',
    )
