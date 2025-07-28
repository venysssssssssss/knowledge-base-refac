"""
Mistral 7B Service with vLLM
Optimized for knowledge base Q&A
"""

import asyncio
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
engine = None

class QueryRequest(BaseModel):
    question: str
    context: str = ""
    max_tokens: int = 512
    temperature: float = 0.7

class QueryResponse(BaseModel):
    answer: str
    tokens_used: int
    processing_time: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the vLLM engine"""
    global engine
    
    # Startup
    logger.info("Initializing Mistral 7B model...")
    
    # Configure engine arguments
    engine_args = AsyncEngineArgs(
        model="models/mistral-7b-instruct-v0.2",  # Update path as needed
        # quantization="awq",  # Disabled for base model
        gpu_memory_utilization=0.85,  # Reduced for unquantized model
        max_model_len=4096,
        enable_chunked_prefill=True,
        max_num_seqs=20,  # Reduced for unquantized model
        tensor_parallel_size=1,  # Single GPU
        trust_remote_code=True
    )
    
    try:
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    if engine:
        logger.info("Shutting down model engine...")
        # Cleanup if needed

app = FastAPI(
    title="Mistral 7B Knowledge Base API",
    description="AI service for document Q&A using Mistral 7B",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "mistral-7b-instruct"}

@app.post("/query", response_model=QueryResponse)
async def query_model(request: QueryRequest):
    """Generate answer based on question and context"""
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    # Format prompt for Mistral
    prompt = format_mistral_prompt(request.question, request.context)
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stop=["</s>", "[/INST]"],
        top_p=0.9
    )
    
    try:
        # Generate response
        results = await engine.generate(prompt, sampling_params, request_id=None)
        
        if not results:
            raise HTTPException(status_code=500, detail="No response generated")
        
        result = results[0]
        answer = result.outputs[0].text.strip()
        
        processing_time = time.time() - start_time
        tokens_used = len(result.outputs[0].token_ids)
        
        return QueryResponse(
            answer=answer,
            tokens_used=tokens_used,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

def format_mistral_prompt(question: str, context: str = "") -> str:
    """Format prompt for Mistral Instruct model"""
    if context:
        prompt = f"""<s>[INST] Com base no contexto fornecido, responda à pergunta de forma clara e precisa.

Contexto:
{context}

Pergunta: {question}

Responda apenas com base nas informações do contexto fornecido. Se a informação não estiver disponível no contexto, diga que não tem informação suficiente. [/INST]"""
    else:
        prompt = f"<s>[INST] {question} [/INST]"
    
    return prompt

@app.get("/stats")
async def get_stats():
    """Get model statistics"""
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get engine stats (implement based on vLLM version)
    return {
        "model": "mistral-7b-instruct",
        "status": "running",
        "max_sequences": 20,
        "gpu_memory_utilization": 0.85
    }

if __name__ == "__main__":
    uvicorn.run(
        "mistral_service:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
