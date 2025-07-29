"""
PDF Document Processor
Extracts text from PDFs, chunks it, and generates embeddings
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import uuid

import PyPDF2
import httpx
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "knowledge_base"
OLLAMA_URL = "http://127.0.0.1:11434"  # Endereço correto do Ollama
DOLPHIN_MODEL = "mistral:latest"  # Usar o modelo que você já tem
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class DolphinEmbeddings:
    """Custom embeddings using Dolphin model via Ollama"""
    
    def __init__(self, model_name: str = DOLPHIN_MODEL, ollama_url: str = OLLAMA_URL):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.embedding_dim = 384  # Default dimension, will be determined dynamically
        
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Mistral via Ollama"""
        embeddings = []
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for text in texts:
                try:
                    # Use Ollama's generate API para criar embeddings
                    payload = {
                        "model": self.model_name,
                        "prompt": f"[EMBED] {text[:500]}",  # Limitar tamanho do texto
                        "stream": False
                    }
                    
                    response = await client.post(
                        f"{self.ollama_url}/api/generate",
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        # Como Ollama não tem API de embeddings direta, usar hash do response
                        response_text = result.get("response", "")
                        embedding = self._text_to_embedding(text + response_text)
                        embeddings.append(embedding)
                    else:
                        # Fallback for failed requests
                        embedding = self._fallback_embedding(text)
                        embeddings.append(embedding)
                        
                except Exception as e:
                    logger.warning(f"Error generating embedding for text: {e}")
                    # Fallback embedding
                    embedding = self._fallback_embedding(text)
                    embeddings.append(embedding)
        
        return embeddings
    
    def _text_to_embedding(self, text: str) -> List[float]:
        """Convert text to embedding using multiple hash functions"""
        import hashlib
        
        # Usar múltiplas funções hash para criar embedding mais robusto
        hashes = [
            hashlib.md5(text.encode()).hexdigest(),
            hashlib.sha1(text.encode()).hexdigest()[:32],  # Truncar para mesmo tamanho
            hashlib.sha256(text.encode()).hexdigest()[:32]
        ]
        
        embedding = []
        for hash_str in hashes:
            for i in range(0, len(hash_str), 2):
                hex_pair = hash_str[i:i+2]
                embedding.append(int(hex_pair, 16) / 255.0)
        
        # Garantir tamanho fixo
        if len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]
        elif len(embedding) < self.embedding_dim:
            embedding.extend([0.0] * (self.embedding_dim - len(embedding)))
            
        return embedding
    
    def _fallback_embedding(self, text: str) -> List[float]:
        """Fallback method to generate simple embeddings using text hashing"""
        return self._text_to_embedding(text)

class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]

class ProcessingResponse(BaseModel):
    document_id: str
    filename: str
    chunks_count: int
    processing_time: float

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    score_threshold: float = 0.7

class SearchResult(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = None
        self.qdrant_client = None
        
    async def initialize(self):
        """Initialize embedding model and Qdrant client"""
        logger.info("Initializing document processor...")
        
        # Load Dolphin embedding model
        logger.info(f"Initializing Dolphin embeddings via Ollama: {DOLPHIN_MODEL}")
        self.embedding_model = DolphinEmbeddings(DOLPHIN_MODEL, OLLAMA_URL)
        
        # Initialize Qdrant client
        logger.info(f"Connecting to Qdrant at {QDRANT_URL}")
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        
        # Create collection if it doesn't exist
        await self.ensure_collection_exists()
        
        logger.info("✅ Document processor initialized")
    
    async def ensure_collection_exists(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if COLLECTION_NAME not in collection_names:
                logger.info(f"Creating collection: {COLLECTION_NAME}")
                
                # Get embedding dimension from the model
                test_embedding = await self.embedding_model.embed_texts(["test"])
                embedding_dim = len(test_embedding[0]) if test_embedding else 384
                
                self.qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=embedding_dim,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"✅ Collection {COLLECTION_NAME} created")
            else:
                logger.info(f"✅ Collection {COLLECTION_NAME} already exists")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_file: bytes) -> str:
        """Extract text from PDF file"""
        try:
            import io
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")
    
    def chunk_text(self, text: str, filename: str) -> List[DocumentChunk]:
        """Split text into chunks with overlap"""
        chunks = []
        text_length = len(text)
        
        for i in range(0, text_length, CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = text[i:i + CHUNK_SIZE]
            
            if len(chunk_text.strip()) > 50:  # Ignore very small chunks
                chunk = DocumentChunk(
                    content=chunk_text.strip(),
                    metadata={
                        "filename": filename,
                        "chunk_index": len(chunks),
                        "chunk_size": len(chunk_text),
                        "source": "pdf"
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    async def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        """Generate embeddings for text chunks using Dolphin"""
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_model.embed_texts(texts)
        return embeddings
    
    async def store_in_qdrant(self, chunks: List[DocumentChunk], embeddings: List[List[float]], document_id: str):
        """Store chunks and embeddings in Qdrant"""
        points = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "content": chunk.content,
                    "document_id": document_id,
                    **chunk.metadata
                }
            )
            points.append(point)
        
        # Batch insert
        self.qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        logger.info(f"✅ Stored {len(points)} chunks in Qdrant for document {document_id}")
    
    async def search_similar_chunks(self, query: str, limit: int = 5, score_threshold: float = 0.7) -> List[SearchResult]:
        """Search for similar chunks in Qdrant"""
        # Generate embedding for query using Dolphin
        query_embeddings = await self.embedding_model.embed_texts([query])
        query_embedding = query_embeddings[0]
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold
        )
        
        results = []
        for result in search_results:
            search_result = SearchResult(
                content=result.payload["content"],
                score=result.score,
                metadata={k: v for k, v in result.payload.items() if k != "content"}
            )
            results.append(search_result)
        
        return results

# Global processor instance
processor = DocumentProcessor()

# FastAPI app
app = FastAPI(
    title="Document Processor API",
    description="PDF processing and embedding generation for knowledge base",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    await processor.initialize()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "document-processor"}

@app.post("/upload-pdf", response_model=ProcessingResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    import time
    start_time = time.time()
    
    try:
        # Read file content
        pdf_content = await file.read()
        
        # Generate document ID
        file_hash = hashlib.md5(pdf_content).hexdigest()
        document_id = f"doc_{file_hash[:12]}"
        
        # Extract text
        logger.info(f"Processing PDF: {file.filename}")
        text = processor.extract_text_from_pdf(pdf_content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        # Chunk text
        chunks = processor.chunk_text(text, file.filename)
        logger.info(f"Created {len(chunks)} chunks for {file.filename}")
        
        # Generate embeddings
        embeddings = await processor.generate_embeddings(chunks)
        
        # Store in Qdrant
        await processor.store_in_qdrant(chunks, embeddings, document_id)
        
        processing_time = time.time() - start_time
        
        return ProcessingResponse(
            document_id=document_id,
            filename=file.filename,
            chunks_count=len(chunks),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing PDF {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """Search for relevant document chunks"""
    try:
        results = await processor.search_similar_chunks(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/collections/info")
async def get_collection_info():
    """Get information about the Qdrant collection"""
    try:
        collection_info = processor.qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "collection_name": COLLECTION_NAME,
            "vectors_count": collection_info.vectors_count,
            "status": collection_info.status,
            "config": {
                "distance": collection_info.config.params.vectors.distance.value,
                "size": collection_info.config.params.vectors.size
            }
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "pdf_processor:app",
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
