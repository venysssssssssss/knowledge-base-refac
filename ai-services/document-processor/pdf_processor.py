"""
PDF Document Processor with Dolphin (ByteDance) for parsing and Sentence Transformers for embeddings
"""

import asyncio
import logging
import os
import hashlib
import httpx
import logging
import httpx
import numpy as np
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

# SOLID principles applied
# SRP: Each class/function has a single responsibility
# OCP: Classes open for extension, closed for modification
# LSP: Subclasses can replace superclasses
# ISP: Specific interfaces for each operation
# DIP: Depend on abstractions

class DocumentProcessorLogger:
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

# Example usage of the logger
doc_logger = DocumentProcessorLogger(__name__)
doc_logger.info('Document processor started.')
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

import PyPDF2
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn
import logging
from pdfminer.high_level import extract_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_URL = "http://qdrant:6333"
COLLECTION_NAME = "knowledge_base_v2"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Modelo robusto para contexto longo e semântica
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = {}

class EmbeddingService:
    """Service for generating embeddings using SentenceTransformers"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 768  # all-MiniLM-L6-v2 dimension
        
    async def initialize(self):
        """Initialize the embedding model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Embedding model {self.model_name} loaded successfully")
        except Exception as e:
            logger.critical(f"❌ Failed to load embedding model: {e}. Embeddings will NOT be generated. Aborting.")
            raise RuntimeError(f"Failed to load embedding model: {e}")
            
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not self.model:
            logger.critical("❌ Embedding model is not loaded. Aborting embedding generation.")
            raise RuntimeError("Embedding model is not loaded.")
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            # Convert tensor to list if needed
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().numpy()
            
            # Ensure embeddings are valid
            if isinstance(embeddings, np.ndarray):
                # Convert numpy array to list of lists
                embeddings_list = embeddings.tolist()
            else:
                # If already list-like
                embeddings_list = embeddings
            
            # Validate each embedding
            for i, emb in enumerate(embeddings_list):
                if not isinstance(emb, (list, tuple)) or len(emb) != self.embedding_dim:
                    logger.warning(f"⚠️ Fixing invalid embedding at index {i}")
                    # Create a fallback embedding if invalid
                    embeddings_list[i] = self._fallback_embedding(texts[i])
            
            return embeddings_list
        except Exception as e:
            logger.critical(f"❌ SentenceTransformer encoding failed: {e}. Using fallback.")
            # Use fallback embeddings for all texts
            return [self._fallback_embedding(text) for text in texts]
    
    def _fallback_embedding(self, text: str) -> List[float]:
        """Fallback method to generate simple embeddings using text hashing"""
        import hashlib
        
        # Use multiple hash functions for more robust embedding
        hashes = [
            hashlib.md5(text.encode()).hexdigest(),
            hashlib.sha1(text.encode()).hexdigest()[:32],
            hashlib.sha256(text.encode()).hexdigest()[:32]
        ]
        
        embedding = []
        for hash_str in hashes:
            for i in range(0, len(hash_str), 2):
                hex_pair = hash_str[i:i+2]
                embedding.append(int(hex_pair, 16) / 255.0)
        
        # Ensure fixed size
        if len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]
        elif len(embedding) < self.embedding_dim:
            embedding.extend([0.0] * (self.embedding_dim - len(embedding)))
            
        return embedding

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
        self.embedding_service = EmbeddingService()
        # Add compatibility mode for older Qdrant server versions
        self.qdrant = QdrantClient(url=QDRANT_URL, check_compatibility=False)
        
    async def initialize(self):
        """Initialize Qdrant client and embedding model"""
        # Initialize embedding model first
        logger.info(f"Initializing SentenceTransformer embeddings: {EMBEDDING_MODEL}")
        self.embedding_model = EmbeddingService(EMBEDDING_MODEL)
        await self.embedding_model.initialize()
        
        # Then initialize Qdrant client
        try:
            self.qdrant_client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
            await self.ensure_collection_exists()
            logger.info("✅ Qdrant client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    
    async def ensure_collection_exists(self):
        """Create Qdrant collection if it doesn't exist, using the correct embedding dimension from the model."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if COLLECTION_NAME not in collection_names:
                logger.info(f"Creating collection: {COLLECTION_NAME}")
                # Descobre a dimensão do modelo carregado
                embedding_dim = self.embedding_model.embedding_dim if hasattr(self.embedding_model, 'embedding_dim') else 768
                self.qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=embedding_dim,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"✅ Collection {COLLECTION_NAME} created with dim {embedding_dim}")
            else:
                # Verifica se a dimensão está correta
                info = self.qdrant_client.get_collection(COLLECTION_NAME)
                current_dim = info.config.params.vectors.size
                expected_dim = self.embedding_model.embedding_dim if hasattr(self.embedding_model, 'embedding_dim') else 768
                if current_dim != expected_dim:
                    logger.error(f"❌ Dimensão da coleção Qdrant ({current_dim}) não corresponde ao modelo ({expected_dim}). Exclua a coleção manualmente ou altere o nome da coleção.")
                    raise Exception(f"Dimensão da coleção Qdrant ({current_dim}) não corresponde ao modelo ({expected_dim}). Exclua a coleção manualmente ou altere o nome da coleção.")
                logger.info(f"✅ Collection {COLLECTION_NAME} already exists with correct dim {current_dim}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    

    def extract_text_with_docling(self, pdf_file: bytes) -> Optional[str]:
        """Extract text from PDF using Docling"""
        try:
            import io
            from docling import Document
            pdf_stream = io.BytesIO(pdf_file)
            doc = Document.from_pdf(pdf_stream)
            texts = [page.text for page in doc.pages]
            logger.info(f"Docling extraiu {len(texts)} páginas.")
            return "\n".join(texts)
        except Exception as e:
            logger.warning(f"Docling parsing failed: {e}")
            return None

    def extract_text_from_pdf(self, pdf_file: bytes) -> str:
        """Extract text from PDF file, prefer Docling, fallback PyPDF2/pdfminer"""
        docling_text = self.extract_text_with_docling(pdf_file)
        if docling_text and len(docling_text.strip()) > 50:
            logger.info("✅ Docling parsing used for PDF")
            return docling_text
        # Fallback para PyPDF2
        try:
            import io
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if len(text.strip()) > 50:
                logger.info("✅ PyPDF2 parsing used for PDF")
                return text.strip()
        except Exception as e:
            logger.warning(f"PyPDF2 parsing failed: {e}")
        # Fallback para pdfminer
        try:
            from pdfminer.high_level import extract_text
            import io
            text = extract_text(io.BytesIO(pdf_file))
            logger.info("✅ pdfminer parsing used for PDF")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")
    
    def chunk_text(self, text: str, filename: str) -> List[DocumentChunk]:
        """
        Chunking inteligente: divide por seções e parágrafos, preservando contexto e metadados estruturados.
        - Detecta títulos/seções por heurística (ex: linhas em maiúsculo, numeradas, etc)
        - Cada chunk = parágrafo, com metadados de seção/título
        - Se parágrafo for muito longo, divide por frases, mas preserva início/fim
        """
        import re
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        # Heurística para seções: linhas em maiúsculo, numeradas, ou com ':'
        section_pattern = re.compile(r'^(\d+\.|[A-Z][A-Z\s\d\-]+:?)$')
        lines = text.splitlines()
        chunks = []
        current_section = ""
        section_index = 0
        paragraph_index = 0
        chunk_index = 0
        paragraph_buffer = []
        for line in lines:
            line = line.strip()
            if not line:
                # Fim de parágrafo
                if paragraph_buffer:
                    paragraph = " ".join(paragraph_buffer).strip()
                    tokens = len(tokenizer.encode(paragraph, add_special_tokens=False))
                    # Se parágrafo for muito longo, divide por frases
                    if tokens > CHUNK_SIZE:
                        sentences = re.split(r'(?<=[.!?]) +', paragraph)
                        sub_chunk = ""
                        sub_tokens = 0
                        sub_index = 0
                        for sent in sentences:
                            sent = sent.strip()
                            if not sent:
                                continue
                            sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
                            if sub_tokens + sent_tokens > CHUNK_SIZE:
                                if len(sub_chunk.strip()) > 30:
                                    chunk = DocumentChunk(
                                        chunk_id=str(uuid.uuid4()),
                                        text=sub_chunk.strip(),
                                        embedding=[],
                                        metadata={
                                            "filename": filename,
                                            "section": current_section,
                                            "section_index": section_index,
                                            "paragraph_index": paragraph_index,
                                            "sub_chunk_index": sub_index,
                                            "chunk_index": chunk_index,
                                            "source": "pdf"
                                        }
                                    )
                                    chunks.append(chunk)
                                    chunk_index += 1
                                    sub_index += 1
                                sub_chunk = sent
                                sub_tokens = sent_tokens
                            else:
                                if sub_chunk:
                                    sub_chunk += " " + sent
                                else:
                                    sub_chunk = sent
                                sub_tokens += sent_tokens
                        if len(sub_chunk.strip()) > 30:
                            chunk = DocumentChunk(
                                chunk_id=str(uuid.uuid4()),
                                text=sub_chunk.strip(),
                                embedding=[],
                                metadata={
                                    "filename": filename,
                                    "section": current_section,
                                    "section_index": section_index,
                                    "paragraph_index": paragraph_index,
                                    "sub_chunk_index": sub_index,
                                    "chunk_index": chunk_index,
                                    "source": "pdf"
                                }
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                    else:
                        if len(paragraph.strip()) > 30:
                            chunk = DocumentChunk(
                                chunk_id=str(uuid.uuid4()),
                                text=paragraph.strip(),
                                embedding=[],
                                metadata={
                                    "filename": filename,
                                    "section": current_section,
                                    "section_index": section_index,
                                    "paragraph_index": paragraph_index,
                                    "chunk_index": chunk_index,
                                    "source": "pdf"
                                }
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                    paragraph_index += 1
                    paragraph_buffer = []
                continue
            # Detecta se é título/seção
            if section_pattern.match(line):
                current_section = line
                section_index += 1
                paragraph_index = 0
                continue
            paragraph_buffer.append(line)
        # Último parágrafo
        if paragraph_buffer:
            paragraph = " ".join(paragraph_buffer).strip()
            tokens = len(tokenizer.encode(paragraph, add_special_tokens=False))
            if tokens > CHUNK_SIZE:
                sentences = re.split(r'(?<=[.!?]) +', paragraph)
                sub_chunk = ""
                sub_tokens = 0
                sub_index = 0
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
                    if sub_tokens + sent_tokens > CHUNK_SIZE:
                        if len(sub_chunk.strip()) > 30:
                            chunk = DocumentChunk(
                                chunk_id=str(uuid.uuid4()),
                                text=sub_chunk.strip(),
                                embedding=[],
                                metadata={
                                    "filename": filename,
                                    "section": current_section,
                                    "section_index": section_index,
                                    "paragraph_index": paragraph_index,
                                    "sub_chunk_index": sub_index,
                                    "chunk_index": chunk_index,
                                    "source": "pdf"
                                }
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                            sub_index += 1
                        sub_chunk = sent
                        sub_tokens = sent_tokens
                    else:
                        if sub_chunk:
                            sub_chunk += " " + sent
                        else:
                            sub_chunk = sent
                        sub_tokens += sent_tokens
                if len(sub_chunk.strip()) > 30:
                    chunk = DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        text=sub_chunk.strip(),
                        embedding=[],
                        metadata={
                            "filename": filename,
                            "section": current_section,
                            "section_index": section_index,
                            "paragraph_index": paragraph_index,
                            "sub_chunk_index": sub_index,
                            "chunk_index": chunk_index,
                            "source": "pdf"
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            else:
                if len(paragraph.strip()) > 30:
                    chunk = DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        text=paragraph.strip(),
                        embedding=[],
                        metadata={
                            "filename": filename,
                            "section": current_section,
                            "section_index": section_index,
                            "paragraph_index": paragraph_index,
                            "chunk_index": chunk_index,
                            "source": "pdf"
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
        logger.info(f"Chunking inteligente gerou {len(chunks)} chunks para {filename}")
        return chunks
    
    async def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks using SentenceTransformers"""
        logger.info(f"Gerando embeddings para {len(chunks)} chunks...")
        embeddings = await self.embedding_model.embed_texts(chunks)
        logger.info(f"Embeddings gerados com sucesso.")
        return embeddings
    
    async def store_in_qdrant(self, chunks: List[DocumentChunk], embeddings: List[List[float]], document_id: str):
        """Store chunks and embeddings in Qdrant, validando embeddings"""
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not isinstance(embedding, (list, tuple)) or len(embedding) != self.embedding_model.embedding_dim:
                logger.critical(f"❌ Embedding inválido para chunk {chunk.chunk_id}: {embedding}")
                raise ValueError(f"Embedding inválido para chunk {chunk.chunk_id}")
            point = models.PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload={
                    "content": chunk.text,
                    "document_id": document_id,
                    **chunk.metadata
                }
            )
            points.append(point)
        self.qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        logger.info(f"✅ Stored {len(points)} chunks in Qdrant for document {document_id}. IDs: {[c.chunk_id for c in chunks]}")
    
    async def search_similar_chunks(self, query: str, limit: int = 5, score_threshold: float = 0.7, document_id: str = None) -> List[SearchResult]:
        """Search for similar chunks in Qdrant, retorna chunks ordenados por relevância e contexto concatenado"""
        logger.info(f"Gerando embedding da pergunta...")
        query_embeddings = await self.embedding_model.embed_texts([query])
        query_embedding = query_embeddings[0]
        logger.info(f"Buscando chunks mais relevantes no Qdrant...")
        
        # Filtros para busca
        search_params = {
            "collection_name": COLLECTION_NAME,
            "query_vector": query_embedding,
            "limit": limit,
            "score_threshold": score_threshold
        }
        
        # Adicionar filtro por document_id se fornecido
        if document_id:
            search_params["query_filter"] = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            )
        
        search_results = self.qdrant_client.search(**search_params)
        
        results = []
        for result in search_results:
            search_result = SearchResult(
                content=result.payload["content"],
                score=result.score,
                metadata={k: v for k, v in result.payload.items() if k != "content"}
            )
            results.append(search_result)
        logger.info(f"Busca retornou {len(results)} chunks relevantes.")
        
        # Ordenar resultados por score
        results_sorted = sorted(results, key=lambda r: r.score, reverse=True)
        
        # Simplifique a busca de parágrafos vizinhos para evitar problemas de versão do Qdrant
        # Em vez de usar filtros complexos, vamos buscar novamente com a mesma query
        # e usar apenas o document_id como filtro, depois filtrar os vizinhos em memória
        if results_sorted and len(results_sorted) > 0:
            top_result = results_sorted[0]
            doc_id = top_result.metadata.get("document_id")
            
            if doc_id:
                try:
                    # Busque todos os chunks do mesmo documento - uma abordagem mais simples
                    # que evita o uso de min_should que está causando o erro
                    doc_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id", 
                                match=models.MatchValue(value=doc_id)
                            )
                        ]
                    )
                    
                    all_doc_chunks = self.qdrant_client.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=query_embedding,
                        query_filter=doc_filter,
                        limit=10  # Buscar mais chunks para ter vizinhos
                    )
                    
                    # Filtrar manualmente os vizinhos que já não estão em results_sorted
                    existing_ids = set(r.metadata.get("chunk_id", "") for r in results_sorted)
                    for chunk in all_doc_chunks:
                        chunk_id = chunk.payload.get("chunk_id", "")
                        if chunk_id not in existing_ids:
                            neighbor_result = SearchResult(
                                content=chunk.payload["content"],
                                score=chunk.score * 0.95,  # Ligeiramente menor relevância
                                metadata={k: v for k, v in chunk.payload.items() if k != "content"}
                            )
                            results_sorted.append(neighbor_result)
                            # Adicione apenas 2 vizinhos no máximo
                            if len(results_sorted) >= len(search_results) + 2:
                                break
                    
                    # Reordenar após adicionar os vizinhos
                    results_sorted = sorted(results_sorted, key=lambda r: r.score, reverse=True)
                except Exception as e:
                    # Se a busca por vizinhos falhar, continue com os resultados originais
                    logger.warning(f"Failed to fetch neighbor chunks: {e}")
        
        return results_sorted

# Global processor instance
processor = DocumentProcessor()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await processor.initialize()
    yield
    # Shutdown - cleanup if needed

# FastAPI app
app = FastAPI(
    title="Document Processor API",
    description="PDF processing and embedding generation for knowledge base",
    version="1.0.0",
    lifespan=lifespan
)

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
        embeddings = await processor.generate_embeddings([chunk.text for chunk in chunks])
        
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

from fastapi.responses import JSONResponse

@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """Search for relevant document chunks, retorna também contexto concatenado para o Mistral"""
    try:
        results = await processor.search_similar_chunks(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        contexto_completo = "\n".join([r.content for r in results])
        return JSONResponse(content={
            "chunks": [r.dict() for r in results],
            "contexto_completo": contexto_completo
        })
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
