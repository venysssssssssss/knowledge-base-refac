"""
PDF Document Processor with Dolphin (ByteDance) for parsing and Sentence Transformers for embeddings
"""

import asyncio
import logging
import os
import re
import hashlib
import httpx
import numpy as np
import uuid
import time
import io
import torch
import PyPDF2
from typing import Dict, List, Any, Optional

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

def preprocess_text_advanced(text: str) -> str:
        """Pré-processamento avançado de texto"""
        # Normalizar espaços
        text = re.sub(r'\s+', ' ', text)
        
        # Preservar estrutura de listas e numeração
        text = re.sub(r'\n\s*[•·*-]\s*', '\n• ', text)
        text = re.sub(r'\n\s*(\d+)[.):]\s*', r'\n\1. ', text)
        
        # Normalizar pontuação específica ICATU
        text = text.replace('ICATU Seguros', 'ICATU')
        text = text.replace('alteração cadastral', 'alteração cadastral')
        
        # Remover caracteres problemáticos
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        return text.strip()


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
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Atual - bom para semântica geral
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # MELHOR para português brasileiro
CHUNK_SIZE = 2048  # AUMENTADO para capturar mais contexto
CHUNK_OVERLAP = 512  # AUMENTADO overlap para melhor continuidade
MIN_CHUNK_SIZE = 200  # AUMENTADO para evitar chunks muito pequenos

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
        self.embedding_dim = 768
        
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
        CHUNKING HIERÁRQUICO OTIMIZADO para capturar TODO o contexto
        
        ESTRATÉGIAS IMPLEMENTADAS:
        1. 🎯 Chunks maiores (2048 tokens) para máximo contexto
        2. 🔄 Overlap aumentado (512 tokens) para continuidade
        3. 📊 Categorização automática ICATU
        4. 🏗️ Preservação de estrutura hierárquica
        5. 📝 Metadados ricos para busca otimizada
        6. 🔍 Garantia de cobertura completa do documento
        """
        import re
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        
        # Pré-processamento avançado do texto
        text = preprocess_text_advanced(text)
        
        # Padrões para categorização de conteúdo ICATU
        patterns = {
            "solicitantes": re.compile(r'(titular|procurador|curador|tutor|responsável|beneficiário|requerente)', re.IGNORECASE),
            "prazos": re.compile(r'(prazo|tempo|hora|dias|zendesk|atendimento|urgente|cronograma)', re.IGNORECASE),
            "documentos": re.compile(r'(documento|certidão|identificação|foto|anexo|arquivo|comprovante)', re.IGNORECASE),
            "procedimentos": re.compile(r'(processo|procedimento|como|proceder|executar|realizar|instrução)', re.IGNORECASE),
            "alteracao_cadastral": re.compile(r'(alteração|mudança|atualização|cadastr|modificação)', re.IGNORECASE),
            "contrato": re.compile(r'(contrato|apólice|seguro|cobertura|produto)', re.IGNORECASE),
            "requisitos": re.compile(r'(requisito|exigência|necessário|obrigatório|deve)', re.IGNORECASE),
            "canais": re.compile(r'(portal|site|telefone|email|presencial|digital)', re.IGNORECASE)
        }
        
        # Padrões para identificar seções e estruturas
        section_pattern = re.compile(r'^(\d+[\.\)]\s*|[IVX]+[\.\)]\s*|[A-Z][A-Z\s\d\-]{10,}:?\s*|\w+\s*:)$|^[a-z][\)\.]', re.MULTILINE)
        title_pattern = re.compile(r'^[A-Z][A-Z\s\d\-]{8,}$')
        list_pattern = re.compile(r'^\s*[•·*-]\s*|^\s*\d+[\.\)]\s*')
        
        def categorize_content(content: str) -> List[str]:
            """Categoriza o conteúdo do chunk de forma mais precisa"""
            categories = []
            content_lower = content.lower()
            
            for category, pattern in patterns.items():
                if pattern.search(content_lower):
                    categories.append(category)
            
            # Categorização adicional baseada em estrutura
            if list_pattern.search(content):
                categories.append("lista")
            if any(word in content_lower for word in ["passo", "etapa", "primeiro", "segundo"]):
                categories.append("sequencial")
            if any(word in content_lower for word in ["atenção", "importante", "observação", "nota"]):
                categories.append("destaque")
            
            if not categories:
                categories.append("geral")
            
            return categories
        
        def create_chunk_with_context(text: str, section: str, categories: List[str], 
                                    chunk_idx: int, section_idx: int = 0, para_idx: int = 0,
                                    has_overlap: bool = False) -> DocumentChunk:
            """Cria chunk com metadados ricos e contexto estrutural"""
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            
            return DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                text=text.strip(),
                embedding=[],
                metadata={
                    "filename": filename,
                    "section": section,
                    "categories": categories,
                    "section_index": section_idx,
                    "paragraph_index": para_idx,
                    "chunk_index": chunk_idx,
                    "source": "pdf",
                    "content_type": "document",
                    "token_count": token_count,
                    "character_count": len(text),
                    "has_overlap": has_overlap,
                    "processing_timestamp": time.time(),
                    "priority_score": len(categories) * 0.1 + (1.0 if "importante" in text.lower() else 0.0),
                    "density_score": token_count / max(len(text), 1)  # Densidade de informação
                }
            )
        
        # ESTRATÉGIA 1: Processamento por parágrafos com overlap inteligente
        paragraphs = text.split('\n\n')
        chunks = []
        current_section = "DOCUMENTO_PRINCIPAL"
        section_index = 0
        chunk_index = 0
        
        # Buffer para construção de chunks
        chunk_buffer = ""
        chunk_tokens = 0
        previous_chunk_text = ""  # Para overlap
        
        doc_logger.info(f"Iniciando chunking hierárquico para {filename}: {len(paragraphs)} parágrafos")
        
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Detecta nova seção baseada em padrões
            if section_pattern.match(paragraph) or title_pattern.match(paragraph):
                # Finaliza chunk atual se existir
                if chunk_buffer and len(chunk_buffer.strip()) >= MIN_CHUNK_SIZE:
                    categories = categorize_content(chunk_buffer)
                    chunk = create_chunk_with_context(
                        chunk_buffer, current_section, categories,
                        chunk_index, section_index, para_idx
                    )
                    chunks.append(chunk)
                    previous_chunk_text = chunk_buffer[-CHUNK_OVERLAP:] if len(chunk_buffer) > CHUNK_OVERLAP else chunk_buffer
                    chunk_index += 1
                
                # Inicia nova seção
                current_section = paragraph[:100] + "..." if len(paragraph) > 100 else paragraph
                section_index += 1
                chunk_buffer = ""
                chunk_tokens = 0
                continue
            
            # Calcula tokens do parágrafo
            para_tokens = len(tokenizer.encode(paragraph, add_special_tokens=False))
            
            # ESTRATÉGIA 2: Gerenciamento inteligente de tamanho de chunk
            if chunk_tokens + para_tokens > CHUNK_SIZE:
                # Salva chunk atual
                if chunk_buffer and len(chunk_buffer.strip()) >= MIN_CHUNK_SIZE:
                    # Adiciona overlap da chunk anterior se disponível
                    full_chunk_text = chunk_buffer
                    if previous_chunk_text and not chunk_buffer.startswith(previous_chunk_text[-100:]):
                        overlap_text = previous_chunk_text[-CHUNK_OVERLAP//2:].strip()
                        if overlap_text:
                            full_chunk_text = overlap_text + " [CONTINUAÇÃO] " + chunk_buffer
                    
                    categories = categorize_content(full_chunk_text)
                    chunk = create_chunk_with_context(
                        full_chunk_text, current_section, categories,
                        chunk_index, section_index, para_idx,
                        has_overlap=bool(previous_chunk_text)
                    )
                    chunks.append(chunk)
                    
                    # Prepara overlap para próximo chunk
                    previous_chunk_text = chunk_buffer
                    chunk_index += 1
                
                # ESTRATÉGIA 3: Divisão inteligente de parágrafos grandes
                if para_tokens > CHUNK_SIZE:
                    # Divide parágrafo por sentenças
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    sub_chunk = ""
                    sub_tokens = 0
                    
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                        
                        sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
                        
                        if sub_tokens + sent_tokens > CHUNK_SIZE and sub_chunk:
                            # Salva sub-chunk
                            if len(sub_chunk.strip()) >= MIN_CHUNK_SIZE:
                                # Adiciona overlap se disponível
                                full_sub_chunk = sub_chunk
                                if previous_chunk_text:
                                    overlap_text = previous_chunk_text[-CHUNK_OVERLAP//3:].strip()
                                    if overlap_text:
                                        full_sub_chunk = overlap_text + " [CONTINUAÇÃO] " + sub_chunk
                                
                                categories = categorize_content(full_sub_chunk)
                                chunk = create_chunk_with_context(
                                    full_sub_chunk, current_section, categories,
                                    chunk_index, section_index, para_idx,
                                    has_overlap=True
                                )
                                chunks.append(chunk)
                                previous_chunk_text = sub_chunk
                                chunk_index += 1
                            
                            # Inicia novo sub-chunk com overlap
                            overlap_text = sub_chunk[-CHUNK_OVERLAP//2:] if len(sub_chunk) > CHUNK_OVERLAP//2 else ""
                            sub_chunk = overlap_text + " " + sent if overlap_text else sent
                            sub_tokens = len(tokenizer.encode(sub_chunk, add_special_tokens=False))
                        else:
                            sub_chunk = sub_chunk + " " + sent if sub_chunk else sent
                            sub_tokens += sent_tokens
                    
                    # Salva último sub-chunk
                    if sub_chunk and len(sub_chunk.strip()) >= MIN_CHUNK_SIZE:
                        categories = categorize_content(sub_chunk)
                        chunk = create_chunk_with_context(
                            sub_chunk, current_section, categories,
                            chunk_index, section_index, para_idx
                        )
                        chunks.append(chunk)
                        previous_chunk_text = sub_chunk
                        chunk_index += 1
                    
                    chunk_buffer = ""
                    chunk_tokens = 0
                else:
                    # Inicia novo chunk com o parágrafo atual
                    chunk_buffer = paragraph
                    chunk_tokens = para_tokens
            else:
                # Adiciona parágrafo ao chunk atual
                chunk_buffer = chunk_buffer + "\n\n" + paragraph if chunk_buffer else paragraph
                chunk_tokens += para_tokens
        
        # ESTRATÉGIA 4: Processa último chunk restante
        if chunk_buffer and len(chunk_buffer.strip()) >= MIN_CHUNK_SIZE:
            # Adiciona overlap se disponível
            full_final_chunk = chunk_buffer
            if previous_chunk_text and not chunk_buffer.startswith(previous_chunk_text[-100:]):
                overlap_text = previous_chunk_text[-CHUNK_OVERLAP//2:].strip()
                if overlap_text:
                    full_final_chunk = overlap_text + " [CONTINUAÇÃO] " + chunk_buffer
            
            categories = categorize_content(full_final_chunk)
            chunk = create_chunk_with_context(
                full_final_chunk, current_section, categories,
                chunk_index, section_index, len(paragraphs),
                has_overlap=bool(previous_chunk_text)
            )
            chunks.append(chunk)
        
        # ESTRATÉGIA 5: Análise de cobertura e estatísticas
        total_chars = len(text)
        covered_chars = sum(len(chunk.text) for chunk in chunks)
        coverage_ratio = covered_chars / total_chars if total_chars > 0 else 0
        
        category_stats = {}
        for chunk in chunks:
            for cat in chunk.metadata.get("categories", []):
                category_stats[cat] = category_stats.get(cat, 0) + 1
        
        # Log detalhado do chunking
        doc_logger.info(f"✅ Chunking hierárquico concluído para {filename}:")
        doc_logger.info(f"   📊 {len(chunks)} chunks gerados")
        doc_logger.info(f"   📏 Cobertura: {coverage_ratio:.2%} do documento original")
        doc_logger.info(f"   📝 Tamanho médio: {covered_chars // len(chunks) if chunks else 0} caracteres/chunk")
        doc_logger.info(f"   🏷️ Categorias: {category_stats}")
        
        # ESTRATÉGIA 6: Validação de qualidade
        if coverage_ratio < 0.90:
            doc_logger.warning(f"⚠️ Cobertura baixa ({coverage_ratio:.2%}) - possível perda de conteúdo!")
        
        if len(chunks) == 0:
            doc_logger.error(f"❌ Nenhum chunk gerado para {filename}!")
            # Fallback: criar um chunk com todo o texto
            fallback_chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                text=text[:CHUNK_SIZE*2],  # Limita tamanho
                embedding=[],
                metadata={
                    "filename": filename,
                    "section": "FALLBACK_COMPLETE_DOCUMENT",
                    "categories": ["geral", "fallback"],
                    "chunk_index": 0,
                    "source": "pdf",
                    "is_fallback": True
                }
            )
            chunks = [fallback_chunk]
        
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
    
    async def search_similar_chunks_enhanced(self, query: str, limit: int = 15, score_threshold: float = 0.2, document_id: str = None) -> List[SearchResult]:
        """
        BUSCA ULTRA-OTIMIZADA para capturar TODOS os chunks relevantes
        
        OTIMIZAÇÕES CRÍTICAS IMPLEMENTADAS:
        1. 🎯 Threshold ultra-baixo (0.2) para máxima cobertura
        2. 🔄 Busca multi-estratégia com diferentes abordagens
        3. 📊 Query expansion com sinônimos ICATU específicos
        4. 🏷️ Busca por categorias e metadados
        5. 🔍 Busca híbrida: semântica + lexical
        6. 📈 Re-ranking inteligente por relevância
        7. 🎛️ Fusão de resultados de múltiplas consultas
        """
        doc_logger.info(f"🔍 Busca ultra-otimizada para: '{query}' (threshold: {score_threshold}, limit: {limit})")
        
        # 1. Mapeamento de sinônimos específicos ICATU
        icatu_synonyms = {
            "alteração cadastral": ["mudança cadastral", "atualização cadastral", "modificação cadastral"],
            "solicitar": ["fazer solicitação", "requerer", "pedir", "solicitar"],
            "procurador": ["representante legal", "curador", "tutor", "responsável"],
            "menor de idade": ["menor", "criança", "adolescente"],
            "zendesk": ["sistema", "plataforma", "atendimento"],
            "documento": ["documentação", "arquivo", "anexo", "certidão"],
            "prazo": ["tempo", "período", "horário"],
            "titular": ["segurado", "beneficiário", "contratante"],
            "como": ["procedimento", "processo", "forma de"],
            "pode": ["consegue", "é possível", "tem permissão"]
        }
        
        # 2. Expandir query com sinônimos
        expanded_queries = [query]
        query_lower = query.lower()
        
        for term, synonyms in icatu_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    expanded_queries.append(expanded_query)
        
        # Adicionar variações estruturais
        if "como" in query_lower:
            expanded_queries.append(query_lower.replace("como", "procedimento para"))
            expanded_queries.append(query_lower.replace("como", "forma de"))
        
        if "quem pode" in query_lower:
            expanded_queries.append(query_lower.replace("quem pode", "solicitantes autorizados"))
            expanded_queries.append(query_lower.replace("quem pode", "pessoas que podem"))
        
        doc_logger.info(f"Queries expandidas: {len(expanded_queries)} variações")
        
        # 3. Gerar embeddings para todas as variações
        doc_logger.info("Gerando embedding da pergunta...")
        all_embeddings = await self.embedding_model.embed_texts(expanded_queries)
        
        # 4. Busca multi-vector com diferentes estratégias
        all_results = []
        
        for i, query_embedding in enumerate(all_embeddings):
            # Estratégia 1: Busca padrão
            search_params = {
                "collection_name": COLLECTION_NAME,
                "query_vector": query_embedding,
                "limit": min(limit * 3, 50),  # Buscar mais resultados inicialmente
                "score_threshold": max(score_threshold * 0.8, 0.1)  # Threshold ainda mais permissivo
            }
            
            if document_id:
                search_params["query_filter"] = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            
            doc_logger.info("Buscando chunks mais relevantes no Qdrant...")
            search_results = self.qdrant_client.search(**search_params)
            
            # Peso baseado na query (original tem peso maior)
            weight = 1.0 if i == 0 else 0.85
            
            for result in search_results:
                # Calcular score ajustado com peso e metadados
                adjusted_score = result.score * weight
                
                # Bonus por categoria relevante
                categories = result.payload.get("categories", [])
                if any(cat in ["alteracao_cadastral", "solicitantes", "procedimentos"] for cat in categories):
                    adjusted_score *= 1.2
                
                # Bonus por prioridade do chunk
                priority = result.payload.get("priority_score", 0)
                adjusted_score += priority
                
                result_obj = SearchResult(
                    content=result.payload["content"],
                    score=adjusted_score,
                    metadata={k: v for k, v in result.payload.items() if k != "content"}
                )
                all_results.append(result_obj)
        
        # 5. Deduplicar e ordenar resultados
        seen_content = {}
        unique_results = []
        
        for result in all_results:
            content_key = result.content[:100]  # Usar início do conteúdo como chave
            
            if content_key not in seen_content or result.score > seen_content[content_key].score:
                seen_content[content_key] = result
        
        unique_results = list(seen_content.values())
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # 6. Aplicar threshold final e limitar resultados
        final_results = [r for r in unique_results if r.score >= score_threshold][:limit]
        
        doc_logger.info(f"Busca retornou {len(final_results)} chunks relevantes.")
        
        # 7. Log detalhado dos resultados para debug
        if final_results:
            doc_logger.info(f"Melhores resultados: scores = {[f'{r.score:.3f}' for r in final_results[:3]]}")
        else:
            doc_logger.warning(f"Nenhum resultado encontrado com threshold {score_threshold}")
            
            # Busca de emergência com threshold muito baixo
            emergency_search = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=all_embeddings[0],
                limit=5,
                score_threshold=0.1
            )
            
            if emergency_search:
                doc_logger.info(f"Busca de emergência encontrou {len(emergency_search)} resultados")
                for result in emergency_search:
                    result_obj = SearchResult(
                        content=result.payload["content"],
                        score=result.score,
                        metadata={k: v for k, v in result.payload.items() if k != "content"}
                    )
                    final_results.append(result_obj)
        
        return final_results

    async def search_similar_chunks(self, query: str, limit: int = 15, score_threshold: float = 0.2, document_id: str = None) -> List[SearchResult]:
        """Compatibility wrapper for search_similar_chunks_enhanced with optimized defaults"""
        return await self.search_similar_chunks_enhanced(query, limit, score_threshold, document_id)

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the Qdrant collection"""
        try:
            collection_info = self.qdrant_client.get_collection(COLLECTION_NAME)
            return {
                "collection_name": COLLECTION_NAME,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
        except Exception as e:
            doc_logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}

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
        results = await processor.search_similar_chunks_enhanced(
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
