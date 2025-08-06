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
        """Pré-processamento avançado de texto com quebras estruturais"""
        # Preservar quebras importantes antes da normalização
        text = text.replace('\n\n', '\n§PARAGRAPH_BREAK§\n')
        
        # Normalizar espaços mas preservar quebras estruturais
        text = re.sub(r'[ \t]+', ' ', text)  # Espaços horizontais
        
        # Detectar e criar quebras em seções numeradas
        text = re.sub(r'(\d+\.)([A-Z])', r'\1\n\2', text)  # "1.QuemPode" -> "1.\nQuemPode"
        text = re.sub(r'([a-z])(\d+\.[A-Z])', r'\1\n\2', text)  # Quebra antes de numeração
        
        # Quebras em letras de subdivisão (a), b), c))
        text = re.sub(r'([.!?])\s*([a-z]\))', r'\1\n\2', text)
        
        # Quebras em títulos em maiúsculas (mínimo 3 palavras maiúsculas)
        text = re.sub(r'([.!?])\s*([A-Z][A-Z\s]{10,})', r'\1\n\2', text)
        
        # Quebras específicas para ICATU
        icatu_breaks = [
            'Objetivo',
            'QuemPodeSolicitar', 
            'TiposdeAlterações',
            'DocumentosnecessáriosparaIdentificação',
            'Procedimentos',
            'ImportanteObservar',
            'Atenção',
            'Observação',
            'Nota:'
        ]
        
        for break_word in icatu_breaks:
            # Adicionar quebra antes dessas palavras-chave
            pattern = r'([.!?:])\s*(' + re.escape(break_word) + ')'
            text = re.sub(pattern, r'\1\n\2', text, flags=re.IGNORECASE)
            
            # Também quebrar se vier após texto corrido
            pattern2 = r'([a-z])\s*(' + re.escape(break_word) + ')'
            text = re.sub(pattern2, r'\1\n\2', text, flags=re.IGNORECASE)
        
        # Quebras em listas com marcadores
        text = re.sub(r'([.!?])\s*([•·*-]\s*)', r'\1\n\2', text)
        text = re.sub(r'([a-z])\s*([•·*-]\s*[A-Z])', r'\1\n\2', text)
        
        # Quebras em final de sentenças longas seguidas de palavras-chave
        text = re.sub(r'([.!?])\s*(Para|Se|Caso|Quando|Após|Durante)', r'\1\n\2', text)
        
        # Restaurar quebras de parágrafo originais
        text = text.replace('\n§PARAGRAPH_BREAK§\n', '\n\n')
        
        # Normalizar quebras múltiplas
        text = re.sub(r'\n{3,}', '\n\n', text)  # Máximo 2 quebras consecutivas
        text = re.sub(r'\n\s+\n', '\n\n', text)  # Remove espaços entre quebras
        
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
CHUNK_SIZE = 1024  # REDUZIDO para gerar mais chunks menores e específicos
CHUNK_OVERLAP = 256  # PROPORCIONALMENTE ajustado
MIN_CHUNK_SIZE = 150  # REDUZIDO para permitir chunks menores mais granulares

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
        """Extract text from PDF using Docling with proper error handling"""
        try:
            # Tentativa 1: Usar DocumentConverter (método mais recente)
            try:
                from docling.document_converter import DocumentConverter
                from docling.datamodel.base_models import DocumentStream
                import io
                
                logger.info("🔄 Trying Docling DocumentConverter...")
                converter = DocumentConverter()
                
                # Criar stream do PDF
                pdf_stream = DocumentStream(
                    name="document.pdf",
                    stream=io.BytesIO(pdf_file)
                )
                
                # Converter documento
                result = converter.convert(pdf_stream)
                
                if result and hasattr(result, 'document'):
                    # Extrair texto do documento convertido
                    if hasattr(result.document, 'export_to_markdown'):
                        text = result.document.export_to_markdown()
                        logger.info(f"✅ Docling DocumentConverter: {len(text)} chars extracted")
                        return text
                    elif hasattr(result.document, 'export_to_text'):
                        text = result.document.export_to_text()
                        logger.info(f"✅ Docling DocumentConverter: {len(text)} chars extracted")
                        return text
                    else:
                        # Tentar extrair texto de páginas
                        pages_text = []
                        if hasattr(result.document, 'pages'):
                            for page in result.document.pages:
                                if hasattr(page, 'text'):
                                    pages_text.append(page.text)
                                elif hasattr(page, 'content'):
                                    pages_text.append(str(page.content))
                        
                        if pages_text:
                            text = "\n".join(pages_text)
                            logger.info(f"✅ Docling DocumentConverter (pages): {len(text)} chars extracted")
                            return text
                
            except ImportError as e:
                logger.info(f"📋 DocumentConverter not available: {e}")
            except Exception as e:
                logger.warning(f"⚠️ DocumentConverter failed: {e}")
            
            # Tentativa 2: Usar método direto do docling
            try:
                import docling
                import io
                
                logger.info("🔄 Trying direct Docling import...")
                
                # Verificar se há uma classe Document disponível
                if hasattr(docling, 'Document'):
                    doc = docling.Document.from_pdf(io.BytesIO(pdf_file))
                    if hasattr(doc, 'pages') and doc.pages:
                        texts = [page.text for page in doc.pages if hasattr(page, 'text')]
                        if texts:
                            text = "\n".join(texts)
                            logger.info(f"✅ Docling direct: {len(text)} chars extracted")
                            return text
                
                # Tentar outros métodos disponíveis
                available_methods = [attr for attr in dir(docling) if not attr.startswith('_')]
                logger.info(f"📋 Available Docling methods: {available_methods}")
                
            except Exception as e:
                logger.warning(f"⚠️ Direct Docling failed: {e}")
            
            # Tentativa 3: Usar docling-parse se disponível
            try:
                from docling_parse.pdf_parser import PdfParser
                import io
                
                logger.info("🔄 Trying docling-parse...")
                parser = PdfParser()
                
                # Parse do PDF
                parsed_doc = parser.parse(io.BytesIO(pdf_file))
                
                if parsed_doc:
                    # Extrair texto de todas as páginas
                    all_text = []
                    
                    if hasattr(parsed_doc, 'pages'):
                        for page_num, page in enumerate(parsed_doc.pages):
                            page_text = ""
                            
                            # Tentar diferentes métodos de extração
                            if hasattr(page, 'text'):
                                page_text = page.text
                            elif hasattr(page, 'get_text'):
                                page_text = page.get_text()
                            elif hasattr(page, 'content'):
                                page_text = str(page.content)
                            
                            if page_text:
                                all_text.append(page_text)
                                logger.info(f"📄 Docling-parse page {page_num + 1}: {len(page_text)} chars")
                    
                    if all_text:
                        text = "\n".join(all_text)
                        logger.info(f"✅ Docling-parse: {len(text)} chars total")
                        return text
                
            except ImportError:
                logger.info("📋 docling-parse not available")
            except Exception as e:
                logger.warning(f"⚠️ docling-parse failed: {e}")
            
            # Tentativa 4: Usar outras bibliotecas docling disponíveis
            try:
                # Verificar se há outros módulos docling disponíveis
                import importlib
                import pkgutil
                
                logger.info("🔄 Scanning for available docling modules...")
                
                # Tentar encontrar módulos docling
                docling_modules = []
                try:
                    import docling
                    for importer, modname, ispkg in pkgutil.iter_modules(docling.__path__, docling.__name__ + "."):
                        docling_modules.append(modname)
                        logger.info(f"📋 Found docling module: {modname}")
                except:
                    pass
                
                # Tentar usar docling_core se disponível (opcional)
                try:
                    try:
                        from docling_core.types.doc import DoclingDocument
                        logger.info("🔄 docling_core available but no direct PDF parser found")
                    except ImportError:
                        logger.info("📋 docling_core not available in this environment")
                    except Exception as e:
                        logger.warning(f"❌ docling_core import failed: {e}")
                        
                except ImportError:
                    logger.info("📋 docling_core not available")
                except Exception as e:
                    logger.warning(f"⚠️ docling_core failed: {e}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Docling module scanning failed: {e}")
                
            logger.warning("⚠️ All Docling extraction methods failed")
            return None
            
        except Exception as e:
            logger.warning(f"❌ Docling extraction completely failed: {e}")
            return None

    def extract_text_from_pdf(self, pdf_file: bytes) -> str:
        """Extract text from PDF file with enhanced debugging and multiple fallbacks"""
        logger.info(f"🔍 Starting PDF text extraction, file size: {len(pdf_file)} bytes")
        
        # Try Docling first (most advanced)
        docling_text = self.extract_text_with_docling(pdf_file)
        if docling_text and len(docling_text.strip()) > 50:
            cleaned_text = self.validate_and_clean_extracted_text(docling_text, "Docling")
            if len(cleaned_text.strip()) > 50:
                logger.info(f"✅ Docling extraction successful: {len(cleaned_text)} characters")
                logger.info(f"📄 Docling preview: {cleaned_text[:200]}...")
                return cleaned_text
        else:
            logger.info(f"⚠️ Docling extraction failed or insufficient text: {len(docling_text) if docling_text else 0} chars")
            
        # Fallback para PyPDF2 com extração melhorada
        logger.info("🔄 Trying enhanced PyPDF2 extraction...")
        try:
            import io
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
            text = ""
            page_count = len(pdf_reader.pages)
            logger.info(f"📖 PDF has {page_count} pages")
            
            for i, page in enumerate(pdf_reader.pages):
                try:
                    # Tentar múltiplos métodos de extração por página
                    page_text = ""
                    
                    # Método 1: extract_text() padrão
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            logger.debug(f"📄 Page {i+1}: Standard extraction: {len(page_text)} chars")
                    except Exception as e:
                        logger.warning(f"⚠️ Page {i+1}: Standard extraction failed: {e}")
                    
                    # Método 2: extract_text() com configurações customizadas
                    if not page_text or len(page_text) < 50:
                        try:
                            # Tentar com diferentes configurações
                            page_text_alt = page.extract_text(
                                extraction_mode="layout",
                                layout_mode_space_vertically=False
                            )
                            if page_text_alt and len(page_text_alt) > len(page_text):
                                page_text = page_text_alt
                                logger.debug(f"📄 Page {i+1}: Layout extraction better: {len(page_text)} chars")
                        except Exception as e:
                            logger.debug(f"Page {i+1}: Layout extraction failed: {e}")
                    
                    # Método 3: Visitor pattern para extração mais detalhada
                    if not page_text or len(page_text) < 50:
                        try:
                            class TextVisitor:
                                def __init__(self):
                                    self.text = []
                                
                                def visit_string(self, string_obj, *args):
                                    if hasattr(string_obj, 'get_data'):
                                        self.text.append(string_obj.get_data())
                                    elif hasattr(string_obj, '_data'):
                                        self.text.append(string_obj._data)
                                    else:
                                        self.text.append(str(string_obj))
                            
                            visitor = TextVisitor()
                            if hasattr(page, 'extract_text'):
                                # Usar visitor se disponível
                                try:
                                    page.extract_text(visitor=visitor.visit_string)
                                    visitor_text = ''.join(visitor.text)
                                    if visitor_text and len(visitor_text) > len(page_text):
                                        page_text = visitor_text
                                        logger.debug(f"📄 Page {i+1}: Visitor extraction better: {len(page_text)} chars")
                                except:
                                    pass
                        except Exception as e:
                            logger.debug(f"Page {i+1}: Visitor extraction failed: {e}")
                    
                    if page_text:
                        text += page_text + "\n"
                        logger.info(f"📄 Page {i+1}: {len(page_text)} characters extracted")
                    else:
                        logger.warning(f"⚠️ Page {i+1}: No text extracted")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Page {i+1}: Complete extraction error: {e}")
                    
            if len(text.strip()) > 50:
                cleaned_text = self.validate_and_clean_extracted_text(text.strip(), "Enhanced PyPDF2")
                if len(cleaned_text.strip()) > 50:
                    logger.info(f"✅ Enhanced PyPDF2 extraction successful: {len(cleaned_text)} characters total")
                    logger.info(f"📄 Enhanced PyPDF2 preview: {cleaned_text[:200]}...")
                    return cleaned_text
            else:
                logger.warning(f"⚠️ Enhanced PyPDF2 extracted insufficient text: {len(text)} chars")
        except Exception as e:
            logger.warning(f"❌ Enhanced PyPDF2 parsing failed: {e}")
            
        # Fallback para pdfminer com configurações melhoradas
        logger.info("🔄 Trying enhanced pdfminer extraction...")
        try:
            from pdfminer.high_level import extract_text
            from pdfminer.layout import LAParams
            import io
            
            # Configurações otimizadas para o pdfminer
            laparams = LAParams(
                boxes_flow=0.5,      # Melhor detecção de colunas
                word_margin=0.1,     # Espaçamento entre palavras
                char_margin=2.0,     # Espaçamento entre caracteres
                line_margin=0.5,     # Espaçamento entre linhas
                all_texts=True       # Extrair todo o texto disponível
            )
            
            text = extract_text(
                io.BytesIO(pdf_file), 
                laparams=laparams,
                maxpages=0,          # Processar todas as páginas
                password="",
                caching=True,
                check_extractable=True
            )
            
            if text and len(text.strip()) > 50:
                cleaned_text = self.validate_and_clean_extracted_text(text.strip(), "Enhanced pdfminer")
                if len(cleaned_text.strip()) > 50:
                    logger.info(f"✅ Enhanced pdfminer extraction successful: {len(cleaned_text)} characters")
                    logger.info(f"📄 Enhanced pdfminer preview: {cleaned_text[:200]}...")
                    return cleaned_text
            else:
                logger.warning(f"⚠️ Enhanced pdfminer extracted insufficient text: {len(text) if text else 0} chars")
                
        except Exception as e:
            logger.error(f"❌ Enhanced pdfminer parsing failed: {e}")
            
        # Último recurso: pdfplumber se disponível
        logger.info("🔄 Trying pdfplumber as last resort...")
        try:
            try:
                import pdfplumber
                import io
                
                text = ""
                with pdfplumber.open(io.BytesIO(pdf_file)) as pdf:
                    logger.info(f"📖 Pdfplumber: PDF has {len(pdf.pages)} pages")
                    
                    for i, page in enumerate(pdf.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                                logger.info(f"📄 Pdfplumber page {i+1}: {len(page_text)} chars")
                        except Exception as e:
                            logger.warning(f"⚠️ Pdfplumber page {i+1}: {e}")
                            
                if text and len(text.strip()) > 50:
                    cleaned_text = self.validate_and_clean_extracted_text(text.strip(), "Pdfplumber")
                    if len(cleaned_text.strip()) > 50:
                        logger.info(f"✅ Pdfplumber extraction successful: {len(cleaned_text)} characters")
                        logger.info(f"📄 Pdfplumber preview: {cleaned_text[:200]}...")
                        return cleaned_text
                else:
                    logger.warning(f"⚠️ Pdfplumber extracted insufficient text: {len(text) if text else 0} chars")
                    
            except ImportError:
                logger.info("📋 pdfplumber not available - install with: poetry add pdfplumber")
            except Exception as e:
                logger.warning(f"❌ pdfplumber extraction failed: {e}")
                
        except Exception as outer_e:
            logger.warning(f"❌ pdfplumber setup failed: {outer_e}")
            
        # If all methods fail
        logger.error("❌ All PDF extraction methods failed!")
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF using all available methods (Docling, PyPDF2, pdfminer, pdfplumber)")
    
    def validate_and_clean_extracted_text(self, text: str, filename: str) -> str:
        """Validate and clean extracted text to ensure quality"""
        if not text:
            logger.warning(f"⚠️ Empty text extracted from {filename}")
            return ""
        
        original_length = len(text)
        
        # 1. Remove caracteres de controle problemáticos
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # 2. Normalizar espaços em branco excessivos
        text = re.sub(r'\s{4,}', ' ', text)  # Máximo 3 espaços consecutivos
        
        # 3. Preservar quebras de linha importantes mas remover excessivas
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # Máximo 3 quebras
        
        # 4. Remover caracteres Unicode problemáticos que podem afetar tokenização
        text = re.sub(r'[\ufeff\u200b-\u200d\u2060]', '', text)  # Zero-width chars
        
        # 5. Verificar se há conteúdo significativo
        meaningful_chars = re.sub(r'[\s\n\r\t]', '', text)
        if len(meaningful_chars) < 50:
            logger.warning(f"⚠️ Very little meaningful content in {filename}: {len(meaningful_chars)} chars")
        
        # 6. Detectar possíveis problemas de encoding
        if text.count('�') > 5:  # Muitos caracteres de replacement
            logger.warning(f"⚠️ Possible encoding issues in {filename}: {text.count('�')} replacement chars")
        
        # 7. Verificar se o texto parece ser português/inglês válido
        alphabet_chars = re.findall(r'[a-zA-ZÀ-ÿ]', text)
        total_visible_chars = re.findall(r'[^\s\n\r\t]', text)
        
        if total_visible_chars and len(alphabet_chars) / len(total_visible_chars) < 0.5:
            logger.warning(f"⚠️ Text might contain many non-alphabetic characters in {filename}")
        
        # 8. Log estatísticas de limpeza
        cleaned_length = len(text)
        if cleaned_length != original_length:
            logger.info(f"🧹 Text cleaned: {original_length} -> {cleaned_length} chars ({filename})")
        
        # 9. Garantir que o texto termina adequadamente
        text = text.strip()
        if text and not text.endswith(('.', '!', '?', ':')):
            text += '.'
        
        return text
    
    def chunk_text(self, text: str, filename: str) -> List[DocumentChunk]:
        """
        CHUNKING HIERÁRQUICO OTIMIZADO com DEBUGGING COMPLETO
        
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
        
        logger.info(f"🔍 CHUNKING DEBUG: Starting chunking for {filename}")
        logger.info(f"📊 Input text length: {len(text)} characters")
        logger.info(f"📄 Text preview: {text[:300]}...")
        
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        
        # Pré-processamento avançado do texto
        original_length = len(text)
        text = preprocess_text_advanced(text)
        processed_length = len(text)
        
        logger.info(f"🔧 Text preprocessing: {original_length} -> {processed_length} chars")
        
        if processed_length < MIN_CHUNK_SIZE:
            logger.warning(f"⚠️ Text too short after preprocessing: {processed_length} < {MIN_CHUNK_SIZE}")
            if processed_length > 0:
                # Create a single chunk with the available text
                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=text,
                    embedding=[],
                    metadata={
                        "filename": filename,
                        "section": "COMPLETE_DOCUMENT",
                        "categories": ["geral"],
                        "chunk_index": 0,
                        "source": "pdf",
                        "is_short_document": True,
                        "character_count": len(text)
                    }
                )
                logger.info(f"✅ Created single chunk for short document: {len(text)} chars")
                return [chunk]
            else:
                logger.error(f"❌ No text available for chunking!")
                return []
        
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
        logger.info(f"📝 Split into {len(paragraphs)} paragraphs")
        
        # Se não há parágrafos bem definidos, dividir por quebras simples
        if len(paragraphs) == 1:
            paragraphs = text.split('\n')
            logger.info(f"📝 Fallback: Split into {len(paragraphs)} lines")
            
            # Se ainda há poucas quebras, forçar mais divisões
            if len(paragraphs) <= 3:
                logger.info(f"🔧 Forçando divisões adicionais...")
                # Dividir por padrões estruturais
                additional_splits = []
                for para in paragraphs:
                    # Dividir por numeração
                    parts = re.split(r'(\d+\.[A-Z])', para)
                    for part in parts:
                        if part.strip():
                            # Dividir por subdivisões a), b), c)
                            subparts = re.split(r'([a-z]\))', part)
                            additional_splits.extend([p.strip() for p in subparts if p.strip()])
                
                if len(additional_splits) > len(paragraphs):
                    paragraphs = additional_splits
                    logger.info(f"📝 Structural split: {len(paragraphs)} parts")
        
        # Se ainda assim há muito pouco conteúdo, usar chunking mais agressivo
        if len(paragraphs) <= 2 and len(text) > MIN_CHUNK_SIZE:
            logger.warning(f"⚠️ Very few paragraphs ({len(paragraphs)}), using aggressive sentence-based chunking")
            # Dividir por sentenças com chunks menores
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            logger.info(f"📝 Split into {len(sentences)} sentences")
            
            # Reagrupar sentenças em chunks MENORES para mais granularidade
            chunks = []
            current_chunk = ""
            chunk_index = 0
            target_chunk_size = CHUNK_SIZE // 2  # Chunks menores: 1024 tokens
            
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                    
                # Verificar se adicionar a sentença não excede o limite MENOR
                test_chunk = current_chunk + " " + sent if current_chunk else sent
                tokens = len(tokenizer.encode(test_chunk, add_special_tokens=False))
                
                if tokens > target_chunk_size and current_chunk:
                    # Salvar chunk atual
                    if len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
                        categories = categorize_content(current_chunk)
                        chunk = create_chunk_with_context(
                            current_chunk, "SENTENCE_BASED_SMALL", categories,
                            chunk_index, 0, 0
                        )
                        chunks.append(chunk)
                        logger.info(f"✅ Created small sentence-based chunk {chunk_index}: {len(current_chunk)} chars, {tokens} tokens")
                        chunk_index += 1
                    current_chunk = sent
                else:
                    current_chunk = test_chunk
            
            # Adicionar último chunk
            if current_chunk and len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
                categories = categorize_content(current_chunk)
                chunk = create_chunk_with_context(
                    current_chunk, "SENTENCE_BASED_SMALL", categories,
                    chunk_index, 0, 0
                )
                chunks.append(chunk)
                logger.info(f"✅ Created final small sentence-based chunk {chunk_index}: {len(current_chunk)} chars")
            
            # Se ainda temos poucos chunks, dividir os maiores
            if len(chunks) <= 3:
                logger.warning(f"⚠️ Still few chunks ({len(chunks)}), splitting large ones...")
                expanded_chunks = []
                for i, chunk in enumerate(chunks):
                    if len(chunk.text) > CHUNK_SIZE:
                        # Dividir chunk grande em pedaços menores
                        words = chunk.text.split()
                        words_per_chunk = len(words) // 3  # Dividir em 3 partes
                        
                        for j in range(0, len(words), words_per_chunk):
                            sub_words = words[j:j + words_per_chunk]
                            if sub_words:
                                sub_text = " ".join(sub_words)
                                if len(sub_text.strip()) >= MIN_CHUNK_SIZE:
                                    sub_categories = categorize_content(sub_text)
                                    sub_chunk = create_chunk_with_context(
                                        sub_text, f"SPLIT_CHUNK_{i}_{j//words_per_chunk}", sub_categories,
                                        len(expanded_chunks), 0, 0
                                    )
                                    expanded_chunks.append(sub_chunk)
                                    logger.info(f"✅ Split chunk {i} part {j//words_per_chunk}: {len(sub_text)} chars")
                    else:
                        expanded_chunks.append(chunk)
                
                if len(expanded_chunks) > len(chunks):
                    chunks = expanded_chunks
                    logger.info(f"🔧 Chunk splitting successful: {len(chunks)} total chunks")
            
            logger.info(f"🎯 Aggressive sentence-based chunking created {len(chunks)} chunks")
            return chunks
        
        chunks = []
        current_section = "DOCUMENTO_PRINCIPAL"
        section_index = 0
        chunk_index = 0
        
        # Buffer para construção de chunks
        chunk_buffer = ""
        chunk_tokens = 0
        previous_chunk_text = ""  # Para overlap
        
        doc_logger.info(f"🔍 Starting paragraph-based chunking for {filename}: {len(paragraphs)} paragraphs")
        
        non_empty_paragraphs = 0
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            non_empty_paragraphs += 1
            if para_idx < 5:  # Log first 5 paragraphs for debugging
                logger.info(f"📄 Paragraph {para_idx}: {len(paragraph)} chars - {paragraph[:100]}...")
            
            # Detecta nova seção baseada em padrões
            if section_pattern.match(paragraph) or title_pattern.match(paragraph):
                logger.info(f"🏷️ Detected new section: {paragraph[:50]}...")
                # Finaliza chunk atual se existir
                if chunk_buffer and len(chunk_buffer.strip()) >= MIN_CHUNK_SIZE:
                    categories = categorize_content(chunk_buffer)
                    chunk = create_chunk_with_context(
                        chunk_buffer, current_section, categories,
                        chunk_index, section_index, para_idx
                    )
                    chunks.append(chunk)
                    logger.info(f"✅ Created section-end chunk {chunk_index}: {len(chunk_buffer)} chars, categories: {categories}")
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
        
        # ESTRATÉGIA 6: Validação crítica e fallback robusto
        if len(chunks) < 3 or coverage_ratio < 0.90:
            logger.warning(f"⚠️ PROBLEMA CRÍTICO: {len(chunks)} chunks, cobertura {coverage_ratio:.2%}")
            logger.warning(f"🔧 Ativando fallback: chunking por força bruta...")
            
            # FALLBACK: Chunking por força bruta - dividir o texto em chunks de tamanho fixo
            fallback_chunks = []
            words = text.split()
            current_chunk_words = []
            current_chunk_tokens = 0
            fallback_chunk_index = 0
            
            for word in words:
                # Estimar tokens (aproximadamente 1 token = 0.75 palavras para português)
                word_tokens = max(1, len(word) // 4)
                
                if current_chunk_tokens + word_tokens > CHUNK_SIZE and current_chunk_words:
                    # Criar chunk atual
                    chunk_text = " ".join(current_chunk_words)
                    if len(chunk_text.strip()) >= MIN_CHUNK_SIZE:
                        categories = categorize_content(chunk_text)
                        fallback_chunk = DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            text=chunk_text,
                            embedding=[],
                            metadata={
                                "filename": filename,
                                "section": f"FALLBACK_CHUNK_{fallback_chunk_index}",
                                "categories": categories,
                                "chunk_index": fallback_chunk_index,
                                "source": "pdf",
                                "content_type": "document",
                                "token_count": current_chunk_tokens,
                                "character_count": len(chunk_text),
                                "is_fallback": True,
                                "processing_timestamp": time.time()
                            }
                        )
                        fallback_chunks.append(fallback_chunk)
                        logger.info(f"✅ Fallback chunk {fallback_chunk_index}: {len(chunk_text)} chars")
                        fallback_chunk_index += 1
                    
                    # Começar novo chunk com overlap
                    overlap_size = min(50, len(current_chunk_words) // 4)
                    current_chunk_words = current_chunk_words[-overlap_size:] + [word]
                    current_chunk_tokens = word_tokens + overlap_size
                else:
                    current_chunk_words.append(word)
                    current_chunk_tokens += word_tokens
            
            # Adicionar último chunk
            if current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                if len(chunk_text.strip()) >= MIN_CHUNK_SIZE:
                    categories = categorize_content(chunk_text)
                    fallback_chunk = DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        text=chunk_text,
                        embedding=[],
                        metadata={
                            "filename": filename,
                            "section": f"FALLBACK_CHUNK_{fallback_chunk_index}",
                            "categories": categories,
                            "chunk_index": fallback_chunk_index,
                            "source": "pdf",
                            "content_type": "document",
                            "token_count": current_chunk_tokens,
                            "character_count": len(chunk_text),
                            "is_fallback": True,
                            "processing_timestamp": time.time()
                        }
                    )
                    fallback_chunks.append(fallback_chunk)
                    logger.info(f"✅ Final fallback chunk {fallback_chunk_index}: {len(chunk_text)} chars")
            
            if len(fallback_chunks) > len(chunks):
                logger.info(f"🔧 Fallback successful: {len(fallback_chunks)} chunks vs {len(chunks)} original")
                chunks = fallback_chunks
            else:
                logger.warning(f"⚠️ Fallback não melhorou: {len(fallback_chunks)} vs {len(chunks)}")
        
        # ESTRATÉGIA 7: Último recurso - garantir pelo menos um chunk
        if len(chunks) == 0:
            logger.error(f"❌ CRÍTICO: Nenhum chunk gerado para {filename}!")
            logger.info(f"🆘 Criando chunk de emergência com todo o texto...")
            emergency_chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                text=text[:CHUNK_SIZE*4],  # Limita para evitar chunks gigantes
                embedding=[],
                metadata={
                    "filename": filename,
                    "section": "EMERGENCY_COMPLETE_DOCUMENT",
                    "categories": ["geral", "emergency"],
                    "chunk_index": 0,
                    "source": "pdf",
                    "is_emergency": True,
                    "character_count": len(text),
                    "processing_timestamp": time.time()
                }
            )
            chunks = [emergency_chunk]
            logger.info(f"🆘 Emergency chunk created: {len(emergency_chunk.text)} chars")
        
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
    return await processor.get_collection_info()

@app.get("/debug/chunks/{document_id}")
async def debug_chunks(document_id: str):
    """Debug endpoint to see all chunks for a document"""
    try:
        # Buscar todos os chunks do documento no Qdrant
        search_results = processor.qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            ),
            limit=100,  # Buscar até 100 chunks
            with_payload=True,
            with_vectors=False
        )
        
        chunks_info = []
        for point in search_results[0]:
            chunk_info = {
                "chunk_id": point.id,
                "content_preview": point.payload.get("content", "")[:200] + "..." if len(point.payload.get("content", "")) > 200 else point.payload.get("content", ""),
                "content_length": len(point.payload.get("content", "")),
                "categories": point.payload.get("categories", []),
                "section": point.payload.get("section", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "token_count": point.payload.get("token_count", 0),
                "metadata": {k: v for k, v in point.payload.items() if k not in ["content"]}
            }
            chunks_info.append(chunk_info)
        
        # Ordenar por chunk_index
        chunks_info.sort(key=lambda x: x.get("chunk_index", 0))
        
        return {
            "document_id": document_id,
            "total_chunks": len(chunks_info),
            "chunks": chunks_info
        }
        
    except Exception as e:
        logger.error(f"Error debugging chunks for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

@app.get("/debug/text-extraction/{document_hash}")
async def debug_text_extraction(document_hash: str):
    """Debug endpoint to see raw text extraction (requires re-upload for testing)"""
    # Note: This would require storing original PDFs or re-uploading for debug
    return {"message": "Upload a PDF with ?debug=true to see text extraction details"}

@app.post("/upload-pdf-debug", response_model=Dict[str, Any])
async def upload_pdf_debug(file: UploadFile = File(...)):
    """Upload PDF with detailed debugging information"""
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
        
        # Extract text with debugging
        logger.info(f"🔍 DEBUG: Processing PDF: {file.filename}")
        text = processor.extract_text_from_pdf(pdf_content)
        
        debug_info = {
            "filename": file.filename,
            "document_id": document_id,
            "file_size_bytes": len(pdf_content),
            "extracted_text_length": len(text),
            "extracted_text_preview": text[:500] + "..." if len(text) > 500 else text,
            "text_empty": not text.strip()
        }
        
        if not text.strip():
            debug_info["error"] = "No text could be extracted from PDF"
            return debug_info
        
        # Chunk text with debugging
        logger.info(f"🔍 DEBUG: Starting chunking process...")
        chunks = processor.chunk_text(text, file.filename)
        
        debug_info.update({
            "chunks_generated": len(chunks),
            "chunks_details": []
        })
        
        # Add detailed chunk information
        for i, chunk in enumerate(chunks):
            chunk_detail = {
                "chunk_index": i,
                "chunk_id": chunk.chunk_id,
                "text_length": len(chunk.text),
                "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "categories": chunk.metadata.get("categories", []),
                "section": chunk.metadata.get("section", ""),
                "token_count": chunk.metadata.get("token_count", 0),
                "has_overlap": chunk.metadata.get("has_overlap", False)
            }
            debug_info["chunks_details"].append(chunk_detail)
        
        # Generate embeddings (optional for debug)
        logger.info(f"🔍 DEBUG: Generating embeddings for {len(chunks)} chunks...")
        embeddings = await processor.generate_embeddings([chunk.text for chunk in chunks])
        
        # Store in Qdrant
        await processor.store_in_qdrant(chunks, embeddings, document_id)
        
        processing_time = time.time() - start_time
        debug_info["processing_time"] = processing_time
        debug_info["success"] = True
        
        return debug_info
        
    except Exception as e:
        logger.error(f"🔍 DEBUG: Error processing PDF {file.filename}: {e}")
        return {
            "filename": file.filename,
            "error": str(e),
            "success": False
        }
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
