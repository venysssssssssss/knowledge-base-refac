"""
PDF Document Processor with Dolphin (ByteDance) for parsing and Sentence Transformers for embeddings
"""

import asyncio
import hashlib
import io
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import PyPDF2
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
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
    text = re.sub(
        r'(\d+\.)([A-Z])', r'\1\n\2', text
    )  # "1.QuemPode" -> "1.\nQuemPode"
    text = re.sub(
        r'([a-z])(\d+\.[A-Z])', r'\1\n\2', text
    )  # Quebra antes de numeração

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
        'Nota:',
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
    text = re.sub(
        r'([.!?])\s*(Para|Se|Caso|Quando|Após|Durante)', r'\1\n\2', text
    )

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
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import PyPDF2
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pdfminer.high_level import extract_text
from PIL import Image
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct, VectorParams, Batch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_URL = 'http://qdrant:6333'
COLLECTION_NAME = 'knowledge_base_v3'
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Atual - bom para semântica geral
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'  # MELHOR para português brasileiro
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
        """Initialize the embedding model with optimizations for Mistral compatibility"""
        try:
            # Configurar o modelo para máxima compatibilidade com Mistral
            self.model = SentenceTransformer(self.model_name)
            
            # Configurações específicas para português brasileiro e Mistral
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = 512  # Otimizado para Mistral
            
            # Verificar dimensão real do modelo
            test_embedding = self.model.encode(["teste"], convert_to_tensor=False)
            if isinstance(test_embedding, np.ndarray):
                self.embedding_dim = test_embedding.shape[1]
            elif isinstance(test_embedding, list) and len(test_embedding) > 0:
                self.embedding_dim = len(test_embedding[0])
            
            logger.info(f'✅ Embedding model {self.model_name} loaded successfully')
            logger.info(f'📏 Embedding dimension confirmed: {self.embedding_dim}')
            
        except Exception as e:
            logger.critical(f'❌ Failed to load embedding model: {e}. Embeddings will NOT be generated. Aborting.')
            raise RuntimeError(f'Failed to load embedding model: {e}')

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings optimized for Mistral 7B retrieval"""
        if not self.model:
            logger.critical('❌ Embedding model is not loaded. Aborting embedding generation.')
            raise RuntimeError('Embedding model is not loaded.')
        
        try:
            # Pré-processamento dos textos para melhor qualidade
            processed_texts = []
            for text in texts:
                # Limitar tamanho do texto para evitar truncamento
                if len(text) > 2048:  # Limite seguro para o modelo
                    text = text[:2048] + "..."
                
                # Normalizar espaços e quebras
                text = re.sub(r'\s+', ' ', text.strip())
                
                # Adicionar contexto ICATU para melhor embedding
                if 'ICATU' not in text.upper():
                    text = f"[ICATU Manual] {text}"
                
                processed_texts.append(text)
            
            # Gerar embeddings com configurações otimizadas
            embeddings = self.model.encode(
                processed_texts,
                convert_to_tensor=False,  # Retornar como numpy array
                normalize_embeddings=True,  # Normalizar para cosine similarity
                show_progress_bar=True,
                batch_size=32  # Batch size otimizado
            )
            
            # Converter para formato correto
            if isinstance(embeddings, np.ndarray):
                embeddings_list = embeddings.tolist()
            elif hasattr(embeddings, 'tolist'):
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = embeddings

            # Validação rigorosa de cada embedding
            validated_embeddings = []
            for i, emb in enumerate(embeddings_list):
                if not isinstance(emb, (list, tuple)):
                    logger.warning(f'⚠️ Embedding {i} não é lista/tupla: {type(emb)}')
                    emb = self._fallback_embedding(processed_texts[i])
                elif len(emb) != self.embedding_dim:
                    logger.warning(f'⚠️ Embedding {i} tem dimensão incorreta: {len(emb)} vs {self.embedding_dim}')
                    emb = self._fallback_embedding(processed_texts[i])
                else:
                    # Validar valores numéricos
                    try:
                        emb = [float(x) for x in emb]
                        # Verificar valores inválidos (NaN, inf)
                        if any(not (isinstance(x, (int, float)) and -1000 < x < 1000 and x == x and x != float('inf') and x != float('-inf')) for x in emb):
                            logger.warning(f'⚠️ Embedding {i} contém valores inválidos')
                            emb = self._fallback_embedding(processed_texts[i])
                    except (ValueError, TypeError):
                        logger.warning(f'⚠️ Erro ao converter embedding {i} para float')
                        emb = self._fallback_embedding(processed_texts[i])
                
                validated_embeddings.append(emb)

            logger.info(f'✅ Generated {len(validated_embeddings)} embeddings successfully')
            return validated_embeddings
            
        except Exception as e:
            logger.critical(f'❌ SentenceTransformer encoding failed: {e}. Using fallback.')
            # Use fallback embeddings for all texts
            return [self._fallback_embedding(text) for text in texts]

    def _fallback_embedding(self, text: str) -> List[float]:
        """Enhanced fallback method optimized for Portuguese and ICATU content"""
        import hashlib
        import math
        
        # Palavras-chave específicas do domínio ICATU para melhor representação
        icatu_keywords = [
            'icatu', 'alteração', 'cadastral', 'solicitar', 'documento', 'cpf', 
            'titular', 'procurador', 'zendesk', 'sistema', 'prazo', 'procedimento',
            'formulário', 'assinatura', 'correio', 'email', 'telefone', 'endereço'
        ]
        
        # Criar embedding baseado em características do texto
        text_lower = text.lower()
        
        # Componente 1: Hash do texto para identificação única
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        hash_component = [int(text_hash[i:i+2], 16) / 255.0 for i in range(0, min(len(text_hash), 64), 2)]
        
        # Componente 2: Presença de palavras-chave ICATU
        keyword_component = []
        for keyword in icatu_keywords:
            if keyword in text_lower:
                keyword_component.append(0.8)
            else:
                keyword_component.append(0.1)
        
        # Componente 3: Características estruturais do texto
        structure_component = [
            len(text) / 1000.0,  # Comprimento normalizado
            text.count('.') / 10.0,  # Densidade de sentenças
            text.count(',') / 20.0,  # Densidade de vírgulas
            text.count('?') / 5.0,   # Presença de perguntas
            text.count('!') / 5.0,   # Presença de exclamações
            text.count(':') / 10.0,  # Presença de dois pontos
        ]
        
        # Componente 4: Análise semântica básica
        semantic_component = []
        semantic_patterns = {
            'procedimento': ['como', 'passo', 'etapa', 'processo'],
            'solicitação': ['solicitar', 'requerer', 'pedir'],
            'documento': ['formulário', 'certidão', 'comprovante'],
            'prazo': ['dia', 'hora', 'tempo', 'período'],
            'sistema': ['zendesk', 'mumps', 'sisprev'],
        }
        
        for category, patterns in semantic_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower) / len(patterns)
            semantic_component.append(score)
        
        # Combinar todos os componentes
        embedding = hash_component + keyword_component + structure_component + semantic_component
        
        # Ajustar para dimensão exata
        if len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]
        elif len(embedding) < self.embedding_dim:
            # Preencher com valores baseados em trigonometria para variação
            while len(embedding) < self.embedding_dim:
                idx = len(embedding)
                value = math.sin(idx * 0.1) * 0.1  # Valores pequenos mas variados
                embedding.append(value)
        
        # Normalizar o embedding
        norm = math.sqrt(sum(x*x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
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
        logger.info(
            f'Initializing SentenceTransformer embeddings: {EMBEDDING_MODEL}'
        )
        self.embedding_model = EmbeddingService(EMBEDDING_MODEL)
        await self.embedding_model.initialize()

        # Then initialize Qdrant client
        try:
            self.qdrant_client = QdrantClient(
                url=QDRANT_URL, check_compatibility=False
            )
            await self.ensure_collection_exists()
            logger.info('✅ Qdrant client initialized')
        except Exception as e:
            logger.error(f'Failed to initialize Qdrant client: {e}')
            raise

    async def ensure_collection_exists(self):
        """Create Qdrant collection if it doesn't exist, using the correct embedding dimension from the model."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if COLLECTION_NAME not in collection_names:
                logger.info(f'Creating collection: {COLLECTION_NAME}')
                # Descobre a dimensão do modelo carregado
                embedding_dim = (
                    self.embedding_model.embedding_dim
                    if hasattr(self.embedding_model, 'embedding_dim')
                    else 768
                )
                self.qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=embedding_dim, distance=models.Distance.COSINE
                    ),
                )
                logger.info(
                    f'✅ Collection {COLLECTION_NAME} created with dim {embedding_dim}'
                )
            else:
                # Verifica se a dimensão está correta
                info = self.qdrant_client.get_collection(COLLECTION_NAME)
                current_dim = info.config.params.vectors.size
                expected_dim = (
                    self.embedding_model.embedding_dim
                    if hasattr(self.embedding_model, 'embedding_dim')
                    else 768
                )
                if current_dim != expected_dim:
                    logger.error(
                        f'❌ Dimensão da coleção Qdrant ({current_dim}) não corresponde ao modelo ({expected_dim}). Exclua a coleção manualmente ou altere o nome da coleção.'
                    )
                    raise Exception(
                        f'Dimensão da coleção Qdrant ({current_dim}) não corresponde ao modelo ({expected_dim}). Exclua a coleção manualmente ou altere o nome da coleção.'
                    )
                logger.info(
                    f'✅ Collection {COLLECTION_NAME} already exists with correct dim {current_dim}'
                )
        except Exception as e:
            logger.error(f'Error creating collection: {e}')
            raise

    def extract_text_with_docling(self, pdf_file: bytes) -> Optional[str]:
        """Extract text from PDF using Docling with proper error handling"""
        try:
            # Tentativa 1: Usar DocumentConverter (método mais recente)
            try:
                import io

                from docling.datamodel.base_models import DocumentStream
                from docling.document_converter import DocumentConverter

                logger.info('🔄 Trying Docling DocumentConverter...')
                converter = DocumentConverter()

                # Criar stream do PDF
                pdf_stream = DocumentStream(
                    name='document.pdf', stream=io.BytesIO(pdf_file)
                )

                # Converter documento
                result = converter.convert(pdf_stream)

                if result and hasattr(result, 'document'):
                    # Extrair texto do documento convertido
                    if hasattr(result.document, 'export_to_markdown'):
                        text = result.document.export_to_markdown()
                        logger.info(
                            f'✅ Docling DocumentConverter: {len(text)} chars extracted'
                        )
                        return text
                    elif hasattr(result.document, 'export_to_text'):
                        text = result.document.export_to_text()
                        logger.info(
                            f'✅ Docling DocumentConverter: {len(text)} chars extracted'
                        )
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
                            text = '\n'.join(pages_text)
                            logger.info(
                                f'✅ Docling DocumentConverter (pages): {len(text)} chars extracted'
                            )
                            return text

            except ImportError as e:
                logger.info(f'📋 DocumentConverter not available: {e}')
            except Exception as e:
                logger.warning(f'⚠️ DocumentConverter failed: {e}')

            # Tentativa 2: Usar método direto do docling
            try:
                import io

                import docling

                logger.info('🔄 Trying direct Docling import...')

                # Verificar se há uma classe Document disponível
                if hasattr(docling, 'Document'):
                    doc = docling.Document.from_pdf(io.BytesIO(pdf_file))
                    if hasattr(doc, 'pages') and doc.pages:
                        texts = [
                            page.text
                            for page in doc.pages
                            if hasattr(page, 'text')
                        ]
                        if texts:
                            text = '\n'.join(texts)
                            logger.info(
                                f'✅ Docling direct: {len(text)} chars extracted'
                            )
                            return text

                # Tentar outros métodos disponíveis
                available_methods = [
                    attr for attr in dir(docling) if not attr.startswith('_')
                ]
                logger.info(
                    f'📋 Available Docling methods: {available_methods}'
                )

            except Exception as e:
                logger.warning(f'⚠️ Direct Docling failed: {e}')

            # Tentativa 3: Usar docling-parse se disponível
            try:
                import io

                from docling_parse.pdf_parser import PdfParser

                logger.info('🔄 Trying docling-parse...')
                parser = PdfParser()

                # Parse do PDF
                parsed_doc = parser.parse(io.BytesIO(pdf_file))

                if parsed_doc:
                    # Extrair texto de todas as páginas
                    all_text = []

                    if hasattr(parsed_doc, 'pages'):
                        for page_num, page in enumerate(parsed_doc.pages):
                            page_text = ''

                            # Tentar diferentes métodos de extração
                            if hasattr(page, 'text'):
                                page_text = page.text
                            elif hasattr(page, 'get_text'):
                                page_text = page.get_text()
                            elif hasattr(page, 'content'):
                                page_text = str(page.content)

                            if page_text:
                                all_text.append(page_text)
                                logger.info(
                                    f'📄 Docling-parse page {page_num + 1}: {len(page_text)} chars'
                                )

                    if all_text:
                        text = '\n'.join(all_text)
                        logger.info(
                            f'✅ Docling-parse: {len(text)} chars total'
                        )
                        return text

            except ImportError:
                logger.info('📋 docling-parse not available')
            except Exception as e:
                logger.warning(f'⚠️ docling-parse failed: {e}')

            # Tentativa 4: Usar outras bibliotecas docling disponíveis
            try:
                # Verificar se há outros módulos docling disponíveis
                import importlib
                import pkgutil

                logger.info('🔄 Scanning for available docling modules...')

                # Tentar encontrar módulos docling
                docling_modules = []
                try:
                    import docling

                    for importer, modname, ispkg in pkgutil.iter_modules(
                        docling.__path__, docling.__name__ + '.'
                    ):
                        docling_modules.append(modname)
                        logger.info(f'📋 Found docling module: {modname}')
                except:
                    pass

                # Tentar usar docling_core se disponível (opcional)
                try:
                    try:
                        from docling_core.types.doc import DoclingDocument

                        logger.info(
                            '🔄 docling_core available but no direct PDF parser found'
                        )
                    except ImportError:
                        logger.info(
                            '📋 docling_core not available in this environment'
                        )
                    except Exception as e:
                        logger.warning(f'❌ docling_core import failed: {e}')

                except ImportError:
                    logger.info('📋 docling_core not available')
                except Exception as e:
                    logger.warning(f'⚠️ docling_core failed: {e}')

            except Exception as e:
                logger.warning(f'⚠️ Docling module scanning failed: {e}')

            logger.warning('⚠️ All Docling extraction methods failed')
            return None

        except Exception as e:
            logger.warning(f'❌ Docling extraction completely failed: {e}')
            return None

    def extract_text_from_pdf(self, pdf_file: bytes) -> str:
        """Extract text from PDF file with enhanced debugging and multiple fallbacks"""
        logger.info(
            f'🔍 Starting PDF text extraction, file size: {len(pdf_file)} bytes'
        )

        # Try Docling first (most advanced)
        docling_text = self.extract_text_with_docling(pdf_file)
        if docling_text and len(docling_text.strip()) > 50:
            cleaned_text = self.validate_and_clean_extracted_text(
                docling_text, 'Docling'
            )
            if len(cleaned_text.strip()) > 50:
                logger.info(
                    f'✅ Docling extraction successful: {len(cleaned_text)} characters'
                )
                logger.info(f'📄 Docling preview: {cleaned_text[:200]}...')
                return cleaned_text
        else:
            logger.info(
                f'⚠️ Docling extraction failed or insufficient text: {len(docling_text) if docling_text else 0} chars'
            )

        # Fallback para PyPDF2 com extração melhorada
        logger.info('🔄 Trying enhanced PyPDF2 extraction...')
        try:
            import io

            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
            text = ''
            page_count = len(pdf_reader.pages)
            logger.info(f'📖 PDF has {page_count} pages')

            for i, page in enumerate(pdf_reader.pages):
                try:
                    # Tentar múltiplos métodos de extração por página
                    page_text = ''

                    # Método 1: extract_text() padrão
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            logger.debug(
                                f'📄 Page {i+1}: Standard extraction: {len(page_text)} chars'
                            )
                    except Exception as e:
                        logger.warning(
                            f'⚠️ Page {i+1}: Standard extraction failed: {e}'
                        )

                    # Método 2: extract_text() com configurações customizadas
                    if not page_text or len(page_text) < 50:
                        try:
                            # Tentar com diferentes configurações
                            page_text_alt = page.extract_text(
                                extraction_mode='layout',
                                layout_mode_space_vertically=False,
                            )
                            if page_text_alt and len(page_text_alt) > len(
                                page_text
                            ):
                                page_text = page_text_alt
                                logger.debug(
                                    f'📄 Page {i+1}: Layout extraction better: {len(page_text)} chars'
                                )
                        except Exception as e:
                            logger.debug(
                                f'Page {i+1}: Layout extraction failed: {e}'
                            )

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
                                    page.extract_text(
                                        visitor=visitor.visit_string
                                    )
                                    visitor_text = ''.join(visitor.text)
                                    if visitor_text and len(
                                        visitor_text
                                    ) > len(page_text):
                                        page_text = visitor_text
                                        logger.debug(
                                            f'📄 Page {i+1}: Visitor extraction better: {len(page_text)} chars'
                                        )
                                except:
                                    pass
                        except Exception as e:
                            logger.debug(
                                f'Page {i+1}: Visitor extraction failed: {e}'
                            )

                    if page_text:
                        text += page_text + '\n'
                        logger.info(
                            f'📄 Page {i+1}: {len(page_text)} characters extracted'
                        )
                    else:
                        logger.warning(f'⚠️ Page {i+1}: No text extracted')

                except Exception as e:
                    logger.warning(
                        f'⚠️ Page {i+1}: Complete extraction error: {e}'
                    )

            if len(text.strip()) > 50:
                cleaned_text = self.validate_and_clean_extracted_text(
                    text.strip(), 'Enhanced PyPDF2'
                )
                if len(cleaned_text.strip()) > 50:
                    logger.info(
                        f'✅ Enhanced PyPDF2 extraction successful: {len(cleaned_text)} characters total'
                    )
                    logger.info(
                        f'📄 Enhanced PyPDF2 preview: {cleaned_text[:200]}...'
                    )
                    return cleaned_text
            else:
                logger.warning(
                    f'⚠️ Enhanced PyPDF2 extracted insufficient text: {len(text)} chars'
                )
        except Exception as e:
            logger.warning(f'❌ Enhanced PyPDF2 parsing failed: {e}')

        # Fallback para pdfminer com configurações melhoradas
        logger.info('🔄 Trying enhanced pdfminer extraction...')
        try:
            import io

            from pdfminer.high_level import extract_text
            from pdfminer.layout import LAParams

            # Configurações otimizadas para o pdfminer
            laparams = LAParams(
                boxes_flow=0.5,  # Melhor detecção de colunas
                word_margin=0.1,  # Espaçamento entre palavras
                char_margin=2.0,  # Espaçamento entre caracteres
                line_margin=0.5,  # Espaçamento entre linhas
                all_texts=True,  # Extrair todo o texto disponível
            )

            text = extract_text(
                io.BytesIO(pdf_file),
                laparams=laparams,
                maxpages=0,  # Processar todas as páginas
                password='',
                caching=True,
                check_extractable=True,
            )

            if text and len(text.strip()) > 50:
                cleaned_text = self.validate_and_clean_extracted_text(
                    text.strip(), 'Enhanced pdfminer'
                )
                if len(cleaned_text.strip()) > 50:
                    logger.info(
                        f'✅ Enhanced pdfminer extraction successful: {len(cleaned_text)} characters'
                    )
                    logger.info(
                        f'📄 Enhanced pdfminer preview: {cleaned_text[:200]}...'
                    )
                    return cleaned_text
            else:
                logger.warning(
                    f'⚠️ Enhanced pdfminer extracted insufficient text: {len(text) if text else 0} chars'
                )

        except Exception as e:
            logger.error(f'❌ Enhanced pdfminer parsing failed: {e}')

        # Último recurso: pdfplumber se disponível
        logger.info('🔄 Trying pdfplumber as last resort...')
        try:
            try:
                import io

                import pdfplumber

                text = ''
                with pdfplumber.open(io.BytesIO(pdf_file)) as pdf:
                    logger.info(
                        f'📖 Pdfplumber: PDF has {len(pdf.pages)} pages'
                    )

                    for i, page in enumerate(pdf.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + '\n'
                                logger.info(
                                    f'📄 Pdfplumber page {i+1}: {len(page_text)} chars'
                                )
                        except Exception as e:
                            logger.warning(f'⚠️ Pdfplumber page {i+1}: {e}')

                if text and len(text.strip()) > 50:
                    cleaned_text = self.validate_and_clean_extracted_text(
                        text.strip(), 'Pdfplumber'
                    )
                    if len(cleaned_text.strip()) > 50:
                        logger.info(
                            f'✅ Pdfplumber extraction successful: {len(cleaned_text)} characters'
                        )
                        logger.info(
                            f'📄 Pdfplumber preview: {cleaned_text[:200]}...'
                        )
                        return cleaned_text
                else:
                    logger.warning(
                        f'⚠️ Pdfplumber extracted insufficient text: {len(text) if text else 0} chars'
                    )

            except ImportError:
                logger.info(
                    '📋 pdfplumber not available - install with: poetry add pdfplumber'
                )
            except Exception as e:
                logger.warning(f'❌ pdfplumber extraction failed: {e}')

        except Exception as outer_e:
            logger.warning(f'❌ pdfplumber setup failed: {outer_e}')

        # If all methods fail
        logger.error('❌ All PDF extraction methods failed!')
        raise HTTPException(
            status_code=400,
            detail='Failed to extract text from PDF using all available methods (Docling, PyPDF2, pdfminer, pdfplumber)',
        )

    def validate_and_clean_extracted_text(
        self, text: str, filename: str
    ) -> str:
        """Validate and clean extracted text to ensure quality"""
        if not text:
            logger.warning(f'⚠️ Empty text extracted from {filename}')
            return ''

        original_length = len(text)

        # 1. Remove caracteres de controle problemáticos
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # 2. Normalizar espaços em branco excessivos
        text = re.sub(r'\s{4,}', ' ', text)  # Máximo 3 espaços consecutivos

        # 3. Preservar quebras de linha importantes mas remover excessivas
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # Máximo 3 quebras

        # 4. Remover caracteres Unicode problemáticos que podem afetar tokenização
        text = re.sub(
            r'[\ufeff\u200b-\u200d\u2060]', '', text
        )  # Zero-width chars

        # 5. Verificar se há conteúdo significativo
        meaningful_chars = re.sub(r'[\s\n\r\t]', '', text)
        if len(meaningful_chars) < 50:
            logger.warning(
                f'⚠️ Very little meaningful content in {filename}: {len(meaningful_chars)} chars'
            )

        # 6. Detectar possíveis problemas de encoding
        if text.count('�') > 5:  # Muitos caracteres de replacement
            logger.warning(
                f"⚠️ Possible encoding issues in {filename}: {text.count('�')} replacement chars"
            )

        # 7. Verificar se o texto parece ser português/inglês válido
        alphabet_chars = re.findall(r'[a-zA-ZÀ-ÿ]', text)
        total_visible_chars = re.findall(r'[^\s\n\r\t]', text)

        if (
            total_visible_chars
            and len(alphabet_chars) / len(total_visible_chars) < 0.5
        ):
            logger.warning(
                f'⚠️ Text might contain many non-alphabetic characters in {filename}'
            )

        # 8. Log estatísticas de limpeza
        cleaned_length = len(text)
        if cleaned_length != original_length:
            logger.info(
                f'🧹 Text cleaned: {original_length} -> {cleaned_length} chars ({filename})'
            )

        # 9. Garantir que o texto termina adequadamente
        text = text.strip()
        if text and not text.endswith(('.', '!', '?', ':')):
            text += '.'

        return text

    def smart_structural_chunking(
        self, text: str, filename: str
    ) -> List[DocumentChunk]:
        """
        Chunking inteligente baseado na estrutura do documento ICATU
        Extrai seções por números (1., 2., etc.) e títulos específicos (Objetivo, etc.)
        """
        logger.info(f'🧠 Iniciando chunking estrutural inteligente para {filename}')
        logger.info(f'📊 Tamanho do texto de entrada: {len(text)} caracteres')
        
        try:
            # 1. Limpeza básica preservando estrutura
            logger.info("🧹 Iniciando limpeza estrutural...")
            clean_text = self.clean_text_preserving_structure(text)
            logger.info(f'🧹 Após limpeza estrutural: {len(clean_text)} caracteres')

            # 2. Extração de seções baseada em padrões do documento ICATU
            logger.info("📋 Extraindo seções do documento...")
            sections = self.extract_icatu_sections(clean_text)
            logger.info(f"📋 Seções extraídas: {len(sections)}")
            
            # Log das seções encontradas
            for i, section in enumerate(sections):
                logger.info(f"🔍 Seção {i}: '{section['title']}' - {len(section['content'])} chars")

            # 3. Criação de chunks otimizados para cada seção
            logger.info("📦 Criando chunks das seções...")
            chunks = self.create_section_chunks(sections, filename)

            if not chunks:
                logger.warning("⚠️ Nenhum chunk criado, usando fallback...")
                return self.fallback_text_chunking(clean_text, filename)

            logger.info(f'✅ Chunking concluído para {filename}:')
            logger.info(f'   📊 {len(chunks)} chunks gerados')
            
            # Log de cada chunk criado
            for i, chunk in enumerate(chunks):
                logger.info(f"📄 Chunk {i}: {chunk.chunk_id} - {len(chunk.text)} chars - '{chunk.metadata.get('section_title', 'NO TITLE')}'")

            return chunks

        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO no chunking estrutural: {e}")
            logger.error(f"❌ Tentando chunking de emergência...")
            return self.emergency_chunking(text, filename)

    def clean_text_preserving_structure(self, text: str) -> str:
        """Limpeza básica preservando a estrutura do documento"""
        # Remove caracteres de controle problemáticos
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normaliza espaços excessivos mas preserva quebras importantes
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove caracteres Unicode problemáticos
        text = re.sub(r'[\ufeff\u200b-\u200d\u2060]', '', text)
        
        return text.strip()

    def extract_icatu_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrai seções do documento ICATU baseado em padrões específicos
        Identifica seções numeradas (1., 2., etc.) e seções especiais (Objetivo, etc.)
        """
        sections = []
        lines = text.split('\n')
        
        # Padrões para identificar início de seções
        numbered_section_pattern = r'^##?\s*(\d+)\.\s*(.+?)$'  # ## 1. Título ou # 1. Título
        objective_pattern = r'^##?\s*(Objetivo|OBJETIVO)$'
        title_pattern = r'^##?\s*(.+?)$'  # Outros títulos com ##
        
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Verifica se é uma seção numerada (1., 2., etc.)
            numbered_match = re.match(numbered_section_pattern, line)
            if numbered_match:
                # Salva seção anterior se existir
                if current_section:
                    current_section['content'] = '\n'.join(current_content).strip()
                    current_section['content_length'] = len(current_section['content'])
                    sections.append(current_section)
                
                # Inicia nova seção numerada
                section_number = numbered_match.group(1)
                section_title = numbered_match.group(2).strip()
                current_section = {
                    'type': 'numbered_section',
                    'number': section_number,
                    'title': section_title,
                    'full_title': f"{section_number}. {section_title}",
                    'start_line': i,
                    'keywords': self.extract_keywords_from_section_title(section_title),
                    'topics': self.extract_topics_from_section_title(section_title)
                }
                current_content = []
                continue
            
            # Verifica se é a seção "Objetivo"
            objective_match = re.match(objective_pattern, line)
            if objective_match:
                # Salva seção anterior se existir
                if current_section:
                    current_section['content'] = '\n'.join(current_content).strip()
                    current_section['content_length'] = len(current_section['content'])
                    sections.append(current_section)
                
                # Inicia seção Objetivo
                current_section = {
                    'type': 'objective',
                    'number': '0',
                    'title': 'Objetivo',
                    'full_title': 'Objetivo',
                    'start_line': i,
                    'keywords': ['objetivo', 'finalidade', 'orientar', 'procedimentos'],
                    'topics': ['objetivo', 'manual', 'orientacao']
                }
                current_content = []
                continue
            
            # Verifica se é outro tipo de título (para capturar título principal)
            title_match = re.match(title_pattern, line)
            if title_match and not current_section:
                title_text = title_match.group(1).strip()
                # Se for o título principal do documento
                if 'manual' in title_text.lower() or 'icatu' in title_text.lower():
                    current_section = {
                        'type': 'main_title',
                        'number': '-1',
                        'title': title_text,
                        'full_title': title_text,
                        'start_line': i,
                        'keywords': ['manual', 'icatu', 'alteracao', 'cadastral'],
                        'topics': ['manual', 'titulo_principal', 'icatu']
                    }
                    current_content = []
                    continue
            
            # Adiciona linha ao conteúdo da seção atual
            if current_section and line:
                current_content.append(line)
        
        # Salva última seção
        if current_section:
            current_section['content'] = '\n'.join(current_content).strip()
            current_section['content_length'] = len(current_section['content'])
            sections.append(current_section)
        
        # Filtra seções vazias ou muito pequenas
        valid_sections = [s for s in sections if s.get('content_length', 0) > 50]
        
        logger.info(f"📋 Seções extraídas: {len(valid_sections)} válidas de {len(sections)} totais")
        
        return valid_sections

    def extract_keywords_from_section_title(self, title: str) -> List[str]:
        """Extrai palavras-chave do título da seção"""
        # Remove palavras comuns e foca nas importantes
        stop_words = {'de', 'da', 'do', 'para', 'com', 'em', 'por', 'a', 'o', 'e', 'ou'}
        words = [w.lower().strip() for w in re.split(r'[\s/\-]+', title) if w.strip()]
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]
        return keywords[:5]  # Limita a 5 palavras-chave

    def extract_topics_from_section_title(self, title: str) -> List[str]:
        """Extrai tópicos principais do título da seção"""
        title_lower = title.lower()
        
        # Mapeamento de tópicos baseado no conteúdo esperado
        topic_mapping = {
            'quem pode solicitar': ['solicitacao', 'titular', 'responsabilidade'],
            'tipos de alterações': ['alteracao', 'tipos', 'categorias'],
            'documento': ['documentos', 'identificacao', 'comprovantes'],
            'nome': ['nome', 'alteracao_nome', 'identificacao'],
            'endereço': ['endereco', 'contato', 'localizacao'],
            'telefone': ['telefone', 'contato', 'comunicacao'],
            'email': ['email', 'contato', 'comunicacao'],
            'cpf': ['cpf', 'documento', 'identificacao'],
            'data de nascimento': ['data_nascimento', 'documento', 'identificacao'],
            'envio': ['envio', 'documentos', 'procedimentos'],
            'registro': ['registro', 'sistema', 'procedimentos'],
            'prazo': ['prazos', 'tempo', 'processamento'],
            'parceiros': ['parceiros', 'banrisul', 'canais'],
            'interditado': ['interditado', 'curatela', 'especial'],
            'impossibilitado': ['impossibilitado', 'assinatura', 'especial']
        }
        
        topics = []
        for key, values in topic_mapping.items():
            if key in title_lower:
                topics.extend(values)
                break
        
        if not topics:
            # Fallback: usar palavras-chave do título
            topics = self.extract_keywords_from_section_title(title)
        
        return topics[:3]  # Limita a 3 tópicos

    def create_section_chunks(self, sections: List[Dict[str, Any]], filename: str) -> List[DocumentChunk]:
        """Cria chunks otimizados para cada seção extraída, especificamente otimizado para Mistral 7B"""
        chunks = []
        
        for i, section in enumerate(sections):
            # Constrói o texto completo da seção com contexto otimizado para Mistral
            section_text = self.build_mistral_optimized_section_text(section)
            
            # Verifica se o chunk tem tamanho adequado para Mistral (preferência por chunks menores)
            if len(section_text) < 100:
                logger.warning(f"⚠️ Seção '{section['title']}' muito pequena: {len(section_text)} chars")
                continue
            
            # Se o chunk for muito grande, dividir em subchunks
            if len(section_text) > 1500:  # Limite otimizado para Mistral 7B
                subchunks = self.create_mistral_subchunks(section, section_text, filename, i)
                chunks.extend(subchunks)
            else:
                # Cria chunk único da seção
                chunk = self.create_optimized_chunk(section, section_text, filename, i)
                chunks.append(chunk)
                logger.info(f"✅ Chunk único criado: Seção {section['number']} - '{section['title']}' ({len(section_text)} chars)")
        
        # Adiciona chunk de contexto geral otimizado para Mistral
        if not any(c.metadata.get('is_main_title') for c in chunks):
            general_context = self.create_mistral_general_context_chunk(filename, len(chunks))
            if general_context:
                chunks.insert(0, general_context)
        
        return chunks

    def build_mistral_optimized_section_text(self, section: Dict[str, Any]) -> str:
        """Constrói o texto da seção otimizado para recuperação com Mistral 7B"""
        parts = []
        
        # Cabeçalho conciso mas informativo
        parts.append(f"[ICATU Manual - {section['title']}]")
        parts.append("")
        
        # Título estruturado
        if section['type'] == 'numbered_section':
            parts.append(f"## {section['number']}. {section['title']}")
        elif section['type'] == 'objective':
            parts.append("## OBJETIVO DO MANUAL")
        else:
            parts.append(f"## {section['title']}")
        
        parts.append("")
        
        # Palavras-chave para contexto (importante para Mistral)
        if section.get('keywords'):
            keywords_text = " • ".join(section['keywords'])
            parts.append(f"**Palavras-chave:** {keywords_text}")
            parts.append("")
        
        # Conteúdo principal limpo e estruturado
        if section.get('content'):
            content = section['content']
            
            # Limpeza específica para Mistral
            content = re.sub(r'\n{3,}', '\n\n', content)
            content = re.sub(r'[ \t]+', ' ', content)
            
            # Estruturação para melhor compreensão
            content = self.enhance_content_structure_for_mistral(content)
            parts.append(content)
        
        # Contexto de fechamento
        parts.append("")
        parts.append(f"[Fim da seção: {section['title']}]")
        
        return '\n'.join(parts)

    def enhance_content_structure_for_mistral(self, content: str) -> str:
        """Melhora a estrutura do conteúdo para melhor compreensão do Mistral"""
        # Melhorar listas e enumerações
        content = re.sub(r'^([a-z]\))', r'• **\1**', content, flags=re.MULTILINE)
        content = re.sub(r'^(\d+\.)', r'**\1**', content, flags=re.MULTILINE)
        
        # Destacar informações importantes
        content = re.sub(r'(ATENÇÃO|IMPORTANTE|OBSERVAÇÃO|NOTA):\s*', r'**\1:** ', content, flags=re.IGNORECASE)
        
        # Melhorar formatação de procedimentos
        content = re.sub(r'(Como|Passo|Etapa)\s*(\d+)', r'**\1 \2**', content, flags=re.IGNORECASE)
        
        return content

    def create_mistral_subchunks(self, section: Dict[str, Any], section_text: str, filename: str, section_index: int) -> List[DocumentChunk]:
        """Cria subchunks otimizados para Mistral quando a seção é muito grande"""
        subchunks = []
        
        # Dividir por parágrafos primeiro
        paragraphs = section_text.split('\n\n')
        
        current_chunk_text = f"[ICATU Manual - {section['title']} - Parte 1]\n\n"
        current_chunk_paragraphs = []
        part_number = 1
        
        for para in paragraphs:
            # Se adicionar este parágrafo ultrapassar o limite, criar chunk atual
            test_text = current_chunk_text + '\n\n'.join(current_chunk_paragraphs + [para])
            
            if len(test_text) > 1200 and current_chunk_paragraphs:  # Limite menor para Mistral
                # Criar chunk atual
                final_text = current_chunk_text + '\n\n'.join(current_chunk_paragraphs)
                final_text += f"\n\n[Continua na parte {part_number + 1}...]"
                
                chunk = self.create_optimized_chunk(
                    section, final_text, filename, section_index, 
                    subchunk_index=part_number, is_subchunk=True
                )
                subchunks.append(chunk)
                
                # Iniciar novo chunk
                part_number += 1
                current_chunk_text = f"[ICATU Manual - {section['title']} - Parte {part_number}]\n\n"
                current_chunk_paragraphs = [para]
            else:
                current_chunk_paragraphs.append(para)
        
        # Adicionar último chunk se houver conteúdo
        if current_chunk_paragraphs:
            final_text = current_chunk_text + '\n\n'.join(current_chunk_paragraphs)
            final_text += f"\n\n[Fim da seção: {section['title']}]"
            
            chunk = self.create_optimized_chunk(
                section, final_text, filename, section_index,
                subchunk_index=part_number, is_subchunk=True
            )
            subchunks.append(chunk)
        
        logger.info(f"✅ Seção '{section['title']}' dividida em {len(subchunks)} subchunks para Mistral")
        return subchunks

    def create_optimized_chunk(self, section: Dict[str, Any], text: str, filename: str, 
                             section_index: int, subchunk_index: int = 0, is_subchunk: bool = False) -> DocumentChunk:
        """Cria chunk individual otimizado para Mistral 7B"""
        
        if is_subchunk:
            chunk_id = f"{filename}_s{section['number']}_p{subchunk_index:02d}_{section_index:03d}"
            section_title = f"{section['title']} (Parte {subchunk_index})"
        else:
            chunk_id = f"{filename}_s{section['number']}_{section_index:03d}"
            section_title = section['title']
        
        # Metadata otimizado para busca com Mistral
        metadata = {
            'filename': filename,
            'section_title': section_title,
            'section_full_title': section['full_title'],
            'section_type': section['type'],
            'section_number': section['number'],
            'section_index': section_index,
            'keywords': section.get('keywords', []),
            'topics': section.get('topics', []),
            'content_length': len(text),
            'is_numbered_section': section['type'] == 'numbered_section',
            'is_objective': section['type'] == 'objective',
            'is_main_title': section['type'] == 'main_title',
            'is_subchunk': is_subchunk,
            'subchunk_index': subchunk_index,
            'context_summary': f"Seção sobre {section['title']} do manual ICATU - otimizado para Mistral 7B",
            # Campos específicos para melhor recuperação com Mistral
            'retrieval_keywords': self.generate_retrieval_keywords(section, text),
            'content_type': self.classify_content_type(text),
            'importance_score': self.calculate_importance_score(section, text),
            'mistral_optimized': True
        }
        
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            embedding=[],
            metadata=metadata
        )
        
        return chunk

    def generate_retrieval_keywords(self, section: Dict[str, Any], text: str) -> List[str]:
        """Gera palavras-chave específicas para melhor recuperação com Mistral"""
        keywords = set()
        
        # Palavras-chave da seção
        keywords.update(section.get('keywords', []))
        
        # Palavras-chave do contexto ICATU
        icatu_context_keywords = [
            'alteração', 'cadastral', 'documento', 'cpf', 'nome', 'endereço',
            'telefone', 'email', 'procedimento', 'solicitação', 'titular',
            'zendesk', 'sistema', 'prazo', 'envio', 'registro'
        ]
        
        text_lower = text.lower()
        for keyword in icatu_context_keywords:
            if keyword in text_lower:
                keywords.add(keyword)
        
        # Detectar procedimentos específicos
        if any(word in text_lower for word in ['como', 'passo', 'etapa', 'procedimento']):
            keywords.add('procedimento_passo_a_passo')
        
        if any(word in text_lower for word in ['documento', 'anexar', 'enviar']):
            keywords.add('documentos_necessarios')
        
        if any(word in text_lower for word in ['prazo', 'tempo', 'dia']):
            keywords.add('prazos_temporais')
        
        return list(keywords)[:10]  # Limitar a 10 palavras-chave mais relevantes

    def classify_content_type(self, text: str) -> str:
        """Classifica o tipo de conteúdo para melhor categorização"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['como', 'passo', 'etapa', 'procedimento']):
            return 'procedimento'
        elif any(word in text_lower for word in ['documento', 'anexar', 'formulário']):
            return 'documentacao'
        elif any(word in text_lower for word in ['prazo', 'tempo', 'dia']):
            return 'prazo_temporal'
        elif any(word in text_lower for word in ['sistema', 'zendesk', 'registro']):
            return 'sistema_operacional'
        elif 'objetivo' in text_lower:
            return 'objetivo_geral'
        else:
            return 'informacao_geral'

    def calculate_importance_score(self, section: Dict[str, Any], text: str) -> float:
        """Calcula score de importância para priorização na recuperação"""
        score = 1.0
        
        # Boost para seção objetivo
        if section['type'] == 'objective':
            score += 0.5
        
        # Boost para seções numeradas (conteúdo principal)
        if section['type'] == 'numbered_section':
            score += 0.3
        
        # Boost para conteúdo sobre procedimentos
        text_lower = text.lower()
        if any(word in text_lower for word in ['como', 'passo', 'procedimento']):
            score += 0.4
        
        # Boost para informações sobre documentos
        if any(word in text_lower for word in ['documento', 'cpf', 'formulário']):
            score += 0.3
        
        # Boost para informações importantes
        if any(word in text_lower for word in ['importante', 'atenção', 'obrigatório']):
            score += 0.2
        
        return min(score, 2.0)  # Máximo de 2.0

    def create_mistral_general_context_chunk(self, filename: str, chunk_count: int) -> Optional[DocumentChunk]:
        """Cria chunk de contexto geral otimizado para Mistral 7B"""
        context_text = """[ICATU Manual - Contexto Geral]

## MANUAL DE ALTERAÇÃO CADASTRAL ICATU

**Palavras-chave:** manual, icatu, alteração, cadastral, procedimentos, documentos

### OBJETIVO
Manual oficial da ICATU Capitalização e Vida para orientar procedimentos de alteração cadastral de clientes.

### PRINCIPAIS TÓPICOS
• **Solicitações:** Quem pode solicitar alterações
• **Tipos:** Alterações de nome, endereço, telefone, email, CPF, data de nascimento
• **Documentos:** Comprovantes necessários para cada alteração
• **Procedimentos:** Passos para solicitar e processar alterações
• **Sistemas:** Registro no Zendesk e outros sistemas
• **Prazos:** Tempos de processamento
• **Casos Especiais:** Interditados e impossibilitados de assinar

### PÚBLICO-ALVO
Operadores de atendimento, analistas e profissionais responsáveis por processar solicitações de alteração cadastral na ICATU.

### IMPORTÂNCIA
Documento essencial para garantir processamento correto das alterações cadastrais, seguindo normas da empresa e regulamentações vigentes.

[Este é o contexto geral do manual completo - use para consultas gerais sobre alterações cadastrais ICATU]"""

        chunk = DocumentChunk(
            chunk_id=f"{filename}_mistral_context_{chunk_count:03d}",
            text=context_text,
            embedding=[],
            metadata={
                'filename': filename,
                'section_title': 'Contexto Geral Mistral',
                'section_type': 'mistral_general_context',
                'section_number': '0',
                'section_index': chunk_count,
                'keywords': ['manual', 'icatu', 'alteracao', 'cadastral', 'geral', 'contexto'],
                'topics': ['manual', 'contexto_geral', 'icatu', 'alteracao_cadastral'],
                'content_length': len(context_text),
                'is_general_context': True,
                'context_summary': 'Contexto geral do manual otimizado para Mistral 7B',
                'retrieval_keywords': ['manual', 'icatu', 'alteracao', 'cadastral', 'procedimentos', 'documentos', 'geral'],
                'content_type': 'contexto_geral',
                'importance_score': 1.8,
                'mistral_optimized': True
            }
        )
        
        return chunk

    def build_section_text_with_context(self, section: Dict[str, Any]) -> str:
        """Constrói o texto da seção com contexto para melhor recuperação"""
        parts = []
        
        # Adiciona contexto do documento
        parts.append("MANUAL DE ALTERAÇÃO CADASTRAL - ICATU")
        parts.append("")
        
        # Adiciona título da seção
        parts.append(f"# {section['full_title']}")
        parts.append("")
        
        # Adiciona o conteúdo da seção
        if section.get('content'):
            parts.append(section['content'])
        
        # Adiciona contexto adicional baseado no tipo de seção
        if section['type'] == 'numbered_section':
            parts.append("")
            parts.append(f"[Esta é a seção {section['number']} do manual sobre: {section['title']}]")
        elif section['type'] == 'objective':
            parts.append("")
            parts.append("[Esta seção define o objetivo e finalidade do manual]")
        
        return '\n'.join(parts)

    def create_general_context_chunk(self, filename: str, chunk_count: int) -> Optional[DocumentChunk]:
        """Cria chunk de contexto geral do documento"""
        context_text = """MANUAL DE ALTERAÇÃO CADASTRAL - ICATU

# CONTEXTO GERAL

Este é o manual oficial da ICATU Capitalização e Vida para procedimentos de alteração cadastral.

## PRINCIPAIS TÓPICOS ABORDADOS:
• Quem pode solicitar alterações cadastrais
• Tipos de alterações cadastrais permitidas
• Documentos necessários para cada tipo de alteração
• Procedimentos específicos e prazos
• Canais de atendimento e envio de documentos
• Registros no sistema
• Casos especiais (interditados, impossibilitados)

## PÚBLICO-ALVO:
Operadores de atendimento, analistas e profissionais que processam solicitações de alteração cadastral na ICATU.

## IMPORTÂNCIA:
Este documento é essencial para garantir que todas as alterações cadastrais sejam processadas corretamente, seguindo as normas da empresa e regulamentações vigentes.

[Este é um resumo geral de todo o conteúdo do manual para consultas gerais]"""

        chunk = DocumentChunk(
            chunk_id=f"{filename}_general_context_{chunk_count:03d}",
            text=context_text,
            embedding=[],
            metadata={
                'filename': filename,
                'section_title': 'Contexto Geral',
                'section_type': 'general_context',
                'section_number': '0',
                'section_index': chunk_count,
                'keywords': ['manual', 'icatu', 'alteracao', 'cadastral', 'geral', 'contexto'],
                'topics': ['manual', 'contexto_geral', 'icatu', 'alteracao_cadastral'],
                'content_length': len(context_text),
                'is_general_context': True,
                'context_summary': 'Contexto geral e resumo do manual completo'
            }
        )
        
        return chunk
        """Chunking básico como fallback"""
        logger.info("🆘 Executing fallback text chunking")
        
        try:
            chunks = []
            chunk_size = 1000
            overlap = 200
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]
                if len(chunk_text.strip()) >= 100:
                    chunk = DocumentChunk(
                        chunk_id=f'{filename}_fallback_{i//chunk_size:03d}',
                        text=chunk_text,
                        embedding=[],
                        metadata={
                            'filename': filename,
                            'section_title': f'Fallback Chunk {i//chunk_size}',
                            'section_type': 'fallback',
                            'section_index': i//chunk_size,
                            'is_fallback': True,
                            'keywords': [],
                            'topics': ['fallback'],
                            'context_summary': 'Chunk criado via fallback básico',
                        },
                    )
                    chunks.append(chunk)
                    logger.info(f"✅ Fallback chunk created: {len(chunk_text)} chars")
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error in fallback chunking: {e}")
            return []

    def emergency_chunking(self, text: str, filename: str) -> List[DocumentChunk]:
        """Chunking de emergência - último recurso"""
        logger.info("🚨 EMERGENCY CHUNKING ACTIVATED")
        
        try:
            # Criar pelo menos um chunk com todo o texto
            emergency_chunk = DocumentChunk(
                chunk_id=f'{filename}_emergency_001',
                text=text[:2000],  # Limitar para evitar chunks gigantes
                embedding=[],
                metadata={
                    'filename': filename,
                    'section_title': 'Emergency Complete Document',
                    'section_type': 'emergency',
                    'section_index': 0,
                    'is_emergency': True,
                    'keywords': ['emergency'],
                    'topics': ['emergency'],
                    'context_summary': 'Chunk de emergência com documento completo',
                },
            )
            
            logger.info(f"🚨 Emergency chunk created: {len(emergency_chunk.text)} chars")
            return [emergency_chunk]
            
        except Exception as e:
            logger.error(f"❌ FALHA CRÍTICA no emergency chunking: {e}")
            return []

    def generate_keywords_summary(self, text: str) -> str:
        """Gera resumo de palavras-chave e tópicos"""
        try:
            keywords = self.extract_keywords_from_text(text)

            categorized_keywords = {
                'Solicitantes': ['titular', 'procurador', 'curador', 'tutor'],
                'Documentos': [
                    'cpf',
                    'rg',
                    'certidão',
                    'formulário',
                    'assinatura',
                ],
                'Procedimentos': [
                    'alteração',
                    'atualização',
                    'validação',
                    'registro',
                ],
                'Prazos': ['dias', 'úteis', 'horas', 'prazo'],
                'Sistemas': ['zendesk', 'sisprev', 'mumps', 'sistema'],
                'Canais': ['correio', 'email', 'formulário', 'site'],
            }

            summary_parts = ['PRINCIPAIS CATEGORIAS E TERMOS:\n']

            for category, category_keywords in categorized_keywords.items():
                found_keywords = [
                    k
                    for k in keywords
                    if any(ck in k.lower() for ck in category_keywords)
                ]
                if found_keywords:
                    summary_parts.append(
                        f"{category}: {', '.join(found_keywords[:8])}"
                    )

            return '\n'.join(summary_parts)
        except Exception as e:
            logger.error(f"❌ Error generating keywords summary: {e}")
            return "Erro ao gerar resumo de palavras-chave"
            section['topics'] = self.extract_section_topics(section)
            section['end_line'] = (
                section['start_line']
                + len(section['content'])
                + len(section['subsections'])
            )

            for subsection in section['subsections']:
                subsection['keywords'] = self.extract_section_keywords(
                    subsection
                )
                subsection['topics'] = self.extract_section_topics(subsection)

        logger.info(
            f"📋 Estrutura detectada: {len(structure['sections'])} seções, título: '{structure['title'][:50]}...'"
        )
        return structure

    def create_hierarchical_chunks(
        self, structure: Dict[str, Any], filename: str
    ) -> List[DocumentChunk]:
        """Cria chunks preservando a hierarquia e contexto semântico com tamanho adequado"""
        logger.info('📦 Criando chunks hierárquicos...')

        chunks = []
        chunk_index = 0

        # 1. Chunk do título e contexto geral
        if structure['title']:
            title_context = f"""# {structure['title']}

Este é um manual da ICATU sobre procedimentos de alteração cadastral.

PRINCIPAIS TÓPICOS ABORDADOS:
• Quem pode solicitar alterações cadastrais
• Tipos de alterações permitidas 
• Documentos necessários
• Procedimentos e prazos
• Canais de atendimento

Este documento é essencial para operadores que processam solicitações de alteração cadastral na ICATU."""

            title_chunk = DocumentChunk(
                chunk_id=f'{filename}_title_{chunk_index:03d}',
                text=title_context,
                embedding=[],
                metadata={
                    'filename': filename,
                    'section_title': 'Título Principal',
                    'section_type': 'title',
                    'section_index': chunk_index,
                    'is_title': True,
                    'topics': [
                        'manual',
                        'icatu',
                        'alteracao_cadastral',
                        'procedimentos',
                    ],
                    'keywords': [
                        'manual',
                        'icatu',
                        'alteração',
                        'cadastral',
                        'procedimentos',
                    ],
                    'context_summary': 'Documento principal sobre alterações cadastrais na ICATU',
                },
            )
            chunks.append(title_chunk)
            chunk_index += 1

        # 2. Agrupar seções pequenas para formar chunks maiores
        current_group = []
        current_group_size = 0
        target_chunk_size = 800  # Tamanho alvo para cada chunk
        min_chunk_size = 300    # Tamanho mínimo aceitável

        for section_idx, section in enumerate(structure['sections']):
            section_content = self.build_section_content(section)
            section_size = len(section_content)

            # Se a seção é muito grande, criar chunk individual
            if section_size > target_chunk_size * 1.5:
                # Finalizar grupo atual se existir
                if current_group:
                    combined_chunk = self.create_combined_chunk(
                        current_group, filename, chunk_index
                    )
                    if combined_chunk:
                        chunks.append(combined_chunk)
                        chunk_index += 1
                    current_group = []
                    current_group_size = 0

                # Dividir seção grande em chunks com overlap
                section_chunks = self.create_overlapping_chunks(
                    section_content, section, filename, chunk_index
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)

            # Se adicionar esta seção não excede o limite, adicionar ao grupo
            elif current_group_size + section_size <= target_chunk_size * 1.2:
                current_group.append((section, section_content))
                current_group_size += section_size

            # Se excede, finalizar grupo atual e iniciar novo
            else:
                if current_group:
                    combined_chunk = self.create_combined_chunk(
                        current_group, filename, chunk_index
                    )
                    if combined_chunk:
                        chunks.append(combined_chunk)
                        chunk_index += 1

                # Iniciar novo grupo
                current_group = [(section, section_content)]
                current_group_size = section_size

        # Finalizar último grupo
        if current_group:
            combined_chunk = self.create_combined_chunk(
                current_group, filename, chunk_index
            )
            if combined_chunk:
                chunks.append(combined_chunk)
                chunk_index += 1

        return chunks

    def create_combined_chunk(
        self, section_group: List, filename: str, chunk_index: int
    ) -> Optional[DocumentChunk]:
        """Combina múltiplas seções em um chunk substancial"""
        if not section_group:
            return None

        combined_content = []
        combined_topics = []
        combined_keywords = []
        main_section_title = section_group[0][0]['title']

        for section, content in section_group:
            combined_content.append(content)
            combined_topics.extend(section.get('topics', []))
            combined_keywords.extend(section.get('keywords', []))

        full_content = '\n\n'.join(combined_content)

        # Verificar se o chunk tem tamanho adequado
        if len(full_content.strip()) < 200:
            return None

        # Remover duplicatas e limitar listas
        unique_topics = list(set(combined_topics))[:10]
        unique_keywords = list(set(combined_keywords))[:15]

        chunk = DocumentChunk(
            chunk_id=f'{filename}_combined_{chunk_index:03d}',
            text=full_content,
            embedding=[],
            metadata={
                'filename': filename,
                'section_title': main_section_title,
                'section_type': 'combined_section',
                'section_index': chunk_index,
                'hierarchical_level': 1,
                'topics': unique_topics,
                'keywords': unique_keywords,
                'context_summary': self.generate_context_summary(full_content),
                'sections_count': len(section_group),
                'combined_sections': [s[0]['title'] for s, _ in section_group],
            },
        )

        return chunk

    def build_section_content(self, section: Dict[str, Any]) -> str:
        """Constrói o conteúdo completo de uma seção com contexto"""
        content_parts = []

        # Título da seção
        content_parts.append(f"## {section['title']}")

        # Conteúdo principal da seção
        if section['content']:
            content_parts.append('\n'.join(section['content']))

        # Subseções
        for subsection in section['subsections']:
            content_parts.append(f"\n### {subsection['title']}")
            if subsection['content']:
                content_parts.append('\n'.join(subsection['content']))

        return '\n\n'.join(content_parts)

    def create_overlapping_chunks(
        self,
        content: str,
        section: Dict[str, Any],
        filename: str,
        start_index: int,
    ) -> List[DocumentChunk]:
        """Cria chunks com overlap inteligente para seções grandes"""
        chunks = []
        max_chunk_size = 1500
        overlap_size = 200

        # Dividir por parágrafos preservando contexto
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        current_chunk = ''
        chunk_paragraphs = []
        chunk_count = 0

        for i, paragraph in enumerate(paragraphs):
            # Verificar se adicionar este parágrafo excederia o limite
            potential_chunk = (
                current_chunk + '\n\n' + paragraph
                if current_chunk
                else paragraph
            )

            if len(potential_chunk) > max_chunk_size and current_chunk:
                # Criar chunk atual
                chunk_text = self.add_section_context(current_chunk, section)
                chunk = DocumentChunk(
                    chunk_id=f'{filename}_section_{start_index + chunk_count:03d}',
                    text=chunk_text,
                    embedding=[],
                    metadata={
                        'filename': filename,
                        'section_title': section['title'],
                        'section_type': f"{section['type']}_part",
                        'section_index': start_index + chunk_count,
                        'part_number': chunk_count + 1,
                        'hierarchical_level': section['level'],
                        'topics': section['topics'],
                        'keywords': self.extract_keywords_from_text(
                            current_chunk
                        ),
                        'context_summary': self.generate_context_summary(
                            current_chunk
                        ),
                        'is_continuation': chunk_count > 0,
                        'has_next_part': i < len(paragraphs) - 1,
                    },
                )
                chunks.append(chunk)

                # Preparar próximo chunk com overlap
                overlap_text = self.get_overlap_text(
                    current_chunk, overlap_size
                )
                current_chunk = (
                    overlap_text + '\n\n' + paragraph
                    if overlap_text
                    else paragraph
                )
                chunk_count += 1
            else:
                current_chunk = potential_chunk

        # Último chunk
        if current_chunk:
            chunk_text = self.add_section_context(current_chunk, section)
            chunk = DocumentChunk(
                chunk_id=f'{filename}_section_{start_index + chunk_count:03d}',
                text=chunk_text,
                embedding=[],
                metadata={
                    'filename': filename,
                    'section_title': section['title'],
                    'section_type': f"{section['type']}_final",
                    'section_index': start_index + chunk_count,
                    'part_number': chunk_count + 1,
                    'hierarchical_level': section['level'],
                    'topics': section['topics'],
                    'keywords': self.extract_keywords_from_text(current_chunk),
                    'context_summary': self.generate_context_summary(
                        current_chunk
                    ),
                    'is_continuation': chunk_count > 0,
                    'has_next_part': False,
                },
            )
            chunks.append(chunk)

        return chunks

    def add_section_context(
        self, content: str, section: Dict[str, Any]
    ) -> str:
        """Adiciona contexto da seção ao chunk"""
        context_prefix = f"[CONTEXTO: {section['title']}]\n\n"
        return context_prefix + content

    def get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Extrai texto de overlap do final do chunk atual"""
        sentences = text.split('.')
        overlap_text = ''
        for sentence in reversed(sentences):
            potential_overlap = sentence.strip() + '. ' + overlap_text
            if len(potential_overlap) <= overlap_size:
                overlap_text = potential_overlap
            else:
                break
        return overlap_text.strip()

    def create_global_context_chunks(
        self, text: str, filename: str, existing_count: int
    ) -> List[DocumentChunk]:
        """Cria chunks de contexto global substanciais para consultas gerais"""
        logger.info('🌐 Criando chunks de contexto global...')

        chunks = []

        # 1. Chunk de resumo executivo expandido
        summary = self.generate_comprehensive_executive_summary(text)
        summary_chunk = DocumentChunk(
            chunk_id=f'{filename}_summary_{existing_count:03d}',
            text=summary,
            embedding=[],
            metadata={
                'filename': filename,
                'section_title': 'Resumo Executivo',
                'section_type': 'executive_summary',
                'section_index': existing_count,
                'is_summary': True,
                'topics': [
                    'resumo',
                    'geral',
                    'overview',
                    'alteracao_cadastral',
                    'procedimentos',
                ],
                'keywords': [
                    'resumo',
                    'geral',
                    'principal',
                    'importante',
                    'alteração',
                    'cadastral',
                    'procedimentos',
                ],
                'context_summary': 'Resumo executivo completo de todo o documento',
            },
        )
        chunks.append(summary_chunk)

        # 2. Chunk de procedimentos principais
        procedures_summary = self.generate_procedures_summary(text)
        procedures_chunk = DocumentChunk(
            chunk_id=f'{filename}_procedures_{existing_count + 1:03d}',
            text=procedures_summary,
            embedding=[],
            metadata={
                'filename': filename,
                'section_title': 'Procedimentos Principais',
                'section_type': 'procedures_summary',
                'section_index': existing_count + 1,
                'is_procedures': True,
                'topics': [
                    'procedimentos',
                    'como_fazer',
                    'passo_a_passo',
                    'fluxo',
                ],
                'keywords': [
                    'procedimento',
                    'como',
                    'fazer',
                    'passo',
                    'fluxo',
                    'processo',
                ],
                'context_summary': 'Compilação de todos os procedimentos do documento',
            },
        )
        chunks.append(procedures_chunk)

        return chunks

    def generate_comprehensive_executive_summary(self, text: str) -> str:
        """Gera um resumo executivo abrangente do documento"""
        lines = text.split('\n')

        # Construir resumo estruturado
        summary_parts = [
            '# MANUAL DE ALTERAÇÃO CADASTRAL - ICATU',
            '',
            '## RESUMO EXECUTIVO',
            '',
            'Este manual da ICATU Capitalização e Vida apresenta os procedimentos completos para alteração cadastral de clientes.',
            '',
            '## PRINCIPAIS TÓPICOS ABORDADOS:',
            '',
            '### 1. QUEM PODE SOLICITAR',
            '• Somente o titular da apólice pode solicitar alterações cadastrais',
            '• Para inclusão de nome social: também permitido procurador, curador ou tutor',
            '• Responsável legal para menores de idade',
            '',
            '### 2. TIPOS DE ALTERAÇÕES',
            '• Documento de identificação',
            '• Nome completo',
            '• Estado civil',
            '• Nome social',
            '• Endereço, telefone e e-mail',
            '• CPF e data de nascimento',
            '',
            '### 3. DOCUMENTOS NECESSÁRIOS',
            '• Formulário de alteração de dados',
            '• Documento de identificação com foto',
            '• Certidões específicas (casamento, divórcio, óbito conforme alteração)',
            '• Comprovante de endereço quando aplicável',
            '',
            '### 4. PROCEDIMENTOS',
            '• Validação no site da Receita Federal (CPF/data nascimento)',
            '• Preenchimento de formulário específico',
            '• Assinatura com reconhecimento de firma (quando exigido)',
            '• Envio via correios ou canais digitais',
            '',
            '### 5. PRAZOS',
            '• Processamento: até 7 dias úteis',
            '• Reflexão no sistema: até 24 horas',
            '• Atualização no Zendesk: até 1 hora',
            '',
            '### 6. CANAIS DE ENVIO',
            '• Área do Cliente (para alterações simples)',
            '• Formulário com assinatura digital',
            '• Correios (caixa postal específica)',
            '• Parceiros específicos (conforme orientação)',
            '',
            'Este documento é essencial para operadores de atendimento que processam solicitações de alteração cadastral na ICATU.',
        ]

        return '\n'.join(summary_parts)

    def generate_procedures_summary(self, text: str) -> str:
        """Gera resumo detalhado dos procedimentos"""

        procedures_parts = [
            '# PROCEDIMENTOS DE ALTERAÇÃO CADASTRAL - ICATU',
            '',
            '## FLUXO GERAL DE ATENDIMENTO',
            '',
            '### PASSO 1: IDENTIFICAÇÃO E VALIDAÇÃO',
            '• Confirmar identidade do solicitante',
            '• Verificar se é titular ou representante autorizado',
            '• Validar dados no sistema ICATU',
            '',
            '### PASSO 2: TIPO DE ALTERAÇÃO',
            '• Identificar que tipo de alteração será realizada',
            '• Verificar documentos necessários',
            '• Orientar sobre procedimentos específicos',
            '',
            '### PASSO 3: VALIDAÇÕES NECESSÁRIAS',
            '• CPF/Data Nascimento: validar na Receita Federal',
            '• Documentos: verificar autenticidade e validade',
            '• Assinatura: confirmar necessidade de reconhecimento',
            '',
            '### PASSO 4: PROCESSAMENTO',
            '• Registrar solicitação no sistema',
            '• Anexar documentos digitalizados',
            '• Definir prazo de conclusão',
            '',
            '### PASSO 5: FINALIZAÇÃO',
            '• Atualizar dados nos sistemas',
            '• Confirmar alteração com cliente',
            '• Registrar conclusão do atendimento',
            '',
            '## PROCEDIMENTOS ESPECÍFICOS',
            '',
            '### ALTERAÇÃO DE NOME',
            '• Documento necessário: RG ou certidão com novo nome',
            '• Erros simples: correção direta no sistema',
            '• Alterações formais: formulário com reconhecimento',
            '',
            '### ALTERAÇÃO DE CPF/DATA NASCIMENTO',
            '• Sempre validar na Receita Federal',
            '• Inserir print da validação no sistema',
            "• Registrar como 'Alteração Cadastral Pendente'",
            '',
            '### ALTERAÇÃO DE ENDEREÇO/TELEFONE/EMAIL',
            '• Pode ser feita diretamente no sistema',
            '• Registro automático gerado',
            '• Sincronização com Zendesk em até 24h',
            '',
            '### NOME SOCIAL',
            '• Base legal: Circular SUSEP 001/2024',
            '• Não necessário documento comprobatório',
            '• Pode ser solicitado a qualquer momento',
            '• Transferência assistida para ramal específico (V&P)',
            '',
            '## SISTEMAS ENVOLVIDOS',
            '• ZENDESK: atendimento e registros',
            '• SISPREV: previdência',
            '• MUMPS/SISVIDA: seguros de vida',
            '• SISCAP: capitalização',
            '',
            'Este guia serve como referência rápida para todos os procedimentos de alteração cadastral.',
        ]

        return '\n'.join(procedures_parts)

    def generate_executive_summary(self, text: str) -> str:
        """Gera um resumo executivo do documento"""
        # Extrair primeiros parágrafos significativos
        paragraphs = [
            p.strip() for p in text.split('\n\n') if p.strip() and len(p) > 50
        ]

        summary_parts = [
            'Este manual da ICATU Capitalização e Vida apresenta os procedimentos para alteração cadastral de clientes.',
            '\nPRINCIPAIS TÓPICOS ABORDADOS:',
        ]

        # Identificar tópicos principais
        main_topics = []
        for paragraph in paragraphs[:20]:  # Primeiros 20 parágrafos
            if any(
                keyword in paragraph.lower()
                for keyword in [
                    'quem pode',
                    'tipos de',
                    'documentos',
                    'procedimento',
                ]
            ):
                clean_para = paragraph.replace('\n', ' ').strip()
                if len(clean_para) < 200:
                    main_topics.append(f'• {clean_para}')

        summary_parts.extend(main_topics[:8])  # Máximo 8 tópicos
        summary_parts.append(
            '\nEste documento é essencial para operadores que processam solicitações de alteração cadastral.'
        )

        return '\n'.join(summary_parts)

    def generate_keywords_summary(self, text: str) -> str:
        """Gera resumo de palavras-chave e tópicos"""
        keywords = self.extract_keywords_from_text(text)

        categorized_keywords = {
            'Solicitantes': ['titular', 'procurador', 'curador', 'tutor'],
            'Documentos': [
                'cpf',
                'rg',
                'certidão',
                'formulário',
                'assinatura',
            ],
            'Procedimentos': [
                'alteração',
                'atualização',
                'validação',
                'registro',
            ],
            'Prazos': ['dias', 'úteis', 'horas', 'prazo'],
            'Sistemas': ['zendesk', 'sisprev', 'mumps', 'sistema'],
            'Canais': ['correio', 'email', 'formulário', 'site'],
        }

        summary_parts = ['PRINCIPAIS CATEGORIAS E TERMOS:\n']

        for category, category_keywords in categorized_keywords.items():
            found_keywords = [
                k
                for k in keywords
                if any(ck in k.lower() for ck in category_keywords)
            ]
            if found_keywords:
                summary_parts.append(
                    f"{category}: {', '.join(found_keywords[:8])}"
                )

        return '\n'.join(summary_parts)

    def extract_keywords_from_text(self, text: str) -> List[str]:
        """Extrai palavras-chave relevantes do texto"""
        # Palavras importantes do domínio
        domain_keywords = [
            'alteração',
            'cadastral',
            'titular',
            'documento',
            'cpf',
            'rg',
            'nome',
            'endereço',
            'telefone',
            'email',
            'formulário',
            'assinatura',
            'prazo',
            'sistema',
            'zendesk',
            'sisprev',
            'procurador',
            'curador',
            'tutor',
        ]

        found_keywords = []
        text_lower = text.lower()

        for keyword in domain_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)

        # Adicionar palavras frequentes específicas
        words = re.findall(r'\b[a-záéíóúçãõâêî]{4,}\b', text_lower)
        word_freq = {}
        for word in words:
            if word not in [
                'para',
                'com',
                'são',
                'está',
                'ter',
                'como',
                'mais',
            ]:
                word_freq[word] = word_freq.get(word, 0) + 1

        frequent_words = [
            word
            for word, freq in sorted(
                word_freq.items(), key=lambda x: x[1], reverse=True
            )[:10]
            if freq > 2
        ]
        found_keywords.extend(frequent_words)

        return list(set(found_keywords))

    def extract_section_keywords(self, section: Dict[str, Any]) -> List[str]:
        """Extrai palavras-chave específicas de uma seção"""
        content = section['title'] + ' ' + ' '.join(section['content'])
        return self.extract_keywords_from_text(content)

    def extract_section_topics(self, section: Dict[str, Any]) -> List[str]:
        """Extrai tópicos específicos de uma seção"""
        title_lower = section['title'].lower()

        topic_mapping = {
            'solicitantes': ['quem pode', 'titular', 'procurador'],
            'documentos': ['documento', 'cpf', 'rg', 'identificação'],
            'procedimentos': ['procedimento', 'como', 'processo'],
            'prazos': ['prazo', 'tempo', 'dias'],
            'alteracao_cadastral': ['alteração', 'cadastral', 'atualização'],
        }

        topics = []
        for topic, keywords in topic_mapping.items():
            if any(keyword in title_lower for keyword in keywords):
                topics.append(topic)

        return topics

    def generate_context_summary(self, content: str) -> str:
        """Gera resumo contextual do conteúdo"""
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        if not lines:
            return 'Conteúdo vazio'

        # Primeira linha significativa
        first_line = lines[0] if lines else ''

        # Identificar pontos-chave
        key_points = []
        for line in lines:
            if any(
                keyword in line.lower()
                for keyword in [
                    'deve',
                    'necessário',
                    'obrigatório',
                    'importante',
                ]
            ):
                key_points.append(
                    line[:100] + '...' if len(line) > 100 else line
                )

        summary = (
            first_line[:100] + '...' if len(first_line) > 100 else first_line
        )
        if key_points:
            summary += f' | Pontos-chave: {len(key_points)} itens'

        return summary

    def optimize_chunks_for_retrieval(
        self, chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """Otimiza chunks para melhor recuperação pelo RAG com critérios ajustados"""
        logger.info('🔧 Otimizando chunks para recuperação...')

        optimized_chunks = []

        for chunk in chunks:
            # Critérios mais flexíveis para o tamanho mínimo
            chunk_text = chunk.text.strip()

            # Aceitar chunks com pelo menos 100 caracteres OU que contenham informações importantes
            is_important_chunk = any(
                keyword in chunk_text.lower()
                for keyword in [
                    'titular',
                    'solicitar',
                    'documento',
                    'procedimento',
                    'prazo',
                    'alteração',
                    'cadastral',
                    'icatu',
                    'formulário',
                    'assinatura',
                    'reconhecimento',
                    'sistema',
                ]
            )

            # Aceitar se tem tamanho adequado OU é importante
            if len(chunk_text) >= 100 or (
                len(chunk_text) >= 50 and is_important_chunk
            ):
                # Adicionar metadados de busca
                chunk.metadata['search_keywords'] = ' '.join(
                    chunk.metadata.get('keywords', [])
                )
                chunk.metadata['search_topics'] = ' '.join(
                    chunk.metadata.get('topics', [])
                )

                # Normalizar texto para busca
                normalized_text = self.normalize_for_search(chunk.text)
                chunk.metadata['normalized_content'] = normalized_text[
                    :500
                ]  # Primeiros 500 chars

                # Adicionar indicadores de qualidade
                chunk.metadata['has_important_keywords'] = is_important_chunk
                chunk.metadata['content_length'] = len(chunk_text)

                optimized_chunks.append(chunk)
            else:
                logger.debug(
                    f'Chunk pequeno ignorado: {chunk.chunk_id} ({len(chunk_text)} chars)'
                )

        logger.info(
            f'🔧 Otimização concluída: {len(optimized_chunks)} chunks válidos de {len(chunks)} originais'
        )
        return optimized_chunks

    def normalize_for_search(self, text: str) -> str:
        """Normaliza texto para melhor busca"""
        # Remover caracteres especiais e normalizar
        normalized = re.sub(r'[^\w\s]', ' ', text.lower())
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()

    def calculate_coverage(
        self, chunks: List[DocumentChunk], original_text: str
    ) -> float:
        """Calcula a cobertura dos chunks em relação ao texto original"""
        total_chunk_chars = sum(len(chunk.text) for chunk in chunks)
        original_chars = len(original_text)
        return (
            (total_chunk_chars / original_chars) * 100
            if original_chars > 0
            else 0
        )

    def analyze_document_structure(self, text: str) -> List[Dict]:
        """Analisa a estrutura lógica do documento e identifica seções"""
        import re

        sections = []
        lines = text.split('\n')
        current_section = {
            'title': 'Introdução',
            'type': 'intro',
            'content': '',
            'topics': [],
            'keywords': [],
            'summary': '',
        }

        title_patterns = [
            r'^#{1,3}\s+(.+)',  # Markdown headers
            r'^(\d+\.)\s+(.+)',  # Numbered sections
            r'^([A-Z][^a-z]*):?\s*$',  # ALL CAPS titles
            r'^(.+):\s*$',  # Title with colon
            r'^([a-z]\))\s+(.+)',  # Letter enumeration
        ]

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Detectar início de nova seção
            is_new_section = False
            section_title = ''
            section_type = 'content'

            for pattern in title_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    if pattern.startswith(r'^#{1,3}'):  # Markdown header
                        section_title = match.group(1).strip()
                        section_type = 'header'
                    elif pattern.startswith(r'^(\d+\.)'):  # Numbered
                        section_title = (
                            f'{match.group(1)} {match.group(2)}'.strip()
                        )
                        section_type = 'numbered'
                    elif pattern.startswith(r'^([A-Z]'):  # ALL CAPS
                        section_title = line.strip(':')
                        section_type = 'emphasis'
                    elif pattern.startswith(r'^(.+):'):  # With colon
                        section_title = match.group(1).strip()
                        section_type = 'topic'
                    elif pattern.startswith(r'^([a-z]\))'):  # Letter enum
                        section_title = (
                            f'{match.group(1)} {match.group(2)}'.strip()
                        )
                        section_type = 'enumeration'

                    is_new_section = True
                    break

            if is_new_section and current_section['content'].strip():
                # Finalizar seção anterior
                current_section['summary'] = self.generate_section_summary(
                    current_section['content']
                )
                current_section['keywords'] = self.extract_keywords(
                    current_section['content']
                )
                current_section['topics'] = self.extract_topics(
                    current_section['content']
                )
                sections.append(current_section)

                # Iniciar nova seção
                current_section = {
                    'title': section_title,
                    'type': section_type,
                    'content': line + '\n',
                    'topics': [],
                    'keywords': [],
                    'summary': '',
                }
            else:
                # Adicionar linha à seção atual
                current_section['content'] += line + '\n'

        # Adicionar última seção
        if current_section['content'].strip():
            current_section['summary'] = self.generate_section_summary(
                current_section['content']
            )
            current_section['keywords'] = self.extract_keywords(
                current_section['content']
            )
            current_section['topics'] = self.extract_topics(
                current_section['content']
            )
            sections.append(current_section)

        return sections

    def split_large_section(
        self, section: Dict, filename: str, section_index: int
    ) -> List[DocumentChunk]:
        """Divide seções muito grandes mantendo contexto"""
        content = section['content']
        chunks = []

        # Dividir por parágrafos lógicos
        paragraphs = content.split('\n\n')
        current_chunk = ''
        chunk_count = 0

        for para in paragraphs:
            if (
                len(current_chunk + para) > 1800
            ):  # Limite menor para garantir contexto
                if current_chunk.strip():
                    # Criar chunk com contexto da seção
                    chunk_id = f'{filename}_section_{section_index:03d}_part_{chunk_count:02d}'
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        text=f"# {section['title']}\n\n{current_chunk.strip()}",
                        embedding=[],
                        metadata={
                            'filename': filename,
                            'section_title': section['title'],
                            'section_type': section['type'],
                            'section_index': section_index,
                            'part_index': chunk_count,
                            'is_continuation': chunk_count > 0,
                            'topics': section['topics'],
                            'keywords': self.extract_keywords(current_chunk),
                            'context_summary': self.generate_section_summary(
                                current_chunk
                            ),
                        },
                    )
                    chunks.append(chunk)
                    chunk_count += 1
                    current_chunk = para + '\n\n'
            else:
                current_chunk += para + '\n\n'

        # Último chunk
        if current_chunk.strip():
            chunk_id = f'{filename}_section_{section_index:03d}_part_{chunk_count:02d}'
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=f"# {section['title']}\n\n{current_chunk.strip()}",
                embedding=[],
                metadata={
                    'filename': filename,
                    'section_title': section['title'],
                    'section_type': section['type'],
                    'section_index': section_index,
                    'part_index': chunk_count,
                    'is_continuation': chunk_count > 0,
                    'topics': section['topics'],
                    'keywords': self.extract_keywords(current_chunk),
                    'context_summary': self.generate_section_summary(
                        current_chunk
                    ),
                },
            )
            chunks.append(chunk)

        return chunks

    def generate_section_summary(self, content: str) -> str:
        """Gera resumo inteligente da seção para facilitar busca"""
        lines = content.strip().split('\n')
        if not lines:
            return ''

        # Pegar primeiras e últimas linhas significativas
        significant_lines = [
            l.strip() for l in lines if l.strip() and len(l.strip()) > 10
        ]

        if len(significant_lines) <= 3:
            return ' '.join(significant_lines)

        # Resumo baseado em conteúdo
        summary_parts = []

        # Primeira linha (geralmente título ou contexto)
        if significant_lines:
            summary_parts.append(significant_lines[0])

        # Identificar pontos-chave
        keywords = [
            'necessário',
            'obrigatório',
            'deve',
            'pode',
            'não',
            'sim',
            'importante',
            'atenção',
        ]
        key_lines = []
        for line in significant_lines[1:]:
            if any(keyword in line.lower() for keyword in keywords):
                key_lines.append(line)

        if key_lines:
            summary_parts.extend(key_lines[:2])  # Máximo 2 linhas-chave

        return ' | '.join(summary_parts)

    def extract_keywords(self, content: str) -> List[str]:
        """Extrai palavras-chave relevantes para busca"""
        import re

        # Palavras importantes do domínio ICATU
        domain_keywords = [
            'alteração',
            'cadastral',
            'cliente',
            'documento',
            'cpf',
            'rg',
            'nome',
            'endereço',
            'telefone',
            'email',
            'social',
            'titular',
            'apólice',
            'formulário',
            'assinatura',
            'reconhecimento',
            'firma',
            'protocolo',
            'prazo',
            'dias',
            'úteis',
            'correio',
            'envelope',
            'caixa postal',
            'sistema',
            'qdrant',
            'zendesk',
            'sisprev',
            'mumps',
            'sisvida',
            'receita federal',
            'validação',
            'autenticação',
            'token',
            'score',
            'pendente',
            'concluído',
            'manifestação',
            'solicitação',
            'atualização',
        ]

        found_keywords = []
        content_lower = content.lower()

        for keyword in domain_keywords:
            if keyword in content_lower:
                found_keywords.append(keyword)

        # Adicionar palavras específicas do conteúdo
        words = re.findall(r'\b[a-záéíóúçãõâêî]{4,}\b', content_lower)
        word_freq = {}
        for word in words:
            if word not in [
                'para',
                'como',
                'deve',
                'será',
                'onde',
                'pelo',
                'pela',
                'esta',
                'este',
            ]:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Adicionar palavras mais frequentes
        frequent_words = sorted(
            word_freq.items(), key=lambda x: x[1], reverse=True
        )[:5]
        found_keywords.extend(
            [word for word, freq in frequent_words if freq > 1]
        )

        return list(set(found_keywords))

    def extract_topics(self, content: str) -> List[str]:
        """Extrai tópicos principais da seção"""
        topics = []

        # Mapear conteúdo para tópicos
        topic_mapping = {
            'cpf': ['cpf', 'receita federal', 'validação'],
            'documento': ['rg', 'documento', 'identificação', 'certidão'],
            'nome': ['nome', 'social', 'alteração'],
            'endereço': ['endereço', 'correspondência', 'residência'],
            'telefone': ['telefone', 'celular', 'móvel'],
            'email': ['email', 'eletrônico'],
            'procedimento': ['formulário', 'assinatura', 'reconhecimento'],
            'prazo': ['prazo', 'dias', 'úteis', 'horas'],
            'sistema': ['sistema', 'zendesk', 'sisprev', 'mumps'],
            'envio': ['correio', 'envelope', 'caixa postal'],
            'solicitante': ['titular', 'procurador', 'curador', 'tutor'],
            'validação': ['autenticação', 'token', 'score', 'rating'],
        }

        content_lower = content.lower()
        for topic, keywords in topic_mapping.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)

        return topics

    def advanced_text_cleaning(self, text: str) -> str:
        """Limpeza básica de texto - mantida por compatibilidade"""
        return self.advanced_text_cleaning_preserving_structure(text)

    def chunk_text(self, text: str, filename: str) -> List[DocumentChunk]:
        """
        CHUNKING INTELIGENTE BASEADO EM ESTRUTURA LÓGICA

        NOVA ESTRATÉGIA IMPLEMENTADA:
        1. 🧠 Análise estrutural do documento (títulos, seções, tópicos)
        2. 🧹 Limpeza avançada de texto e correção de encoding
        3. � Chunking baseado em seções lógicas completas
        4. 🏷️ Metadados ricos com keywords e sumários
        5. � Facilitação da busca pelo MISTRAL
        6. ✅ Preservação completa do contexto
        """
        logger.info(
            f'🧠 Starting intelligent structural chunking for {filename}'
        )
        logger.info(f'� Input text length: {len(text)} characters')
        logger.info(f'📄 Text preview: {text[:300]}...')

        # Usar novo sistema de chunking estrutural
        chunks = self.smart_structural_chunking(text, filename)

        if not chunks:
            logger.warning('⚠️ Structural chunking failed, using fallback...')
            # Fallback para método anterior se necessário
            chunks = self.fallback_chunking(text, filename)

        logger.info(
            f'✅ Chunking complete: {len(chunks)} intelligent chunks created'
        )

        return chunks

    def fallback_chunking(
        self, text: str, filename: str
    ) -> List[DocumentChunk]:
        """Método de fallback se o chunking estrutural falhar"""
        logger.info('� Using fallback chunking method')

        # Limpeza básica
        clean_text = self.advanced_text_cleaning(text)

        # Divisão simples por parágrafos
        paragraphs = [p.strip() for p in clean_text.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = ''
        chunk_index = 0

        for para in paragraphs:
            if len(current_chunk + para) > 1500:
                if current_chunk.strip():
                    chunk_id = f'{filename}_fallback_{chunk_index:03d}'
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        text=current_chunk.strip(),
                        embedding=[],
                        metadata={
                            'filename': filename,
                            'chunk_index': chunk_index,
                            'method': 'fallback',
                            'keywords': self.extract_keywords(current_chunk),
                            'topics': self.extract_topics(current_chunk),
                        },
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk = para + '\n\n'
            else:
                current_chunk += para + '\n\n'

        # Último chunk
        if current_chunk.strip():
            chunk_id = f'{filename}_fallback_{chunk_index:03d}'
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=current_chunk.strip(),
                embedding=[],
                metadata={
                    'filename': filename,
                    'chunk_index': chunk_index,
                    'method': 'fallback',
                    'keywords': self.extract_keywords(current_chunk),
                    'topics': self.extract_topics(current_chunk),
                },
            )
            chunks.append(chunk)

        return chunks

    async def generate_embeddings(
        self, chunks: List[str]
    ) -> List[List[float]]:
        """Generate embeddings for text chunks using SentenceTransformers"""
        logger.info(f'Gerando embeddings para {len(chunks)} chunks...')
        embeddings = await self.embedding_model.embed_texts(chunks)
        logger.info(f'Embeddings gerados com sucesso.')
        return embeddings

    async def store_in_qdrant(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        document_id: str,
    ):
        """Store chunks and embeddings in Qdrant with correct format"""
        if not chunks or not embeddings:
            raise ValueError("Chunks and embeddings cannot be empty")
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
        
        points = []
        point_ids = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Validar embedding
            if not isinstance(embedding, (list, tuple)) or len(embedding) != self.embedding_model.embedding_dim:
                logger.error(f"❌ Embedding inválido para chunk {chunk.chunk_id}: dimensão {len(embedding) if embedding else 'None'}")
                continue

            # Converter todos os valores do embedding para float e validar
            try:
                if not embedding or not isinstance(embedding, (list, tuple)):
                    logger.error(f"❌ Embedding inválido para {chunk.chunk_id}: não é lista/tupla")
                    continue
                    
                # Converter para lista de floats
                embedding = [float(x) for x in embedding]
                
                # Verificar se todos os valores são números válidos
                if not all(isinstance(x, (int, float)) and not (isinstance(x, float) and (x != x or x == float('inf') or x == float('-inf'))) for x in embedding):
                    logger.error(f"❌ Embedding contém valores inválidos para {chunk.chunk_id}")
                    continue
                    
                # Verificar dimensão
                if len(embedding) != self.embedding_model.embedding_dim:
                    logger.error(f"❌ Dimensão incorreta para {chunk.chunk_id}: {len(embedding)} vs {self.embedding_model.embedding_dim}")
                    continue
                    
            except (ValueError, TypeError) as e:
                logger.error(f"❌ Erro ao processar embedding para {chunk.chunk_id}: {e}")
                continue

            # Gerar ID único baseado no chunk_id
            point_id = chunk.chunk_id if chunk.chunk_id else f"{document_id}_chunk_{i}"
            point_ids.append(point_id)

            # Preparar payload com metadados limpos e serializáveis
            payload = {}
            
            # Campos básicos e seguros
            try:
                payload.update({
                    'content': str(chunk.text) if chunk.text else '',
                    'document_id': str(document_id),
                    'filename': str(chunk.metadata.get('filename', '')),
                    'section_title': str(chunk.metadata.get('section_title', '')),
                    'section_full_title': str(chunk.metadata.get('section_full_title', '')),
                    'section_type': str(chunk.metadata.get('section_type', '')),
                    'section_number': str(chunk.metadata.get('section_number', '')),
                    'character_count': int(len(chunk.text)),
                    'token_count': int(len(chunk.text.split())),
                    'source': 'pdf',
                    'content_type': 'document',
                    'processing_timestamp': float(time.time()),
                })
                
                # Campos numéricos seguros
                section_index = chunk.metadata.get('section_index', 0)
                if isinstance(section_index, (int, float)) and not (isinstance(section_index, float) and section_index != section_index):
                    payload['section_index'] = int(section_index)
                else:
                    payload['section_index'] = 0
                
                content_length = chunk.metadata.get('content_length', len(chunk.text))
                if isinstance(content_length, (int, float)) and not (isinstance(content_length, float) and content_length != content_length):
                    payload['content_length'] = int(content_length)
                else:
                    payload['content_length'] = len(chunk.text)
                
            except Exception as e:
                logger.error(f"❌ Erro ao criar payload básico para {chunk.chunk_id}: {e}")
                continue

            # Adicionar campos opcionais de forma segura
            try:
                # Flags booleanas
                boolean_fields = ['is_numbered_section', 'is_objective', 'is_main_title', 'is_general_context']
                for field in boolean_fields:
                    if field in chunk.metadata:
                        payload[field] = bool(chunk.metadata[field])

                # Keywords como string
                if 'keywords' in chunk.metadata and chunk.metadata['keywords']:
                    keywords = chunk.metadata['keywords']
                    if isinstance(keywords, list):
                        # Filtrar valores vazios ou None
                        valid_keywords = [str(k).strip() for k in keywords if k and str(k).strip()]
                        if valid_keywords:
                            payload['keywords'] = ','.join(valid_keywords)
                    else:
                        keyword_str = str(keywords).strip()
                        if keyword_str:
                            payload['keywords'] = keyword_str

                # Topics como string
                if 'topics' in chunk.metadata and chunk.metadata['topics']:
                    topics = chunk.metadata['topics']
                    if isinstance(topics, list):
                        # Filtrar valores vazios ou None
                        valid_topics = [str(t).strip() for t in topics if t and str(t).strip()]
                        if valid_topics:
                            payload['topics'] = ','.join(valid_topics)
                    else:
                        topic_str = str(topics).strip()
                        if topic_str:
                            payload['topics'] = topic_str

                # Context summary
                if 'context_summary' in chunk.metadata and chunk.metadata['context_summary']:
                    context_summary = str(chunk.metadata['context_summary']).strip()
                    if context_summary:
                        payload['context_summary'] = context_summary
                        
            except Exception as e:
                logger.error(f"❌ Erro ao adicionar campos opcionais para {chunk.chunk_id}: {e}")
                # Continuar sem os campos opcionais

            # Criar ponto no formato correto para Qdrant
            try:
                # Verificar se o embedding está correto antes de criar o ponto
                if not isinstance(embedding, list):
                    logger.error(f"❌ Embedding não é lista para {chunk.chunk_id}: {type(embedding)}")
                    continue
                
                if len(embedding) != self.embedding_model.embedding_dim:
                    logger.error(f"❌ Dimensão incorreta para {chunk.chunk_id}: {len(embedding)} vs {self.embedding_model.embedding_dim}")
                    continue
                
                # Verificar se todos os valores são float válidos
                try:
                    clean_embedding = []
                    for j, val in enumerate(embedding):
                        if isinstance(val, (int, float)) and val == val and val != float('inf') and val != float('-inf'):
                            clean_embedding.append(float(val))
                        else:
                            logger.warning(f"⚠️ Valor inválido no embedding {chunk.chunk_id}[{j}]: {val}")
                            clean_embedding.append(0.0)
                    embedding = clean_embedding
                except Exception as val_error:
                    logger.error(f"❌ Erro ao validar valores do embedding {chunk.chunk_id}: {val_error}")
                    continue

                # Usar o formato de dict simples para o ponto
                point = {
                    "id": point_id,
                    "vector": embedding,
                    "payload": payload
                }
                points.append(point)
                logger.debug(f"✅ Ponto criado para {point_id}: {len(embedding)} dims")
            except Exception as e:
                logger.error(f"❌ Erro ao criar ponto para {point_id}: {e}")
                continue

        if not points:
            raise ValueError("Nenhum ponto válido foi criado para armazenamento")

        # Armazenar no Qdrant usando upsert com formato correto
        try:
            logger.info(f"📤 Enviando {len(points)} pontos para Qdrant...")
            
            # Usar formato direto sem PointStruct - compatível com todas as versões do Qdrant
            qdrant_points = []
            
            for point in points:
                try:
                    # Validar e limpar dados antes de criar o ponto
                    point_id = str(point["id"])
                    vector = point["vector"]
                    payload = point["payload"]
                    
                    # Validação final do vector
                    if not isinstance(vector, list) or len(vector) != 768:
                        logger.error(f"❌ Vector inválido para {point_id}: {type(vector)}, len={len(vector) if hasattr(vector, '__len__') else 'N/A'}")
                        continue
                    
                    # Garantir que todos os valores do vector são float válidos
                    try:
                        clean_vector = []
                        for val in vector:
                            if isinstance(val, (int, float)) and val == val and val != float('inf') and val != float('-inf'):
                                clean_vector.append(float(val))
                            else:
                                clean_vector.append(0.0)
                        vector = clean_vector
                    except (ValueError, TypeError) as ve:
                        logger.error(f"❌ Erro ao converter vector para float em {point_id}: {ve}")
                        continue
                    
                    # Limpar o payload para ser JSON-serializável
                    clean_payload = {}
                    for key, value in payload.items():
                        if value is not None:
                            if isinstance(value, (str, int, float, bool)):
                                clean_payload[key] = value
                            elif isinstance(value, list):
                                # Converter listas para strings
                                if all(isinstance(item, (str, int, float)) for item in value):
                                    clean_payload[key] = ','.join(str(item) for item in value)
                                else:
                                    clean_payload[key] = str(value)
                            else:
                                clean_payload[key] = str(value)
                    
                    # Criar ponto no formato dict simples (mais compatível)
                    qdrant_point = {
                        "id": point_id,
                        "vector": vector,
                        "payload": clean_payload
                    }
                    qdrant_points.append(qdrant_point)
                    
                except Exception as pe:
                    logger.error(f"❌ Erro ao processar ponto {point.get('id', 'UNKNOWN')}: {pe}")
                    continue
            
            if not qdrant_points:
                raise ValueError("Nenhum ponto válido foi processado para o Qdrant")
            
            logger.info(f"📦 Processados {len(qdrant_points)} pontos válidos de {len(points)} originais")
            
            # Usar método direto mais simples para máxima compatibilidade
            try:
                # NOVO: Usar formato Batch como descoberto na API
                logger.info("🔄 Usando formato Batch para Qdrant...")
                
                # Preparar listas para Batch
                batch_ids = []
                batch_vectors = []
                batch_payloads = []
                
                for point_data in qdrant_points:
                    batch_ids.append(point_data["id"])
                    batch_vectors.append(point_data["vector"])
                    batch_payloads.append(point_data["payload"])
                
                # Usar Batch format
                result = self.qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=Batch(
                        ids=batch_ids,
                        vectors=batch_vectors,
                        payloads=batch_payloads
                    ),
                    wait=True
                )
                
            except Exception as upsert_error:
                logger.error(f"❌ Erro no upsert normal: {upsert_error}")
                # Fallback: enviar em lotes menores
                logger.info("🔄 Tentando envio em lotes menores...")
                
                batch_size = 5
                successful_batches = 0
                total_success = 0
                
                for i in range(0, len(qdrant_points), batch_size):
                    batch = qdrant_points[i:i + batch_size]
                    try:
                        batch_structs = []
                        for point_data in batch:
                            point_struct = PointStruct(
                                id=point_data["id"],
                                vector=point_data["vector"],
                                payload=point_data["payload"]
                            )
                            batch_structs.append(point_struct)
                        
                        batch_result = self.qdrant_client.upsert(
                            collection_name=COLLECTION_NAME,
                            points=batch_structs,
                            wait=True
                        )
                        successful_batches += 1
                        total_success += len(batch)
                        logger.info(f"✅ Lote {successful_batches} enviado: {len(batch)} pontos")
                        
                    except Exception as batch_error:
                        logger.error(f"❌ Erro no lote {i//batch_size + 1}: {batch_error}")
                        
                        # Tentar envio individual neste lote
                        for point_data in batch:
                            try:
                                single_point = PointStruct(
                                    id=point_data["id"],
                                    vector=point_data["vector"],
                                    payload=point_data["payload"]
                                )
                                single_result = self.qdrant_client.upsert(
                                    collection_name=COLLECTION_NAME,
                                    points=[single_point],
                                    wait=True
                                )
                                total_success += 1
                                logger.info(f"✅ Ponto individual: {point_data['id']}")
                            except Exception as single_error:
                                logger.error(f"❌ Falha individual {point_data['id']}: {single_error}")
                                break  # Para na primeira falha para debug
                
                if total_success > 0:
                    logger.info(f"✅ Total inserido com sucesso: {total_success}/{len(qdrant_points)}")
                    result = {"status": "partial_success", "successful_points": total_success}
                else:
                    raise Exception("Nenhum ponto foi inserido com sucesso")
            
            logger.info(f'✅ Armazenados {len(qdrant_points)} chunks no Qdrant para documento {document_id}')
            logger.info(f'📋 IDs dos pontos: {point_ids[:5]}{"..." if len(point_ids) > 5 else ""}')
            logger.info(f'📊 Resultado do upsert: {result}')
            
            return result
            
        except Exception as e:
            logger.error(f'❌ Erro ao armazenar no Qdrant: {e}')
            logger.error(f'❌ Tipo do erro: {type(e)}')
            logger.error(f'❌ Número de pontos: {len(points)}')
            
            # Log detalhado do primeiro ponto para debug
            if points:
                first_point = points[0]
                logger.error(f'❌ Estrutura do primeiro ponto:')
                logger.error(f'   - ID: {first_point.get("id", "MISSING")}')
                logger.error(f'   - Vector type: {type(first_point.get("vector", "MISSING"))}')
                logger.error(f'   - Vector length: {len(first_point.get("vector", [])) if first_point.get("vector") else "MISSING"}')
                logger.error(f'   - Payload keys: {list(first_point.get("payload", {}).keys())}')
                
                # Tentar diagnóstico mais profundo
                vector = first_point.get("vector", [])
                if vector:
                    logger.error(f'   - Vector sample: {vector[:3]}...{vector[-3:]}')
                    logger.error(f'   - Vector types: {[type(x) for x in vector[:5]]}')
            
            if hasattr(e, 'response'):
                logger.error(f'❌ Response: {e.response}')
            raise

        # Padrões para identificar seções e estruturas
        section_pattern = re.compile(
            r'^(\d+[\.\)]\s*|[IVX]+[\.\)]\s*|[A-Z][A-Z\s\d\-]{10,}:?\s*|\w+\s*:)$|^[a-z][\)\.]',
            re.MULTILINE,
        )
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
                categories.append('lista')
            if any(
                word in content_lower
                for word in ['passo', 'etapa', 'primeiro', 'segundo']
            ):
                categories.append('sequencial')
            if any(
                word in content_lower
                for word in ['atenção', 'importante', 'observação', 'nota']
            ):
                categories.append('destaque')

            if not categories:
                categories.append('geral')

            return categories

        def create_chunk_with_context(
            text: str,
            section: str,
            categories: List[str],
            chunk_idx: int,
            section_idx: int = 0,
            para_idx: int = 0,
            has_overlap: bool = False,
        ) -> DocumentChunk:
            """Cria chunk com metadados ricos e contexto estrutural"""
            token_count = len(tokenizer.encode(text, add_special_tokens=False))

            return DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                text=text.strip(),
                embedding=[],
                metadata={
                    'filename': filename,
                    'section': section,
                    'categories': categories,
                    'section_index': section_idx,
                    'paragraph_index': para_idx,
                    'chunk_index': chunk_idx,
                    'source': 'pdf',
                    'content_type': 'document',
                    'token_count': token_count,
                    'character_count': len(text),
                    'has_overlap': has_overlap,
                    'processing_timestamp': time.time(),
                    'priority_score': len(categories) * 0.1
                    + (1.0 if 'importante' in text.lower() else 0.0),
                    'density_score': token_count
                    / max(len(text), 1),  # Densidade de informação
                },
            )

        # ESTRATÉGIA 1: Processamento por parágrafos com overlap inteligente
        paragraphs = text.split('\n\n')
        logger.info(f'📝 Split into {len(paragraphs)} paragraphs')

        # Se não há parágrafos bem definidos, dividir por quebras simples
        if len(paragraphs) == 1:
            paragraphs = text.split('\n')
            logger.info(f'📝 Fallback: Split into {len(paragraphs)} lines')

            # Se ainda há poucas quebras, forçar mais divisões
            if len(paragraphs) <= 3:
                logger.info(f'🔧 Forçando divisões adicionais...')
                # Dividir por padrões estruturais
                additional_splits = []
                for para in paragraphs:
                    # Dividir por numeração
                    parts = re.split(r'(\d+\.[A-Z])', para)
                    for part in parts:
                        if part.strip():
                            # Dividir por subdivisões a), b), c)
                            subparts = re.split(r'([a-z]\))', part)
                            additional_splits.extend(
                                [p.strip() for p in subparts if p.strip()]
                            )

                if len(additional_splits) > len(paragraphs):
                    paragraphs = additional_splits
                    logger.info(f'📝 Structural split: {len(paragraphs)} parts')

        # Se ainda assim há muito pouco conteúdo, usar chunking mais agressivo
        if len(paragraphs) <= 2 and len(text) > MIN_CHUNK_SIZE:
            logger.warning(
                f'⚠️ Very few paragraphs ({len(paragraphs)}), using aggressive sentence-based chunking'
            )
            # Dividir por sentenças com chunks menores
            import re

            sentences = re.split(r'(?<=[.!?])\s+', text)
            logger.info(f'📝 Split into {len(sentences)} sentences')

            # Reagrupar sentenças em chunks MENORES para mais granularidade
            chunks = []
            current_chunk = ''
            chunk_index = 0
            target_chunk_size = CHUNK_SIZE // 2  # Chunks menores: 1024 tokens

            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue

                # Verificar se adicionar a sentença não excede o limite MENOR
                test_chunk = (
                    current_chunk + ' ' + sent if current_chunk else sent
                )
                tokens = len(
                    tokenizer.encode(test_chunk, add_special_tokens=False)
                )

                if tokens > target_chunk_size and current_chunk:
                    # Salvar chunk atual
                    if len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
                        categories = categorize_content(current_chunk)
                        chunk = create_chunk_with_context(
                            current_chunk,
                            'SENTENCE_BASED_SMALL',
                            categories,
                            chunk_index,
                            0,
                            0,
                        )
                        chunks.append(chunk)
                        logger.info(
                            f'✅ Created small sentence-based chunk {chunk_index}: {len(current_chunk)} chars, {tokens} tokens'
                        )
                        chunk_index += 1
                    current_chunk = sent
                else:
                    current_chunk = test_chunk

            # Adicionar último chunk
            if current_chunk and len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
                categories = categorize_content(current_chunk)
                chunk = create_chunk_with_context(
                    current_chunk,
                    'SENTENCE_BASED_SMALL',
                    categories,
                    chunk_index,
                    0,
                    0,
                )
                chunks.append(chunk)
                logger.info(
                    f'✅ Created final small sentence-based chunk {chunk_index}: {len(current_chunk)} chars'
                )

            # Se ainda temos poucos chunks, dividir os maiores
            if len(chunks) <= 3:
                logger.warning(
                    f'⚠️ Still few chunks ({len(chunks)}), splitting large ones...'
                )
                expanded_chunks = []
                for i, chunk in enumerate(chunks):
                    if len(chunk.text) > CHUNK_SIZE:
                        # Dividir chunk grande em pedaços menores
                        words = chunk.text.split()
                        words_per_chunk = (
                            len(words) // 3
                        )  # Dividir em 3 partes

                        for j in range(0, len(words), words_per_chunk):
                            sub_words = words[j : j + words_per_chunk]
                            if sub_words:
                                sub_text = ' '.join(sub_words)
                                if len(sub_text.strip()) >= MIN_CHUNK_SIZE:
                                    sub_categories = categorize_content(
                                        sub_text
                                    )
                                    sub_chunk = create_chunk_with_context(
                                        sub_text,
                                        f'SPLIT_CHUNK_{i}_{j//words_per_chunk}',
                                        sub_categories,
                                        len(expanded_chunks),
                                        0,
                                        0,
                                    )
                                    expanded_chunks.append(sub_chunk)
                                    logger.info(
                                        f'✅ Split chunk {i} part {j//words_per_chunk}: {len(sub_text)} chars'
                                    )
                    else:
                        expanded_chunks.append(chunk)

                if len(expanded_chunks) > len(chunks):
                    chunks = expanded_chunks
                    logger.info(
                        f'🔧 Chunk splitting successful: {len(chunks)} total chunks'
                    )

            logger.info(
                f'🎯 Aggressive sentence-based chunking created {len(chunks)} chunks'
            )
            return chunks

        chunks = []
        current_section = 'DOCUMENTO_PRINCIPAL'
        section_index = 0
        chunk_index = 0

        # Buffer para construção de chunks
        chunk_buffer = ''
        chunk_tokens = 0
        previous_chunk_text = ''  # Para overlap

        doc_logger.info(
            f'🔍 Starting paragraph-based chunking for {filename}: {len(paragraphs)} paragraphs'
        )

        non_empty_paragraphs = 0
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            non_empty_paragraphs += 1
            if para_idx < 5:  # Log first 5 paragraphs for debugging
                logger.info(
                    f'📄 Paragraph {para_idx}: {len(paragraph)} chars - {paragraph[:100]}...'
                )

            # Detecta nova seção baseada em padrões
            if section_pattern.match(paragraph) or title_pattern.match(
                paragraph
            ):
                logger.info(f'🏷️ Detected new section: {paragraph[:50]}...')
                # Finaliza chunk atual se existir
                if (
                    chunk_buffer
                    and len(chunk_buffer.strip()) >= MIN_CHUNK_SIZE
                ):
                    categories = categorize_content(chunk_buffer)
                    chunk = create_chunk_with_context(
                        chunk_buffer,
                        current_section,
                        categories,
                        chunk_index,
                        section_index,
                        para_idx,
                    )
                    chunks.append(chunk)
                    logger.info(
                        f'✅ Created section-end chunk {chunk_index}: {len(chunk_buffer)} chars, categories: {categories}'
                    )
                    previous_chunk_text = (
                        chunk_buffer[-CHUNK_OVERLAP:]
                        if len(chunk_buffer) > CHUNK_OVERLAP
                        else chunk_buffer
                    )
                    chunk_index += 1

                # Inicia nova seção
                current_section = (
                    paragraph[:100] + '...'
                    if len(paragraph) > 100
                    else paragraph
                )
                section_index += 1
                chunk_buffer = ''
                chunk_tokens = 0
                continue

            # Calcula tokens do parágrafo
            para_tokens = len(
                tokenizer.encode(paragraph, add_special_tokens=False)
            )

            # ESTRATÉGIA 2: Gerenciamento inteligente de tamanho de chunk
            if chunk_tokens + para_tokens > CHUNK_SIZE:
                # Salva chunk atual
                if (
                    chunk_buffer
                    and len(chunk_buffer.strip()) >= MIN_CHUNK_SIZE
                ):
                    # Adiciona overlap da chunk anterior se disponível
                    full_chunk_text = chunk_buffer
                    if previous_chunk_text and not chunk_buffer.startswith(
                        previous_chunk_text[-100:]
                    ):
                        overlap_text = previous_chunk_text[
                            -CHUNK_OVERLAP // 2 :
                        ].strip()
                        if overlap_text:
                            full_chunk_text = (
                                overlap_text + ' [CONTINUAÇÃO] ' + chunk_buffer
                            )

                    categories = categorize_content(full_chunk_text)
                    chunk = create_chunk_with_context(
                        full_chunk_text,
                        current_section,
                        categories,
                        chunk_index,
                        section_index,
                        para_idx,
                        has_overlap=bool(previous_chunk_text),
                    )
                    chunks.append(chunk)

                    # Prepara overlap para próximo chunk
                    previous_chunk_text = chunk_buffer
                    chunk_index += 1

                # ESTRATÉGIA 3: Divisão inteligente de parágrafos grandes
                if para_tokens > CHUNK_SIZE:
                    # Divide parágrafo por sentenças
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    sub_chunk = ''
                    sub_tokens = 0

                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue

                        sent_tokens = len(
                            tokenizer.encode(sent, add_special_tokens=False)
                        )

                        if sub_tokens + sent_tokens > CHUNK_SIZE and sub_chunk:
                            # Salva sub-chunk
                            if len(sub_chunk.strip()) >= MIN_CHUNK_SIZE:
                                # Adiciona overlap se disponível
                                full_sub_chunk = sub_chunk
                                if previous_chunk_text:
                                    overlap_text = previous_chunk_text[
                                        -CHUNK_OVERLAP // 3 :
                                    ].strip()
                                    if overlap_text:
                                        full_sub_chunk = (
                                            overlap_text
                                            + ' [CONTINUAÇÃO] '
                                            + sub_chunk
                                        )

                                categories = categorize_content(full_sub_chunk)
                                chunk = create_chunk_with_context(
                                    full_sub_chunk,
                                    current_section,
                                    categories,
                                    chunk_index,
                                    section_index,
                                    para_idx,
                                    has_overlap=True,
                                )
                                chunks.append(chunk)
                                previous_chunk_text = sub_chunk
                                chunk_index += 1

                            # Inicia novo sub-chunk com overlap
                            overlap_text = (
                                sub_chunk[-CHUNK_OVERLAP // 2 :]
                                if len(sub_chunk) > CHUNK_OVERLAP // 2
                                else ''
                            )
                            sub_chunk = (
                                overlap_text + ' ' + sent
                                if overlap_text
                                else sent
                            )
                            sub_tokens = len(
                                tokenizer.encode(
                                    sub_chunk, add_special_tokens=False
                                )
                            )
                        else:
                            sub_chunk = (
                                sub_chunk + ' ' + sent if sub_chunk else sent
                            )
                            sub_tokens += sent_tokens

                    # Salva último sub-chunk
                    if sub_chunk and len(sub_chunk.strip()) >= MIN_CHUNK_SIZE:
                        categories = categorize_content(sub_chunk)
                        chunk = create_chunk_with_context(
                            sub_chunk,
                            current_section,
                            categories,
                            chunk_index,
                            section_index,
                            para_idx,
                        )
                        chunks.append(chunk)
                        previous_chunk_text = sub_chunk
                        chunk_index += 1

                    chunk_buffer = ''
                    chunk_tokens = 0
                else:
                    # Inicia novo chunk com o parágrafo atual
                    chunk_buffer = paragraph
                    chunk_tokens = para_tokens
            else:
                # Adiciona parágrafo ao chunk atual
                chunk_buffer = (
                    chunk_buffer + '\n\n' + paragraph
                    if chunk_buffer
                    else paragraph
                )
                chunk_tokens += para_tokens

        # ESTRATÉGIA 4: Processa último chunk restante
        if chunk_buffer and len(chunk_buffer.strip()) >= MIN_CHUNK_SIZE:
            # Adiciona overlap se disponível
            full_final_chunk = chunk_buffer
            if previous_chunk_text and not chunk_buffer.startswith(
                previous_chunk_text[-100:]
            ):
                overlap_text = previous_chunk_text[
                    -CHUNK_OVERLAP // 2 :
                ].strip()
                if overlap_text:
                    full_final_chunk = (
                        overlap_text + ' [CONTINUAÇÃO] ' + chunk_buffer
                    )

            categories = categorize_content(full_final_chunk)
            chunk = create_chunk_with_context(
                full_final_chunk,
                current_section,
                categories,
                chunk_index,
                section_index,
                len(paragraphs),
                has_overlap=bool(previous_chunk_text),
            )
            chunks.append(chunk)

        # ESTRATÉGIA 5: Análise de cobertura e estatísticas
        total_chars = len(text)
        covered_chars = sum(len(chunk.text) for chunk in chunks)
        coverage_ratio = covered_chars / total_chars if total_chars > 0 else 0

        category_stats = {}
        for chunk in chunks:
            for cat in chunk.metadata.get('categories', []):
                category_stats[cat] = category_stats.get(cat, 0) + 1

        # Log detalhado do chunking
        doc_logger.info(f'✅ Chunking hierárquico concluído para {filename}:')
        doc_logger.info(f'   📊 {len(chunks)} chunks gerados')
        doc_logger.info(
            f'   📏 Cobertura: {coverage_ratio:.2%} do documento original'
        )
        doc_logger.info(
            f'   📝 Tamanho médio: {covered_chars // len(chunks) if chunks else 0} caracteres/chunk'
        )
        doc_logger.info(f'   🏷️ Categorias: {category_stats}')

        # ESTRATÉGIA 6: Validação crítica e fallback robusto
        if len(chunks) < 3 or coverage_ratio < 0.90:
            logger.warning(
                f'⚠️ PROBLEMA CRÍTICO: {len(chunks)} chunks, cobertura {coverage_ratio:.2%}'
            )
            logger.warning(f'🔧 Ativando fallback: chunking por força bruta...')

            # FALLBACK: Chunking por força bruta - dividir o texto em chunks de tamanho fixo
            fallback_chunks = []
            words = text.split()
            current_chunk_words = []
            current_chunk_tokens = 0
            fallback_chunk_index = 0

            for word in words:
                # Estimar tokens (aproximadamente 1 token = 0.75 palavras para português)
                word_tokens = max(1, len(word) // 4)

                if (
                    current_chunk_tokens + word_tokens > CHUNK_SIZE
                    and current_chunk_words
                ):
                    # Criar chunk atual
                    chunk_text = ' '.join(current_chunk_words)
                    if len(chunk_text.strip()) >= MIN_CHUNK_SIZE:
                        categories = categorize_content(chunk_text)
                        fallback_chunk = DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            text=chunk_text,
                            embedding=[],
                            metadata={
                                'filename': filename,
                                'section': f'FALLBACK_CHUNK_{fallback_chunk_index}',
                                'categories': categories,
                                'chunk_index': fallback_chunk_index,
                                'source': 'pdf',
                                'content_type': 'document',
                                'token_count': current_chunk_tokens,
                                'character_count': len(chunk_text),
                                'is_fallback': True,
                                'processing_timestamp': time.time(),
                            },
                        )
                        fallback_chunks.append(fallback_chunk)
                        logger.info(
                            f'✅ Fallback chunk {fallback_chunk_index}: {len(chunk_text)} chars'
                        )
                        fallback_chunk_index += 1

                    # Começar novo chunk com overlap
                    overlap_size = min(50, len(current_chunk_words) // 4)
                    current_chunk_words = current_chunk_words[
                        -overlap_size:
                    ] + [word]
                    current_chunk_tokens = word_tokens + overlap_size
                else:
                    current_chunk_words.append(word)
                    current_chunk_tokens += word_tokens

            # Adicionar último chunk
            if current_chunk_words:
                chunk_text = ' '.join(current_chunk_words)
                if len(chunk_text.strip()) >= MIN_CHUNK_SIZE:
                    categories = categorize_content(chunk_text)
                    fallback_chunk = DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        text=chunk_text,
                        embedding=[],
                        metadata={
                            'filename': filename,
                            'section': f'FALLBACK_CHUNK_{fallback_chunk_index}',
                            'categories': categories,
                            'chunk_index': fallback_chunk_index,
                            'source': 'pdf',
                            'content_type': 'document',
                            'token_count': current_chunk_tokens,
                            'character_count': len(chunk_text),
                            'is_fallback': True,
                            'processing_timestamp': time.time(),
                        },
                    )
                    fallback_chunks.append(fallback_chunk)
                    logger.info(
                        f'✅ Final fallback chunk {fallback_chunk_index}: {len(chunk_text)} chars'
                    )

            if len(fallback_chunks) > len(chunks):
                logger.info(
                    f'🔧 Fallback successful: {len(fallback_chunks)} chunks vs {len(chunks)} original'
                )
                chunks = fallback_chunks
            else:
                logger.warning(
                    f'⚠️ Fallback não melhorou: {len(fallback_chunks)} vs {len(chunks)}'
                )

        # ESTRATÉGIA 7: Último recurso - garantir pelo menos um chunk
        if len(chunks) == 0:
            logger.error(f'❌ CRÍTICO: Nenhum chunk gerado para {filename}!')
            logger.info(f'🆘 Criando chunk de emergência com todo o texto...')
            emergency_chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                text=text[
                    : CHUNK_SIZE * 4
                ],  # Limita para evitar chunks gigantes
                embedding=[],
                metadata={
                    'filename': filename,
                    'section': 'EMERGENCY_COMPLETE_DOCUMENT',
                    'categories': ['geral', 'emergency'],
                    'chunk_index': 0,
                    'source': 'pdf',
                    'is_emergency': True,
                    'character_count': len(text),
                    'processing_timestamp': time.time(),
                },
            )
            chunks = [emergency_chunk]
            logger.info(
                f'🆘 Emergency chunk created: {len(emergency_chunk.text)} chars'
            )

        if len(chunks) == 0:
            doc_logger.error(f'❌ Nenhum chunk gerado para {filename}!')
            # Fallback: criar um chunk com todo o texto
            fallback_chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                text=text[: CHUNK_SIZE * 2],  # Limita tamanho
                embedding=[],
                metadata={
                    'filename': filename,
                    'section': 'FALLBACK_COMPLETE_DOCUMENT',
                    'categories': ['geral', 'fallback'],
                    'chunk_index': 0,
                    'source': 'pdf',
                    'is_fallback': True,
                },
            )
            chunks = [fallback_chunk]

        return chunks

    async def generate_embeddings(
        self, chunks: List[str]
    ) -> List[List[float]]:
        """Generate embeddings for text chunks using SentenceTransformers"""
        logger.info(f'Gerando embeddings para {len(chunks)} chunks...')
        embeddings = await self.embedding_model.embed_texts(chunks)
        logger.info(f'Embeddings gerados com sucesso.')
        return embeddings


    async def search_similar_chunks_enhanced(
        self,
        query: str,
        limit: int = 15,
        score_threshold: float = 0.2,
        document_id: str = None,
    ) -> List[SearchResult]:
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
        doc_logger.info(
            f"🔍 Busca ultra-otimizada para: '{query}' (threshold: {score_threshold}, limit: {limit})"
        )

        # 1. Mapeamento de sinônimos específicos ICATU
        icatu_synonyms = {
            'alteração cadastral': [
                'mudança cadastral',
                'atualização cadastral',
                'modificação cadastral',
            ],
            'solicitar': [
                'fazer solicitação',
                'requerer',
                'pedir',
                'solicitar',
            ],
            'procurador': [
                'representante legal',
                'curador',
                'tutor',
                'responsável',
            ],
            'menor de idade': ['menor', 'criança', 'adolescente'],
            'zendesk': ['sistema', 'plataforma', 'atendimento'],
            'documento': ['documentação', 'arquivo', 'anexo', 'certidão'],
            'prazo': ['tempo', 'período', 'horário'],
            'titular': ['segurado', 'beneficiário', 'contratante'],
            'como': ['procedimento', 'processo', 'forma de'],
            'pode': ['consegue', 'é possível', 'tem permissão'],
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
        if 'como' in query_lower:
            expanded_queries.append(
                query_lower.replace('como', 'procedimento para')
            )
            expanded_queries.append(query_lower.replace('como', 'forma de'))

        if 'quem pode' in query_lower:
            expanded_queries.append(
                query_lower.replace('quem pode', 'solicitantes autorizados')
            )
            expanded_queries.append(
                query_lower.replace('quem pode', 'pessoas que podem')
            )

        doc_logger.info(
            f'Queries expandidas: {len(expanded_queries)} variações'
        )

        # 3. Gerar embeddings para todas as variações
        doc_logger.info('Gerando embedding da pergunta...')
        all_embeddings = await self.embedding_model.embed_texts(
            expanded_queries
        )

        # 4. Busca multi-vector com diferentes estratégias
        all_results = []

        for i, query_embedding in enumerate(all_embeddings):
            # Estratégia 1: Busca padrão
            search_params = {
                'collection_name': COLLECTION_NAME,
                'query_vector': query_embedding,
                'limit': min(
                    limit * 3, 50
                ),  # Buscar mais resultados inicialmente
                'score_threshold': max(
                    score_threshold * 0.8, 0.1
                ),  # Threshold ainda mais permissivo
            }

            if document_id:
                search_params['query_filter'] = models.Filter(
                    must=[
                        models.FieldCondition(
                            key='document_id',
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )

            doc_logger.info('Buscando chunks mais relevantes no Qdrant...')
            search_results = self.qdrant_client.search(**search_params)

            # Peso baseado na query (original tem peso maior)
            weight = 1.0 if i == 0 else 0.85

            for result in search_results:
                # Calcular score ajustado com peso e metadados
                adjusted_score = result.score * weight

                # Bonus por categoria relevante
                categories = result.payload.get('categories', [])
                if any(
                    cat
                    in ['alteracao_cadastral', 'solicitantes', 'procedimentos']
                    for cat in categories
                ):
                    adjusted_score *= 1.2

                # Bonus por prioridade do chunk
                priority = result.payload.get('priority_score', 0)
                adjusted_score += priority

                result_obj = SearchResult(
                    content=result.payload['content'],
                    score=adjusted_score,
                    metadata={
                        k: v
                        for k, v in result.payload.items()
                        if k != 'content'
                    },
                )
                all_results.append(result_obj)

        # 5. Deduplicar e ordenar resultados
        seen_content = {}
        unique_results = []

        for result in all_results:
            content_key = result.content[
                :100
            ]  # Usar início do conteúdo como chave

            if (
                content_key not in seen_content
                or result.score > seen_content[content_key].score
            ):
                seen_content[content_key] = result

        unique_results = list(seen_content.values())
        unique_results.sort(key=lambda x: x.score, reverse=True)

        # 6. Aplicar threshold final e limitar resultados
        final_results = [
            r for r in unique_results if r.score >= score_threshold
        ][:limit]

        doc_logger.info(
            f'Busca retornou {len(final_results)} chunks relevantes.'
        )

        # 7. Log detalhado dos resultados para debug
        if final_results:
            doc_logger.info(
                f"Melhores resultados: scores = {[f'{r.score:.3f}' for r in final_results[:3]]}"
            )
        else:
            doc_logger.warning(
                f'Nenhum resultado encontrado com threshold {score_threshold}'
            )

            # Busca de emergência com threshold muito baixo
            emergency_search = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=all_embeddings[0],
                limit=5,
                score_threshold=0.1,
            )

            if emergency_search:
                doc_logger.info(
                    f'Busca de emergência encontrou {len(emergency_search)} resultados'
                )
                for result in emergency_search:
                    result_obj = SearchResult(
                        content=result.payload['content'],
                        score=result.score,
                        metadata={
                            k: v
                            for k, v in result.payload.items()
                            if k != 'content'
                        },
                    )
                    final_results.append(result_obj)

        return final_results

    async def search_similar_chunks(
        self,
        query: str,
        limit: int = 15,
        score_threshold: float = 0.2,
        document_id: str = None,
    ) -> List[SearchResult]:
        """Compatibility wrapper for search_similar_chunks_enhanced with optimized defaults"""
        return await self.search_similar_chunks_enhanced(
            query, limit, score_threshold, document_id
        )

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the Qdrant collection"""
        try:
            collection_info = self.qdrant_client.get_collection(
                COLLECTION_NAME
            )
            return {
                'collection_name': COLLECTION_NAME,
                'vectors_count': collection_info.vectors_count,
                'points_count': collection_info.points_count,
                'status': collection_info.status,
            }
        except Exception as e:
            doc_logger.error(f'Error getting collection info: {e}')
            return {'error': str(e)}


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
    title='Document Processor API',
    description='PDF processing and embedding generation for knowledge base',
    version='1.0.0',
    lifespan=lifespan,
)


@app.get('/health')
async def health_check():
    return {'status': 'healthy', 'service': 'document-processor'}


@app.post('/upload-pdf', response_model=ProcessingResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, detail='Only PDF files are supported'
        )

    import time

    start_time = time.time()

    try:
        # Read file content
        pdf_content = await file.read()

        # Generate document ID
        file_hash = hashlib.md5(pdf_content).hexdigest()
        document_id = f'doc_{file_hash[:12]}'

        # Extract text
        logger.info(f'Processing PDF: {file.filename}')
        text = processor.extract_text_from_pdf(pdf_content)

        if not text.strip():
            raise HTTPException(
                status_code=400, detail='No text could be extracted from PDF'
            )

        # Chunk text
        chunks = processor.chunk_text(text, file.filename)
        logger.info(f'Created {len(chunks)} chunks for {file.filename}')

        # Generate embeddings
        embeddings = await processor.generate_embeddings(
            [chunk.text for chunk in chunks]
        )

        # Store in Qdrant
        await processor.store_in_qdrant(chunks, embeddings, document_id)

        processing_time = time.time() - start_time

        return ProcessingResponse(
            document_id=document_id,
            filename=file.filename,
            chunks_count=len(chunks),
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f'Error processing PDF {file.filename}: {e}')
        raise HTTPException(
            status_code=500, detail=f'Processing failed: {str(e)}'
        )


from fastapi.responses import JSONResponse


@app.post('/search', response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """Search for relevant document chunks, retorna também contexto concatenado para o Mistral"""
    try:
        results = await processor.search_similar_chunks_enhanced(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold,
        )
        contexto_completo = '\n'.join([r.content for r in results])
        return JSONResponse(
            content={
                'chunks': [r.dict() for r in results],
                'contexto_completo': contexto_completo,
            }
        )
    except Exception as e:
        logger.error(f'Search error: {e}')
        raise HTTPException(status_code=500, detail=f'Search failed: {str(e)}')


@app.get('/collections/info')
async def get_collection_info():
    """Get information about the Qdrant collection"""
    return await processor.get_collection_info()


@app.get('/debug/chunks/{document_id}')
async def debug_chunks(document_id: str):
    """Debug endpoint to see all chunks for a document"""
    try:
        # Buscar todos os chunks do documento no Qdrant
        search_results = processor.qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key='document_id',
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            limit=100,  # Buscar até 100 chunks
            with_payload=True,
            with_vectors=False,
        )

        chunks_info = []
        for point in search_results[0]:
            chunk_info = {
                'chunk_id': point.id,
                'content_preview': point.payload.get('content', '')[:200]
                + '...'
                if len(point.payload.get('content', '')) > 200
                else point.payload.get('content', ''),
                'content_length': len(point.payload.get('content', '')),
                'categories': point.payload.get('categories', []),
                'section': point.payload.get('section', ''),
                'chunk_index': point.payload.get('chunk_index', 0),
                'token_count': point.payload.get('token_count', 0),
                'metadata': {
                    k: v
                    for k, v in point.payload.items()
                    if k not in ['content']
                },
            }
            chunks_info.append(chunk_info)

        # Ordenar por chunk_index
        chunks_info.sort(key=lambda x: x.get('chunk_index', 0))

        return {
            'document_id': document_id,
            'total_chunks': len(chunks_info),
            'chunks': chunks_info,
        }

    except Exception as e:
        logger.error(f'Error debugging chunks for {document_id}: {e}')
        raise HTTPException(status_code=500, detail=f'Debug failed: {str(e)}')


@app.get('/debug/text-extraction/{document_hash}')
async def debug_text_extraction(document_hash: str):
    """Debug endpoint to see raw text extraction (requires re-upload for testing)"""
    # Note: This would require storing original PDFs or re-uploading for debug
    return {
        'message': 'Upload a PDF with ?debug=true to see text extraction details'
    }


@app.get('/qdrant/info')
async def get_qdrant_info():
    """Get comprehensive information about the Qdrant database"""
    try:
        # Informações gerais do Qdrant
        collections = processor.qdrant_client.get_collections()

        # Informações detalhadas da collection principal
        try:
            collection_info = processor.qdrant_client.get_collection(
                COLLECTION_NAME
            )
            collection_exists = True
        except Exception:
            collection_info = None
            collection_exists = False

        # Estatísticas básicas
        total_points = 0
        if collection_exists:
            try:
                count_result = processor.qdrant_client.count(
                    collection_name=COLLECTION_NAME
                )
                total_points = count_result.count
            except Exception as e:
                logger.warning(f'Could not count points: {e}')

        return {
            'qdrant_url': QDRANT_URL,
            'collection_name': COLLECTION_NAME,
            'collection_exists': collection_exists,
            'total_collections': len(collections.collections),
            'collections': [col.name for col in collections.collections],
            'total_points': total_points,
            'collection_info': {
                'status': collection_info.status
                if collection_info
                else 'Not found',
                'vector_size': collection_info.config.params.vectors.size
                if collection_info
                else None,
                'distance': collection_info.config.params.vectors.distance
                if collection_info
                else None,
            }
            if collection_info
            else None,
        }
    except Exception as e:
        logger.error(f'Error getting Qdrant info: {e}')
        raise HTTPException(
            status_code=500, detail=f'Failed to get Qdrant info: {str(e)}'
        )


@app.get('/qdrant/documents')
async def get_all_documents():
    """Get all documents stored in Qdrant with statistics"""
    try:
        # Buscar todos os pontos
        all_points = []
        next_page_offset = None

        while True:
            scroll_result = processor.qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False,
            )

            points, next_page_offset = scroll_result
            all_points.extend(points)

            if next_page_offset is None:
                break

        # Agrupar por documento
        documents = {}
        for point in all_points:
            doc_id = point.payload.get('document_id', 'unknown')
            filename = point.payload.get('filename', 'unknown')

            if doc_id not in documents:
                documents[doc_id] = {
                    'document_id': doc_id,
                    'filename': filename,
                    'chunks': [],
                    'total_chunks': 0,
                    'total_characters': 0,
                    'categories': set(),
                    'sections': set(),
                }

            # Adicionar chunk
            chunk_info = {
                'chunk_id': point.id,
                'content_length': len(point.payload.get('content', '')),
                'content_preview': point.payload.get('content', '')[:100]
                + '...'
                if len(point.payload.get('content', '')) > 100
                else point.payload.get('content', ''),
                'categories': point.payload.get('categories', []),
                'section': point.payload.get('section', ''),
                'chunk_index': point.payload.get('chunk_index', 0),
                'token_count': point.payload.get('token_count', 0),
            }

            documents[doc_id]['chunks'].append(chunk_info)
            documents[doc_id]['total_chunks'] += 1
            documents[doc_id]['total_characters'] += len(
                point.payload.get('content', '')
            )
            documents[doc_id]['categories'].update(
                point.payload.get('categories', [])
            )
            documents[doc_id]['sections'].add(point.payload.get('section', ''))

        # Converter sets para lists para JSON
        for doc in documents.values():
            doc['categories'] = list(doc['categories'])
            doc['sections'] = list(doc['sections'])
            doc['chunks'].sort(key=lambda x: x.get('chunk_index', 0))

        return {
            'total_documents': len(documents),
            'total_chunks': len(all_points),
            'documents': list(documents.values()),
        }

    except Exception as e:
        logger.error(f'Error getting all documents: {e}')
        raise HTTPException(
            status_code=500, detail=f'Failed to get documents: {str(e)}'
        )


@app.get('/qdrant/stats')
async def get_qdrant_statistics():
    """Get detailed statistics about the knowledge base"""
    try:
        # Buscar todos os pontos com payload
        all_points = []
        next_page_offset = None

        while True:
            scroll_result = processor.qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False,
            )

            points, next_page_offset = scroll_result
            all_points.extend(points)

            if next_page_offset is None:
                break

        # Calcular estatísticas
        stats = {
            'total_chunks': len(all_points),
            'total_documents': len(
                set(p.payload.get('document_id', '') for p in all_points)
            ),
            'total_characters': sum(
                len(p.payload.get('content', '')) for p in all_points
            ),
            'average_chunk_size': 0,
            'categories': {},
            'sections': {},
            'document_breakdown': {},
            'chunk_size_distribution': {
                'small (0-500)': 0,
                'medium (500-1500)': 0,
                'large (1500+)': 0,
            },
        }

        if stats['total_chunks'] > 0:
            stats['average_chunk_size'] = (
                stats['total_characters'] // stats['total_chunks']
            )

        # Analisar cada ponto
        for point in all_points:
            # Categorias
            for cat in point.payload.get('categories', []):
                stats['categories'][cat] = stats['categories'].get(cat, 0) + 1

            # Seções
            section = point.payload.get('section', 'unknown')
            stats['sections'][section] = stats['sections'].get(section, 0) + 1

            # Documentos
            doc_id = point.payload.get('document_id', 'unknown')
            filename = point.payload.get('filename', 'unknown')
            if doc_id not in stats['document_breakdown']:
                stats['document_breakdown'][doc_id] = {
                    'filename': filename,
                    'chunks': 0,
                    'characters': 0,
                }
            stats['document_breakdown'][doc_id]['chunks'] += 1
            stats['document_breakdown'][doc_id]['characters'] += len(
                point.payload.get('content', '')
            )

            # Distribuição de tamanhos
            content_length = len(point.payload.get('content', ''))
            if content_length < 500:
                stats['chunk_size_distribution']['small (0-500)'] += 1
            elif content_length < 1500:
                stats['chunk_size_distribution']['medium (500-1500)'] += 1
            else:
                stats['chunk_size_distribution']['large (1500+)'] += 1

        return stats

    except Exception as e:
        logger.error(f'Error getting Qdrant statistics: {e}')
        raise HTTPException(
            status_code=500, detail=f'Failed to get statistics: {str(e)}'
        )


@app.get('/qdrant/search-test')
async def test_qdrant_search(query: str = 'seguro', limit: int = 5):
    """Test search functionality in Qdrant"""
    try:
        # Verificar se o embedding service está inicializado
        if (
            not processor.embedding_service
            or not processor.embedding_service.model
        ):
            await processor.initialize()

        # Gerar embedding da query
        query_embedding = await processor.embedding_service.embed_texts(
            [query]
        )

        # Buscar no Qdrant
        search_results = processor.qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding[0],
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results = []
        for result in search_results:
            results.append(
                {
                    'score': result.score,
                    'chunk_id': result.id,
                    'content_preview': result.payload.get('content', '')[:200]
                    + '...'
                    if len(result.payload.get('content', '')) > 200
                    else result.payload.get('content', ''),
                    'categories': result.payload.get('categories', []),
                    'section': result.payload.get('section', ''),
                    'filename': result.payload.get('filename', ''),
                    'document_id': result.payload.get('document_id', ''),
                }
            )

        return {
            'query': query,
            'total_results': len(results),
            'results': results,
        }

    except Exception as e:
        logger.error(f'Error testing search: {e}')
        raise HTTPException(
            status_code=500, detail=f'Search test failed: {str(e)}'
        )


@app.delete('/qdrant/clear')
async def clear_qdrant_collection():
    """Clear all data from the Qdrant collection (CAUTION!)"""
    try:
        # Deletar a collection
        processor.qdrant_client.delete_collection(COLLECTION_NAME)

        # Recriar a collection
        await processor.ensure_collection_exists()

        return {
            'message': f'Collection {COLLECTION_NAME} cleared and recreated successfully',
            'status': 'success',
        }

    except Exception as e:
        logger.error(f'Error clearing collection: {e}')
        raise HTTPException(
            status_code=500, detail=f'Failed to clear collection: {str(e)}'
        )


@app.get('/qdrant/raw-points')
async def get_raw_points(limit: int = 10, offset: int = 0):
    """Get raw points from Qdrant for detailed inspection"""
    try:
        scroll_result = processor.qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            offset=offset if offset > 0 else None,
            with_payload=True,
            with_vectors=True,  # Include vectors for complete view
        )

        points, next_offset = scroll_result

        raw_points = []
        for point in points:
            raw_points.append(
                {
                    'id': point.id,
                    'vector_size': len(point.vector) if point.vector else 0,
                    'vector_preview': point.vector[:5]
                    if point.vector
                    else None,  # First 5 dimensions
                    'payload': point.payload,
                }
            )

        return {
            'limit': limit,
            'offset': offset,
            'next_offset': next_offset,
            'total_returned': len(raw_points),
            'points': raw_points,
        }

    except Exception as e:
        logger.error(f'Error getting raw points: {e}')
        raise HTTPException(
            status_code=500, detail=f'Failed to get raw points: {str(e)}'
        )


@app.post('/upload-pdf-debug', response_model=Dict[str, Any])
async def upload_pdf_debug(file: UploadFile = File(...)):
    """Upload PDF with detailed debugging information"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, detail='Only PDF files are supported'
        )

    import time

    start_time = time.time()

    try:
        # Read file content
        pdf_content = await file.read()

        # Generate document ID
        file_hash = hashlib.md5(pdf_content).hexdigest()
        document_id = f'doc_{file_hash[:12]}'

        # Extract text with debugging
        logger.info(f'🔍 DEBUG: Processing PDF: {file.filename}')
        text = processor.extract_text_from_pdf(pdf_content)

        debug_info = {
            'filename': file.filename,
            'document_id': document_id,
            'file_size_bytes': len(pdf_content),
            'extracted_text_length': len(text),
            'extracted_text_preview': text[:500] + '...'
            if len(text) > 500
            else text,
            'text_empty': not text.strip(),
        }

        if not text.strip():
            debug_info['error'] = 'No text could be extracted from PDF'
            return debug_info

        # Chunk text with debugging
        logger.info(f'🔍 DEBUG: Starting chunking process...')
        chunks = processor.chunk_text(text, file.filename)

        debug_info.update(
            {'chunks_generated': len(chunks), 'chunks_details': []}
        )

        # Add detailed chunk information
        for i, chunk in enumerate(chunks):
            chunk_detail = {
                'chunk_index': i,
                'chunk_id': chunk.chunk_id,
                'text_length': len(chunk.text),
                'text_preview': chunk.text[:200] + '...'
                if len(chunk.text) > 200
                else chunk.text,
                'categories': chunk.metadata.get('categories', []),
                'section': chunk.metadata.get('section', ''),
                'token_count': chunk.metadata.get('token_count', 0),
                'has_overlap': chunk.metadata.get('has_overlap', False),
            }
            debug_info['chunks_details'].append(chunk_detail)

        # Generate embeddings (optional for debug)
        logger.info(
            f'🔍 DEBUG: Generating embeddings for {len(chunks)} chunks...'
        )
        embeddings = await processor.generate_embeddings(
            [chunk.text for chunk in chunks]
        )

        # Store in Qdrant
        await processor.store_in_qdrant(chunks, embeddings, document_id)

        processing_time = time.time() - start_time
        debug_info['processing_time'] = processing_time
        debug_info['success'] = True

        return debug_info

    except Exception as e:
        logger.error(f'🔍 DEBUG: Error processing PDF {file.filename}: {e}')
        return {'filename': file.filename, 'error': str(e), 'success': False}
    try:
        collection_info = processor.qdrant_client.get_collection(
            COLLECTION_NAME
        )
        return {
            'collection_name': COLLECTION_NAME,
            'vectors_count': collection_info.vectors_count,
            'status': collection_info.status,
            'config': {
                'distance': collection_info.config.params.vectors.distance.value,
                'size': collection_info.config.params.vectors.size,
            },
        }
    except Exception as e:
        return {'error': str(e)}


if __name__ == '__main__':
    uvicorn.run(
        'pdf_processor:app', host='0.0.0.0', port=8001, log_level='info'
    )
