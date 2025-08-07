"""
RAG (Retrieval-Augmented Generation) Service
Combines document search with Mistral 7B for contextual answers
"""

import asyncio
import logging
import re
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
MISTRAL_SERVICE_URL = (
    'http://10.117.0.19:8003'  # Updated to avoid port conflict
)
DOCUMENT_PROCESSOR_URL = 'http://10.117.0.19:8001'


class RAGRequest(BaseModel):
    question: str
    max_tokens: int = 512
    temperature: float = 0.7
    search_limit: int = 8
    score_threshold: float = 0.5
    document_id: str = None  # Novo campo opcional para filtrar por documento


class FullContextRAGRequest(BaseModel):
    question: str
    max_tokens: int = 1024
    temperature: float = 0.3
    document_id: str = None  # Opcional: usar documento específico
    use_full_manual: bool = True  # Se deve usar todo o manual como contexto


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
        rag_logger.info('✅ RAG service initialized')

    async def cleanup(self):
        """Cleanup resources"""
        if self.http_client:
            await self.http_client.aclose()

    async def get_full_manual_content(self, document_id: str = None) -> str:
        """
        Extrai todo o conteúdo do manual ICATU para usar como contexto completo.
        Se document_id for fornecido, extrai apenas desse documento específico.
        """
        try:
            # Preparar payload para buscar todos os chunks do documento
            search_payload = {
                'query': 'alteração cadastral procedimento documento',  # Query ampla para pegar tudo
                'limit': 1000,  # Limite alto para pegar todos os chunks
                'score_threshold': 0.1,  # Threshold muito baixo para incluir tudo
            }
            if document_id:
                search_payload['document_id'] = document_id

            response = await self.http_client.post(
                f'{DOCUMENT_PROCESSOR_URL}/search', json=search_payload
            )

            if response.status_code != 200:
                rag_logger.error(
                    f'Erro ao buscar conteúdo completo: {response.status_code}'
                )
                return ''

            response_data = response.json()
            if isinstance(response_data, dict) and 'chunks' in response_data:
                all_chunks = response_data['chunks']
            else:
                all_chunks = response_data

            if not all_chunks:
                rag_logger.warning(
                    'Nenhum chunk encontrado para contexto completo'
                )
                return ''

            # Organizar chunks por ordem de documento/página se possível
            sorted_chunks = sorted(
                all_chunks,
                key=lambda x: (
                    x.get('metadata', {}).get('page_number', 0),
                    x.get('metadata', {}).get('chunk_index', 0),
                ),
            )

            # Construir texto completo do manual
            manual_sections = []
            current_section = ''
            last_page = -1

            for chunk in sorted_chunks:
                content = chunk.get('content', '').strip()
                if not content:
                    continue

                metadata = chunk.get('metadata', {})
                page_number = metadata.get('page_number', 0)
                section_title = metadata.get('section_title', '')

                # Adicionar quebra de seção se mudou de página ou há título de seção
                if page_number != last_page and page_number > 0:
                    if current_section:
                        manual_sections.append(current_section)
                    current_section = f'\n--- PÁGINA {page_number} ---\n'
                    last_page = page_number

                # Adicionar título de seção se disponível
                if section_title and section_title not in current_section:
                    current_section += f'\n## {section_title}\n'

                # Adicionar conteúdo
                current_section += f'{content}\n'

            # Adicionar última seção
            if current_section:
                manual_sections.append(current_section)

            full_manual = '\n'.join(manual_sections)

            rag_logger.info(
                f'📖 Conteúdo completo extraído: {len(full_manual)} caracteres de {len(sorted_chunks)} chunks'
            )

            return full_manual

        except Exception as e:
            rag_logger.error(
                f'Erro ao extrair conteúdo completo do manual: {e}'
            )
            return ''

    async def search_documents(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.5,
        document_id: str = None,
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents with enhanced multi-query strategy"""
        try:
            # 1. Expandir consulta com variações semânticas
            expanded_queries = self.expand_search_query(query)
            rag_logger.info(
                f'🔍 Busca expandida: {len(expanded_queries)} variações da consulta'
            )

            all_results = []

            # 2. Buscar com múltiplas variações
            for search_query in expanded_queries:
                search_payload = {
                    'query': search_query,
                    'limit': min(limit * 2, 10),  # Mais resultados por query
                    'score_threshold': max(
                        score_threshold - 0.1, 0.3
                    ),  # Threshold mais permissivo
                }
                if document_id:
                    search_payload['document_id'] = document_id

                response = await self.http_client.post(
                    f'{DOCUMENT_PROCESSOR_URL}/search', json=search_payload
                )

                if response.status_code == 200:
                    response_data = response.json()
                    if (
                        isinstance(response_data, dict)
                        and 'chunks' in response_data
                    ):
                        results = response_data['chunks']
                    else:
                        results = response_data
                    all_results.extend(results)

            # 3. Deduplificar e ranquear resultados
            deduplicated_results = self.deduplicate_and_rank_results(
                all_results, query
            )

            # 4. Retornar melhores resultados
            final_results = deduplicated_results[:limit]
            rag_logger.info(
                f'📊 Busca finalizada: {len(final_results)} chunks selecionados de {len(all_results)} encontrados'
            )

            return final_results

        except Exception as e:
            rag_logger.error(f'Error searching documents: {e}')
            return []

    def expand_search_query(self, query: str) -> List[str]:
        """Expande a consulta com variações semânticas e sinônimos"""
        expanded_queries = [query]  # Consulta original

        # Mapeamento de sinônimos específicos do domínio ICATU
        synonym_mapping = {
            'quem pode': [
                'quem consegue',
                'quem tem permissão',
                'autorizado',
                'habilitado',
            ],
            'solicitar': ['pedir', 'requerer', 'realizar', 'fazer'],
            'alteração': ['mudança', 'modificação', 'atualização', 'correção'],
            'cadastral': ['cadastro', 'dados pessoais', 'informações'],
            'documento': ['documentação', 'papéis', 'comprovante'],
            'procedimento': [
                'processo',
                'fluxo',
                'passo a passo',
                'como fazer',
            ],
            'prazo': ['tempo', 'período', 'duração', 'quando'],
            'titular': ['portador', 'segurado', 'cliente'],
            'cpf': ['documento federal', 'receita federal'],
            'endereço': ['residência', 'correspondência', 'localização'],
            'telefone': ['contato', 'celular', 'número'],
            'email': ['correio eletrônico', 'e-mail'],
        }

        query_lower = query.lower()

        # Adicionar variações com sinônimos
        for term, synonyms in synonym_mapping.items():
            if term in query_lower:
                for synonym in synonyms:
                    variant = query_lower.replace(term, synonym)
                    if variant != query_lower:
                        expanded_queries.append(variant)

        # Adicionar consultas focadas em palavras-chave
        keywords = self.extract_query_keywords(query)
        if len(keywords) > 1:
            # Consulta com apenas palavras-chave principais
            expanded_queries.append(' '.join(keywords[:3]))

            # Consultas individuais por palavra-chave
            for keyword in keywords[:2]:
                expanded_queries.append(keyword)

        # Adicionar variações específicas por tipo de pergunta
        if any(word in query_lower for word in ['quem', 'pode', 'solicitar']):
            expanded_queries.extend(
                [
                    'titular pode solicitar',
                    'procurador autorizado',
                    'quem está habilitado',
                ]
            )
        elif any(
            word in query_lower
            for word in ['como', 'procedimento', 'processo']
        ):
            expanded_queries.extend(
                ['passo a passo', 'fluxo procedimento', 'como realizar']
            )
        elif any(word in query_lower for word in ['prazo', 'tempo', 'quando']):
            expanded_queries.extend(
                ['dias úteis', 'tempo necessário', 'duração processo']
            )
        elif any(
            word in query_lower
            for word in ['documento', 'necessário', 'exigido']
        ):
            expanded_queries.extend(
                [
                    'documentos obrigatórios',
                    'papéis necessários',
                    'comprovantes exigidos',
                ]
            )

        # Remover duplicatas e limitar
        unique_queries = []
        for q in expanded_queries:
            if q not in unique_queries:
                unique_queries.append(q)

        return unique_queries[:8]  # Máximo 8 variações

    def extract_query_keywords(self, query: str) -> List[str]:
        """Extrai palavras-chave importantes da consulta"""
        # Palavras irrelevantes
        stop_words = {
            'o',
            'a',
            'os',
            'as',
            'de',
            'da',
            'do',
            'em',
            'na',
            'no',
            'para',
            'por',
            'com',
            'como',
            'que',
            'é',
            'são',
            'tem',
            'ter',
            'pode',
            'posso',
        }

        words = re.findall(r'\b[a-záéíóúçãõâêî]{3,}\b', query.lower())
        keywords = [word for word in words if word not in stop_words]

        return keywords

    def deduplicate_and_rank_results(
        self, results: List[Dict], original_query: str
    ) -> List[Dict]:
        """Remove duplicatas e ranqueia resultados por relevância"""
        if not results:
            return []

        # 1. Remover duplicatas por conteúdo
        seen_content = set()
        unique_results = []

        for result in results:
            content_key = result.get('content', '')[
                :100
            ]  # Primeiros 100 chars como chave
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)

        # 2. Calcular relevância aprimorada
        for result in unique_results:
            relevance_score = self.calculate_enhanced_relevance(
                result, original_query
            )
            result['enhanced_score'] = relevance_score

        # 3. Ordenar por relevância aprimorada
        unique_results.sort(
            key=lambda x: x.get('enhanced_score', 0), reverse=True
        )

        return unique_results

    def calculate_enhanced_relevance(self, result: Dict, query: str) -> float:
        """Calcula score de relevância aprimorado"""
        base_score = result.get('score', 0)
        content = result.get('content', '').lower()
        metadata = result.get('metadata', {})

        # Fatores de relevância
        relevance_factors = []

        # 1. Score base do embedding
        relevance_factors.append(base_score * 0.4)

        # 2. Correspondência exata de termos
        query_terms = self.extract_query_keywords(query)
        exact_matches = sum(1 for term in query_terms if term in content)
        term_score = (
            (exact_matches / len(query_terms)) * 0.3 if query_terms else 0
        )
        relevance_factors.append(term_score)

        # 3. Presença de palavras-chave importantes
        important_keywords = metadata.get('keywords', [])
        keyword_overlap = len(set(query_terms) & set(important_keywords))
        keyword_score = (keyword_overlap / max(len(query_terms), 1)) * 0.2
        relevance_factors.append(keyword_score)

        # 4. Tipo de seção (algumas são mais importantes)
        section_type = metadata.get('section_type', '')
        section_bonus = 0
        if 'title' in section_type or 'summary' in section_type:
            section_bonus = 0.1
        elif 'main_section' in section_type:
            section_bonus = 0.05
        relevance_factors.append(section_bonus)

        # 5. Comprimento adequado do conteúdo
        content_length = len(content)
        length_score = 0
        if 200 <= content_length <= 1500:  # Tamanho ideal
            length_score = 0.1
        elif content_length > 50:  # Mínimo aceitável
            length_score = 0.05
        relevance_factors.append(length_score)

        return sum(relevance_factors)

    def build_context_intelligent(
        self, search_results: List[Dict[str, Any]], question: str
    ) -> str:
        """
        Construção inteligente de contexto otimizada para máxima precisão
        Estratégia: Hierarquia semântica + Deduplicação + Contexto estruturado
        """
        if not search_results:
            return ''

        rag_logger.info(
            f"🧠 Construindo contexto inteligente para: '{question[:50]}...'"
        )
        rag_logger.info(
            f'📊 Processando {len(search_results)} chunks encontrados'
        )

        # 1. Filtrar e classificar resultados por qualidade
        high_quality_results = self.filter_high_quality_results(
            search_results, question
        )
        rag_logger.info(
            f'✅ {len(high_quality_results)} chunks de alta qualidade selecionados'
        )

        # 2. Agrupar por relevância semântica
        grouped_results = self.group_by_semantic_relevance(
            high_quality_results, question
        )

        # 3. Construir contexto hierárquico estruturado
        structured_context = self.build_hierarchical_context(
            grouped_results, question
        )

        # 4. Otimizar tamanho do contexto
        final_context = self.optimize_context_size(
            structured_context, max_chars=3000
        )

        rag_logger.info(f'📝 Contexto final: {len(final_context)} caracteres')
        return final_context

    def filter_high_quality_results(
        self, results: List[Dict[str, Any]], question: str
    ) -> List[Dict[str, Any]]:
        """Filtra resultados de alta qualidade"""
        if not results:
            return []

        filtered_results = []
        query_keywords = self.extract_query_keywords(question.lower())

        for result in results:
            content = result.get('content', '').strip()
            score = result.get('score', 0)
            metadata = result.get('metadata', {})

            # Critérios de qualidade
            quality_score = 0

            # 1. Score de embedding base
            if score > 0.8:
                quality_score += 3
            elif score > 0.6:
                quality_score += 2
            elif score > 0.4:
                quality_score += 1

            # 2. Tamanho adequado do conteúdo
            content_length = len(content)
            if 100 <= content_length <= 2000:  # Tamanho ideal
                quality_score += 2
            elif content_length >= 50:  # Mínimo aceitável
                quality_score += 1

            # 3. Correspondência de palavras-chave
            content_lower = content.lower()
            keyword_matches = sum(
                1 for keyword in query_keywords if keyword in content_lower
            )
            if keyword_matches >= 2:
                quality_score += 3
            elif keyword_matches >= 1:
                quality_score += 1

            # 4. Tipo de seção importante
            section_type = metadata.get('section_type', '')
            if any(
                important_type in section_type
                for important_type in ['title', 'summary', 'main_section']
            ):
                quality_score += 2

            # 5. Presença de informações estruturadas
            if any(
                indicator in content_lower
                for indicator in [
                    'procedimento',
                    'documento',
                    'prazo',
                    'quem pode',
                ]
            ):
                quality_score += 1

            # Aceitar apenas resultados com qualidade suficiente
            if quality_score >= 3:
                result['quality_score'] = quality_score
                filtered_results.append(result)

        # Ordenar por qualidade e score combinados
        filtered_results.sort(
            key=lambda x: (x['quality_score'], x.get('score', 0)), reverse=True
        )

        return filtered_results[:8]  # Máximo 8 chunks de alta qualidade

    def group_by_semantic_relevance(
        self, results: List[Dict[str, Any]], question: str
    ) -> Dict[str, List[Dict]]:
        """Agrupa resultados por relevância semântica"""
        grouped = {
            'direct_answer': [],  # Resposta direta à pergunta
            'procedure': [],  # Procedimentos e processos
            'requirements': [],  # Requisitos e documentos
            'timeframes': [],  # Prazos e tempos
            'authorization': [],  # Quem pode fazer
            'context': [],  # Contexto geral
        }

        question_lower = question.lower()

        for result in results:
            content = result.get('content', '').lower()
            metadata = result.get('metadata', {})
            section_title = metadata.get('section_title', '').lower()

            # Classificar por tipo de informação
            categorized = False

            # Perguntas sobre autorização/permissão
            if any(
                word in question_lower
                for word in ['quem', 'pode', 'autorizado', 'permitido']
            ):
                if any(
                    word in content
                    for word in [
                        'titular',
                        'pode',
                        'autorizado',
                        'procurador',
                        'curador',
                    ]
                ):
                    grouped['authorization'].append(result)
                    categorized = True

            # Perguntas sobre procedimentos
            if any(
                word in question_lower
                for word in ['como', 'procedimento', 'processo', 'passo']
            ):
                if any(
                    word in content
                    for word in [
                        'procedimento',
                        'processo',
                        'seguir',
                        'realizar',
                        'fluxo',
                    ]
                ):
                    grouped['procedure'].append(result)
                    categorized = True

            # Perguntas sobre documentos/requisitos
            if any(
                word in question_lower
                for word in [
                    'documento',
                    'necessário',
                    'exigido',
                    'obrigatório',
                ]
            ):
                if any(
                    word in content
                    for word in [
                        'documento',
                        'necessário',
                        'exigido',
                        'obrigatório',
                        'formulário',
                    ]
                ):
                    grouped['requirements'].append(result)
                    categorized = True

            # Perguntas sobre prazos
            if any(
                word in question_lower
                for word in ['prazo', 'tempo', 'quando', 'duração']
            ):
                if any(
                    word in content
                    for word in ['prazo', 'dias', 'horas', 'tempo', 'úteis']
                ):
                    grouped['timeframes'].append(result)
                    categorized = True

            # Se tem alta relevância, considerar resposta direta
            if (
                result.get('score', 0) > 0.8
                or result.get('quality_score', 0) >= 5
            ):
                grouped['direct_answer'].append(result)
                categorized = True

            # Caso contrário, adicionar ao contexto geral
            if not categorized:
                grouped['context'].append(result)

        return grouped

    def build_hierarchical_context(
        self, grouped_results: Dict[str, List[Dict]], question: str
    ) -> str:
        """Constrói contexto hierárquico baseado na pergunta"""
        context_parts = []

        # Determinar prioridade baseada na pergunta
        question_lower = question.lower()

        if any(
            word in question_lower for word in ['quem', 'pode', 'autorizado']
        ):
            priority_order = [
                'authorization',
                'direct_answer',
                'requirements',
                'procedure',
                'timeframes',
                'context',
            ]
        elif any(
            word in question_lower
            for word in ['como', 'procedimento', 'processo']
        ):
            priority_order = [
                'procedure',
                'direct_answer',
                'requirements',
                'authorization',
                'timeframes',
                'context',
            ]
        elif any(
            word in question_lower
            for word in ['documento', 'necessário', 'obrigatório']
        ):
            priority_order = [
                'requirements',
                'direct_answer',
                'procedure',
                'authorization',
                'timeframes',
                'context',
            ]
        elif any(
            word in question_lower for word in ['prazo', 'tempo', 'quando']
        ):
            priority_order = [
                'timeframes',
                'direct_answer',
                'procedure',
                'requirements',
                'authorization',
                'context',
            ]
        else:
            priority_order = [
                'direct_answer',
                'procedure',
                'requirements',
                'authorization',
                'timeframes',
                'context',
            ]

        # Construir contexto seguindo a prioridade
        section_headers = {
            'authorization': '👥 QUEM PODE SOLICITAR',
            'procedure': '📋 PROCEDIMENTOS',
            'requirements': '📄 DOCUMENTOS NECESSÁRIOS',
            'timeframes': '⏰ PRAZOS',
            'direct_answer': '🎯 INFORMAÇÃO PRINCIPAL',
            'context': '📌 CONTEXTO ADICIONAL',
        }

        used_content = set()
        total_chars = 0
        max_chars_per_section = {
            'direct_answer': 800,
            'authorization': 600,
            'procedure': 700,
            'requirements': 600,
            'timeframes': 400,
            'context': 400,
        }

        for category in priority_order:
            if total_chars >= 2500:  # Limite global
                break

            results = grouped_results.get(category, [])
            if not results:
                continue

            section_content = []
            section_chars = 0
            max_section_chars = max_chars_per_section.get(category, 500)

            for result in results[:3]:  # Máximo 3 itens por categoria
                content = result.get('content', '').strip()
                content_key = content[:100]  # Chave para deduplicação

                if content_key in used_content:
                    continue

                if section_chars + len(content) > max_section_chars:
                    # Truncar conteúdo se necessário
                    remaining_chars = max_section_chars - section_chars
                    if (
                        remaining_chars > 200
                    ):  # Só truncar se sobrar espaço razoável
                        content = content[:remaining_chars] + '...'
                    else:
                        break

                used_content.add(content_key)
                score = result.get('score', 0)
                filename = result.get('metadata', {}).get('filename', 'ICATU')

                formatted_content = f'• {content} [Relevância: {score:.2f}]'
                section_content.append(formatted_content)
                section_chars += len(formatted_content)

                if section_chars >= max_section_chars:
                    break

            if section_content:
                header = section_headers.get(category, f'� {category.upper()}')
                section_text = f'\n{header}:\n' + '\n'.join(section_content)
                context_parts.append(section_text)
                total_chars += len(section_text)

        if not context_parts:
            return 'Informação específica não encontrada nos documentos disponíveis.'

        # Adicionar cabeçalho contextual
        context_header = f"""CONTEXTO ICATU - Alteração Cadastral
Pergunta: {question}
Fontes encontradas: {len([r for results in grouped_results.values() for r in results])} documentos

INFORMAÇÕES RELEVANTES:"""

        full_context = context_header + '\n' + '\n'.join(context_parts)

        return full_context

    def optimize_context_size(
        self, context: str, max_chars: int = 3000
    ) -> str:
        """Otimiza o tamanho do contexto mantendo as informações mais importantes"""
        if len(context) <= max_chars:
            return context

        # Dividir em seções
        sections = context.split('\n\n')

        # Priorizar seções por importância (baseado nos emojis/headers)
        priority_order = ['🎯', '👥', '📋', '📄', '⏰', '📌']

        important_sections = []
        remaining_sections = []

        for section in sections:
            is_important = any(
                emoji in section for emoji in priority_order[:3]
            )
            if is_important:
                important_sections.append(section)
            else:
                remaining_sections.append(section)

        # Construir contexto otimizado
        optimized_context = ''

        # Adicionar seções importantes primeiro
        for section in important_sections:
            if len(optimized_context) + len(section) <= max_chars:
                optimized_context += section + '\n\n'
            else:
                # Truncar seção se necessário
                remaining_space = max_chars - len(optimized_context) - 50
                if remaining_space > 200:
                    optimized_context += section[:remaining_space] + '...\n\n'
                break

        # Adicionar seções restantes se houver espaço
        for section in remaining_sections:
            if len(optimized_context) + len(section) <= max_chars:
                optimized_context += section + '\n\n'
            else:
                break

        return optimized_context.strip()

    async def generate_answer_enhanced(
        self,
        question: str,
        context: str,
        max_tokens: int = 500,
        temperature: float = 0.1,
    ) -> dict:
        """Generate enhanced answer using Mistral with advanced prompting strategy"""
        try:
            # Prompt engineering otimizado para máxima precisão
            enhanced_prompt = self.build_optimal_prompt(question, context)

            mistral_payload = {
                'question': question,
                'context': '',  # Deixar vazio para forçar uso do manual hardcoded
                'max_tokens': max_tokens,
                'temperature': min(
                    temperature, 0.15
                ),  # Temperatura muito baixa para máxima precisão
                'instructions': '',  # Sem instruções adicionais - usar apenas o manual hardcoded
            }

            response = await self.http_client.post(
                f'{MISTRAL_SERVICE_URL}/query', json=mistral_payload
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f'Mistral service error: {response.text}',
                )

            result = response.json()
            rag_logger.debug(f'Mistral response: {result}')

            # Pós-processamento da resposta
            processed_answer = self.post_process_answer(
                result.get('answer', ''), question, context
            )
            result['answer'] = processed_answer

            return result

        except Exception as e:
            rag_logger.error(f'Error generating answer: {e}')
            raise HTTPException(
                status_code=500, detail=f'Answer generation failed: {str(e)}'
            )

    def build_optimal_prompt(self, question: str, context: str) -> str:
        """Constrói o prompt otimizado para o Mistral"""

        # Determinar tipo de pergunta para personalizar o prompt
        question_type = self.classify_question_type(question)

        # Prompts especializados por tipo de pergunta
        specialized_instructions = {
            'authorization': """
Você é um especialista em regulamentações ICATU. Responda com precisão sobre QUEM pode solicitar alterações cadastrais.
FOQUE EM: autorização, permissões, quem está habilitado, limitações.
FORMATO: Liste claramente quem pode e quem não pode, com base APENAS no contexto.
""",
            'procedure': """
Você é um especialista em processos ICATU. Explique COMO realizar procedimentos de alteração cadastral.
FOQUE EM: passos, fluxos, sequência de ações, orientações práticas.
FORMATO: Liste os passos de forma ordenada e clara, baseado APENAS no contexto.
""",
            'requirements': """
Você é um especialista em documentação ICATU. Liste QUAIS documentos são necessários.
FOQUE EM: documentos obrigatórios, formulários, comprovantes necessários.
FORMATO: Liste todos os documentos exigidos, baseado APENAS no contexto.
""",
            'timeframes': """
Você é um especialista em prazos ICATU. Informe QUANDO/QUANTO TEMPO leva cada processo.
FOQUE EM: prazos específicos, duração, tempos de processamento.
FORMATO: Indique prazos claros e específicos, baseado APENAS no contexto.
""",
            'general': """
Você é um especialista em procedimentos ICATU. Responda de forma abrangente e precisa.
FOQUE EM: informação mais relevante para a pergunta específica.
FORMATO: Resposta clara e direta, baseado APENAS no contexto.
""",
        }

        instruction = specialized_instructions.get(
            question_type, specialized_instructions['general']
        )

        # Prompt estruturado final
        enhanced_prompt = f"""SISTEMA: {instruction}

REGRAS CRÍTICAS:
1. Use APENAS informações do contexto fornecido abaixo
2. Se a informação não estiver no contexto: "A informação solicitada não está disponível nos documentos fornecidos"
3. Seja específico e objetivo
4. Mantenha terminologia técnica ICATU
5. Cite procedimentos e prazos exatos quando disponíveis
6. NÃO invente ou extrapole informações

CONTEXTO ICATU OFICIAL:
{context}

PERGUNTA ESPECÍFICA: {question}

RESPOSTA TÉCNICA (baseada exclusivamente no contexto acima):"""

        return enhanced_prompt

    def classify_question_type(self, question: str) -> str:
        """Classifica o tipo de pergunta para personalizar o prompt"""
        question_lower = question.lower()

        if any(
            word in question_lower
            for word in [
                'quem',
                'pode',
                'autorizado',
                'permitido',
                'habilitado',
            ]
        ):
            return 'authorization'
        elif any(
            word in question_lower
            for word in [
                'como',
                'procedimento',
                'processo',
                'passo',
                'realizar',
            ]
        ):
            return 'procedure'
        elif any(
            word in question_lower
            for word in [
                'documento',
                'necessário',
                'obrigatório',
                'exigido',
                'formulário',
            ]
        ):
            return 'requirements'
        elif any(
            word in question_lower
            for word in ['prazo', 'tempo', 'quando', 'duração', 'demora']
        ):
            return 'timeframes'
        else:
            return 'general'

    def post_process_answer(
        self, answer: str, question: str, context: str
    ) -> str:
        """Pós-processa a resposta para melhorar qualidade e precisão"""
        if not answer or len(answer.strip()) < 10:
            return 'Não foi possível gerar uma resposta adequada com base no contexto fornecido.'

        # Limpar resposta
        processed_answer = answer.strip()

        # Remover repetições desnecessárias
        lines = processed_answer.split('\n')
        unique_lines = []
        seen_content = set()

        for line in lines:
            line_clean = line.strip().lower()
            if line_clean and line_clean not in seen_content:
                seen_content.add(line_clean)
                unique_lines.append(line.strip())

        processed_answer = '\n'.join(unique_lines)

        # Adicionar formatação se necessário
        question_type = self.classify_question_type(question)

        if (
            question_type == 'authorization'
            and 'titular' in processed_answer.lower()
        ):
            if not processed_answer.startswith('**'):
                processed_answer = (
                    f'**QUEM PODE SOLICITAR:**\n{processed_answer}'
                )

        elif question_type == 'procedure' and any(
            word in processed_answer.lower()
            for word in ['procedimento', 'processo']
        ):
            if not processed_answer.startswith('**'):
                processed_answer = f'**PROCEDIMENTO:**\n{processed_answer}'

        elif (
            question_type == 'requirements'
            and 'documento' in processed_answer.lower()
        ):
            if not processed_answer.startswith('**'):
                processed_answer = (
                    f'**DOCUMENTOS NECESSÁRIOS:**\n{processed_answer}'
                )

        elif question_type == 'timeframes' and any(
            word in processed_answer.lower()
            for word in ['prazo', 'dias', 'horas']
        ):
            if not processed_answer.startswith('**'):
                processed_answer = f'**PRAZOS:**\n{processed_answer}'

        # Garantir que não seja muito longo
        if len(processed_answer) > 800:
            # Truncar preservando frases completas
            sentences = processed_answer.split('.')
            truncated = ''
            for sentence in sentences:
                if len(truncated + sentence + '.') <= 750:
                    truncated += sentence + '.'
                else:
                    break
            if truncated:
                processed_answer = (
                    truncated
                    + '\n\n[Resposta truncada - consulte o documento completo para mais detalhes]'
                )

        return processed_answer

    async def generate_answer_with_full_context(
        self,
        question: str,
        full_manual_content: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> dict:
        """Generate answer using full manual content as context"""
        try:
            mistral_payload = {
                'question': question,
                'full_document_topics': full_manual_content,
                'max_tokens': max_tokens,
                'temperature': temperature,
            }

            response = await self.http_client.post(
                f'{MISTRAL_SERVICE_URL}/query-full-context',
                json=mistral_payload,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f'Mistral service error: {response.text}',
                )

            result = response.json()
            rag_logger.debug(f'Full context Mistral response: {result}')

            return result

        except Exception as e:
            rag_logger.error(f'Error generating answer with full context: {e}')
            raise HTTPException(
                status_code=500,
                detail=f'Full context answer generation failed: {str(e)}',
            )

    async def process_full_context_rag_query(
        self, request: FullContextRAGRequest
    ) -> RAGResponse:
        """Process a RAG query using the full manual as context"""
        start_time = time.time()

        rag_logger.info(
            f'🔄 Processando consulta com contexto completo: {request.question}'
        )

        # Step 1: Get full manual content
        search_start = time.time()
        full_manual_content = await self.get_full_manual_content(
            request.document_id
        )
        search_time = time.time() - search_start

        if not full_manual_content:
            rag_logger.warning('Nenhum conteúdo do manual encontrado')
            return RAGResponse(
                question=request.question,
                answer='Não foi possível acessar o conteúdo do manual para responder à pergunta.',
                sources=[],
                tokens_used=0,
                processing_time=time.time() - start_time,
                search_time=search_time,
                generation_time=0.0,
            )

        rag_logger.info(
            f'📖 Manual completo carregado: {len(full_manual_content)} caracteres'
        )

        # Step 2: Generate answer with full context
        generation_start = time.time()
        mistral_response = await self.generate_answer_with_full_context(
            question=request.question,
            full_manual_content=full_manual_content,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time

        # Step 3: Prepare sources (indicating full manual was used)
        sources = [
            {
                'content_preview': 'Manual completo ICATU - Alteração Cadastral utilizado como contexto',
                'score': 1.0,
                'metadata': {
                    'source_type': 'full_manual',
                    'content_length': len(full_manual_content),
                    'document_id': request.document_id or 'all_documents',
                },
            }
        ]

        rag_logger.info(
            f'✅ Consulta com contexto completo processada em {total_time:.2f}s'
        )

        return RAGResponse(
            question=request.question,
            answer=mistral_response.get('answer', ''),
            sources=sources,
            tokens_used=mistral_response.get('tokens_used', 0),
            processing_time=total_time,
            search_time=search_time,
            generation_time=generation_time,
        )

    async def generate_answer(
        self,
        question: str,
        context: str,
        max_tokens: int = 500,
        temperature: float = 0.1,
    ) -> dict:
        """Generate answer using the enhanced method - compatibility wrapper"""
        response = await self.generate_answer_enhanced(
            question, context, max_tokens, temperature
        )

        # If response is a dict (expected), return it directly
        if isinstance(response, dict):
            return response

        # If response is a string, wrap it in the expected format
        return {
            'answer': str(response),
            'tokens_used': len(str(response).split()) if response else 0,
            'processing_time': 0.0,
        }

    async def process_rag_query(self, request: RAGRequest) -> RAGResponse:
        """Process a complete RAG query, optionally filtering by document_id"""
        start_time = time.time()
        # Step 1: Search for relevant documents
        search_start = time.time()
        search_results = await self.search_documents(
            query=request.question,
            limit=min(request.search_limit, 8),
            score_threshold=max(
                request.score_threshold, 0.2
            ),  # Threshold mais baixo
            document_id=request.document_id,
        )
        search_time = time.time() - search_start
        rag_logger.info(
            f'Found {len(search_results)} relevant documents in {search_time:.2f}s'
        )

        # Step 2: Build context
        context = self.build_context_intelligent(
            search_results, request.question
        )

        # Log the context length for debugging
        rag_logger.debug(f'Built context with {len(context)} characters')

        # Step 3: Generate answer with Mistral
        generation_start = time.time()
        if context:
            mistral_response = await self.generate_answer(
                question=request.question,
                context=context,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        else:
            # Fallback: answer without context
            rag_logger.warning(
                'No relevant documents found, answering without context'
            )
            mistral_response = await self.generate_answer(
                question=request.question,
                context='',
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        # Step 4: Prepare sources information
        sources = []
        for result in search_results:
            source = {
                'content_preview': result.get('content', '')[:200] + '...'
                if len(result.get('content', '')) > 200
                else result.get('content', ''),
                'score': result.get('score', 0),
                'metadata': result.get('metadata', {}),
            }
            sources.append(source)
        return RAGResponse(
            question=request.question,
            answer=mistral_response.get('answer', ''),
            sources=sources,
            tokens_used=mistral_response.get('tokens_used', 0),
            processing_time=total_time,
            search_time=search_time,
            generation_time=generation_time,
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
            mistral_response = await client.get(
                f'{MISTRAL_SERVICE_URL}/health'
            )
            if mistral_response.status_code == 200:
                rag_logger.info('✅ Mistral service is accessible')
            else:
                rag_logger.warning('⚠️ Mistral service not accessible')

            # Test Document processor
            doc_response = await client.get(f'{DOCUMENT_PROCESSOR_URL}/health')
            if doc_response.status_code == 200:
                rag_logger.info('✅ Document processor is accessible')
            else:
                rag_logger.warning('⚠️ Document processor not accessible')

    except Exception as e:
        rag_logger.error(f'Service connection test failed: {e}')

    yield

    # Shutdown
    await rag_service.cleanup()


# FastAPI app
app = FastAPI(
    title='RAG Knowledge Base API',
    description='Retrieval-Augmented Generation for document-based Q&A',
    version='1.0.0',
    lifespan=lifespan,
)


@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'service': 'rag-service',
        'dependencies': {
            'mistral_service': MISTRAL_SERVICE_URL,
            'document_processor': DOCUMENT_PROCESSOR_URL,
        },
    }


@app.post('/ask', response_model=RAGResponse)
async def ask_question(request: RAGRequest):
    """Ask a question with RAG (Retrieval-Augmented Generation)"""
    try:
        rag_logger.info(f'Recebida pergunta: {request.question}')
        response = await rag_service.process_rag_query(request)
        rag_logger.info(f'Resposta gerada para pergunta: {request.question}')
        return response
    except Exception as e:
        rag_logger.error(f'RAG query failed: {e}')
        raise HTTPException(
            status_code=500, detail=f'Query processing failed: {str(e)}'
        )


@app.post('/ask-full-context', response_model=RAGResponse)
async def ask_question_full_context(request: FullContextRAGRequest):
    """Ask a question using the full manual as context for more comprehensive answers"""
    try:
        rag_logger.info(
            f'Recebida pergunta com contexto completo: {request.question}'
        )
        response = await rag_service.process_full_context_rag_query(request)
        rag_logger.info(
            f'Resposta com contexto completo gerada para: {request.question}'
        )
        return response
    except Exception as e:
        rag_logger.error(f'Full context RAG query failed: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Full context query processing failed: {str(e)}',
        )


@app.get('/services/status')
async def get_services_status():
    """Check status of all dependent services"""
    status = {
        'mistral_service': {'url': MISTRAL_SERVICE_URL, 'status': 'unknown'},
        'document_processor': {
            'url': DOCUMENT_PROCESSOR_URL,
            'status': 'unknown',
        },
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check Mistral service
            try:
                response = await client.get(f'{MISTRAL_SERVICE_URL}/health')
                status['mistral_service']['status'] = (
                    'healthy' if response.status_code == 200 else 'unhealthy'
                )
                status['mistral_service']['details'] = (
                    response.json()
                    if response.status_code == 200
                    else response.text
                )
            except Exception as e:
                status['mistral_service']['status'] = 'error'
                status['mistral_service']['error'] = str(e)

            # Check Document processor
            try:
                response = await client.get(f'{DOCUMENT_PROCESSOR_URL}/health')
                status['document_processor']['status'] = (
                    'healthy' if response.status_code == 200 else 'unhealthy'
                )
                status['document_processor']['details'] = (
                    response.json()
                    if response.status_code == 200
                    else response.text
                )
            except Exception as e:
                status['document_processor']['status'] = 'error'
                status['document_processor']['error'] = str(e)

    except Exception as e:
        rag_logger.error(f'Error checking services: {e}')

    return status


if __name__ == '__main__':
    uvicorn.run('rag_service:app', host='0.0.0.0', port=8002, log_level='info')
