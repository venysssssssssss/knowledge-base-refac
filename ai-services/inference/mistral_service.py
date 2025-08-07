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


class FullContextQueryRequest(BaseModel):
    question: str
    full_document_topics: str = ''  # Todo o conteúdo do manual
    max_tokens: int = 1024
    temperature: float = 0.3
    instructions: str = ''


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


@app.post('/query-full-context', response_model=QueryResponse)
async def query_with_full_context(request: FullContextQueryRequest):
    """
    Gera resposta baseada na pergunta usando TODO O CONTEÚDO do manual ICATU como contexto.
    Essa abordagem permite que o modelo tenha acesso a todas as informações disponíveis
    para fornecer respostas mais completas e precisas.
    """
    start_time = time.time()
    
    # Usar instruções específicas para contexto completo se não fornecidas
    if not request.instructions:
        request.instructions = """
        Você é um especialista em procedimentos ICATU com acesso a TODO O MANUAL de Alteração Cadastral.
        Responda à pergunta do usuário com base EXCLUSIVAMENTE nas informações do manual completo fornecido.
        
        DIRETRIZES:
        1. Use APENAS informações do manual fornecido
        2. Seja específico e detalhado quando a informação estiver disponível
        3. Se a informação não estiver no manual, diga claramente que não está disponível
        4. Mantenha a terminologia técnica oficial do ICATU
        5. Cite procedimentos, prazos e requisitos exatos
        6. Organize a resposta de forma clara e estruturada
        7. Se houver múltiplas seções relevantes, combine-as de forma coerente
        
        Responda de forma profissional e precisa, como um especialista em processos ICATU.
        """
    
    # Formatar prompt otimizado para contexto completo
    prompt = format_full_context_prompt(
        request.question, 
        request.full_document_topics, 
        request.instructions
    )
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:  # Timeout maior para contexto completo
            payload = {
                'model': MODEL_NAME,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': request.temperature,
                    'num_predict': request.max_tokens,
                    'top_p': 0.9,
                    'num_ctx': 8192,  # Aumentar contexto para acomodar manual completo
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
            
            inference_logger.info(f'Full context query processed in {processing_time:.2f}s')
            
            return QueryResponse(
                answer=answer,
                tokens_used=int(tokens_used),
                processing_time=processing_time,
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail='Request timeout - try with shorter context')
    except Exception as e:
        inference_logger.error(f'Full context generation error: {e}')
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
        prompt = f"""<s>[INST] {final_instructions}\n\nContexto:\n1. Quem Pode Solicitar
Somente o titular da apólice pode solicitar alterações cadastrais. Para inclusão de nome
social, também é permitido o pedido por Procurador, Curador ou Tutor.
2. Tipos de Alterações Cadastrais
a) Documento de Identificação / Nome / Estado Civil
Documentos exigidos:
Nome: cópia do documento de identificação com foto.
Estado Civil: certidão de casamento, averbação de separação/divórcio ou certidão de óbito.
Documento de identificação: cópia simples.
Importante:
Erros simples (ex: Ana Silvia → Ana Silva) podem ser corrigidos diretamente no sistema.
Se o cliente alegar erro na proposta, seguir procedimento de reclamação de implantação.
b) Nome Social
Base legal: Ofício-Circular nº 001/2024/DIR2/SUSEP.
Não é necessário documento comprobatório.
Pode ser solicitado a qualquer momento.
Sem restrições de nomes.
Se o cliente tiver Previdência e Vida, realizar transferência assistida para o ramal: TRANS
NOME SOCIAL V&P.
Nome social não aparecerá:
No site “Área do Cliente”.
Em comunicações de marketing.
Nos boletos (apenas nome de registro).
Será incluído:
Propostas, fichas de cadastro, apólices, certificados e títulos de capitalização.
Registro:
Tipo de Motivo: Solicitação
Motivo: Atualização Cadastral
Razão: Nome Social
Ação Tomada: Realizada ou Pendente
c) Endereço / Telefone / E-mail
Alteração feita diretamente no sistema.
Registro automático é gerado.
Se houver erro no sistema, registrar protocolo pendente e incluir observação.
Não é possível ter endereços de correspondência e cobrança diferentes.
Telefone celular deve ser marcado como tipo “Móvel”.
Corrigir proativamente se o tipo estiver incorreto.
d) CPF / Data de Nascimento
Seguir os mesmos critérios de validação e registro.
e) Interditado / Impossibilitado de Assinar
Procedimentos específicos devem ser seguidos (detalhes provavelmente nos próximos
arquivos).
3. Envio de Documentos
Parceiros Rio Grande (Banrisul)
E-mail: formularioscap@riograndeseguradora.com.br
Correio:
Rua Siqueira Campos, 1163 – 6º andar
Porto Alegre – RS
CEP: 90010-001
Agências Banrisul
Outros Parceiros
E-mail: documentos@capitalizacao.com
Correio:
Caixa Postal 6577
Rio de Janeiro – RJ
CEP: 20030-970
Observação: Documentos que exigem reconhecimento de firma devem ser enviados
obrigatoriamente pelos correios.
4. Registro no Sistema
Tipo de Motivo: Solicitação
Motivo: Atualização Cadastral Cliente
Razão: CPF / Data de Nascimento / Nome Social / Endereço / E-mail / Nome / RG /
Telefone / Estado Civil
Ação Tomada: Concluído / Pendente / Não Realizado
Prazo: 07 dias úteis
5. Alteração de CPF / Data de Nascimento
Confirmar o dado correto com o cliente.
Validar no site da Receita Federal:
Se os dados estiverem corretos:
Verificar se os campos da aba “Cliente” e “Documentos” (CPF e RG: número, órgão expedidor
e data de expedição) estão preenchidos.
Se sim, registrar manifestação como Alteração Cadastral Pendente, indicando no protocolo:
Que a consulta à Receita foi feita.
Data e horário da consulta.
Se não, orientar o envio de:
Formulário de alteração.
CPF.
Documento de identificação (RG, certidão de nascimento, passaporte etc.).
6. Interditado / Impossibilitado de Assinar
a) Interditado
Documentos necessários:
Cópia do RG e CPF.
Curatela do curador nomeado.
Formulário: Alteração de Dados.
Registro: Ação Tomada: Não Realizada.
Assinatura:
Se possui discernimento: assinatura do proponente e/ou curador.
Se não possui discernimento: apenas o curador assina.
b) Impossibilitado de Assinar
Com coleta de impressão digital:
Inserir a digital do cliente no formulário.
Assinatura de:
Uma pessoa identificada que assina a pedido do cliente.
Duas testemunhas.
Sem coleta de impressão digital:
Assinatura do representante legal ou procurador.
Documentos necessários:
Cópia de documento de identificação do representante (CNH, RG, CTPS ou passaporte).
Procuração.
7. Envio de Documentos (Reforço)
Banrisul: Clique aqui
Demais parceiros: Clique aqui
8. Registro no Sistema (Reforço)
Tipo de Motivo: Solicitação
Motivo: Atualização Cadastral Cliente
Razão: CPF / Data de Nascimento / Nome Social / Endereço / E-mail / Nome / RG /
Telefone / Estado Civil
Ação Tomada: Concluído / Pendente / Não Realizado
Prazo: 07 dias úteis
9. Quem Pode Solicitar
Titular maior de idade.
Responsável legal ou tutor, no caso de titular menor de idade.
Procurador, curador ou tutor, conforme o cenário.
10. Tipos de Alterações e Procedimentos Específicos
a) Clientes do Parceiro PICPAY
A alteração não se reflete no app PICPAY.
O cliente deve atualizar os dados diretamente no aplicativo.
b) Documento de Identificação / Nome / Estado Civil
Documentos necessários:
Nome: cópia do documento com foto.
Estado Civil: certidão de casamento, averbação de separação/divórcio ou certidão de óbito.
Importante:
Erros simples (ex: Ana Silvia → Ana Silva): alterar diretamente no sistema.
Alegação de preenchimento correto na proposta: seguir fluxo de reclamação de
implantação.
Para menores de idade, a solicitação deve ser feita via formulário específico.
c) Endereço / Telefone / E-mail
Cenários:
Dados atualizados no sistema, mas desatualizados no Zendesk:
Copiar os dados do sistema para o Zendesk.
Se houver erro ao gravar, fazer uma pequena alteração no sistema (ex: “Rua” → “R.”) para
forçar a sincronização.
Cliente com telefone sem prefixo 9:
Atualizar conforme padrão nacional.
Não é necessário realizar identificação positiva ou autenticação de segurança nesses casos.
d) CPF / Data de Nascimento / Nome Social
Seguir os mesmos critérios já descritos na seção de Capitalização.
e) Cliente Reprovado com Advertência
Não realizar fluxo de Token/Rating/Score.
Orientar o cliente a seguir uma das opções:
Área do Cliente (exceto Sicredi e HDI).
Formulário com assinatura digital (ICP-Brasil ou Gov.br).
Formulário com firma reconhecida.
11. Envio de Documentos
As formas de envio variam conforme o parceiro. O manual pode conter um anexo ou link
com as opções atualizadas (ex: “Clique aqui para consultar as opções disponíveis”).
12. Registro no Sistema
Marca
Forma de Contato
Tipo de Público
Tipo de Relação
ID Positiva
Linha de Negócio: Automático
Parceiro: Automático
Produto localizado Previdência: Automático
Produto Texto: Automático
Tipo de Contato: Solicitação
Motivo do Contato: Alteração
Submotivo de Contato 1 e 2: conforme a solicitação
Número do Certificado: Automático
Aceitou a alteração cadastral?
Resultado da Manifestação
13. Prazos
Alterações refletem no sistema e no Zendesk em até 24 horas.
Prazo geral para conclusão da solicitação: 07 dias úteis.
14. Cliente com Telefone Celular sem Prefixo 9
Verificação: Se o número de celular não possui o dígito 9, ele deve ser incluído.
Onde alterar:
Para um único cliente: diretamente no Zendesk.
Para múltiplos clientes: diretamente no sistema do produto.
Não é necessário realizar identificação positiva ou autenticação de segurança.
15. Cliente com Dados Atualizados no Sistema, mas Desatualizados no
Zendesk
Ação: Copiar os dados corretos do sistema para o Zendesk.
Se houver erro ao gravar:
Realizar uma pequena alteração no sistema (ex: abreviações como “Rua” → “R.”).
Isso força a sincronização com a base de dados.
Resultado: A alteração será refletida automaticamente no Zendesk.
16. Cliente com Dados Desatualizados (Sistema e Zendesk) sem
Advertência
Fluxo necessário: Token/Rating/Score (autenticação).
Se aprovado:
Realizar a alteração diretamente no Zendesk.
Se o app não permitir alteração:
Verificar se há:
Mais de um cadastro com datas de nascimento diferentes.
Plano de dependente vinculado ao CPF.
Se sim, seguir os procedimentos específicos para alteração de data de nascimento ou CPF do
dependente.
Se não, realizar a alteração diretamente no sistema do produto.
Prazos para refletir a alteração:
Zendesk: até 1 hora.
Sistema do produto: até 24 horas.
17. Alteração de CPF (Detalhamento)
Não realizar fluxo de Token/Rating/Score.
Confirmar o dado com o cliente.
Validar no site da Receita Federal.
Ações conforme o sistema:
MUMPS/SISVIDA: inserir print da Receita no ticket e registrar manifestação como pendente.
TELEMARKETING/SISCAP: orientar o cliente a contatar a Central de Capitalização.
PGBL/SISPREV:
Se campos de identificação estiverem preenchidos: registrar manifestação como pendente.
Se não: solicitar formulário, CPF e documento de identificação.
Observação: A alteração de data de nascimento é feita apenas nas informações cadastrais,
não no certificado.
18. Alteração de Data de Nascimento
Procedimento Geral:
Não realizar fluxo de Token/Rating/Score.
Confirmar o dado correto com o cliente.
Validar os dados no site da Receita Federal.
Ações conforme o sistema:
MUMPS / SISVIDA:
Inserir print da Receita Federal no ticket.
Registrar manifestação como Alteração Cadastral Pendente, independentemente do status
do certificado (ativo ou cancelado).
TELEMARKETING / SISCAP:
Orientar o cliente a entrar em contato com a Central de Capitalização.
PGBL / SISPREV:
Verificar se os campos de documento de identificação, data de expedição e natureza do
documento estão preenchidos.
Se não estiverem preenchidos: solicitar envio de formulário, CPF e documento de
identificação (RG, certidão de nascimento, passaporte etc.).
Se estiverem preenchidos: inserir print da Receita Federal no ticket e realizar a alteração
diretamente no SISPREV.
Prazo para refletir a alteração no site e Zendesk: até 24 horas.
Observação:
A alteração de data de nascimento é feita apenas nas informações cadastrais do cliente.
Mesmo que o cliente tenha um certificado de risco vinculado ao de acumulação, isso não
afeta o capital segurado.
Finalização do Registro
Para todos os cenários, o registro deve conter:
Marca
Forma de Contato
Tipo de Público
Tipo de Relação
ID Positiva
Linha de Negócio: Automático
Parceiro: Automático
Produto localizado Previdência: Automático
Produto Texto: Automático
Tipo de Contato: Solicitação
Motivo do Contato: Alteração
Submotivo de Contato 1 e 2: conforme a solicitação
Número do Certificado: Automático
Aceitou a alteração cadastral?
Resultado da Manifestação
Atualizações do Procedimento
13/01/2025: Atualização geral dos procedimentos.
11/03/2025: Atualização sobre sincronização entre sistema e Zendesk

Pergunta: {question}

Responda apenas com base nas informações do contexto fornecido. [/INST]"""
    else:
        prompt = f'<s>[INST] {question} [/INST]'
    return prompt


def format_full_context_prompt(
    question: str, full_document_content: str, instructions: str = ''
) -> str:
    """
    Formata o prompt para o Mistral Instruct usando TODO O CONTEÚDO do manual como contexto.
    Otimizado para trabalhar com o manual completo de alteração cadastral ICATU.
    """
    default_instructions = """
    Você é um especialista em procedimentos ICATU com acesso ao MANUAL COMPLETO de Alteração Cadastral.
    Responda à pergunta do usuário com base EXCLUSIVAMENTE nas informações do manual fornecido abaixo.
    
    DIRETRIZES IMPORTANTES:
    1. Use APENAS informações do manual fornecido - não invente ou extrapole
    2. Seja específico e detalhado quando a informação estiver disponível
    3. Se a informação não estiver no manual, diga claramente: "A informação solicitada não está disponível no manual fornecido"
    4. Mantenha a terminologia técnica oficial do ICATU
    5. Cite procedimentos, prazos e requisitos exatos conforme o manual
    6. Organize a resposta de forma clara e estruturada
    7. Se houver múltiplas seções relevantes, combine-as de forma coerente
    8. Priorize informações mais específicas e detalhadas
    
    Responda como um especialista em processos ICATU, de forma profissional e precisa.
    """
    
    # Use instruções personalizadas se fornecidas
    final_instructions = instructions if instructions else default_instructions
    
    # Verificar se o contexto não está vazio
    if not full_document_content or len(full_document_content.strip()) == 0:
        return f'<s>[INST] {question}\n\nObs: Nenhum contexto do manual foi fornecido. [/INST]'
    
    # Truncar contexto se for muito longo (mantendo as partes mais importantes)
    max_context_length = 6000  # Limite para evitar overflow
    if len(full_document_content) > max_context_length:
        # Priorizar início e fim do documento, que geralmente contém informações importantes
        first_part = full_document_content[:max_context_length//2]
        last_part = full_document_content[-max_context_length//2:]
        full_document_content = f"{first_part}\n\n[... CONTEÚDO TRUNCADO ...]\n\n{last_part}"
    
    prompt = f"""<s>[INST] {final_instructions}

MANUAL COMPLETO ICATU - ALTERAÇÃO CADASTRAL:
{full_document_content}

PERGUNTA DO USUÁRIO: {question}

Responda com base EXCLUSIVAMENTE nas informações do manual acima. [/INST]"""
    
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
