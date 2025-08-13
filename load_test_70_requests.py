#!/usr/bin/env python3
"""
Script para testar 70 consultas simultâneas no endpoint /api/ai/query do frontend
Testa a capacidade de load do sistema com múltiplas consultas concorrentes
"""

import asyncio
import httpx
import time
import json
import random
from typing import List, Dict, Any
from datetime import datetime

# Configuração
FRONTEND_URL = "http://localhost:3000"
ENDPOINT = "/api/ai/query"
NUM_REQUESTS = 10
TIMEOUT = 15  # 60 segundos de timeout

# Lista de perguntas variadas para testar
TEST_QUESTIONS = [
    "Quem pode solicitar alterações cadastrais?",
    "Quais documentos são necessários para alteração de nome?",
    "Qual é o prazo para conclusão de uma alteração cadastral?",
    "Como funciona a alteração de nome social?",
    "Quais são os procedimentos para cliente interditado?",
    "Como enviar documentos para o Banrisul?",
    "O que fazer quando o cliente tem telefone sem prefixo 9?",
    "Quais são os sistemas mencionados no manual?",
    "Como é feita a validação no site da Receita Federal?",
    "Quais são os tipos de alterações cadastrais disponíveis?",
    "Qual email usar para envio de documentos?",
    "Como alterar CPF no sistema?",
    "Quais são os prazos do sistema Zendesk?",
    "Como funciona a alteração de estado civil?",
    "Procedimentos para cliente impossibilitado de assinar?",
    "Como fazer transferência assistida?",
    "Quais são as formas de envio pelo correio?",
    "Como registrar manifestação pendente?",
    "Procedimentos para menor de idade?",
    "Como funciona a sincronização de dados?",
    "Qual é o fluxo de Token/Rating/Score?",
    "Como atualizar dados no Zendesk?",
    "Procedimentos para cliente PICPAY?",
    "Como fazer alteração no SISPREV?",
    "Quais documentos para curatela?",
    "Como incluir impressão digital?",
    "Procedimentos para procuração?",
    "Como validar dados na Receita Federal?",
    "Alteração para certificado cancelado?",
    "Como corrigir dados proativamente?"
]

async def make_single_request(
    client: httpx.AsyncClient, 
    question: str, 
    request_id: int
) -> Dict[str, Any]:
    """Faz uma única requisição ao endpoint"""
    
    payload = {
        "question": question,
        "max_tokens": 512,
        "temperature": 0.7,
        "search_limit": 3,
        "score_threshold": 0.6
    }
    
    start_time = time.time()
    
    try:
        response = await client.post(
            f"{FRONTEND_URL}{ENDPOINT}",
            json=payload,
            timeout=TIMEOUT
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "request_id": request_id,
                "status": "success",
                "status_code": response.status_code,
                "response_time": response_time,
                "question": question,
                "answer_length": len(data.get("answer", "")),
                "tokens_used": data.get("tokens_used", 0),
                "sources_count": len(data.get("sources", [])),
                "processing_time": data.get("processing_time", 0),
                "search_time": data.get("search_time", 0),
                "generation_time": data.get("generation_time", 0),
                "error": None
            }
        else:
            error_text = response.text
            return {
                "request_id": request_id,
                "status": "error",
                "status_code": response.status_code,
                "response_time": response_time,
                "question": question,
                "error": f"HTTP {response.status_code}: {error_text}"
            }
            
    except asyncio.TimeoutError:
        end_time = time.time()
        return {
            "request_id": request_id,
            "status": "timeout",
            "response_time": end_time - start_time,
            "question": question,
            "error": "Request timeout"
        }
        
    except Exception as e:
        end_time = time.time()
        return {
            "request_id": request_id,
            "status": "error",
            "response_time": end_time - start_time,
            "question": question,
            "error": str(e)
        }

async def run_load_test():
    """Executa o teste de carga com 70 requisições simultâneas"""
    
    print("🚀 INICIANDO TESTE DE CARGA - 70 CONSULTAS SIMULTÂNEAS")
    print("=" * 80)
    print(f"📍 Endpoint: {FRONTEND_URL}{ENDPOINT}")
    print(f"🔢 Número de requisições: {NUM_REQUESTS}")
    print(f"⏱️ Timeout por requisição: {TIMEOUT}s")
    print(f"🕒 Horário de início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Preparar as perguntas (repetindo a lista se necessário)
    questions = []
    for i in range(NUM_REQUESTS):
        question = TEST_QUESTIONS[i % len(TEST_QUESTIONS)]
        questions.append(question)
    
    # Randomizar a ordem das perguntas
    random.shuffle(questions)
    
    # Configurar cliente HTTP
    connector_limits = httpx.Limits(
        max_keepalive_connections=100,
        max_connections=150,
        keepalive_expiry=30.0
    )
    
    start_time = time.time()
    
    async with httpx.AsyncClient(limits=connector_limits) as client:
        # Criar todas as tasks simultaneamente
        tasks = []
        for i, question in enumerate(questions):
            task = make_single_request(client, question, i + 1)
            tasks.append(task)
        
        print(f"🔄 Enviando {len(tasks)} requisições simultâneas...")
        
        # Executar todas as requisições simultaneamente
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Processar resultados
    successful_requests = []
    failed_requests = []
    timeout_requests = []
    
    for result in results:
        if isinstance(result, Exception):
            failed_requests.append({
                "error": str(result),
                "status": "exception"
            })
        elif result["status"] == "success":
            successful_requests.append(result)
        elif result["status"] == "timeout":
            timeout_requests.append(result)
        else:
            failed_requests.append(result)
    
    # Calcular estatísticas
    total_requests = len(results)
    success_count = len(successful_requests)
    error_count = len(failed_requests)
    timeout_count = len(timeout_requests)
    success_rate = (success_count / total_requests) * 100
    
    # Estatísticas de tempo de resposta
    if successful_requests:
        response_times = [r["response_time"] for r in successful_requests]
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        # Estatísticas do processamento
        processing_times = [r["processing_time"] for r in successful_requests if r["processing_time"]]
        search_times = [r["search_time"] for r in successful_requests if r["search_time"]]
        generation_times = [r["generation_time"] for r in successful_requests if r["generation_time"]]
        
        avg_processing = sum(processing_times) / len(processing_times) if processing_times else 0
        avg_search = sum(search_times) / len(search_times) if search_times else 0
        avg_generation = sum(generation_times) / len(generation_times) if generation_times else 0
        
        total_tokens = sum(r["tokens_used"] for r in successful_requests)
        avg_tokens = total_tokens / len(successful_requests)
    else:
        avg_response_time = min_response_time = max_response_time = 0
        avg_processing = avg_search = avg_generation = 0
        total_tokens = avg_tokens = 0
    
    # Imprimir resultados
    print("\n" + "=" * 80)
    print("📊 RESULTADOS DO TESTE DE CARGA")
    print("=" * 80)
    
    print(f"⏱️ Tempo total de execução: {total_time:.2f}s")
    print(f"🚀 Requisições por segundo: {total_requests / total_time:.2f} req/s")
    print()
    
    print(f"📈 ESTATÍSTICAS GERAIS:")
    print(f"   Total de requisições: {total_requests}")
    print(f"   ✅ Sucessos: {success_count} ({success_rate:.1f}%)")
    print(f"   ❌ Erros: {error_count}")
    print(f"   ⏰ Timeouts: {timeout_count}")
    print()
    
    if successful_requests:
        print(f"⚡ TEMPOS DE RESPOSTA (Sucessos):")
        print(f"   Mínimo: {min_response_time:.2f}s")
        print(f"   Máximo: {max_response_time:.2f}s")
        print(f"   Média: {avg_response_time:.2f}s")
        print()
        
        print(f"🔍 PROCESSAMENTO INTERNO:")
        print(f"   Tempo médio de busca: {avg_search:.2f}s")
        print(f"   Tempo médio de geração: {avg_generation:.2f}s")
        print(f"   Tempo médio total: {avg_processing:.2f}s")
        print()
        
        print(f"🧠 TOKENS E CONTEÚDO:")
        print(f"   Total de tokens usados: {total_tokens:,}")
        print(f"   Média de tokens por resposta: {avg_tokens:.1f}")
        print()
    
    # Mostrar alguns erros se houver
    if failed_requests:
        print(f"❌ AMOSTRA DE ERROS ({min(5, len(failed_requests))} de {len(failed_requests)}):")
        for i, error in enumerate(failed_requests[:5]):
            print(f"   {i+1}. {error.get('error', 'Erro desconhecido')}")
        print()
    
    # Análise de distribuição de tempos
    if successful_requests:
        print(f"📊 DISTRIBUIÇÃO DE TEMPOS DE RESPOSTA:")
        response_times = [r["response_time"] for r in successful_requests]
        response_times.sort()
        
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            index = int((p / 100) * len(response_times)) - 1
            if index >= 0:
                print(f"   P{p}: {response_times[index]:.2f}s")
        print()
    
    # Recomendações
    print(f"💡 ANÁLISE E RECOMENDAÇÕES:")
    if success_rate >= 95:
        print("   ✅ Excelente! Sistema suporta bem a carga.")
    elif success_rate >= 80:
        print("   ⚠️  Bom, mas pode ser otimizado.")
    else:
        print("   ❌ Sistema com dificuldades sob carga.")
    
    if avg_response_time < 5:
        print("   ✅ Tempos de resposta muito bons.")
    elif avg_response_time < 10:
        print("   ⚠️  Tempos de resposta aceitáveis.")
    else:
        print("   ❌ Tempos de resposta altos.")
    
    rps = total_requests / total_time
    if rps > 10:
        print("   ✅ Boa capacidade de throughput.")
    elif rps > 5:
        print("   ⚠️  Capacidade de throughput moderada.")
    else:
        print("   ❌ Baixa capacidade de throughput.")
    
    print("\n" + "=" * 80)
    print("✅ TESTE DE CARGA CONCLUÍDO")
    print("=" * 80)
    
    # Salvar resultados detalhados em arquivo
    results_data = {
        "test_info": {
            "endpoint": f"{FRONTEND_URL}{ENDPOINT}",
            "num_requests": NUM_REQUESTS,
            "total_time": total_time,
            "timestamp": datetime.now().isoformat()
        },
        "summary": {
            "total_requests": total_requests,
            "successful_requests": success_count,
            "failed_requests": error_count,
            "timeout_requests": timeout_count,
            "success_rate": success_rate,
            "requests_per_second": rps,
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time
        },
        "detailed_results": results
    }
    
    with open(f"load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"📁 Resultados detalhados salvos em: load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

# Comando curl equivalente para uma única requisição
def show_curl_example():
    """Mostra exemplo de comando curl para teste individual"""
    
    print("\n" + "=" * 80)
    print("🔧 COMANDO CURL PARA TESTE INDIVIDUAL")
    print("=" * 80)
    
    curl_command = f'''curl -X POST {FRONTEND_URL}{ENDPOINT} \\
  -H "Content-Type: application/json" \\
  -d '{{
    "question": "Quem pode solicitar alterações cadastrais?",
    "max_tokens": 512,
    "temperature": 0.7,
    "search_limit": 3,
    "score_threshold": 0.6
  }}' \\
  -w "\\nTempo de resposta: %{{time_total}}s\\nCódigo HTTP: %{{http_code}}\\n"'''
    
    print(curl_command)
    print("\n" + "=" * 80)

if __name__ == "__main__":
    print("🧪 TESTE DE CARGA - SISTEMA RAG ICATU")
    print("Pressione Ctrl+C para cancelar a qualquer momento\n")
    
    try:
        # Mostrar exemplo de curl primeiro
        show_curl_example()
        
        # Executar teste de carga
        asyncio.run(run_load_test())
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Teste cancelado pelo usuário.")
    except Exception as e:
        print(f"\n\n❌ Erro durante o teste: {e}")
