# 🧪 Testes de Carga - 70 Consultas Simultâneas

Este diretório contém scripts para testar a capacidade do sistema RAG ICATU com **70 consultas simultâneas**.

## 🚀 Opções de Teste

### 1. Script Python (Recomendado)
```bash
# Instalar dependências se necessário
pip install httpx asyncio

# Executar teste completo com estatísticas detalhadas
python3 load_test_70_requests.py
```

**Características:**
- ✅ Estatísticas completas e análise detalhada
- ✅ Controle de timeout e error handling robusto
- ✅ Salva resultados em JSON
- ✅ Percentis de tempo de resposta
- ✅ Análise de throughput e recomendações

### 2. Script Bash com Curl
```bash
# Executar teste com curl nativo
./load_test_curl.sh
```

**Características:**
- ✅ Não requer dependências Python
- ✅ Usa GNU parallel se disponível
- ✅ Logs detalhados por requisição
- ✅ Análise básica de resultados

### 3. Curl Individual (Teste Rápido)
```bash
# Teste de uma única requisição
curl -X POST http://localhost:3000/api/ai/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quem pode solicitar alterações cadastrais?",
    "max_tokens": 512,
    "temperature": 0.7,
    "search_limit": 3,
    "score_threshold": 0.6
  }' \
  -w "\nTempo: %{time_total}s | HTTP: %{http_code}\n"
```

### 4. Teste com Multiple Curl (Background)
```bash
# Executar 70 curls em background simultaneamente
for i in {1..70}; do
  curl -X POST http://localhost:3000/api/ai/query \
    -H "Content-Type: application/json" \
    -d '{
      "question": "Pergunta teste '$i'",
      "max_tokens": 512,
      "temperature": 0.7
    }' \
    -w "Request '$i': %{time_total}s\n" \
    -s -o response_$i.json &
done

# Aguardar todas terminarem
wait

# Ver resumo
echo "Respostas recebidas: $(ls response_*.json | wc -l)"
rm response_*.json
```

## 📊 Métricas Coletadas

### Métricas de Performance
- **Tempo total de execução**
- **Requisições por segundo (RPS)**
- **Taxa de sucesso (%)**
- **Tempos de resposta (min/max/média)**
- **Percentis (P50, P75, P90, P95, P99)**

### Métricas de Sistema
- **Tokens utilizados**
- **Tempo de busca (search_time)**
- **Tempo de geração (generation_time)**
- **Número de fontes encontradas**
- **Códigos de erro HTTP**

### Métricas de Carga
- **Timeouts**
- **Conexões falhadas**
- **Distribuição de erros**
- **Throughput sustentado**

## 🎯 Resultados Esperados

### Sistema Saudável
- ✅ **Taxa de sucesso**: > 95%
- ✅ **Tempo médio de resposta**: < 5s
- ✅ **RPS sustentado**: > 10 req/s
- ✅ **P95**: < 10s

### Sistema Sob Estresse
- ⚠️ **Taxa de sucesso**: 80-95%
- ⚠️ **Tempo médio de resposta**: 5-10s
- ⚠️ **RPS sustentado**: 5-10 req/s
- ⚠️ **P95**: < 15s

### Sistema Sobrecarregado
- ❌ **Taxa de sucesso**: < 80%
- ❌ **Tempo médio de resposta**: > 10s
- ❌ **RPS sustentado**: < 5 req/s
- ❌ **Muitos timeouts**

## 🔧 Configuração dos Serviços

Antes de executar os testes, certifique-se de que todos os serviços estão rodando:

```bash
# Verificar status
curl http://localhost:3000/api/ai/health
curl http://localhost:8002/health  # RAG Service
curl http://localhost:8003/health  # Mistral Service
curl http://localhost:8001/health  # Document Processor

# Iniciar serviços se necessário
docker-compose up -d
# ou
cd ai-services && python -m uvicorn rag.rag_service:app --host 0.0.0.0 --port 8002
```

## 📈 Interpretação dos Resultados

### Gargalos Comuns

1. **Alto tempo de search_time**:
   - Problema no Qdrant ou Document Processor
   - Solução: Otimizar embeddings ou índices

2. **Alto generation_time**:
   - Problema no Mistral/Ollama
   - Solução: Verificar GPU/CPU, ajustar parâmetros

3. **Muitos timeouts**:
   - Sobrecarga geral do sistema
   - Solução: Aumentar recursos ou implementar rate limiting

4. **Erros HTTP 5xx**:
   - Falhas nos serviços backend
   - Solução: Verificar logs dos serviços

### Otimizações Possíveis

1. **Cache**: Implementar cache Redis para respostas frequentes
2. **Load Balancing**: Múltiplas instâncias dos serviços
3. **Connection Pooling**: Otimizar conexões HTTP
4. **Batching**: Agrupar requisições similares
5. **Rate Limiting**: Controlar carga de entrada

## 🏃‍♂️ Execução Rápida

Para um teste rápido de validação:

```bash
# Teste básico (1 requisição)
curl -X POST http://localhost:3000/api/ai/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Teste rápido"}' \
  -w "Tempo: %{time_total}s\n"

# Teste médio (10 requisições)
for i in {1..10}; do curl -X POST http://localhost:3000/api/ai/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Teste '$i'"}' \
  -w "$i: %{time_total}s\n" -s -o /dev/null & done; wait

# Teste completo (70 requisições)
python3 load_test_70_requests.py
```

---

**💡 Dica**: Execute primeiro um teste com 1-5 requisições para validar que o sistema está funcionando antes de executar o teste completo com 70 requisições.
