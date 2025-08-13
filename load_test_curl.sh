#!/bin/bash

# Script para testar 70 consultas simultâneas usando curl
# Usage: ./load_test_curl.sh

FRONTEND_URL="http://localhost:3000"
ENDPOINT="/api/ai/query"
NUM_REQUESTS=70
TIMEOUT=60

# Array de perguntas para teste
QUESTIONS=(
    "Quem pode solicitar alterações cadastrais?"
    "Quais documentos são necessários para alteração de nome?"
    "Qual é o prazo para conclusão de uma alteração cadastral?"
    "Como funciona a alteração de nome social?"
    "Quais são os procedimentos para cliente interditado?"
    "Como enviar documentos para o Banrisul?"
    "O que fazer quando o cliente tem telefone sem prefixo 9?"
    "Quais são os sistemas mencionados no manual?"
    "Como é feita a validação no site da Receita Federal?"
    "Quais são os tipos de alterações cadastrais disponíveis?"
    "Qual email usar para envio de documentos?"
    "Como alterar CPF no sistema?"
    "Quais são os prazos do sistema Zendesk?"
    "Como funciona a alteração de estado civil?"
    "Procedimentos para cliente impossibilitado de assinar?"
    "Como fazer transferência assistida?"
    "Quais são as formas de envio pelo correio?"
    "Como registrar manifestação pendente?"
    "Procedimentos para menor de idade?"
    "Como funciona a sincronização de dados?"
)

echo "🚀 INICIANDO TESTE DE CARGA COM CURL - $NUM_REQUESTS CONSULTAS SIMULTÂNEAS"
echo "============================================================================="
echo "📍 Endpoint: $FRONTEND_URL$ENDPOINT"
echo "🔢 Número de requisições: $NUM_REQUESTS"
echo "⏱️ Timeout por requisição: ${TIMEOUT}s"
echo "🕒 Horário de início: $(date)"
echo "============================================================================="

# Criar diretório para logs
mkdir -p load_test_logs
LOG_DIR="load_test_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Função para fazer uma requisição
make_request() {
    local request_id=$1
    local question="${QUESTIONS[$((request_id % ${#QUESTIONS[@]}))]}"
    local start_time=$(date +%s.%N)
    
    # Preparar payload JSON
    local payload=$(cat <<EOF
{
    "question": "$question",
    "max_tokens": 512,
    "temperature": 0.7,
    "search_limit": 3,
    "score_threshold": 0.6
}
EOF
)
    
    # Fazer requisição com curl
    local response=$(curl -s -w "\nHTTP_CODE:%{http_code}\nTIME_TOTAL:%{time_total}\nTIME_CONNECT:%{time_connect}\nTIME_PRETRANSFER:%{time_pretransfer}\nTIME_STARTTRANSFER:%{time_starttransfer}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        --max-time $TIMEOUT \
        "$FRONTEND_URL$ENDPOINT" \
        2>&1)
    
    local end_time=$(date +%s.%N)
    local total_time=$(echo "$end_time - $start_time" | bc -l)
    
    # Extrair métricas da resposta
    local http_code=$(echo "$response" | grep "HTTP_CODE:" | cut -d: -f2)
    local time_total=$(echo "$response" | grep "TIME_TOTAL:" | cut -d: -f2)
    local time_connect=$(echo "$response" | grep "TIME_CONNECT:" | cut -d: -f2)
    local time_starttransfer=$(echo "$response" | grep "TIME_STARTTRANSFER:" | cut -d: -f2)
    
    # Remover métricas da resposta para obter apenas o JSON
    local json_response=$(echo "$response" | sed '/HTTP_CODE:/,$d')
    
    # Salvar resultado
    echo "REQUEST_$request_id: HTTP $http_code, Time: ${time_total}s, Question: $question" >> "$LOG_DIR/summary.log"
    echo "$json_response" > "$LOG_DIR/response_$request_id.json"
    
    # Log detalhado
    cat << EOF >> "$LOG_DIR/detailed_$request_id.log"
REQUEST_ID: $request_id
QUESTION: $question
HTTP_CODE: $http_code
TIME_TOTAL: $time_total
TIME_CONNECT: $time_connect
TIME_STARTTRANSFER: $time_starttransfer
SCRIPT_TOTAL_TIME: $total_time
RESPONSE: $json_response
EOF
    
    # Output para monitoramento em tempo real
    if [[ "$http_code" == "200" ]]; then
        echo "✅ Request $request_id: ${time_total}s"
    else
        echo "❌ Request $request_id: HTTP $http_code"
    fi
}

# Executar requisições em paralelo
echo "🔄 Enviando $NUM_REQUESTS requisições simultâneas..."
echo "📁 Logs sendo salvos em: $LOG_DIR"
echo ""

start_time=$(date +%s.%N)

# Usar GNU parallel se disponível, senão usar background jobs
if command -v parallel &> /dev/null; then
    echo "🚀 Usando GNU parallel para execução otimizada..."
    export -f make_request
    export FRONTEND_URL ENDPOINT TIMEOUT LOG_DIR
    export QUESTIONS
    seq 1 $NUM_REQUESTS | parallel -j $NUM_REQUESTS make_request
else
    echo "🔄 Usando background jobs (instale 'parallel' para melhor performance)..."
    
    # Executar em background
    for i in $(seq 1 $NUM_REQUESTS); do
        make_request $i &
    done
    
    # Aguardar todas as requisições terminarem
    wait
fi

end_time=$(date +%s.%N)
total_time=$(echo "$end_time - $start_time" | bc -l)

echo ""
echo "============================================================================="
echo "📊 ANÁLISE DOS RESULTADOS"
echo "============================================================================="

# Analisar resultados
total_requests=$NUM_REQUESTS
successful_requests=$(grep -c "HTTP 200" "$LOG_DIR/summary.log" 2>/dev/null || echo "0")
failed_requests=$((total_requests - successful_requests))
success_rate=$(echo "scale=1; ($successful_requests * 100) / $total_requests" | bc -l)

echo "⏱️ Tempo total de execução: ${total_time}s"
echo "🚀 Requisições por segundo: $(echo "scale=2; $total_requests / $total_time" | bc -l) req/s"
echo ""
echo "📈 ESTATÍSTICAS GERAIS:"
echo "   Total de requisições: $total_requests"
echo "   ✅ Sucessos: $successful_requests (${success_rate}%)"
echo "   ❌ Falhas: $failed_requests"
echo ""

# Estatísticas de tempo (apenas sucessos)
if [[ $successful_requests -gt 0 ]]; then
    echo "⚡ TEMPOS DE RESPOSTA (apenas sucessos):"
    
    # Extrair tempos de resposta dos sucessos
    grep "HTTP 200" "$LOG_DIR/summary.log" | \
    sed 's/.*Time: \([0-9.]*\)s.*/\1/' | \
    sort -n > "$LOG_DIR/response_times.txt"
    
    if [[ -s "$LOG_DIR/response_times.txt" ]]; then
        local min_time=$(head -n1 "$LOG_DIR/response_times.txt")
        local max_time=$(tail -n1 "$LOG_DIR/response_times.txt")
        local avg_time=$(awk '{sum+=$1} END {print sum/NR}' "$LOG_DIR/response_times.txt")
        
        echo "   Mínimo: ${min_time}s"
        echo "   Máximo: ${max_time}s"
        echo "   Média: $(printf "%.2f" $avg_time)s"
        
        # Percentis
        local total_lines=$(wc -l < "$LOG_DIR/response_times.txt")
        echo "   P50: $(sed -n "$((total_lines * 50 / 100 + 1))p" "$LOG_DIR/response_times.txt")s"
        echo "   P90: $(sed -n "$((total_lines * 90 / 100 + 1))p" "$LOG_DIR/response_times.txt")s"
        echo "   P95: $(sed -n "$((total_lines * 95 / 100 + 1))p" "$LOG_DIR/response_times.txt")s"
    fi
fi

echo ""

# Mostrar alguns erros se houver
if [[ $failed_requests -gt 0 ]]; then
    echo "❌ AMOSTRA DE ERROS:"
    grep -v "HTTP 200" "$LOG_DIR/summary.log" | head -5
    echo ""
fi

# Recomendações
echo "💡 ANÁLISE E RECOMENDAÇÕES:"
if (( $(echo "$success_rate >= 95" | bc -l) )); then
    echo "   ✅ Excelente! Sistema suporta bem a carga."
elif (( $(echo "$success_rate >= 80" | bc -l) )); then
    echo "   ⚠️  Bom, mas pode ser otimizado."
else
    echo "   ❌ Sistema com dificuldades sob carga."
fi

echo ""
echo "============================================================================="
echo "✅ TESTE DE CARGA CONCLUÍDO"
echo "============================================================================="
echo "📁 Logs detalhados disponíveis em: $LOG_DIR"
echo "📊 Arquivo de resumo: $LOG_DIR/summary.log"
echo ""

# Comando curl individual para teste
echo "🔧 COMANDO CURL PARA TESTE INDIVIDUAL:"
echo "============================================================================="
cat << 'EOF'
curl -X POST http://localhost:3000/api/ai/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quem pode solicitar alterações cadastrais?",
    "max_tokens": 512,
    "temperature": 0.7,
    "search_limit": 3,
    "score_threshold": 0.6
  }' \
  -w "\nTempo de resposta: %{time_total}s\nCódigo HTTP: %{http_code}\n"
EOF
echo "============================================================================="
