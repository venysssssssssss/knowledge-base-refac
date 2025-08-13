#!/bin/bash

# Comandos Curl para Testar 70 Consultas Simultâneas no Frontend RAG

echo "🚀 COMANDOS CURL PARA TESTE DE 70 CONSULTAS SIMULTÂNEAS"
echo "============================================================"

echo ""
echo "1️⃣ TESTE INDIVIDUAL (validação):"
echo "============================================================"
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
  -w "\nTempo: %{time_total}s | HTTP: %{http_code}\n"
EOF

echo ""
echo "2️⃣ TESTE RÁPIDO (5 consultas simultâneas):"
echo "============================================================"
cat << 'EOF'
for i in {1..5}; do
  curl -X POST http://localhost:3000/api/ai/query \
    -H "Content-Type: application/json" \
    -d '{"question": "Teste '${i}' - Quem pode solicitar alterações?", "max_tokens": 256}' \
    -w "Request $i: %{time_total}s\n" \
    -s -o /dev/null &
done; wait
EOF

echo ""
echo "3️⃣ TESTE MÉDIO (20 consultas simultâneas):"
echo "============================================================"
cat << 'EOF'
for i in {1..20}; do
  curl -X POST http://localhost:3000/api/ai/query \
    -H "Content-Type: application/json" \
    -d '{
      "question": "Pergunta '${i}' - Quais são os procedimentos?",
      "max_tokens": 512,
      "temperature": 0.7
    }' \
    -w "Request $i: %{time_total}s (HTTP: %{http_code})\n" \
    -s -o response_$i.json &
done

echo "Aguardando todas as requisições..."
wait

echo "Respostas recebidas: $(ls response_*.json 2>/dev/null | wc -l)"
echo "Sucessos (HTTP 200): $(grep -l '"answer"' response_*.json 2>/dev/null | wc -l)"

# Limpeza
rm -f response_*.json
EOF

echo ""
echo "4️⃣ TESTE COMPLETO (70 consultas simultâneas) - MANUAL:"
echo "============================================================"
cat << 'EOF'
# Criar array de perguntas
declare -a QUESTIONS=(
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
)

# Executar 70 requisições
echo "Iniciando 70 consultas simultâneas em $(date)"
start_time=$(date +%s)

for i in {1..70}; do
  question_index=$((i % ${#QUESTIONS[@]}))
  question="${QUESTIONS[$question_index]}"
  
  curl -X POST http://localhost:3000/api/ai/query \
    -H "Content-Type: application/json" \
    -d '{
      "question": "'"$question"'",
      "max_tokens": 512,
      "temperature": 0.7,
      "search_limit": 3,
      "score_threshold": 0.6
    }' \
    -w "Request $i: %{time_total}s (HTTP: %{http_code})\n" \
    -s -o response_$i.json &
done

echo "Aguardando todas as 70 requisições terminarem..."
wait

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo ""
echo "=== RESULTADOS ==="
echo "Tempo total: ${total_time}s"
echo "Respostas recebidas: $(ls response_*.json 2>/dev/null | wc -l)"
echo "Sucessos: $(grep -l '"answer"' response_*.json 2>/dev/null | wc -l)"
echo "Falhas: $(grep -L '"answer"' response_*.json 2>/dev/null | wc -l)"
echo "RPS: $(echo "scale=2; 70 / $total_time" | bc -l) req/s"

# Análise de tempos (opcional)
echo ""
echo "=== DISTRIBUIÇÃO DE TEMPOS ==="
grep "Request.*:" /dev/stdout | sort -k3 -n | head -5 | echo "5 mais rápidas:"
grep "Request.*:" /dev/stdout | sort -k3 -n | tail -5 | echo "5 mais lentas:"

# Limpeza
read -p "Remover arquivos de resposta? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  rm -f response_*.json
fi
EOF

echo ""
echo "5️⃣ VERSÃO SIMPLIFICADA (70 consultas em uma linha):"
echo "============================================================"
cat << 'EOF'
for i in {1..70}; do curl -X POST http://localhost:3000/api/ai/query -H "Content-Type: application/json" -d '{"question": "Pergunta '${i}'", "max_tokens": 256}' -w "$i:%{time_total}s " -s -o /dev/null & done; wait; echo -e "\n✅ 70 consultas concluídas!"
EOF

echo ""
echo "6️⃣ SCRIPTS AUTOMATIZADOS (recomendado):"
echo "============================================================"
echo "# Para análise completa com estatísticas:"
echo "python3 load_test_70_requests.py"
echo ""
echo "# Para teste com logs detalhados:"
echo "./load_test_curl.sh"

echo ""
echo "7️⃣ COMANDOS DE MONITORAMENTO:"
echo "============================================================"
cat << 'EOF'
# Verificar saúde dos serviços
curl http://localhost:3000/api/ai/health

# Monitorar serviços individuais
curl http://localhost:8002/health  # RAG
curl http://localhost:8003/health  # Mistral
curl http://localhost:8001/health  # Document Processor

# Ver logs em tempo real (em terminal separado)
docker-compose logs -f
EOF

echo ""
echo "💡 DICAS DE USO:"
echo "============================================================"
echo "1. Comece com teste individual para validar"
echo "2. Execute teste rápido (5 consultas) para verificar concorrência"
echo "3. Execute o teste completo apenas quando sistema estiver estável"
echo "4. Use os scripts Python/Bash para análise detalhada"
echo "5. Monitore logs dos serviços durante os testes"
echo ""
echo "⚠️  ATENÇÃO:"
echo "- Teste pode demorar 2-10 minutos para completar"
echo "- Sistema pode ficar lento durante o teste"
echo "- Certifique-se que todos os serviços estão funcionando"
