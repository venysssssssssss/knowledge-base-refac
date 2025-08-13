# 🧠 Contexto Completo - Manual ICATU

## Visão Geral

Esta implementação permite que o modelo Mistral 7B responda perguntas usando **TODO O CONTEÚDO** do manual de Alteração Cadastral ICATU como contexto, ao invés de apenas alguns chunks específicos encontrados pela busca semântica.

## 🆚 Comparação das Abordagens

### RAG Tradicional (`/query`)
- **Como funciona**: Busca 3-8 chunks mais relevantes → Usa como contexto
- **Vantagens**: Rápido, preciso para perguntas específicas
- **Limitações**: Pode perder informações importantes que estão em outros chunks
- **Melhor para**: Perguntas diretas e específicas

### RAG Contexto Completo (`/query-full`)
- **Como funciona**: Usa TODO o manual como contexto → Resposta abrangente
- **Vantagens**: Resposta mais completa, não perde informações relacionadas
- **Limitações**: Mais lento, usa mais tokens
- **Melhor para**: Perguntas complexas que exigem informações de múltiplas seções

## 🚀 Novos Endpoints Implementados

### 1. Serviço Mistral - `/query-full-context`
```bash
POST http://localhost:8003/query-full-context
Content-Type: application/json

{
  "question": "Quem pode solicitar alterações cadastrais?",
  "full_document_topics": "CONTEÚDO_COMPLETO_DO_MANUAL",
  "max_tokens": 1024,
  "temperature": 0.3,
  "instructions": "Instruções específicas (opcional)"
}
```

### 2. Serviço RAG - `/ask-full-context`
```bash
POST http://localhost:8002/ask-full-context
Content-Type: application/json

{
  "question": "Quem pode solicitar alterações cadastrais?",
  "max_tokens": 1024,
  "temperature": 0.3,
  "document_id": "opcional",
  "use_full_manual": true
}
```

### 3. Frontend API - `/api/ai/query-full`
```bash
POST http://localhost:3000/api/ai/query-full
Content-Type: application/json

{
  "question": "Quem pode solicitar alterações cadastrais?",
  "max_tokens": 1024,
  "temperature": 0.3
}
```

## 📋 Parâmetros Principais

| Parâmetro | Tradicional | Contexto Completo | Descrição |
|-----------|-------------|-------------------|-----------|
| `max_tokens` | 512 | 1024 | Tokens máximos na resposta |
| `temperature` | 0.7 | 0.3 | Criatividade (menor = mais preciso) |
| `search_limit` | 8 | N/A | Chunks na busca tradicional |
| `score_threshold` | 0.5 | N/A | Threshold de relevância |

## 🧪 Como Testar

### Teste Rápido
```bash
# 1. Executar script de teste
python3 test_full_context.py

# 2. Teste manual via curl
curl -X POST http://localhost:8002/ask-full-context \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quais são todos os tipos de alterações cadastrais disponíveis?",
    "max_tokens": 1024,
    "temperature": 0.3
  }'
```

### Teste via Frontend
```javascript
// Contexto completo
fetch('/api/ai/query-full', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: "Explique todos os procedimentos para alteração de CPF",
    max_tokens: 1024
  })
})

// Tradicional
fetch('/api/ai/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: "Quem pode solicitar alteração de CPF?",
    max_tokens: 512
  })
})
```

## 🎯 Casos de Uso Recomendados

### Use Contexto Completo (`/query-full`) para:
- ✅ "Explique todos os tipos de alterações cadastrais"
- ✅ "Quais são todos os procedimentos mencionados no manual?"
- ✅ "Como funciona o processo completo de alteração?"
- ✅ "Quais são todas as formas de envio de documentos?"
- ✅ "Liste todos os sistemas mencionados no manual"

### Use RAG Tradicional (`/query`) para:
- ✅ "Qual o prazo para alteração de nome?"
- ✅ "Quem pode solicitar nome social?"
- ✅ "Email do Banrisul para envio de documentos"
- ✅ "Documentos necessários para estado civil"

## 📊 Resultados Esperados

### Exemplo: "Quem pode solicitar alterações cadastrais?"

**RAG Tradicional**:
```
Somente o titular da apólice pode solicitar alterações cadastrais. Para nome social, também é permitido procurador, curador ou tutor.
```

**Contexto Completo**:
```
**QUEM PODE SOLICITAR:**

1. **Geral**: Somente o titular da apólice pode solicitar alterações cadastrais.

2. **Nome Social**: Para inclusão de nome social, também é permitido o pedido por Procurador, Curador ou Tutor.

3. **Detalhamento**:
   - Titular maior de idade
   - Responsável legal ou tutor (titular menor de idade)
   - Procurador, curador ou tutor (conforme cenário específico)

4. **Exceções**:
   - Clientes do PICPAY: alteração não reflete no app
   - Menores de idade: solicitação via formulário específico
   - Interditados: curador nomeado pode assinar
```

## ⚙️ Configurações Técnicas

### Limites do Mistral
- **Contexto máximo**: 8192 tokens
- **Truncamento**: Se manual > 6000 chars, usa início + fim
- **Timeout**: 60 segundos (vs 30s tradicional)

### Otimizações Implementadas
1. **Prompt Engineering**: Instruções específicas para contexto completo
2. **Estruturação**: Manual organizado por seções e páginas
3. **Deduplicação**: Remove conteúdo repetitivo
4. **Priorização**: Foca nas informações mais relevantes

## 🚨 Considerações Importantes

### Performance
- **Tempo**: 2-5x mais lento que RAG tradicional
- **Tokens**: Usa 2-3x mais tokens
- **Memória**: Carrega todo o manual em memória

### Quando NÃO Usar
- ❌ Perguntas simples e diretas
- ❌ Consultas frequentes e rápidas
- ❌ Quando velocidade é prioridade
- ❌ Recursos limitados de compute

### Monitoramento
```bash
# Verificar logs do serviço
docker logs knowledge-base-refac-ai-services-inference-1

# Verificar status
curl http://localhost:8002/health
curl http://localhost:8003/health
```

## 🔧 Troubleshooting

### Erro "Request timeout"
- **Causa**: Contexto muito grande ou serviço sobrecarregado
- **Solução**: Reduzir `max_tokens` ou usar RAG tradicional

### Erro "No response generated"
- **Causa**: Modelo não conseguiu processar o contexto
- **Solução**: Verificar se Ollama está rodando e modelo carregado

### Resposta incompleta
- **Causa**: `max_tokens` muito baixo para contexto completo
- **Solução**: Aumentar para 1024+ tokens

## 🎉 Benefícios da Implementação

1. **✅ Respostas Mais Completas**: Acesso a todo o manual
2. **✅ Não Perde Informações**: Não depende apenas da busca semântica
3. **✅ Melhor para Perguntas Complexas**: Combina informações de múltiplas seções
4. **✅ Flexibilidade**: Usuário pode escolher a abordagem
5. **✅ Compatibilidade**: Não quebra funcionalidade existente

---

**💡 Dica**: Comece testando com o script `test_full_context.py` para ver a diferença entre as abordagens!
