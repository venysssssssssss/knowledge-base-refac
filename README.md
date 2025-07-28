![Architecture Diagram](deepseek_mermaid_20250728_e36e87.png)

# Knowledge Base Refactoring

Um sistema inteligente de base de conhecimento usando Mistral 7B para processamento de documentos e resposta a perguntas.

## Estrutura Técnica

### 1. Modelo de IA
**Principal:** Mistral 7B Instruct (4-bit AWQ)

**Alternativas:**
- Zephyr-7B-Beta (fine-tune para chat)
- Llama-3-8B-Instruct (instruções em inglês)
- Phi-3-mini (3.8B para menor consumo)

**Otimizações:**
- vLLM + continuous batching
- Flash Attention 2
- Quantização 4-bit (AWQ)

### 2. Processamento de Documentos
**Bibliotecas:**
- unstructured (parser PDF)
- langchain (chunking inteligente)
- DocArray + Dolphin (embeddings a5-base)

### 3. Estrutura do Projeto
```
projeto/
├── frontend/       # Next.js + React
├── backend/        # NestJS + TypeScript
├── ai-services/    # Python
│   ├── embeddings/
│   ├── inference/
│   └── workers/
├── vector-db/      # Qdrant config
└── ops/            # Docker + Monitoramento
```

### 4. Configuração vLLM
```python
# vLLM Config (mistral_service.py)
engine = AsyncLLMEngine(
    model="models/mistral-7b-awq",
    quantization="awq",
    gpu_memory_utilization=0.92,
    max_model_len=4096,
    enable_chunked_prefill=True,  # Continuous batching
    max_num_seqs=40               # Máx 40 chats concorrentes
)
```

## Workflow Operacional

### Upload de Documentos
1. Operador envia PDF via interface React
2. Backend salva no S3/MinIO e enfileira processamento

### Indexação
Serviço Python:
```python
chunks = split_document(pdf, chunk_size=512)
embeddings = dolphin.encode(chunks)
qdrant.upsert(vectors=embeddings)
```

## Checklist Inicial
- [ ] Configurar VM com Docker + NVIDIA Container Toolkit
- [ ] Clonar repositório base
- [ ] Subir Qdrant: `docker compose up -d qdrant`
- [ ] Baixar modelo Mistral AWQ: `huggingface-cli download TheBloke/Mistral-7B-AWQ`
- [ ] Testar inferência: `python ai-services/inference/test_load.py`

## 🚀 Primeiro Passo: Deploy do Modelo Mistral 7B

**Próxima Etapa:** Configurar fila Redis para processamento assíncrono!
