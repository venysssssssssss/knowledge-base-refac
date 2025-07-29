# 🚀 RAG Pipeline Setup Guide - Mistral 7B + Ollama

## Arquitetura do Sistema

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Upload    │────│  Document        │────│    Qdrant      │
│     (User)      │    │  Processor       │    │  (Embeddings)   │
└─────────────────┘    │  (Port 8001)     │    │  (Port 6333)    │
                       └──────────────────┘    └─────────────────┘
                                │
                                │ Embeddings
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Questions     │────│   RAG Service    │────│  Mistral 7B     │
│    (User)       │    │   (Port 8002)    │    │  via Ollama     │
└─────────────────┘    └──────────────────┘    │  (Port 11434)   │
                                │               └─────────────────┘
                                │ Context
                                ▼
                       ┌──────────────────┐
                       │   Final Answer   │
                       │    to User       │
                       └──────────────────┘
```

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 🎯 Deploy Rápido

### 1. Clone e Setup
```bash
git clone https://github.com/venysssssssssss/knowledge-base-refac.git
cd knowledge-base-refac
./scripts/setup.sh
```

### 2. Download do Modelo
```bash
./scripts/download_model.sh
```

### 3. Teste do Modelo
```bash
# Instalar dependências Python
pip install -r ai-services/requirements.txt

# Testar carregamento
python ai-services/inference/test_load.py
```

### 4. Start Completo
```bash
# Subir todos os serviços
docker compose up -d

# Verificar status
docker compose ps
```

## 🔧 Configuração Avançada

### Ajuste de GPU Memory
Edite `ai-services/inference/mistral_service.py`:
```python
# Para GPUs com menos VRAM
gpu_memory_utilization=0.7  # Em vez de 0.92

# Para múltiplas GPUs
tensor_parallel_size=2  # Para 2 GPUs
```

### Otimização de Performance
```python
# Mais sequências concorrentes (se tiver mais VRAM)
max_num_seqs=60

# Chunking mais agressivo
enable_chunked_prefill=True
max_num_batched_tokens=8192
```

## 🧪 Teste da API

### Health Check
```bash
curl http://localhost:8000/health
```

### Query de Teste
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "Como reiniciar um servidor?",
       "context": "Para reiniciar um servidor Linux, use o comando sudo reboot ou sudo systemctl reboot.",
       "max_tokens": 200
     }'
```

## 📊 Monitoramento

### Logs dos Serviços
```bash
# AI Service
docker compose logs -f ai-services

# Qdrant
docker compose logs -f qdrant

# Todos os serviços
docker compose logs -f
```

### Dashboards
- **Qdrant**: http://localhost:6333/dashboard
- **MinIO**: http://localhost:9001 (minioadmin/minioadmin123)
- **API Docs**: http://localhost:8000/docs

## 🔄 Operações

### Restart Serviços
```bash
# Restart AI service apenas
docker compose restart ai-services

# Restart completo
docker compose down && docker compose up -d
```

### Backup
```bash
# Backup dados Qdrant
tar -czf backup-qdrant-$(date +%Y%m%d).tar.gz data/qdrant/

# Backup MinIO
tar -czf backup-minio-$(date +%Y%m%d).tar.gz data/minio/
```

### Update Modelo
```bash
# Baixar nova versão
rm -rf models/mistral-7b-instruct-v0.2
./scripts/download_model.sh

# Restart AI service
docker compose restart ai-services
```

## 🚨 Troubleshooting

### Modelo não carrega
```bash
# Verificar CUDA
nvidia-smi

# Verificar espaço em disco
df -h

# Logs detalhados
docker compose logs ai-services | grep -i error
```

### Performance baixa
```bash
# Monitorar GPU
watch -n 1 nvidia-smi

# Verificar memory leaks
docker stats ai-services
```

### Qdrant connection issues
```bash
# Testar conexão
curl http://localhost:6333/health

# Restart Qdrant
docker compose restart qdrant
```

## 📈 Scaling

### Horizontal (Múltiplas Instâncias)
```yaml
# docker-compose.yml
ai-services:
  deploy:
    replicas: 2
```

### Load Balancer (Nginx)
```nginx
upstream ai_backend {
    server localhost:8000;
    server localhost:8001;
}

server {
    location /api/ai/ {
        proxy_pass http://ai_backend;
    }
}
```

## 🔐 Produção

### Segurança
```bash
# Mudar credenciais MinIO
export MINIO_ROOT_USER=your_admin
export MINIO_ROOT_PASSWORD=your_secure_password

# Firewall
sudo ufw allow 6333  # Qdrant (apenas interno)
sudo ufw allow 8000  # AI API
```

### SSL/HTTPS
```bash
# Nginx com Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```
