# 🐬 Dolphin Integration Setup

## Git Submodules

Este projeto usa o **ByteDance Dolphin** como um Git submodule para parsing avançado de documentos.

## Clonando o Projeto

### Para novos usuários:
```bash
# Clone com submodules
git clone --recursive https://github.com/venysssssssssss/knowledge-base-refac.git

# OU clone normal + inicializar submodules
git clone https://github.com/venysssssssssss/knowledge-base-refac.git
cd knowledge-base-refac
git submodule init
git submodule update
```

### Para repositórios existentes:
```bash
# Se você já tem o repo clonado
cd knowledge-base-refac
git submodule init
git submodule update
```

### Atualizando Submodules:
```bash
# Atualizar para a versão mais recente do Dolphin
git submodule update --remote external/dolphin
```

## Verificando se está funcionando

```bash
# Verificar se o Dolphin foi baixado
ls -la external/dolphin/

# Deve mostrar os arquivos do repositório Dolphin
# README.md, demo_page.py, requirements.txt, etc.
```

## Integração com o Sistema

O Dolphin está integrado no nosso sistema de duas formas:

1. **Parsing Avançado**: Para extrair tabelas, fórmulas e layout complexo de PDFs
2. **SentenceTransformers**: Para embeddings de alta qualidade (sistema principal)

## Estrutura

```
knowledge-base-refac/
├── external/
│   └── dolphin/                 # Git submodule (ByteDance Dolphin)
│       ├── demo_page.py
│       ├── requirements.txt
│       └── ...
├── ai-services/
│   ├── document-processor/      # Usa SentenceTransformers + opcionalmente Dolphin
│   ├── inference/               # Mistral via Ollama
│   └── rag/                     # Orquestração RAG
└── scripts/
    └── install_dependencies.sh  # Setup automático
```

## Troubleshooting

### Submodule vazio:
```bash
git submodule init
git submodule update
```

### Erro de permissões:
```bash
chmod +x scripts/install_dependencies.sh
```

### Atualizar Dolphin:
```bash
git submodule update --remote external/dolphin
git add external/dolphin
git commit -m "Update Dolphin submodule"
```
