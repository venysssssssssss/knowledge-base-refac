# Dockerfile para Mistral Service
FROM python:3.12-slim

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
COPY . .

# Instala Poetry
RUN pip install poetry

# Instala dependências do projeto
RUN poetry install --no-root


EXPOSE 8003

CMD ["poetry", "run", "python", "mistral_service.py"]
