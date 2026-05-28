cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код API
COPY api/ ./api/
COPY models/ ./models/

# Создаем папку для моделей
RUN mkdir -p /app/models

# Запускаем сервер
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]
EOF
