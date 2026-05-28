.PHONY: help build up down train test health clean

help:
	@echo "Available commands:"
	@echo "  make build   - Build Docker images"
	@echo "  make up      - Start all services"
	@echo "  make down    - Stop all services"
	@echo "  make train   - Train model"
	@echo "  make test    - Test API endpoints"
	@echo "  make health  - Check API health"
	@echo "  make clean   - Clean temporary files"

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "Services started. API available at http://localhost:8000"

down:
	docker-compose down

train:
	docker-compose exec api python src/train.py

test:
	curl -s http://localhost:8000/health | jq .
	@echo "\nTesting prediction endpoint..."
	curl -s -X POST http://localhost:8000/predict \
	  -H "Content-Type: application/json" \
	  -d '{"features": [0.5, 0.3, -0.2, 1.0, -0.5], "customer_id": "test123"}' | jq .

health:
	curl -s http://localhost:8000/health | jq .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
