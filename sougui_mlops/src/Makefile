# Makefile pour sougui_mlops

.PHONY: help run build docker-build docker-run docker-stop clean test

help:
	@echo "Commandes disponibles :"
	@echo "  make run          - Lancer l'API localement"
	@echo "  make build        - Construire l'image Docker"
	@echo "  make docker-run   - Lancer le conteneur Docker"
	@echo "  make docker-stop  - Arrêter le conteneur Docker"
	@echo "  make mlflow       - Lancer l'interface MLflow"
	@echo "  make test         - Tester l'API"
	@echo "  make clean        - Nettoyer les fichiers temporaires"

# Lancer l'API localement
run:
	uvicorn src.app:app --reload --port 8000

# Lancer MLflow
mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

# Construire l'image Docker
docker-build:
	docker build -t sougui_mlops:v1 .

# Lancer le conteneur Docker
docker-run:
	docker run -d -p 8000:8000 --name sougui_app sougui_mlops:v1

# Arrêter le conteneur Docker
docker-stop:
	docker stop sougui_app
	docker rm sougui_app

# Tester l'API
test:
	curl.exe http://localhost:8000/health

# Nettoyer
clean:
	rm -rf __pycache__
	rm -rf src/__pycache__