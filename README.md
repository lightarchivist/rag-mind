# Rag-Mind

Ask questions across PDFs, books, and documents using a fully local
RAG pipeline — no cloud, no API keys.

## Setup Reference

### macOS (M1/M2/M3)
•	brew install ollama
•	ollama pull llama3 && ollama pull nomic-embed-text
•	git clone git@github.com:lightarchivist/rag-mind.git
•	cd rag-mind && python3 -m venv venv && source venv/bin/activate
•	pip install -r requirements.txt -r web/requirements-web.txt
•	./start.sh

### Docker (any platform)
•	docker pull lightarchivist/rag-mind:latest
•	ollama pull llama3 && ollama pull nomic-embed-text
•	docker run -p 8000:8000 -v ragmind_data:/data -e OLLAMA_BASE=http://host.docker.internal:11434 --add-host host.docker.internal:host-gateway lightarchivist/rag-mind:latest
•	Open http://localhost:8000

### Linux Installation (Ubuntu / Debian)

1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
ollama pull nomic-embed-text
ollama serve &

2. Install Docker
sudo apt update && sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER && newgrp docker

3. Clone the repository
git clone git@github.com:lightarchivist/rag-mind.git
cd rag-mind

4a. Run with Docker (recommended)
cd web
docker compose up --build -d

Open: http://localhost:8000 or http://<your-ip>:8000

4b. Run without Docker (Python direct)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r web/requirements-web.txt
chmod +x start.sh && ./start.sh

5. Or pull from Docker Hub (no build needed)
docker pull lightarchivist/rag-mind:latest
docker run -p 8000:8000 -v ragmind_data:/data \
  -e OLLAMA_BASE=http://host.docker.internal:11434 \
  --add-host host.docker.internal:host-gateway \
  lightarchivist/rag-mind:latest
