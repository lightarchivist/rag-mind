# RAG-MIND 

> Started as an LLM CLI for academic paper analysis — now a full local RAG system for any document — no cloud, no API keys, your data stays yours.

Ask natural language questions across PDFs, EPUBs, books, and documents using Llama 3 running entirely on your own machine.

[![Licence: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-lightarchivist%2Frag--mind-blue)](https://hub.docker.com/r/lightarchivist/rag-mind)

## Acknowledgements

What started as a small assignment at ATU has grown into a fully functional local RAG system — and may continue to evolve. Developed with AI assistance. All code has been reviewed, tested, and deployed by the author.

---

## Features

-  Upload PDF, EPUB, DOCX, TXT, Markdown
-  Semantic search using nomic-embed-text embeddings
-  Streaming answers grounded strictly in your documents
-  Index multiple documents, switch context per conversation
-  Access from any device on your local network
-  100% local — nothing leaves your machine
-  CLI and web interface included

---

## Quick Start

### macOS (M1/M2/M3)

```bash
brew install ollama
ollama pull llama3
ollama pull nomic-embed-text

git clone git@github.com:lightarchivist/rag-mind.git
cd rag-mind
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r web/requirements-web.txt
./start.sh
```

Open `http://localhost:8000`

---

### Docker (any platform)

```bash
ollama pull llama3
ollama pull nomic-embed-text

docker pull lightarchivist/rag-mind:latest

docker run -p 8000:8000 \
  -v ragmind_data:/data \
  -e OLLAMA_BASE=http://host.docker.internal:11434 \
  --add-host host.docker.internal:host-gateway \
  lightarchivist/rag-mind:latest
```

Open `http://localhost:8000`

---

### Linux (Ubuntu / Debian)

**1. Install Ollama**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
ollama pull nomic-embed-text
ollama serve &
```

**2. Install Docker**
```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER && newgrp docker
```

**3. Clone the repo**
```bash
git clone git@github.com:lightarchivist/rag-mind.git
cd rag-mind
```

**4a. Run with Docker (recommended)**
```bash
cd web
docker compose up --build -d
```

Open `http://localhost:8000` or `http://<your-ip>:8000`

**4b. Run without Docker**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r web/requirements-web.txt
chmod +x start.sh && ./start.sh
```

---

## Project Structure

```
rag-mind/
├── rag_cli.py              ← CLI version
├── requirements.txt        ← CLI dependencies
├── start.sh                ← Startup script
├── web/                    ← Web app (Docker)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements-web.txt
│   └── app/
│       ├── main.py         ← FastAPI backend
│       └── static/
│           └── index.html  ← Web UI
└── pdf/                    ← Drop books here (gitignored)
```

---

## CLI Usage

```bash
source venv/bin/activate

python rag_cli.py add paper.pdf
python rag_cli.py folder ./my_papers
python rag_cli.py chat
python rag_cli.py list
python rag_cli.py clear
```

---

## Licence Notes

- **RAG-MIND** — GNU General Public License v3.0
- **Llama 3** — [Meta Community Licence](https://llama.meta.com/llama3/license/) — requires attribution to Meta
- **ebooklib** (EPUB support) — AGPL-3.0 — optional dependency
- All other dependencies — MIT / Apache 2.0 / BSD

---

## Repositories

| | |
|---|---|
| Source code | github.com/lightarchivist/rag-mind |
| Docker image | hub.docker.com/r/lightarchivist/rag-mind |
