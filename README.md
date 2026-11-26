# RAG-MIND 

> Started as an LLM CLI for academic paper analysis — now a full local RAG system for any document — no cloud, no API keys, your data stays yours.

Ask natural language questions across PDFs, EPUBs, books, and documents using Llama 3 running entirely on your own machine.

[![Licence: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-lightarchivist%2Frag--mind-blue)](https://hub.docker.com/r/lightarchivist/rag-mind)
[![Version](https://img.shields.io/badge/version-2.1.0-green)](CHANGELOG.md)

**Version 2.1.0.** See [CHANGELOG.md](CHANGELOG.md) for what's new and what was fixed in each
version. The web app shows the current version and its changes under **About**, and the CLI
prints it with `python rag_cli.py --version`.

## Acknowledgements

What started as a small assignment at ATU has grown into a fully functional local RAG system — and may continue to evolve. Developed with AI assistance. All code has been reviewed, tested, and deployed by the author.

---

## Features

-  Upload PDF, EPUB, DOCX, TXT, Markdown
-  Semantic search using nomic-embed-text embeddings
-  Streaming answers grounded strictly in your documents, citing file and page
-  Switch between any chat models installed in Ollama
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

git clone https://github.com/lightarchivist/rag-mind.git
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
git clone https://github.com/lightarchivist/rag-mind.git
cd rag-mind
```

**4a. Run with Docker (recommended)**
```bash
cd web
docker compose up --build -d
```

Open `http://localhost:8000` or `http://<your-ip>:8000`

To build the image yourself instead (from the repo root):
```bash
docker build --network=host -f web/Dockerfile -t lightarchivist/rag-mind:latest .
```
`--network=host` avoids `pip install` timeouts that some Linux setups hit inside
Docker builds; the compose file already uses it.

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
├── ragmind/                ← Shared core used by the CLI and the web app
│   ├── config.py           ← Settings (defaults → config file → env vars)
│   ├── ollama.py           ← Ollama client: models, embeddings, streaming, clear errors
│   ├── extract.py          ← PDF / DOCX / EPUB / HTML / text extraction (with page numbers)
│   ├── chunking.py         ← Overlapping, sentence-aware chunks
│   ├── prompt.py           ← Prompt building within the model's context window
│   ├── pipeline.py         ← Retrieval: follow-up rewriting, search, filtering, grading
│   ├── numpy_index.py      ← The web app's vector index
│   ├── thinking.py         ← Hides reasoning models' thinking
│   └── changelog.py        ← Reads CHANGELOG.md for the About box
├── rag_cli.py              ← CLI version (ChromaDB)
├── requirements.txt        ← CLI dependencies
├── CHANGELOG.md            ← What changed in each version
├── start.sh                ← Startup script
├── tests/                  ← Offline tests (fake Ollama; no models needed)
├── web/                    ← Web app (Docker)
│   ├── Dockerfile          ← build from the repo root: docker build -f web/Dockerfile .
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

python rag_cli.py add paper.pdf book.epub     # a bad file is reported and skipped
python rag_cli.py folder ./my_papers -r
python rag_cli.py chat                        # or: chat --model qwen2.5
python rag_cli.py list
python rag_cli.py reindex                     # rebuild after upgrading or changing EMBED_MODEL
python rag_cli.py clear                       # asks first; --yes to skip
```

Answers cite the file (and page, for PDFs). Follow-up questions such as
"and its weight?" are rewritten into a full question before searching, and the
rewritten search is shown.

### Choosing the model

Any chat model installed in Ollama can answer questions:

```bash
python rag_cli.py models                      # installed chat models, * = current
python rag_cli.py models --default qwen2.5    # remember a default
python rag_cli.py chat --model mistral        # this session only
```

Inside a chat, `/model` lists the models and `/model qwen2.5` switches (and
remembers the choice). In the web app, pick the model from the menu beside the
status light at the top of the page; your choice is remembered in the browser.
To get more models: `ollama pull <name>` (see https://ollama.com/library).

The embedding model (`EMBED_MODEL`, default `nomic-embed-text`) is different:
changing it makes the existing index unusable, so RagMind detects the change and
asks you to rebuild (`reindex` in the CLI, **Rebuild index** in the web app).

### Settings

Environment variables override `~/.ragmind/config.json` (the web app reads
`$DATA_DIR/config.json`).

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE` | `http://localhost:11434` | Ollama address |
| `LLM_MODEL` | `llama3` | Default chat model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model (changing it needs a rebuild) |
| `RAG_CLI_DB` | `~/.rag_cli_db` | CLI index folder |
| `RAGMIND_MAX_CTX` | `8192` | Largest context window to request (more = more memory) |
| `RAGMIND_TOP_K` | `6` | Passages retrieved per question |
| `RAGMIND_MIN_SCORE` | `0.25` | Ignore passages less similar than this |
| `RAGMIND_REWRITE` | `1` | Rewrite follow-up questions before searching |
| `RAGMIND_GRADE` | `0` | Ask the model to filter passages (more accurate, slower) |
| `RAGMIND_TIMEOUT` | `600` | Seconds to wait for Ollama |

### Upgrading from 1.x

Indexes built by earlier versions are detected and must be rebuilt once
(`python rag_cli.py reindex`, or **Rebuild index** in the web app), because
vectors are now made with the embedding model's task prefixes, which improves
search. Files that have moved keep their stored text.

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests use a fake Ollama server, so no models are needed.

---

## Releasing a new version

1. Choose the number: **MAJOR** when users must act (e.g. rebuild the index),
   **MINOR** for new features, **PATCH** for fixes only.
2. Add an entry at the top of `CHANGELOG.md` (`## [x.y.z] - YYYY-MM-DD`, with
   `### Added` / `### Changed` / `### Fixed` lists).
3. Set the same number in `ragmind/__init__.py` (`__version__`) and in the version
   badge at the top of this README. The tests check that the changelog and
   `__version__` agree.
4. Commit, tag and push:
   ```bash
   python -m pytest
   git commit -am "Release x.y.z"
   git tag vx.y.z
   git push && git push --tags
   ```
5. Publish the Docker image under both the version and `latest`:
   ```bash
   docker build --network=host -f web/Dockerfile \
     -t lightarchivist/rag-mind:x.y.z -t lightarchivist/rag-mind:latest .
   docker push lightarchivist/rag-mind:x.y.z
   docker push lightarchivist/rag-mind:latest
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
