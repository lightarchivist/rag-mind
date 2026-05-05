#!/bin/bash

echo "🚀 Starting RAG-MIND (Linux)..."

DATA_DIR="/home/$(whoami)/ragmind-data"
echo "📁 Data directory: $DATA_DIR"

# Start Ollama if not running
if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "⚡ Starting Ollama..."
    ollama serve &
    sleep 5
else
    echo "✅ Ollama already running"
fi

# Activate venv
cd ~/GIT/rag-mind
source venv/bin/activate

# Start web app
echo "🌐 Starting RAG-MIND web app at http://$(hostname -I | awk '{print $1}'):8000"
cd web/app
DATA_DIR=$DATA_DIR \
OLLAMA_BASE=http://127.0.0.1:11434 \
uvicorn main:app --host 0.0.0.0 --port 8000
