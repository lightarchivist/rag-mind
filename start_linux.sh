#!/bin/bash
# Start RAG-MIND on Linux. Works from wherever the repo was cloned.
set -e

echo " Starting RAG-MIND (Linux)..."

# Data folder: override with DATA_DIR=/some/path ./start_linux.sh
DATA_DIR="${DATA_DIR:-$HOME/ragmind-data}"
echo " Data directory: $DATA_DIR"

# Start Ollama if not running
if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "⚡ Starting Ollama..."
    ollama serve &
    sleep 5
else
    echo " Ollama already running"
fi

# Run from the repo folder (the folder this script lives in)
cd "$(dirname "$0")"

# Activate the venv if there is one
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

# Start web app
echo " Starting RAG-MIND web app at http://$(hostname -I | awk '{print $1}'):8000"
cd web/app
DATA_DIR="$DATA_DIR" \
OLLAMA_BASE=http://127.0.0.1:11434 \
uvicorn main:app --host 0.0.0.0 --port 8000
