# Build from the repository root (the image needs the shared ragmind/ package):
#   docker build -f web/Dockerfile -t lightarchivist/rag-mind .
# or from web/:  docker compose up --build -d
FROM python:3.12-slim

WORKDIR /app

COPY web/requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY ragmind/ ./ragmind/
COPY CHANGELOG.md ./
COPY web/app/ ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
