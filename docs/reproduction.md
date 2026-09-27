# Reproducibility Guide

This guide documents the minimum steps for another developer to reproduce the local demo.

## Requirements
- Python 3.13
- Ollama
- `qwen3:8b`
- `nomic-embed-text`
- TigerGraph Savanna for GraphRAG/Agentic GraphRAG

## Setup
```powershell
python -m venv .venv

# Windows:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

ollama pull qwen3:8b
ollama pull nomic-embed-text
```

## Environment
Create a `.env` file in the project root:
```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434

TIGERGRAPH_HOST=https://your-instance.i.tgcloud.io
TIGERGRAPH_GRAPH=GRAPHRAG
TIGERGRAPH_SECRET=your_gsql_secret
```
*(Never include a real secret in documentation or source control).*

## Run
From the project root:
```powershell
$env:PYTHONPATH="."
python dashboard/app.py
```
Then navigate to: `http://localhost:8080`

*(Note: The complete three-way comparison requires TigerGraph connectivity).*
