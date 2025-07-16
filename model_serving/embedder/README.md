# 🔗 vLLM Embedding Server

This project hosts the [AITeamVN/Vietnamese_Embedding_v2](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2) model via vLLM.

## Features ✨

- Embedding via vLLM Python library (`vllm.encode`)  
- FastAPI server exposing `/v1/embeddings` endpoint  
- OpenAI-compatible request/response schema  
- Production-ready structure with Docker support

## Setup 🚀

1. 🐍 (Optional) Create Python venv:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. 📦 Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. ⚙️ Set environment variables in `.env`:
   ```dotenv
   EMBEDDING_MODEL=AITeamVN/Vietnamese_Embedding_v2
   ```

4. 🏃‍♂️Run locally:
   ```bash
   uvicorn server_app:app --host 0.0.0.0 --port 8020 --loop asyncio
   ```

## Docker 🐳

🔨 Build and run with Docker Compose:
```bash
# Create external network (if not exists)
docker network create chatbot

# Start the service
docker compose up --env-file .env -d --build

# Check logs
docker compose logs -f embedder-server

# Stop the service
docker compose down
```

🔧 Manual Docker build:
```bash
docker build -t vllm-embed-server .
docker run -p 8020:8000 \
  -e EMBEDDING_MODEL=AITeamVN/Vietnamese_Embedding_v2 \
  vllm-embed-server
```

## Usage 📝

```bash
curl -X POST http://localhost:8020/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["xin chao", "toi la ChatGPT"]}'
```