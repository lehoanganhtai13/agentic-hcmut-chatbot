# 🔗 vLLM Embedding Server

This project hosts the [AITeamVN/Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker) model via vLLM.

## Features ✨

- Embedding via vLLM Python library (`vllm.score`)  
- FastAPI server exposing `/v1/reranking` endpoint  
- Automatic sorting by relevance score (descending)
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
   RERANKER_MODEL=AITeamVN/Vietnamese_Reranker
   ```

4. 🏃‍♂️Run locally:
   ```bash
   uvicorn server_app:app --host 0.0.0.0 --port 8030 --loop asyncio
   ```

## Docker 🐳

🔨 Build and run with Docker Compose:
```bash
# Create external network (if not exists)
docker network create chatbot

# Start the service
docker compose --env-file .env up -d --build

# Check logs
docker compose logs -f reranker-server

# Stop the service
docker compose down
```

🔧 Manual Docker build:
```bash
docker build -t vllm-embed-server .
docker run -p 8030:8000 \
  -e RERANKER_MODEL=AITeamVN/Vietnamese_Reranker \
  vllm-embed-server
```

## Usage 📝

```bash
curl -X POST http://localhost:8030/v1/reranking \
   -H "Content-Type: application/json" \  
   -d '{
      "query": "What is artificial intelligence?",
      "documents": [
      "AI is a field of computer science",
      "Cats are cute animals",
      "Machine learning is a subset of AI"
      ]
   }'
```