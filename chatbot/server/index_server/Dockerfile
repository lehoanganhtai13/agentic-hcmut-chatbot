FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends iputils-ping && \
    apt-get install -y curl && \
    apt-get clean

RUN pip install --upgrade pip

COPY ./server/requirements.txt .
RUN pip install uv
RUN uv pip install --system --no-cache-dir -r requirements.txt \
    && uv pip install --system --no-cache-dir 'nltk>=3.9.1' \
    && uv pip install --system --no-cache-dir 'uvicorn[standard]'

ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python - <<'PY'
import nltk, os, pathlib
dst=os.environ["NLTK_DATA"]; pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
nltk.download('punkt', download_dir=dst, quiet=True)
nltk.download('punkt_tab', download_dir=dst, quiet=True)
nltk.download('stopwords', download_dir=dst, quiet=True)
PY

# RUN pip install docling==2.31.0

COPY ./config /app/chatbot/config
COPY ./core /app/chatbot/core
COPY ./indexing /app/chatbot/indexing
COPY ./prompts /app/chatbot/prompts
COPY ./utils /app/chatbot/utils
COPY ./workflow /app/chatbot/workflow
COPY ./server/index_server /app/chatbot/server/index_server

EXPOSE 8000
