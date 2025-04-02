FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY ./../requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends iputils-ping && \
    apt-get clean

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install 'uvicorn[standard]'

COPY ./config /app/chatbot/config
COPY ./core /app/chatbot/core
COPY ./prompt /app/chatbot/prompt
COPY ./utils /app/chatbot/utils
COPY ./workflow /app/chatbot/workflow
COPY ./server /app/chatbot/server

EXPOSE 8000
