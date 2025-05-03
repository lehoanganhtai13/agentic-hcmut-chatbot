FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    # Download the Google signing key
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    # Add the Google Chrome repository to the sources list
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    # Update the package list after adding the repo
    && apt-get update \
    # Install Google Chrome Stable
    && apt-get install -y google-chrome-stable \
    # All dependencies for Chrome and Chromedriver
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    libfreetype6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libpango-1.0-0 \
    libasound2 \
    # Clean up cache to reduce image size
    && rm -rf /var/lib/apt/lists/* \
    # Remove the Google Chrome repository from the sources list
    && rm -f /etc/apt/sources.list.d/google-chrome.list

COPY ./requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends iputils-ping && \
    apt-get clean

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install fastapi 'uvicorn[standard]'

COPY . /app/web_crawler

EXPOSE 8000
