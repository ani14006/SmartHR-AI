FROM python:3.11-slim

WORKDIR /app

# System libs needed by opencv-python-headless on Debian slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects $PORT at runtime — shell form so the variable is expanded
CMD gunicorn app:app --workers 1 --threads 4 --timeout 300 --bind 0.0.0.0:$PORT
