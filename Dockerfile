FROM python:3.11-slim

# System dependencies for OCR and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libgl1-mesa-glx \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/chroma data/uploads data

ENV PYTHONUNBUFFERED=1
ENV CHROMA_DB_PATH=/app/data/chroma
ENV SQLITE_DB_PATH=/app/data/psl.db
ENV UPLOAD_DIR=/app/data/uploads

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
