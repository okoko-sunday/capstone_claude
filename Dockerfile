# Inference image: serves the web UI + API with a trained model baked in.
#   docker build -t catdog .
#   docker run -p 8000:8000 catdog
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /app

COPY requirements.txt .
RUN sed -i 's/^tensorflow>=/tensorflow-cpu>=/' requirements.txt \
 && pip install --no-cache-dir -r requirements.txt

COPY catdog ./catdog
COPY web ./web
COPY artifacts ./artifacts

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if b'ok' in urllib.request.urlopen('http://127.0.0.1:8000/api/health').read() else 1)"
CMD ["python", "-m", "catdog", "serve", "--host", "0.0.0.0", "--port", "8000"]
