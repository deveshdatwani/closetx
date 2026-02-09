FROM python:3.10-slim
WORKDIR /app
COPY worker ./worker
COPY api ./api
COPY shared ./shared
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=/app
CMD ["celery","-A","worker.celery_app.app","worker","--loglevel=info"]