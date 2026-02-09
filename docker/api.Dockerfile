FROM python:3.10-slim
WORKDIR /app
COPY api ./api
COPY shared ./shared
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=/app
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]