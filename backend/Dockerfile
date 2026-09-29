FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY backend/ backend/
COPY src/models/health_score_and_resale.py src/models/health_score_and_resale.py
COPY saved_models/ saved_models/
EXPOSE 8000
CMD ["/bin/sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
