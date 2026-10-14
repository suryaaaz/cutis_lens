# UNTESTED template for Hugging Face Spaces (Docker) / generic hosting. Verify before relying on it.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
ENV PORT=7860
EXPOSE 7860
CMD gunicorn "app:create_app()" --bind 0.0.0.0:${PORT} --workers 1 --timeout 120
