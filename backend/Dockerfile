# Stage 1: Build the frontend Angular application
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Create the final Python application image
FROM python:3.10

WORKDIR /app

# Copy and install dependencies first
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary backend application and MLOps pipeline files
COPY backend/config.yml .
COPY backend/main.py .
COPY backend/app.py .
COPY backend/steps/ ./steps/
COPY backend/models/ ./models/
COPY backend/data/ ./data/

# Copy compiled Angular browser bundle from the frontend builder stage
COPY --from=frontend-builder /frontend/dist/frontend/browser/ ./frontend-dist/

# Expose FastAPI server port
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


