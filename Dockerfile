FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Default command: show usage or run api
CMD ["uvicorn", "module_4_decision_dashboard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
