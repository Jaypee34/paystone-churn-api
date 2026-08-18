# =============================================================================
# PAYSTONE CUSTOMER CHURN PREDICTION
# FASTAPI DOCKER DEPLOYMENT
# =============================================================================

# Use lightweight Python 3.11 image
FROM python:3.11-slim

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Display Python output immediately in Docker logs
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# =============================================================================
# INSTALL PYTHON DEPENDENCIES
# =============================================================================

# Copy requirements file
COPY requirements.txt .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# COPY APPLICATION FILES
# =============================================================================

# Copy FastAPI application
COPY app.py .

# Copy feature configuration
COPY feature_columns.json .

# Copy trained CatBoost model
COPY model ./model

# =============================================================================
# FASTAPI CONFIGURATION
# =============================================================================

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI with Uvicorn
# 0.0.0.0 allows external connections
# =============================================================================
# FASTAPI CONFIGURATION
# =============================================================================

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]