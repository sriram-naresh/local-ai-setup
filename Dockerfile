FROM python:3.13-slim

# ============================================================
# ENVIRONMENT
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ============================================================
# WORKING DIRECTORY
# ============================================================

WORKDIR /app

# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# APPLICATION FILES
# ============================================================

COPY app.py .

# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

EXPOSE 8501

# ============================================================
# START APPLICATION
# ============================================================

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]