FROM mcr.microsoft.com/playwright/python:v1.50.0-noble

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for caching
COPY pyproject.toml .

# Install dependencies (no venv needed in container)
RUN uv pip install --system --no-cache -e "."

# Install Playwright chromium browser
RUN python -m playwright install chromium

# Copy application code
COPY . .

# Create data directory for SQLite
RUN mkdir -p /data

# Default environment variables
ENV DB_PATH=/data/poly_weather.db
ENV PAPER_MODE=true
ENV LOG_LEVEL=INFO
ENV PORT=8000

EXPOSE 8000

CMD ["python", "main.py"]
