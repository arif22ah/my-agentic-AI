FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY my_agent ./my_agent
RUN pip install . pytest pytest-timeout ruff numpy

EXPOSE 8000
# Runs as root so it can talk to the mounted Docker socket (see README: security notes).
CMD ["uvicorn", "my_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
