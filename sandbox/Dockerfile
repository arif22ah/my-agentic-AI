# Image in which LLM-generated code and its tests are executed.
# CPU-only PyTorch keeps it small; the container runs with no network.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install numpy pytest pytest-timeout ruff

# World-readable work dir; files are streamed in at runtime and run as `nobody`.
RUN mkdir /work && chmod 755 /work
WORKDIR /work
