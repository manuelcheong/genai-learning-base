FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install uv && \
    uv venv && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt && \
    uv clean
ENV PATH="/app/.venv/bin:$PATH"