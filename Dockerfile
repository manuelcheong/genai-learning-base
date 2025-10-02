FROM python:3.12-slim

WORKDIR /app
ARG GIT_TOKEN


RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install uv && \
    uv venv && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt && \
    uv clean
ENV PATH="/app/.venv/bin:$PATH"