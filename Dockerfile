FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini config.yaml ./

RUN pip install --upgrade pip \
    && pip install .

CMD ["newsletter-agent", "run"]
