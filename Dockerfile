FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry

# Copy pyproject.toml
COPY pyproject.toml /app/

# Install dependencies (disable virtualenv to install globally in docker)
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only main --no-interaction --no-ansi

# Copy the source code
COPY . /app

# Install the package itself in non-editable mode
RUN poetry install --only main --no-interaction --no-ansi

# Run the CLI by default
ENTRYPOINT ["python", "-m", "model_inference.inference"]
